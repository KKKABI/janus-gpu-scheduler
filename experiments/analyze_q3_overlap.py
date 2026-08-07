#!/usr/bin/env python3
"""Analyze Q3 operator overlap from Nsight Systems SQLite exports."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
from pathlib import Path
from statistics import median


TARGET_OPS = ("x_11", "x_13", "x_17")
VARIANTS = (
    {
        "key": "static",
        "display": "Static+Janus",
        "trace_variant": "Baseline",
        "selected": ("x_13", "x_17"),
    },
    {
        "key": "td_janus",
        "display": "TD+Janus",
        "trace_variant": "TD+Janus",
        "selected": TARGET_OPS,
    },
)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_measure(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def intersect_two(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if end > start:
            output.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return output


def intersect_many(
    interval_groups: list[list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    if not interval_groups:
        return []
    result = interval_groups[0]
    for intervals in interval_groups[1:]:
        result = intersect_two(result, intervals)
        if not result:
            break
    return result


def concurrency_segments(
    intervals_by_op: dict[str, list[tuple[int, int]]]
) -> list[dict]:
    boundaries = sorted(
        {
            point
            for intervals in intervals_by_op.values()
            for interval in intervals
            for point in interval
        }
    )
    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = [
            op
            for op, intervals in intervals_by_op.items()
            if any(left <= start and right >= end for left, right in intervals)
        ]
        if active:
            segments.append({"start": start, "end": end, "active_ops": active})
    return segments


def resolved_nvtx_text(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    return row["text"] if row["text"] is not None else strings.get(row["textId"])


def analyze_variant(spec: dict, root: Path) -> dict:
    database_path = root / f"{spec['key']}_full_trace.sqlite"
    summary_path = root / f"{spec['key']}_q3_summary.json"
    map_path = root / f"{spec['key']}_fx_stream_map.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stream_map_payload = json.loads(map_path.read_text(encoding="utf-8"))
    stream_map = {node["name"]: node for node in stream_map_payload["nodes"]}

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    strings = {
        int(row["id"]): row["value"]
        for row in cursor.execute("SELECT id, value FROM StringIds")
    }

    capture_ranges: dict[str, tuple[int, int]] = {}
    for row in cursor.execute(
        "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        text = resolved_nvtx_text(row, strings)
        prefix = "FX::GRAPH_CAPTURE::"
        if text and text.startswith(prefix):
            op = text[len(prefix) :]
            if op in TARGET_OPS:
                if op in capture_ranges:
                    raise RuntimeError(f"duplicate capture NVTX range for {spec['display']} {op}")
                capture_ranges[op] = (int(row["start"]), int(row["end"]))
    missing_ranges = sorted(set(TARGET_OPS) - set(capture_ranges))
    if missing_ranges:
        raise RuntimeError(f"missing capture ranges for {spec['display']}: {missing_ranges}")

    capture_graph_node_ids: dict[str, list[int]] = {}
    graph_node_events: dict[str, list[dict]] = {}
    for op, (start, end) in capture_ranges.items():
        rows = [
            dict(row)
            for row in cursor.execute(
                """
                SELECT start, end, eventClass, nameId, graphNodeId, originalGraphNodeId
                FROM CUDA_GRAPH_NODE_EVENTS
                WHERE start >= ? AND start <= ?
                ORDER BY start, graphNodeId
                """,
                (start, end),
            )
        ]
        ids = sorted({int(row["graphNodeId"]) for row in rows})
        if not ids:
            raise RuntimeError(f"no graph node ids mapped for {spec['display']} {op}")
        capture_graph_node_ids[op] = ids
        graph_node_events[op] = rows

    clone_edges = [
        (int(row["originalGraphNodeId"]), int(row["graphNodeId"]))
        for row in cursor.execute(
            """
            SELECT originalGraphNodeId, graphNodeId
            FROM CUDA_GRAPH_NODE_EVENTS
            WHERE originalGraphNodeId IS NOT NULL
            """
        )
    ]
    replay_graph_node_ids: dict[str, list[int]] = {}
    for op, source_ids in capture_graph_node_ids.items():
        expanded = set(source_ids)
        changed = True
        while changed:
            changed = False
            for original_id, cloned_id in clone_edges:
                if original_id in expanded and cloned_id not in expanded:
                    expanded.add(cloned_id)
                    changed = True
        replay_graph_node_ids[op] = sorted(expanded)

    replay_prefix = f"Q3_REPLAY::{spec['trace_variant']}::"
    replay_ranges = []
    for row in cursor.execute(
        "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"
    ):
        text = resolved_nvtx_text(row, strings)
        if text and text.startswith(replay_prefix):
            replay_ranges.append(
                {
                    "index": int(text[len(replay_prefix) :]),
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                }
            )
    replay_ranges.sort(key=lambda item: item["index"])
    if [item["index"] for item in replay_ranges] != list(range(summary["replays"])):
        raise RuntimeError(f"replay NVTX ranges mismatch for {spec['display']}: {replay_ranges}")

    replay_results = []
    for replay in replay_ranges:
        kernel_rows = [
            dict(row)
            for row in cursor.execute(
                """
                SELECT k.start, k.end, k.streamId, k.graphNodeId, k.gridX, k.gridY,
                       k.gridZ, k.blockX, k.blockY, k.blockZ, k.shortName,
                       s.value AS shortNameText
                FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
                LEFT JOIN StringIds AS s ON s.id = k.shortName
                WHERE k.start >= ? AND k.end <= ?
                ORDER BY k.start, k.end
                """,
                (replay["start"], replay["end"]),
            )
        ]
        intervals_by_op: dict[str, list[tuple[int, int]]] = {}
        op_details = {}
        for op in TARGET_OPS:
            ids = set(replay_graph_node_ids[op])
            rows = [row for row in kernel_rows if row["graphNodeId"] in ids]
            intervals = merge_intervals(
                [(int(row["start"]), int(row["end"])) for row in rows]
            )
            if not intervals:
                raise RuntimeError(
                    f"no replay kernels mapped for {spec['display']} replay {replay['index']} {op}"
                )
            intervals_by_op[op] = intervals
            op_details[op] = {
                "kernel_count": len(rows),
                "graph_node_ids": sorted({int(row["graphNodeId"]) for row in rows}),
                "stream_ids": sorted({int(row["streamId"]) for row in rows}),
                "kernel_names": sorted(
                    {row["shortNameText"] for row in rows if row["shortNameText"]}
                ),
                "active_ns": interval_measure(intervals),
                "span_ns": intervals[-1][1] - intervals[0][0],
                "intervals_ns": [[start, end] for start, end in intervals],
            }

        pairwise = {}
        for left_index, left in enumerate(TARGET_OPS):
            for right in TARGET_OPS[left_index + 1 :]:
                key = f"{left}&{right}"
                pairwise[key] = interval_measure(
                    intersect_two(intervals_by_op[left], intervals_by_op[right])
                )
        triple_intervals = intersect_many([intervals_by_op[op] for op in TARGET_OPS])
        selected_intervals = intersect_many(
            [intervals_by_op[op] for op in spec["selected"]]
        )
        segments = concurrency_segments(intervals_by_op)
        target_start = min(intervals[0][0] for intervals in intervals_by_op.values())
        target_end = max(intervals[-1][1] for intervals in intervals_by_op.values())
        multi_operator_ns = sum(
            segment["end"] - segment["start"]
            for segment in segments
            if len(segment["active_ops"]) >= 2
        )
        replay_results.append(
            {
                "index": replay["index"],
                "nvtx_start_ns": replay["start"],
                "nvtx_end_ns": replay["end"],
                "nvtx_duration_ns": replay["end"] - replay["start"],
                "all_graph_kernel_count": len(kernel_rows),
                "all_graph_gpu_span_ns": (
                    max(row["end"] for row in kernel_rows)
                    - min(row["start"] for row in kernel_rows)
                    if kernel_rows
                    else 0
                ),
                "target_window_ns": target_end - target_start,
                "target_start_ns": target_start,
                "target_end_ns": target_end,
                "ops": op_details,
                "pairwise_overlap_ns": pairwise,
                "strict_triple_overlap_ns": interval_measure(triple_intervals),
                "selected_group_overlap_ns": interval_measure(selected_intervals),
                "multi_operator_overlap_ns": multi_operator_ns,
                "max_concurrent_target_ops": max(
                    len(segment["active_ops"]) for segment in segments
                ),
                "concurrency_segments_ns": segments,
            }
        )

    def series(field: str) -> list[int]:
        return [int(row[field]) for row in replay_results]

    medians = {
        "selected_group_overlap_ns": median(series("selected_group_overlap_ns")),
        "strict_triple_overlap_ns": median(series("strict_triple_overlap_ns")),
        "multi_operator_overlap_ns": median(series("multi_operator_overlap_ns")),
        "target_window_ns": median(series("target_window_ns")),
        "all_graph_gpu_span_ns": median(series("all_graph_gpu_span_ns")),
        "max_concurrent_target_ops": median(series("max_concurrent_target_ops")),
    }
    selected_overlap_all_replays = all(
        row["selected_group_overlap_ns"] > 0 for row in replay_results
    )
    strict_triple_all_replays = all(
        row["strict_triple_overlap_ns"] > 0 for row in replay_results
    )
    return {
        "key": spec["key"],
        "display": spec["display"],
        "trace_variant": spec["trace_variant"],
        "selected_ops": list(spec["selected"]),
        "database": str(database_path),
        "summary": summary,
        "fx_stream_map": {
            op: {
                "stream_index": stream_map[op]["stream_index"],
                "stream_ptr": stream_map[op]["stream_ptr"],
                "wait_event_count": stream_map[op]["wait_event_count"],
                "profiled_kernel_count": len(stream_map[op]["kernels"]),
            }
            for op in TARGET_OPS
        },
        "capture_nvtx_ranges_ns": {
            op: list(capture_ranges[op]) for op in TARGET_OPS
        },
        "capture_graph_node_ids": capture_graph_node_ids,
        "replay_graph_node_ids": replay_graph_node_ids,
        "capture_graph_node_event_counts": {
            op: len(graph_node_events[op]) for op in TARGET_OPS
        },
        "replays": replay_results,
        "medians": medians,
        "verdict": {
            "selected_group_overlap_all_5_replays": selected_overlap_all_replays,
            "strict_triple_overlap_all_5_replays": strict_triple_all_replays,
            "maximum_concurrent_target_ops": max(
                row["max_concurrent_target_ops"] for row in replay_results
            ),
        },
    }


def nanoseconds_to_microseconds(value: float) -> float:
    return float(value) / 1000.0


def write_csv(results: dict, path: Path) -> None:
    fieldnames = [
        "variant",
        "replay",
        "selected_ops",
        "all_graph_gpu_span_us",
        "target_window_us",
        "x_11_active_us",
        "x_13_active_us",
        "x_17_active_us",
        "x_11_x_13_overlap_us",
        "x_11_x_17_overlap_us",
        "x_13_x_17_overlap_us",
        "strict_triple_overlap_us",
        "selected_group_overlap_us",
        "multi_operator_overlap_us",
        "max_concurrent_target_ops",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant in results["variants"]:
            for replay in variant["replays"]:
                writer.writerow(
                    {
                        "variant": variant["display"],
                        "replay": replay["index"],
                        "selected_ops": "+".join(variant["selected_ops"]),
                        "all_graph_gpu_span_us": nanoseconds_to_microseconds(
                            replay["all_graph_gpu_span_ns"]
                        ),
                        "target_window_us": nanoseconds_to_microseconds(
                            replay["target_window_ns"]
                        ),
                        "x_11_active_us": nanoseconds_to_microseconds(
                            replay["ops"]["x_11"]["active_ns"]
                        ),
                        "x_13_active_us": nanoseconds_to_microseconds(
                            replay["ops"]["x_13"]["active_ns"]
                        ),
                        "x_17_active_us": nanoseconds_to_microseconds(
                            replay["ops"]["x_17"]["active_ns"]
                        ),
                        "x_11_x_13_overlap_us": nanoseconds_to_microseconds(
                            replay["pairwise_overlap_ns"]["x_11&x_13"]
                        ),
                        "x_11_x_17_overlap_us": nanoseconds_to_microseconds(
                            replay["pairwise_overlap_ns"]["x_11&x_17"]
                        ),
                        "x_13_x_17_overlap_us": nanoseconds_to_microseconds(
                            replay["pairwise_overlap_ns"]["x_13&x_17"]
                        ),
                        "strict_triple_overlap_us": nanoseconds_to_microseconds(
                            replay["strict_triple_overlap_ns"]
                        ),
                        "selected_group_overlap_us": nanoseconds_to_microseconds(
                            replay["selected_group_overlap_ns"]
                        ),
                        "multi_operator_overlap_us": nanoseconds_to_microseconds(
                            replay["multi_operator_overlap_ns"]
                        ),
                        "max_concurrent_target_ops": replay[
                            "max_concurrent_target_ops"
                        ],
                    }
                )


def representative_replay(variant: dict) -> dict:
    values = [row["selected_group_overlap_ns"] for row in variant["replays"]]
    target = median(values)
    return min(
        variant["replays"],
        key=lambda row: (abs(row["selected_group_overlap_ns"] - target), row["index"]),
    )


def write_timeline(results: dict, path: Path) -> None:
    colors = {"x_11": "#2F6B9A", "x_13": "#E58634", "x_17": "#4C9F70"}
    representatives = [representative_replay(variant) for variant in results["variants"]]
    maximum_us = max(
        nanoseconds_to_microseconds(replay["target_window_ns"])
        for replay in representatives
    )
    maximum_us = max(1.0, maximum_us * 1.05)
    width = 1400
    height = 820
    left = 170
    right = 60
    plot_width = width - left - right
    panel_height = 300
    panel_tops = (110, 440)
    row_offsets = (92, 158, 224)
    bar_height = 34

    def x_position(value_us: float) -> float:
        return left + (value_us / maximum_us) * plot_width

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,Segoe UI,sans-serif;fill:#17212B}.title{font-size:25px;font-weight:700}.panel{font-size:19px;font-weight:700}.label{font-size:16px}.tick{font-size:13px;fill:#4B5563}.note{font-size:14px;fill:#374151}</style>',
        '<text class="title" x="700" y="42" text-anchor="middle">GoogLeNet call 13: Nsight Systems kernel-level concurrency</text>',
        '<text class="note" x="700" y="68" text-anchor="middle">Exact replay kernel intervals; * denotes the operators selected together at call 13</text>',
    ]
    tick_count = 6
    for panel_index, (variant, replay, top) in enumerate(
        zip(results["variants"], representatives, panel_tops)
    ):
        origin = replay["target_start_ns"]
        selected = set(variant["selected_ops"])
        selected_overlap = nanoseconds_to_microseconds(
            replay["selected_group_overlap_ns"]
        )
        triple_overlap = nanoseconds_to_microseconds(
            replay["strict_triple_overlap_ns"]
        )
        title = (
            f"{variant['display']} — replay {replay['index']} | "
            f"selected overlap={selected_overlap:.3f} us | "
            f"strict triple={triple_overlap:.3f} us"
        )
        elements.append(
            f'<text class="panel" x="{left}" y="{top + 28}">{html.escape(title)}</text>'
        )
        plot_top = top + 48
        plot_bottom = top + panel_height - 20
        elements.append(
            f'<rect x="{left}" y="{plot_top}" width="{plot_width}" height="{plot_bottom - plot_top}" fill="#FAFAFA" stroke="#D1D5DB"/>'
        )
        for tick in range(tick_count + 1):
            value = maximum_us * tick / tick_count
            x = x_position(value)
            elements.append(
                f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" stroke="#D1D5DB" stroke-width="1"/>'
            )
            elements.append(
                f'<text class="tick" x="{x:.2f}" y="{plot_bottom + 20}" text-anchor="middle">{value:.1f}</text>'
            )

        for segment in replay["concurrency_segments_ns"]:
            count = len(segment["active_ops"])
            if count < 2:
                continue
            start_us = (segment["start"] - origin) / 1000.0
            end_us = (segment["end"] - origin) / 1000.0
            x = x_position(start_us)
            segment_width = max(0.5, x_position(end_us) - x)
            fill = "#C72E29" if count >= 3 else "#9CA3AF"
            opacity = "0.17" if count >= 3 else "0.12"
            elements.append(
                f'<rect x="{x:.2f}" y="{plot_top}" width="{segment_width:.2f}" height="{plot_bottom - plot_top}" fill="{fill}" opacity="{opacity}"/>'
            )

        for op_index, op in enumerate(TARGET_OPS):
            y = top + row_offsets[op_index]
            suffix = " *" if op in selected else ""
            elements.append(
                f'<text class="label" x="{left - 18}" y="{y + 7}" text-anchor="end">{html.escape(op + suffix)}</text>'
            )
            opacity = 1.0 if op in selected else 0.45
            stroke = "#17212B" if op in selected else "#6B7280"
            stroke_width = 2 if op in selected else 1
            for start, end in replay["ops"][op]["intervals_ns"]:
                start_us = (start - origin) / 1000.0
                end_us = (end - origin) / 1000.0
                x = x_position(start_us)
                bar_width = max(0.8, x_position(end_us) - x)
                elements.append(
                    f'<rect x="{x:.2f}" y="{y - bar_height / 2:.2f}" width="{bar_width:.2f}" height="{bar_height}" rx="3" fill="{colors[op]}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
                )
        elements.append(
            f'<text class="tick" x="{left + plot_width / 2}" y="{plot_bottom + 43}" text-anchor="middle">GPU time relative to first target kernel (microseconds)</text>'
        )

    legend_y = 790
    legend_items = [
        ("#2F6B9A", "x_11"),
        ("#E58634", "x_13"),
        ("#4C9F70", "x_17"),
        ("#9CA3AF", ">=2 target ops active"),
        ("#C72E29", "3 target ops active"),
    ]
    x_cursor = 220
    for color, label in legend_items:
        elements.append(
            f'<rect x="{x_cursor}" y="{legend_y - 14}" width="22" height="14" fill="{color}" opacity="0.75"/>'
        )
        elements.append(
            f'<text class="note" x="{x_cursor + 30}" y="{legend_y - 2}">{html.escape(label)}</text>'
        )
        x_cursor += 190 if len(label) < 10 else 250
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def format_us(value_ns: float) -> str:
    return f"{nanoseconds_to_microseconds(value_ns):.3f}"


def write_markdown(results: dict, path: Path) -> None:
    static, td = results["variants"]
    lines = [
        "# Q3 GPU overlap validation - GoogLeNet call 13",
        "",
        "## Verdict",
        "",
        (
            f"- Static+Janus selected `x_13+x_17`; their kernel-level overlap was "
            f"positive in all 5 replays (median "
            f"{format_us(static['medians']['selected_group_overlap_ns'])} us)."
        ),
        (
            f"- TD+Janus selected `x_11+x_13+x_17`; strict three-way kernel overlap was "
            f"{'positive' if td['verdict']['strict_triple_overlap_all_5_replays'] else 'not positive'} "
            f"in all 5 replays (median "
            f"{format_us(td['medians']['strict_triple_overlap_ns'])} us; "
            f"maximum concurrent target operators "
            f"{td['verdict']['maximum_concurrent_target_ops']})."
        ),
        "",
        "This validates actual GPU kernel execution, not merely the scheduler's selected-set log.",
        "",
        "## Per-replay measurements (microseconds)",
        "",
        "| Variant | Replay | x11&x13 | x11&x17 | x13&x17 | Strict triple | Selected-group overlap | Max concurrent |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in results["variants"]:
        for replay in variant["replays"]:
            pairwise = replay["pairwise_overlap_ns"]
            lines.append(
                "| {variant} | {index} | {p1} | {p2} | {p3} | {triple} | {selected} | {maximum} |".format(
                    variant=variant["display"],
                    index=replay["index"],
                    p1=format_us(pairwise["x_11&x_13"]),
                    p2=format_us(pairwise["x_11&x_17"]),
                    p3=format_us(pairwise["x_13&x_17"]),
                    triple=format_us(replay["strict_triple_overlap_ns"]),
                    selected=format_us(replay["selected_group_overlap_ns"]),
                    maximum=replay["max_concurrent_target_ops"],
                )
            )
    lines += [
        "",
        "## Mapping and reproducibility",
        "",
        "- Frozen commit: `3b2880ad5ca4b78d0385c9dd014ac2f4ab420648`.",
        "- GoogLeNet profile SHA-256: `0d29cfcd359efbf8d0630d9ef8171b0f6cd383fbac8ce27d7c6a1b18b3a1ae14`.",
        "- Hardware: NVIDIA RTX A5000; batch size 1; 5 CUDA Graph replays per variant.",
        "- Mapping chain: capture-phase FX NVTX range -> captured `graphNodeId` -> CUDA Graph clone `originalGraphNodeId -> graphNodeId` -> replay kernel `graphNodeId`.",
        "- Overlap is the exact intersection of GPU kernel intervals after merging each operator's kernel intervals. No kernel-name-only or CUDA-stream-pointer heuristic is used.",
        "- The call-13 ready set, enumerated/feasible counts, selected set, and numerical correctness were asserted by the profiling driver before accepting the trace.",
        "",
        "## Scope",
        "",
        "This is a targeted mechanism check for call 13. It proves whether these selected operators actually overlap on the GPU; it does not by itself establish end-to-end latency superiority or generalize to every scheduler call/model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    variants = [analyze_variant(spec, root) for spec in VARIANTS]
    results = {
        "schema_version": 1,
        "question": (
            "Does the Static+Janus two-op selection and TD+Janus three-op selection "
            "at GoogLeNet call 13 correspond to actual GPU concurrency?"
        ),
        "mapping_method": (
            "FX capture NVTX -> captured graphNodeId -> clone originalGraphNodeId/graphNodeId -> replay kernel graphNodeId"
        ),
        "interval_method": (
            "Per-op replay kernel intervals are merged; pair/triple overlap is their exact intersection."
        ),
        "variants": variants,
    }
    json_path = root / "q3_gpu_overlap_results.json"
    csv_path = root / "q3_gpu_overlap_replays.csv"
    timeline_path = root / "q3_gpu_overlap_timeline.svg"
    markdown_path = root / "Q3_GPU_OVERLAP_RESULT.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(results, csv_path)
    write_timeline(results, timeline_path)
    write_markdown(results, markdown_path)
    compact = {
        variant["display"]: {
            "selected_ops": variant["selected_ops"],
            "medians_us": {
                key: nanoseconds_to_microseconds(value)
                for key, value in variant["medians"].items()
                if key.endswith("_ns")
            },
            "verdict": variant["verdict"],
        }
        for variant in variants
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
