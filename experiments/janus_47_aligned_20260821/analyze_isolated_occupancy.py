#!/usr/bin/env python3
"""Analyze per-group isolated Nsight Systems GPU occupancy captures."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics


METRIC_NAME = "Compute Warps in Flight [Throughput %]"


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = math.floor(position)
    right = math.ceil(position)
    if left == right:
        return float(ordered[left])
    return float(ordered[left] * (right - position) + ordered[right] * (position - left))


def resolved(row, strings):
    return row["text"] if row["text"] is not None else strings.get(row["textId"])


def merge(spans):
    output = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not output or start > output[-1][1]:
            output.append([start, end])
        else:
            output[-1][1] = max(output[-1][1], end)
    return [(int(start), int(end)) for start, end in output]


def intersect(left, right):
    output = []
    i = j = 0
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


def strict_overlap(intervals_by_op):
    common = list(intervals_by_op.values())[0]
    for spans in list(intervals_by_op.values())[1:]:
        common = intersect(common, spans)
        if not common:
            return False
    return bool(common)


def metric_id(db):
    matches = [
        int(row[0])
        for row in db.execute(
            "SELECT metricId FROM TARGET_INFO_GPU_METRICS WHERE metricName=?",
            (METRIC_NAME,),
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(f"metric lookup failed: {matches}")
    return matches[0]


def analyze_trace(path: Path, case, replay_count):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = {int(row["id"]): row["value"] for row in db.execute("SELECT id,value FROM StringIds")}
    nvtx = list(db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"))
    clone_edges = [
        (int(row["originalGraphNodeId"]), int(row["graphNodeId"]))
        for row in db.execute(
            "SELECT originalGraphNodeId,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
            "WHERE originalGraphNodeId IS NOT NULL AND graphNodeId IS NOT NULL"
        )
    ]
    replay_ranges = [
        (int(row["start"]), int(row["end"]))
        for row in nvtx if resolved(row, strings) == case["marker"]
    ]
    if len(replay_ranges) != 1:
        raise RuntimeError(f"{case['case_id']}: replay marker count={len(replay_ranges)}")
    replay_start, replay_end = replay_ranges[0]
    kernels = [
        dict(row)
        for row in db.execute(
            "SELECT start,end,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL ORDER BY start,end",
            (replay_start, replay_end),
        )
    ]
    kernels_by_node = defaultdict(list)
    for row in kernels:
        kernels_by_node[int(row["graphNodeId"])].append(row)

    nodes_by_op = {}
    missing = []
    for name in case["group"]:
        marker = f"ISOLATED_FX_CAPTURE::{case['case_id']}::{name}"
        ranges = [
            (int(row["start"]), int(row["end"]))
            for row in nvtx if resolved(row, strings) == marker
        ]
        if len(ranges) != 1:
            missing.append(name)
            continue
        start, end = ranges[0]
        ids = {
            int(row["graphNodeId"])
            for row in db.execute(
                "SELECT graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                "WHERE start>=? AND start<=? AND graphNodeId IS NOT NULL",
                (start, end),
            )
        }
        expanded = set(ids)
        changed = True
        while changed:
            changed = False
            for original, clone in clone_edges:
                if original in expanded and clone not in expanded:
                    expanded.add(clone)
                    changed = True
        active_ids = sorted(node for node in expanded if node in kernels_by_node)
        if not active_ids:
            missing.append(name)
        nodes_by_op[name] = active_ids

    strict_replays = 0
    if not missing:
        counts = {
            len(kernels_by_node[node])
            for nodes in nodes_by_op.values() for node in nodes
        }
        if counts != {replay_count}:
            raise RuntimeError(f"{case['case_id']}: graph-node replay counts={sorted(counts)}")
        for replay_index in range(replay_count):
            intervals = {
                name: merge([
                    (
                        int(kernels_by_node[node][replay_index]["start"]),
                        int(kernels_by_node[node][replay_index]["end"]),
                    )
                    for node in nodes
                ])
                for name, nodes in nodes_by_op.items()
            }
            strict_replays += strict_overlap(intervals)

    mid = metric_id(db)
    samples = [
        float(row["value"])
        for row in db.execute(
            "SELECT value FROM GPU_METRICS WHERE metricId=? AND timestamp>=? AND timestamp<=?",
            (mid, replay_start, replay_end),
        )
    ]
    nonzero = [value for value in samples if value > 0]
    db.close()
    return {
        "auditable": not missing,
        "missing_ops": missing,
        "strict_parallel_replays": strict_replays,
        "strict_parallel_rate": strict_replays / replay_count,
        "metric_sample_count": len(samples),
        "nonzero_metric_sample_count": len(nonzero),
        "actual_compute_warps_p95_pct": percentile(nonzero, 0.95),
        "actual_compute_warps_median_pct": statistics.median(nonzero) if nonzero else None,
        "actual_compute_warps_max_pct": max(nonzero) if nonzero else None,
    }


def summary_row(method, model, rows):
    auditable = [row for row in rows if row["auditable"]]
    sampled = [row for row in auditable if row["actual_compute_warps_p95_pct"] is not None]
    errors = [
        abs(row["actual_compute_warps_p95_pct"] - row["predicted_occupancy_pct"])
        for row in sampled
    ]
    return {
        "method": method,
        "model": model,
        "selected_groups": len(rows),
        "auditable_groups": len(auditable),
        "occupancy_sampled_groups": len(sampled),
        "strict_parallel_any_replay_groups": sum(row["strict_parallel_replays"] > 0 for row in auditable),
        "strict_parallel_all_replays_groups": sum(row["strict_parallel_replays"] == row["metrics_replays"] for row in auditable),
        "occupancy_mae_percentage_points": statistics.fmean(errors) if errors else None,
        "occupancy_max_error_percentage_points": max(errors) if errors else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_csv.exists():
        raise FileExistsError("refusing to overwrite output")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    planned = {row["case_id"]: row for row in manifest["cases"]}
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    replay_count = int(summary["gpu_metrics_replays"])
    capture_index = 0
    groups = []
    method_rows = defaultdict(list)
    for case in summary["cases"]:
        base = planned[case["case_id"]]
        if case.get("capture_status") != "captured":
            measured = {
                "auditable": False,
                "missing_ops": list(base["group"]),
                "strict_parallel_replays": 0,
                "strict_parallel_rate": 0.0,
                "metric_sample_count": 0,
                "nonzero_metric_sample_count": 0,
                "actual_compute_warps_p95_pct": None,
                "actual_compute_warps_median_pct": None,
                "actual_compute_warps_max_pct": None,
            }
            trace_path = None
        else:
            capture_index += 1
            trace_path = args.trace_dir / f"full_trace.{capture_index}.sqlite"
            if not trace_path.is_file():
                raise FileNotFoundError(trace_path)
            measured = analyze_trace(trace_path, case, replay_count)
        group = {
            "case_id": case["case_id"],
            "model": base["model"],
            "operators": list(base["group"]),
            "width": int(base["size"]),
            "selected_for_methods": list(base["selected_for_methods"]),
            "predicted_occupancy_by_method": dict(base["predicted_occupancy_by_method"]),
            "metrics_replays": replay_count,
            "trace_sqlite": str(trace_path.resolve()) if trace_path else None,
            **measured,
        }
        groups.append(group)
        for method in group["selected_for_methods"]:
            row = {
                **group,
                "method": method,
                "predicted_occupancy_pct": float(base["predicted_occupancy_by_method"][method]) * 100.0,
            }
            method_rows[(method, base["model"])].append(row)
            method_rows[(method, "ALL")].append(row)
    trace_count = len(list(args.trace_dir.glob("full_trace.*.sqlite")))
    if trace_count != capture_index:
        raise RuntimeError(f"trace count differs: files={trace_count}, captured={capture_index}")
    summaries = [
        summary_row(method, model, rows)
        for (method, model), rows in sorted(method_rows.items())
    ]
    payload = {
        "schema_version": 1,
        "protocol": "janus_section_4_7_isolated_gpu_occupancy_v1",
        "model": summary["model"],
        "gpu_metric": METRIC_NAME,
        "metric_frequency_hz": 10000,
        "metrics_replays_per_group": replay_count,
        "important_boundary": (
            "GA10x Nsight Systems exposes Compute Warps in Flight throughput rather than a metric literally named SM Occupancy; p95 of nonzero isolated samples is compared with predicted occupancy."
        ),
        "summaries": summaries,
        "groups": groups,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "method", "model", "selected_groups", "auditable_groups",
        "occupancy_sampled_groups", "strict_parallel_any_replay_groups",
        "strict_parallel_all_replays_groups", "occupancy_mae_percentage_points",
        "occupancy_max_error_percentage_points",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
