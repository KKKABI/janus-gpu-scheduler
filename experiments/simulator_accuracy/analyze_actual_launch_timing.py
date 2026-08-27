#!/usr/bin/env python3
"""Extract per-operator replay timing from isolated exact-group NSYS traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3


def merged(spans):
    output = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not output or start > output[-1][1]:
            output.append([start, end])
        else:
            output[-1][1] = max(output[-1][1], end)
    return [(int(start), int(end)) for start, end in output]


def load_td_predictions(root: Path):
    output = {}
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row["case_id"] in output:
                raise RuntimeError(f"duplicate TD result: {row['case_id']}")
            output[row["case_id"]] = row["gap_results"]["0.0020"]
    return output


def analyze_model(model_root: Path, predictions: dict):
    summary = json.loads(
        (model_root / "artifacts" / "summary.json").read_text(encoding="utf-8")
    )
    db = sqlite3.connect(model_root / "full_trace.sqlite")
    db.row_factory = sqlite3.Row
    strings = {
        int(row["id"]): row["value"]
        for row in db.execute("SELECT id,value FROM StringIds")
    }

    def nvtx_text(row):
        return row["text"] if row["text"] is not None else strings.get(row["textId"])

    nvtx = list(
        db.execute(
            "SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
        )
    )
    clone_edges = [
        (int(row["originalGraphNodeId"]), int(row["graphNodeId"]))
        for row in db.execute(
            "SELECT originalGraphNodeId,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
            "WHERE originalGraphNodeId IS NOT NULL AND graphNodeId IS NOT NULL"
        )
    ]
    rows = []
    for case in summary["cases"]:
        if case.get("capture_status") != "captured":
            continue
        replay_ranges = [
            (int(row["start"]), int(row["end"]))
            for row in nvtx
            if nvtx_text(row) == case["marker"]
        ]
        if len(replay_ranges) != 1:
            raise RuntimeError(
                f"{case['case_id']}: replay marker count={len(replay_ranges)}"
            )
        replay_start, replay_end = replay_ranges[0]
        kernels = [
            dict(row)
            for row in db.execute(
                "SELECT start,end,streamId,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL ORDER BY start,end",
                (replay_start, replay_end),
            )
        ]
        by_op = {}
        missing = []
        for name in case["group"]:
            marker = f"ISOLATED_FX_CAPTURE::{case['case_id']}::{name}"
            capture_ranges = [
                (int(row["start"]), int(row["end"]))
                for row in nvtx
                if nvtx_text(row) == marker
            ]
            if len(capture_ranges) != 1:
                missing.append(name)
                continue
            start, end = capture_ranges[0]
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
            matched = [
                row for row in kernels if int(row["graphNodeId"]) in expanded
            ]
            spans = merged((int(row["start"]), int(row["end"])) for row in matched)
            if not ids or not spans:
                missing.append(name)
                continue
            by_op[name] = {
                "first_start_ns": min(start for start, _ in spans),
                "last_end_ns": max(end for _, end in spans),
                "active_duration_ns": sum(end - start for start, end in spans),
                "span_duration_ns": max(end for _, end in spans)
                - min(start for start, _ in spans),
                "kernel_count": len(matched),
                "stream_ids": sorted({int(row["streamId"]) for row in matched}),
            }
        if missing:
            continue
        group_origin = min(item["first_start_ns"] for item in by_op.values())
        launch_offsets = {
            name: item["first_start_ns"] - group_origin for name, item in by_op.items()
        }
        ordered_offsets = sorted(launch_offsets.values())
        adjacent_gaps = [
            right - left for left, right in zip(ordered_offsets, ordered_offsets[1:])
        ]
        intervals = [
            (item["first_start_ns"], item["last_end_ns"])
            for item in by_op.values()
        ]
        strict_span_ns = max(
            0,
            min(end for _, end in intervals) - max(start for start, _ in intervals),
        )
        prediction = predictions.get(case["case_id"], {})
        rows.append(
            {
                "case_id": case["case_id"],
                "model": case["model"],
                "size": len(case["group"]),
                "group": list(case["group"]),
                "launch_offsets_ns": launch_offsets,
                "adjacent_launch_gaps_ns": adjacent_gaps,
                "max_launch_offset_ns": max(ordered_offsets),
                "operators": by_op,
                "strict_span_ns": strict_span_ns,
                "strict_parallel": strict_span_ns > 0,
                "td_prediction": bool(prediction.get("strict_parallel", False)),
                "td_failure_reason": prediction.get("failure_reason"),
                "td_predicted_overlap_ms": prediction.get(
                    "strict_overlap_duration", 0.0
                ),
            }
        )
    db.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardware-root", type=Path, required=True)
    parser.add_argument("--td-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    predictions = load_td_predictions(args.td_root)
    rows = []
    for sqlite_path in sorted(args.hardware_root.glob("*/full_trace.sqlite")):
        rows.extend(analyze_model(sqlite_path.parent, predictions))
    if not rows:
        raise RuntimeError("no auditable timing rows")
    gaps = [gap for row in rows for gap in row["adjacent_launch_gaps_ns"]]
    gaps.sort()

    def percentile(values, fraction):
        if not values:
            return None
        index = round((len(values) - 1) * fraction)
        return values[index]

    payload = {
        "schema_version": 1,
        "definition": (
            "Per-operator first/last CUDA kernel timing inside one isolated "
            "common-start CUDA Graph replay. Span overlap is diagnostic; the "
            "existing strict label still uses the union of actual kernel intervals."
        ),
        "case_count": len(rows),
        "adjacent_launch_gap_ns": {
            "count": len(gaps),
            "min": min(gaps) if gaps else None,
            "p25": percentile(gaps, 0.25),
            "median": percentile(gaps, 0.50),
            "p75": percentile(gaps, 0.75),
            "p95": percentile(gaps, 0.95),
            "max": max(gaps) if gaps else None,
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_id",
                "model",
                "size",
                "group",
                "strict_parallel",
                "strict_span_ns",
                "max_launch_offset_ns",
                "adjacent_launch_gaps_ns",
                "td_prediction",
                "td_failure_reason",
                "td_predicted_overlap_ms",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["case_id"],
                    row["model"],
                    row["size"],
                    "+".join(row["group"]),
                    row["strict_parallel"],
                    row["strict_span_ns"],
                    row["max_launch_offset_ns"],
                    ";".join(str(value) for value in row["adjacent_launch_gaps_ns"]),
                    row["td_prediction"],
                    row["td_failure_reason"],
                    row["td_predicted_overlap_ms"],
                ]
            )
    print(json.dumps({key: payload[key] for key in ("case_count", "adjacent_launch_gap_ns")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
