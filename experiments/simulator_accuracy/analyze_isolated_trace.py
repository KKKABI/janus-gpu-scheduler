#!/usr/bin/env python3
"""Map isolated CUDA Graph nodes to one replay and measure strict overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def merge(spans):
    output = []
    for start, end in sorted(spans):
        if end <= start:
            continue
        if not output or start > output[-1][1]:
            output.append([start, end])
        else:
            output[-1][1] = max(output[-1][1], end)
    return [(a, b) for a, b in output]


def intersect(left, right):
    output = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            output.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    db = sqlite3.connect(args.sqlite)
    db.row_factory = sqlite3.Row
    strings = {int(row["id"]): row["value"] for row in db.execute("SELECT id,value FROM StringIds")}

    def text(row):
        return row["text"] if row["text"] is not None else strings.get(row["textId"])

    nvtx = list(db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"))
    clone_edges = [
        (int(row["originalGraphNodeId"]), int(row["graphNodeId"]))
        for row in db.execute(
            "SELECT originalGraphNodeId,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
            "WHERE originalGraphNodeId IS NOT NULL AND graphNodeId IS NOT NULL"
        )
    ]
    results = []
    for case in summary["cases"]:
        if case.get("capture_status") != "captured":
            results.append(
                {
                    **case,
                    "auditable": False,
                    "missing_ops": list(case["group"]),
                    "strict_overlap_ns": 0,
                    "isolated_strict_parallel": False,
                    "streams_by_op": {},
                }
            )
            continue
        replay_ranges = [(int(row["start"]), int(row["end"])) for row in nvtx if text(row) == case["marker"]]
        if len(replay_ranges) != 1:
            raise RuntimeError(f"{case['case_id']}: replay marker count={len(replay_ranges)}")
        replay_start, replay_end = replay_ranges[0]
        kernels = [
            dict(row)
            for row in db.execute(
                "SELECT start,end,streamId,graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL ORDER BY start,end",
                (replay_start, replay_end),
            )
        ]
        intervals = {}
        streams = {}
        missing = []
        for name in case["group"]:
            marker = f"ISOLATED_FX_CAPTURE::{case['case_id']}::{name}"
            capture_ranges = [(int(row["start"]), int(row["end"])) for row in nvtx if text(row) == marker]
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
            matched = [row for row in kernels if int(row["graphNodeId"]) in expanded]
            intervals[name] = merge((int(row["start"]), int(row["end"])) for row in matched)
            streams[name] = sorted({int(row["streamId"]) for row in matched})
            if not ids or not intervals[name]:
                missing.append(name)
        common = list(intervals.values())[0] if intervals and not missing else []
        for spans in list(intervals.values())[1:]:
            common = intersect(common, spans)
        strict_ns = sum(end - start for start, end in common)
        results.append(
            {
                **case,
                "auditable": not missing,
                "missing_ops": missing,
                "strict_overlap_ns": strict_ns,
                "isolated_strict_parallel": not missing and strict_ns > 0,
                "streams_by_op": streams,
            }
        )
    db.close()
    auditable = [row for row in results if row["auditable"]]
    positive = [row for row in auditable if row["isolated_strict_parallel"]]
    payload = {
        "schema_version": 1,
        "protocol": summary["protocol"],
        "model": summary["model"],
        "case_count": len(results),
        "auditable_cases": len(auditable),
        "isolated_strict_parallel_cases": len(positive),
        "recovery_rate": len(positive) / len(auditable) if auditable else None,
        "cases": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("model", "case_count", "auditable_cases", "isolated_strict_parallel_cases", "recovery_rate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
