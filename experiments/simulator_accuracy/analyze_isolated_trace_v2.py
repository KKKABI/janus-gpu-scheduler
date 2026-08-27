#!/usr/bin/env python3
"""Audit isolated CUDA Graph groups and retain per-OP timing evidence.

The primary truth is strict full-group kernel overlap.  Unlike the original
analyzer, this version retains the merged intervals, common intervals, stream
IDs, and every matched kernel so that each verdict is independently auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            output.append((int(start), int(end)))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return output


def relative_spans(spans, origin):
    return [
        {
            "start_ns": int(start),
            "end_ns": int(end),
            "duration_ns": int(end - start),
            "start_from_replay_ns": int(start - origin),
            "end_from_replay_ns": int(end - origin),
        }
        for start, end in spans
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--rep", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for required in (args.sqlite, args.summary):
        if not required.is_file():
            raise FileNotFoundError(required)
    if args.rep is not None and not args.rep.is_file():
        raise FileNotFoundError(args.rep)
    if args.output.exists():
        raise FileExistsError(args.output)

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    db = sqlite3.connect(args.sqlite)
    db.row_factory = sqlite3.Row
    strings = {
        int(row["id"]): row["value"]
        for row in db.execute("SELECT id,value FROM StringIds")
    }

    def text(row):
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
                    "replay_interval": None,
                    "streams_by_op": {},
                    "intervals_by_op": {},
                    "kernels_by_op": {},
                    "common_intervals": [],
                }
            )
            continue

        replay_ranges = [
            (int(row["start"]), int(row["end"]))
            for row in nvtx
            if text(row) == case["marker"]
        ]
        if len(replay_ranges) != 1:
            raise RuntimeError(
                f"{case['case_id']}: replay marker count={len(replay_ranges)}"
            )
        replay_start, replay_end = replay_ranges[0]
        kernels = [
            dict(row)
            for row in db.execute(
                "SELECT start,end,streamId,graphNodeId,shortName,demangledName "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL "
                "ORDER BY start,end",
                (replay_start, replay_end),
            )
        ]

        intervals = {}
        streams = {}
        kernels_by_op = {}
        missing = []
        for name in case["group"]:
            marker = f"ISOLATED_FX_CAPTURE::{case['case_id']}::{name}"
            capture_ranges = [
                (int(row["start"]), int(row["end"]))
                for row in nvtx
                if text(row) == marker
            ]
            if len(capture_ranges) != 1:
                missing.append(name)
                continue
            capture_start, capture_end = capture_ranges[0]
            ids = {
                int(row["graphNodeId"])
                for row in db.execute(
                    "SELECT graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                    "WHERE start>=? AND start<=? AND graphNodeId IS NOT NULL",
                    (capture_start, capture_end),
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
            intervals[name] = merge(
                (int(row["start"]), int(row["end"])) for row in matched
            )
            streams[name] = sorted({int(row["streamId"]) for row in matched})
            kernels_by_op[name] = [
                {
                    "start_ns": int(row["start"]),
                    "end_ns": int(row["end"]),
                    "duration_ns": int(row["end"]) - int(row["start"]),
                    "start_from_replay_ns": int(row["start"]) - replay_start,
                    "end_from_replay_ns": int(row["end"]) - replay_start,
                    "stream_id": int(row["streamId"]),
                    "graph_node_id": int(row["graphNodeId"]),
                    "short_name": strings.get(int(row["shortName"])),
                    "demangled_name": strings.get(int(row["demangledName"])),
                }
                for row in matched
            ]
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
                "strict_overlap_ns": int(strict_ns),
                "isolated_strict_parallel": not missing and strict_ns > 0,
                "replay_interval": {
                    "start_ns": replay_start,
                    "end_ns": replay_end,
                    "duration_ns": replay_end - replay_start,
                },
                "streams_by_op": streams,
                "intervals_by_op": {
                    name: relative_spans(spans, replay_start)
                    for name, spans in intervals.items()
                },
                "kernels_by_op": kernels_by_op,
                "common_intervals": relative_spans(common, replay_start),
            }
        )

    db.close()
    auditable = [row for row in results if row["auditable"]]
    positive = [row for row in auditable if row["isolated_strict_parallel"]]
    evidence = {
        "sqlite_path": str(args.sqlite.resolve()),
        "sqlite_sha256": sha256_file(args.sqlite),
        "sqlite_size_bytes": args.sqlite.stat().st_size,
        "rep_path": str(args.rep.resolve()) if args.rep is not None else None,
        "rep_sha256": sha256_file(args.rep) if args.rep is not None else None,
        "rep_size_bytes": args.rep.stat().st_size if args.rep is not None else None,
        "summary_path": str(args.summary.resolve()),
        "summary_sha256": sha256_file(args.summary),
    }
    payload = {
        "schema_version": 2,
        "protocol": summary["protocol"],
        "truth_definition": (
            "every target OP maps to replay kernels and the intersection of "
            "all merged OP kernel intervals has positive duration"
        ),
        "model": summary["model"],
        "case_count": len(results),
        "auditable_cases": len(auditable),
        "isolated_strict_parallel_cases": len(positive),
        "positive_precision": len(positive) / len(auditable) if auditable else None,
        "evidence": evidence,
        "cases": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "model",
                    "case_count",
                    "auditable_cases",
                    "isolated_strict_parallel_cases",
                    "positive_precision",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
