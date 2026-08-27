#!/usr/bin/env python3
"""Map solo CUDA Graph replay kernels back to sampled FX operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


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
    for operator in summary["operators"]:
        if operator.get("capture_status") != "captured":
            results.append({**operator, "auditable": False, "kernels": []})
            continue
        capture_ranges = [
            (int(row["start"]), int(row["end"]))
            for row in nvtx
            if text(row) == operator["capture_marker"]
        ]
        replay_ranges = [
            (int(row["start"]), int(row["end"]))
            for row in nvtx
            if text(row) == operator["replay_marker"]
        ]
        if len(capture_ranges) != 1 or len(replay_ranges) != 1:
            results.append(
                {
                    **operator,
                    "auditable": False,
                    "kernels": [],
                    "mapping_error": (
                        f"capture_ranges={len(capture_ranges)}, "
                        f"replay_ranges={len(replay_ranges)}"
                    ),
                }
            )
            continue
        capture_start, capture_end = capture_ranges[0]
        replay_start, replay_end = replay_ranges[0]
        capture_ids = {
            int(row["graphNodeId"])
            for row in db.execute(
                "SELECT graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                "WHERE start>=? AND start<=? AND graphNodeId IS NOT NULL",
                (capture_start, capture_end),
            )
        }
        expanded = set(capture_ids)
        changed = True
        while changed:
            changed = False
            for original, clone in clone_edges:
                if original in expanded and clone not in expanded:
                    expanded.add(clone)
                    changed = True
        kernels = [
            dict(row)
            for row in db.execute(
                "SELECT start,end,streamId,graphNodeId,shortName,demangledName "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL "
                "ORDER BY start,end",
                (replay_start, replay_end),
            )
            if int(row["graphNodeId"]) in expanded
        ]
        safe_kernels = []
        for row in kernels:
            safe_kernels.append(
                {
                    "start_ns": int(row["start"]),
                    "end_ns": int(row["end"]),
                    "duration_ns": int(row["end"]) - int(row["start"]),
                    "stream_id": int(row["streamId"]),
                    "graph_node_id": int(row["graphNodeId"]),
                    "short_name": strings.get(int(row["shortName"])),
                    "demangled_name": strings.get(int(row["demangledName"])),
                }
            )
        active_ns = sum(row["duration_ns"] for row in safe_kernels)
        span_ns = (
            max(row["end_ns"] for row in safe_kernels)
            - min(row["start_ns"] for row in safe_kernels)
            if safe_kernels
            else 0
        )
        results.append(
            {
                **operator,
                "auditable": bool(capture_ids and safe_kernels),
                "capture_graph_node_count": len(capture_ids),
                "kernel_count": len(safe_kernels),
                "active_duration_ns": active_ns,
                "span_duration_ns": span_ns,
                "kernels": safe_kernels,
            }
        )
    db.close()
    payload = {
        "schema_version": 1,
        "protocol": summary["protocol"],
        "model": summary["model"],
        "git_head": summary["git_head"],
        "profile_sha256": summary["profile_sha256"],
        "target_count": len(results),
        "auditable_count": sum(row["auditable"] for row in results),
        "operators": results,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "model": payload["model"],
                "target_count": payload["target_count"],
                "auditable_count": payload["auditable_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
