#!/usr/bin/env python3
"""Measure actual concurrency of final selected groups in one NSYS replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    output: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not output or start > output[-1][1]:
            output.append([start, end])
        else:
            output[-1][1] = max(output[-1][1], end)
    return [(start, end) for start, end in output]


def intersect_two(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
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


def strict_intersection_ns(
    intervals_by_op: dict[str, list[tuple[int, int]]]
) -> int:
    groups = list(intervals_by_op.values())
    result = groups[0] if groups else []
    for group in groups[1:]:
        result = intersect_two(result, group)
        if not result:
            break
    return sum(end - start for start, end in result)


def concurrency_metrics(
    intervals_by_op: dict[str, list[tuple[int, int]]]
) -> tuple[int, int]:
    boundaries = sorted({
        point for spans in intervals_by_op.values() for span in spans for point in span
    })
    maximum = 0
    any_pair_ns = 0
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        active = sum(
            any(left <= start and right >= end for left, right in spans)
            for spans in intervals_by_op.values()
        )
        maximum = max(maximum, active)
        if active >= 2:
            any_pair_ns += end - start
    return maximum, any_pair_ns


def resolved_text(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    return row["text"] if row["text"] is not None else strings.get(row["textId"])


def trace_data(db_path: Path, trace_tag: str):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    strings = {
        int(row["id"]): row["value"]
        for row in db.execute("SELECT id,value FROM StringIds")
    }
    prefix = "FX::GRAPH_CAPTURE::"
    capture_ranges = {}
    for row in db.execute(
        "SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        value = resolved_text(row, strings)
        if value and value.startswith(prefix):
            capture_ranges[value[len(prefix):]] = (
                int(row["start"]), int(row["end"])
            )

    capture_ids = {}
    for op, (start, end) in capture_ranges.items():
        ids = {
            int(row["graphNodeId"])
            for row in db.execute(
                "SELECT graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                "WHERE start>=? AND start<=? AND graphNodeId IS NOT NULL",
                (start, end),
            )
        }
        if ids:
            capture_ids[op] = ids

    clone_edges = [
        (int(row["originalGraphNodeId"]), int(row["graphNodeId"]))
        for row in db.execute(
            "SELECT originalGraphNodeId,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
            "WHERE originalGraphNodeId IS NOT NULL AND graphNodeId IS NOT NULL"
        )
    ]
    replay_ids = {}
    for op, source_ids in capture_ids.items():
        expanded = set(source_ids)
        changed = True
        while changed:
            changed = False
            for original, clone in clone_edges:
                if original in expanded and clone not in expanded:
                    expanded.add(clone)
                    changed = True
        replay_ids[op] = expanded

    marker = f"JANUS_PRECISION_REPLAY::{trace_tag}"
    ranges = []
    for row in db.execute(
        "SELECT start,end,text,textId FROM NVTX_EVENTS "
        "WHERE end IS NOT NULL ORDER BY start"
    ):
        value = resolved_text(row, strings)
        if value == marker:
            ranges.append((int(row["start"]), int(row["end"])))
    if len(ranges) != 1:
        raise RuntimeError(f"expected one replay marker {marker!r}, got {len(ranges)}")
    replay_start, replay_end = ranges[0]
    kernels = [
        dict(row)
        for row in db.execute(
            "SELECT start,end,streamId,graphNodeId FROM "
            "CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? "
            "AND graphNodeId IS NOT NULL ORDER BY start,end",
            (replay_start, replay_end),
        )
    ]
    db.close()
    return replay_ids, kernels, {
        "marker": marker,
        "start": replay_start,
        "end": replay_end,
        "kernel_count": len(kernels),
    }


def main() -> int:
    args = parse_args()
    db_path = args.sqlite.resolve()
    artifacts = args.artifacts_dir.resolve()
    output_json = args.output_json.resolve()
    output_csv = args.output_csv.resolve()
    if output_json.exists() or output_csv.exists():
        raise FileExistsError("refusing to overwrite analyzer outputs")
    summary_path = artifacts / "summary.json"
    calls_path = artifacts / "scheduler_calls.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calls = json.loads(calls_path.read_text(encoding="utf-8"))
    if not summary["correctness"].get("ok"):
        raise RuntimeError("captured graph output correctness failed")
    controls = summary["paper_aligned_controls"]
    expected_controls = {
        "all_lp_forced_to_hp": True,
        "max_ready": 6,
        "final_selected_groups_only": True,
        "nsys_replays_per_group": 1,
        "rejected_candidates_evaluated": False,
        "kernel_bearing_operators_only": True,
        "ready_filter": "highest_predicted_achieved_occupancy",
    }
    if controls != expected_controls:
        raise RuntimeError(f"protocol controls differ: {controls}")

    replay_ids, kernels, replay = trace_data(db_path, summary["trace_tag"])
    call_results = []
    for call in calls:
        ops = list(call.get("selected_gpu_resource") or [])
        if len(ops) < 2:
            continue
        missing_capture = [op for op in ops if op not in replay_ids]
        intervals_by_op = {}
        streams_by_op = {}
        missing_replay = []
        if not missing_capture:
            for op in ops:
                matched = [
                    kernel for kernel in kernels
                    if int(kernel["graphNodeId"]) in replay_ids[op]
                ]
                intervals_by_op[op] = merge_intervals([
                    (int(kernel["start"]), int(kernel["end"]))
                    for kernel in matched
                ])
                streams_by_op[op] = sorted({
                    int(kernel["streamId"]) for kernel in matched
                })
                if not intervals_by_op[op]:
                    missing_replay.append(op)
        auditable = not missing_capture and not missing_replay
        strict_ns = 0
        max_concurrent = 0
        any_pair_ns = 0
        if auditable:
            strict_ns = strict_intersection_ns(intervals_by_op)
            max_concurrent, any_pair_ns = concurrency_metrics(intervals_by_op)
        item = {
            "call": int(call["call"]),
            "ready_count": int(call.get("ready_count", 0)),
            "ready_used_count": int(call.get("ready_used_count", 0)),
            "ready_ops": list(call.get("ready_ops") or []),
            "selected_ops": ops,
            "selected_size": len(ops),
            "enumerated_count": int(call.get("enumerated_count", 0)),
            "feasible_count": int(call.get("feasible_count", 0)),
            "scoring_candidate_count": int(
                call.get("scoring_candidate_count", 0)
            ),
            "auditable": auditable,
            "missing_capture_ops": missing_capture,
            "missing_replay_ops": missing_replay,
            "strict_all_selected_overlap_ns": strict_ns,
            "actual_full_group_concurrency": (
                auditable and strict_ns > 0 and max_concurrent == len(ops)
            ),
            "any_pair_overlap_ns": any_pair_ns,
            "actual_any_pair_concurrency": auditable and any_pair_ns > 0,
            "max_concurrent_selected_ops": max_concurrent,
            "streams_by_op": streams_by_op,
        }
        call_results.append(item)

    auditable = [item for item in call_results if item["auditable"]]
    strict_positive = [
        item for item in auditable if item["actual_full_group_concurrency"]
    ]
    any_pair_positive = [
        item for item in auditable if item["actual_any_pair_concurrency"]
    ]
    result = {
        "schema_version": 1,
        "protocol": summary["protocol"],
        "model": summary["model"],
        "variant": summary["variant"],
        "configuration": summary["configuration"],
        "paper_aligned_controls": controls,
        "git_head": summary["git_head"],
        "profile_sha256": summary["profile_sha256"],
        "correctness": summary["correctness"],
        "sqlite": str(db_path),
        "sqlite_sha256": sha256_file(db_path),
        "summary_sha256": sha256_file(summary_path),
        "scheduler_calls_sha256": sha256_file(calls_path),
        "replay": replay,
        "planned_multi_operator_groups": len(call_results),
        "auditable_groups": len(auditable),
        "unmapped_groups": len(call_results) - len(auditable),
        "audit_coverage": (
            len(auditable) / len(call_results) if call_results else None
        ),
        "actual_full_group_concurrent_groups": len(strict_positive),
        "paper_like_positive_precision": (
            len(strict_positive) / len(auditable) if auditable else None
        ),
        "actual_any_pair_concurrent_groups": len(any_pair_positive),
        "any_pair_positive_precision": (
            len(any_pair_positive) / len(auditable) if auditable else None
        ),
        "calls": call_results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "call", "ready_count", "ready_used_count", "selected_ops",
        "selected_size", "auditable", "actual_full_group_concurrency",
        "strict_all_selected_overlap_ns", "actual_any_pair_concurrency",
        "any_pair_overlap_ns", "max_concurrent_selected_ops",
        "missing_capture_ops", "missing_replay_ops",
    ]
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in call_results:
            writer.writerow({
                key: (
                    " + ".join(item[key]) if isinstance(item[key], list)
                    else item[key]
                )
                for key in fields
            })
    print(json.dumps({
        "model": result["model"],
        "configuration": result["configuration"],
        "planned": result["planned_multi_operator_groups"],
        "auditable": result["auditable_groups"],
        "actual": result["actual_full_group_concurrent_groups"],
        "precision": result["paper_like_positive_precision"],
        "coverage": result["audit_coverage"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
