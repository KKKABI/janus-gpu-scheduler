#!/usr/bin/env python3
"""Verify strict target-operator overlap from an Nsight Systems SQLite export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--execution-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return [(start, end) for start, end in result]


def strict_intersection_ns(intervals_by_op: dict[str, list[tuple[int, int]]]) -> int:
    events = []
    for op, intervals in intervals_by_op.items():
        for start, end in merged(intervals):
            events.append((start, 1, op))
            events.append((end, -1, op))
    events.sort(key=lambda item: (item[0], item[1]))  # end before start on ties
    active: set[str] = set()
    required = set(intervals_by_op)
    previous = None
    total = 0
    for timestamp, delta, op in events:
        if previous is not None and active == required:
            total += timestamp - previous
        if delta < 0:
            active.discard(op)
        else:
            active.add(op)
        previous = timestamp
    return total


def resolved_text(row: sqlite3.Row, strings: dict[int, str]) -> str | None:
    if row["text"] is not None:
        return str(row["text"])
    if row["textId"] is not None:
        return strings.get(int(row["textId"]))
    return None


def main() -> int:
    args = parse_args()
    sqlite_path = args.sqlite.resolve()
    execution_json = args.execution_json.resolve()
    output_json = args.output_json.resolve()
    if output_json.exists():
        raise FileExistsError(f"refusing to overwrite {output_json}")
    execution = json.loads(execution_json.read_text(encoding="utf-8"))
    if execution.get("mode") != "trace":
        raise RuntimeError(f"execution mode must be trace, got {execution.get('mode')}")
    expected_ops = list(execution["group"])

    connection = sqlite3.connect(str(sqlite_path))
    connection.row_factory = sqlite3.Row
    try:
        for table in (
            "NVTX_EVENTS",
            "CUDA_GRAPH_NODE_EVENTS",
            "CUPTI_ACTIVITY_KIND_KERNEL",
        ):
            if not table_exists(connection, table):
                raise RuntimeError(f"required Nsight table is missing: {table}")
        strings = {
            int(row["id"]): str(row["value"])
            for row in connection.execute("SELECT id,value FROM StringIds")
        } if table_exists(connection, "StringIds") else {}
        nvtx_rows = connection.execute(
            "SELECT start,end,text,textId FROM NVTX_EVENTS "
            "WHERE end IS NOT NULL ORDER BY start"
        ).fetchall()

        capture_ranges: dict[str, tuple[int, int]] = {}
        capture_prefix = "JANUS_GRAPH_CAPTURE_OP:"
        solo_capture_ranges: dict[str, tuple[int, int]] = {}
        solo_capture_prefix = "JANUS_SOLO_GRAPH_CAPTURE_OP:"
        replay_rows = []
        replay_prefix = "JANUS_GROUP_REPLAY:"
        solo_replay_rows: dict[str, list[dict[str, Any]]] = {
            op: [] for op in expected_ops
        }
        solo_replay_prefix = "JANUS_SOLO_REPLAY:"
        for row in nvtx_rows:
            value = resolved_text(row, strings)
            if value and value.startswith(capture_prefix):
                capture_ranges[value[len(capture_prefix):]] = (
                    int(row["start"]), int(row["end"])
                )
            elif value and value.startswith(solo_capture_prefix):
                solo_capture_ranges[value[len(solo_capture_prefix):]] = (
                    int(row["start"]), int(row["end"])
                )
            elif value and value.startswith(replay_prefix):
                replay_rows.append({
                    "text": value,
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                })
            elif value and value.startswith(solo_replay_prefix):
                suffix = value[len(solo_replay_prefix):]
                op, index = suffix.rsplit(":", 1)
                if op in solo_replay_rows:
                    solo_replay_rows[op].append({
                        "index": int(index),
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                    })
        missing_ranges = sorted(set(expected_ops) - set(capture_ranges))
        if missing_ranges:
            raise RuntimeError(f"CUDA Graph capture NVTX ranges are missing: {missing_ranges}")
        missing_solo_ranges = sorted(set(expected_ops) - set(solo_capture_ranges))
        if missing_solo_ranges:
            raise RuntimeError(
                f"solo CUDA Graph capture NVTX ranges are missing: {missing_solo_ranges}"
            )
        if len(replay_rows) != len(execution["trace_replays"]):
            raise RuntimeError(
                f"NVTX replay count differs: sqlite={len(replay_rows)}, "
                f"execution={len(execution['trace_replays'])}"
            )
        for op in expected_ops:
            expected_count = len(execution["solo_trace_replays"][op])
            if len(solo_replay_rows[op]) != expected_count:
                raise RuntimeError(
                    f"{op}: solo replay count differs: "
                    f"sqlite={len(solo_replay_rows[op])}, execution={expected_count}"
                )

        capture_ids: dict[str, set[int]] = {}
        for op in expected_ops:
            start, end = capture_ranges[op]
            ids = {
                int(row["graphNodeId"])
                for row in connection.execute(
                    "SELECT graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                    "WHERE start>=? AND start<=? AND graphNodeId IS NOT NULL",
                    (start, end),
                )
            }
            if not ids:
                raise RuntimeError(f"{op}: capture range owns no CUDA Graph nodes")
            capture_ids[op] = ids
        solo_capture_ids: dict[str, set[int]] = {}
        for op in expected_ops:
            start, end = solo_capture_ranges[op]
            ids = {
                int(row["graphNodeId"])
                for row in connection.execute(
                    "SELECT graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                    "WHERE start>=? AND start<=? AND graphNodeId IS NOT NULL",
                    (start, end),
                )
            }
            if not ids:
                raise RuntimeError(f"{op}: solo capture range owns no CUDA Graph nodes")
            solo_capture_ids[op] = ids

        clone_edges = [
            (int(row["originalGraphNodeId"]), int(row["graphNodeId"]))
            for row in connection.execute(
                "SELECT originalGraphNodeId,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
                "WHERE originalGraphNodeId IS NOT NULL AND graphNodeId IS NOT NULL"
            )
        ]
        def expand_clone_ids(source_ids: set[int]) -> set[int]:
            expanded = set(source_ids)
            changed = True
            while changed:
                changed = False
                for original, clone in clone_edges:
                    if original in expanded and clone not in expanded:
                        expanded.add(clone)
                        changed = True
            return expanded

        replay_ids = {
            op: expand_clone_ids(source_ids)
            for op, source_ids in capture_ids.items()
        }
        solo_replay_ids = {
            op: expand_clone_ids(source_ids)
            for op, source_ids in solo_capture_ids.items()
        }

        replay_results = []
        for replay_index, replay in enumerate(replay_rows):
            intervals_by_op: dict[str, list[tuple[int, int]]] = {}
            kernel_rows_by_op: dict[str, list[dict[str, Any]]] = {}
            kernels = connection.execute(
                "SELECT start,end,streamId,correlationId,shortName,graphNodeId "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL "
                "ORDER BY start,end",
                (int(replay["start"]), int(replay["end"])),
            ).fetchall()
            for op in expected_ops:
                kernel_rows = [
                    row for row in kernels
                    if int(row["graphNodeId"]) in replay_ids[op]
                ]
                if not kernel_rows:
                    raise RuntimeError(
                        f"replay {replay_index} {op}: no mapped CUDA Graph kernels"
                    )
                intervals_by_op[op] = [
                    (int(row["start"]), int(row["end"])) for row in kernel_rows
                ]
                kernel_rows_by_op[op] = [
                    {
                        "start": int(row["start"]),
                        "end": int(row["end"]),
                        "duration_ns": int(row["end"]) - int(row["start"]),
                        "stream_id": int(row["streamId"]),
                        "correlation_id": int(row["correlationId"]),
                        "short_name_id": int(row["shortName"]),
                        "graph_node_id": int(row["graphNodeId"]),
                    }
                    for row in kernel_rows
                ]
            overlap_ns = strict_intersection_ns(intervals_by_op)
            replay_results.append({
                "index": replay_index,
                "strict_all_op_overlap": overlap_ns > 0,
                "strict_intersection_ns": overlap_ns,
                "operators": {
                    op: {
                        "kernel_count": len(kernel_rows_by_op[op]),
                        "streams": sorted({row["stream_id"] for row in kernel_rows_by_op[op]}),
                        "first_start": min(start for start, _ in intervals_by_op[op]),
                        "last_end": max(end for _, end in intervals_by_op[op]),
                        "span_ns": (
                            max(end for _, end in intervals_by_op[op])
                            - min(start for start, _ in intervals_by_op[op])
                        ),
                        "active_union_ns": sum(
                            end - start for start, end in merged(intervals_by_op[op])
                        ),
                    }
                    for op in expected_ops
                },
            })

        solo_results: dict[str, list[dict[str, Any]]] = {
            op: [] for op in expected_ops
        }
        for op in expected_ops:
            for replay in sorted(solo_replay_rows[op], key=lambda row: row["index"]):
                kernels = connection.execute(
                    "SELECT start,end,streamId,graphNodeId "
                    "FROM CUPTI_ACTIVITY_KIND_KERNEL "
                    "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL "
                    "ORDER BY start,end",
                    (int(replay["start"]), int(replay["end"])),
                ).fetchall()
                matched = [
                    row for row in kernels
                    if int(row["graphNodeId"]) in solo_replay_ids[op]
                ]
                if not matched:
                    raise RuntimeError(
                        f"solo replay {replay['index']} {op}: no mapped kernels"
                    )
                intervals = [
                    (int(row["start"]), int(row["end"])) for row in matched
                ]
                solo_results[op].append({
                    "index": int(replay["index"]),
                    "kernel_count": len(matched),
                    "streams": sorted({int(row["streamId"]) for row in matched}),
                    "span_ns": max(end for _, end in intervals) - min(
                        start for start, _ in intervals
                    ),
                    "active_union_ns": sum(
                        end - start for start, end in merged(intervals)
                    ),
                })
    finally:
        connection.close()

    strict_count = sum(row["strict_all_op_overlap"] for row in replay_results)
    slowdown = {}
    for op in expected_ops:
        solo_median = statistics.median(row["span_ns"] for row in solo_results[op])
        group_median = statistics.median(
            row["operators"][op]["span_ns"] for row in replay_results
        )
        slowdown[op] = {
            "solo_span_median_ns": solo_median,
            "group_span_median_ns": group_median,
            "slowdown": group_median / solo_median - 1.0,
        }
    output = {
        "schema_version": 1,
        "model": execution["model"],
        "call": execution["call"],
        "group": expected_ops,
        "sqlite": str(sqlite_path),
        "sqlite_sha256": sha256_file(sqlite_path),
        "execution_json": str(execution_json),
        "execution_sha256": sha256_file(execution_json),
        "replay_count": len(replay_results),
        "strict_overlap_count": strict_count,
        "strict_all_replays": strict_count == len(replay_results),
        "capture_graph_node_ids": {
            op: sorted(ids) for op, ids in capture_ids.items()
        },
        "solo_capture_graph_node_ids": {
            op: sorted(ids) for op, ids in solo_capture_ids.items()
        },
        "solo_replays": solo_results,
        "per_operator_slowdown": slowdown,
        "replays": replay_results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "ok",
        "strict_overlap": f"{strict_count}/{len(replay_results)}",
        "output_json": str(output_json),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
