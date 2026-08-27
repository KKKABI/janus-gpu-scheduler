#!/usr/bin/env python3
"""Align Janus selected groups with Nsight Systems GPU occupancy samples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sqlite3
import statistics


METRIC_NAME = "Compute Warps in Flight [Throughput %]"


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


def strict_overlap_ns(intervals_by_op):
    common = list(intervals_by_op.values())[0]
    for spans in list(intervals_by_op.values())[1:]:
        common = intersect(common, spans)
        if not common:
            return 0
    return sum(end - start for start, end in common)


def load_reference_mapping(path: Path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    strings = {int(row["id"]): row["value"] for row in db.execute("SELECT id,value FROM StringIds")}
    prefix = "FX::GRAPH_CAPTURE::"
    capture_ranges = {}
    for row in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        text = resolved(row, strings)
        if text and text.startswith(prefix):
            capture_ranges[text[len(prefix):]] = (int(row["start"]), int(row["end"]))
    source_ids = {}
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
            source_ids[op] = ids
    clone_edges = [
        (int(row["originalGraphNodeId"]), int(row["graphNodeId"]))
        for row in db.execute(
            "SELECT originalGraphNodeId,graphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
            "WHERE originalGraphNodeId IS NOT NULL AND graphNodeId IS NOT NULL"
        )
    ]
    expanded = {}
    for op, ids in source_ids.items():
        values = set(ids)
        changed = True
        while changed:
            changed = False
            for original, clone in clone_edges:
                if original in values and clone not in values:
                    values.add(clone)
                    changed = True
        expanded[op] = values
    topology = Counter(
        (
            int(row["graphNodeId"]),
            int(row["originalGraphNodeId"]) if row["originalGraphNodeId"] is not None else None,
        )
        for row in db.execute(
            "SELECT graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
            "WHERE graphNodeId IS NOT NULL"
        )
    )
    db.close()
    return expanded, topology


def metric_id(db):
    row = db.execute(
        "SELECT data FROM GENERIC_EVENT_TYPES ORDER BY length(data) DESC LIMIT 1"
    ).fetchone()
    fields = json.loads(row[0])["Fields"]
    matches = [index for index, field in enumerate(fields) if field["Name"] == METRIC_NAME]
    if len(matches) != 1:
        raise RuntimeError(f"metric lookup failed for {METRIC_NAME}: {matches}")
    return matches[0]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-sqlite", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--metrics-sqlite", type=Path, required=True)
    parser.add_argument("--metrics-summary", type=Path, required=True)
    parser.add_argument("--metrics-calls", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_csv.exists():
        raise FileExistsError("refusing to overwrite output")
    reference_summary = json.loads(args.reference_summary.read_text(encoding="utf-8"))
    metrics_summary = json.loads(args.metrics_summary.read_text(encoding="utf-8"))
    for key in ("model", "configuration", "profile_sha256", "trace_tag"):
        if reference_summary[key] != metrics_summary[key]:
            raise RuntimeError(f"summary identity differs for {key}")
    replay_count = int(metrics_summary["paper_aligned_controls"]["gpu_metrics_replays"])
    if replay_count != 100:
        raise RuntimeError(f"expected 100 metrics replays, got {replay_count}")
    op_graph_ids, reference_topology = load_reference_mapping(args.reference_sqlite)

    db = sqlite3.connect(args.metrics_sqlite)
    db.row_factory = sqlite3.Row
    strings = {int(row["id"]): row["value"] for row in db.execute("SELECT id,value FROM StringIds")}
    metrics_topology = Counter(
        (
            int(row["graphNodeId"]),
            int(row["originalGraphNodeId"]) if row["originalGraphNodeId"] is not None else None,
        )
        for row in db.execute(
            "SELECT graphNodeId,originalGraphNodeId FROM CUDA_GRAPH_NODE_EVENTS "
            "WHERE graphNodeId IS NOT NULL"
        )
    )
    if metrics_topology != reference_topology:
        raise RuntimeError("CUDA Graph topology/IDs differ from the identity-matched reference trace")
    marker = f"JANUS_PRECISION_REPLAY::{metrics_summary['trace_tag']}"
    replay_ranges = [
        (int(row["start"]), int(row["end"]))
        for row in db.execute("SELECT start,end,text,textId FROM NVTX_EVENTS WHERE end IS NOT NULL")
        if resolved(row, strings) == marker
    ]
    if len(replay_ranges) != 1:
        raise RuntimeError(f"replay marker count={len(replay_ranges)}")
    replay_start, replay_end = replay_ranges[0]
    kernels = [
        dict(row)
        for row in db.execute(
            "SELECT start,end,graphNodeId,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE start>=? AND end<=? AND graphNodeId IS NOT NULL ORDER BY start,end",
            (replay_start, replay_end),
        )
    ]
    kernels_by_node = defaultdict(list)
    for row in kernels:
        kernels_by_node[int(row["graphNodeId"])].append(row)
    used_node_counts = {len(rows) for rows in kernels_by_node.values()}
    if used_node_counts != {replay_count}:
        raise RuntimeError(f"kernel graph-node replay counts differ: {sorted(used_node_counts)}")
    selected_metric_id = metric_id(db)
    metric_samples = [
        (int(row["timestamp"]), float(row["value"]))
        for row in db.execute(
            "SELECT timestamp,value FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp",
            (selected_metric_id,),
        )
    ]
    calls = json.loads(args.metrics_calls.read_text(encoding="utf-8"))
    selected_calls = [
        row for row in calls
        if int(row.get("selected_gpu_resource_size", 0) or 0) >= 2
    ]
    admission_by_call = {
        int(row["call"]): row.get("admission")
        for row in metrics_summary.get("selected_admission_trace", [])
    }

    results = []
    for call in selected_calls:
        names = list(call["selected_gpu_resource"])
        missing = [name for name in names if name not in op_graph_ids]
        node_ids_by_op = {
            name: sorted(node for node in op_graph_ids.get(name, set()) if node in kernels_by_node)
            for name in names
        }
        missing.extend(name for name, ids in node_ids_by_op.items() if not ids and name not in missing)
        if missing:
            results.append({
                "call": int(call["call"]),
                "operators": names,
                "width": len(names),
                "auditable": False,
                "missing_ops": sorted(set(missing)),
            })
            continue
        per_replay_peaks = []
        per_replay_means = []
        strict_replays = 0
        sampled_replays = 0
        for replay_index in range(replay_count):
            intervals_by_op = {}
            for name, node_ids in node_ids_by_op.items():
                spans = [
                    (
                        int(kernels_by_node[node][replay_index]["start"]),
                        int(kernels_by_node[node][replay_index]["end"]),
                    )
                    for node in node_ids
                ]
                intervals_by_op[name] = merge(spans)
            strict_replays += strict_overlap_ns(intervals_by_op) > 0
            all_spans = [span for spans in intervals_by_op.values() for span in spans]
            window_start = min(start for start, _ in all_spans)
            window_end = max(end for _, end in all_spans)
            values = [value for timestamp, value in metric_samples if window_start <= timestamp <= window_end]
            if values:
                sampled_replays += 1
                per_replay_peaks.append(max(values))
                per_replay_means.append(statistics.fmean(values))
        admission = admission_by_call.get(int(call["call"])) or {}
        predicted = admission.get("initial_utilization", call.get("occ_max"))
        predicted_pct = float(predicted) * 100.0 if predicted is not None else None
        actual_peak_median = statistics.median(per_replay_peaks) if per_replay_peaks else None
        results.append({
            "call": int(call["call"]),
            "operators": names,
            "width": len(names),
            "auditable": True,
            "missing_ops": [],
            "predicted_occupancy_pct": predicted_pct,
            "actual_compute_warps_peak_median_pct": actual_peak_median,
            "actual_compute_warps_peak_p95_pct": percentile(per_replay_peaks, 0.95),
            "actual_compute_warps_mean_median_pct": statistics.median(per_replay_means) if per_replay_means else None,
            "occupancy_sampled_replays": sampled_replays,
            "strict_parallel_replays": strict_replays,
            "strict_parallel_rate": strict_replays / replay_count,
            "absolute_occupancy_error_pct_points": (
                abs(actual_peak_median - predicted_pct)
                if actual_peak_median is not None and predicted_pct is not None else None
            ),
            "admission_source": admission.get("admission_source"),
        })
    db.close()
    occupancy_rows = [
        row for row in results if row.get("absolute_occupancy_error_pct_points") is not None
    ]
    payload = {
        "schema_version": 1,
        "protocol": "janus_section_4_7_sm_occupancy_metrics_v1",
        "model": metrics_summary["model"],
        "configuration": metrics_summary["configuration"],
        "profile_sha256": metrics_summary["profile_sha256"],
        "gpu_metric": METRIC_NAME,
        "metric_frequency_hz": 200000,
        "metrics_replays": replay_count,
        "selected_multi_operator_groups": len(results),
        "auditable_groups": sum(row["auditable"] for row in results),
        "occupancy_sampled_groups": len(occupancy_rows),
        "strict_parallel_groups_all_100": sum(row.get("strict_parallel_replays") == replay_count for row in results),
        "strict_parallel_groups_any_of_100": sum(row.get("strict_parallel_replays", 0) > 0 for row in results),
        "occupancy_mae_percentage_points": (
            statistics.fmean(row["absolute_occupancy_error_pct_points"] for row in occupancy_rows)
            if occupancy_rows else None
        ),
        "important_boundary": (
            "Nsight Systems on GA10x exposes Compute Warps in Flight throughput, not a field literally named SM Occupancy; this is the closest hardware occupancy signal and is reported by its exact metric name."
        ),
        "groups": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "call", "operators", "width", "auditable", "missing_ops",
        "predicted_occupancy_pct", "actual_compute_warps_peak_median_pct",
        "actual_compute_warps_peak_p95_pct", "actual_compute_warps_mean_median_pct",
        "occupancy_sampled_replays", "strict_parallel_replays", "strict_parallel_rate",
        "absolute_occupancy_error_pct_points", "admission_source",
    ]
    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({
                **row,
                "operators": " + ".join(row["operators"]),
                "missing_ops": " + ".join(row["missing_ops"]),
            })
    print(json.dumps({key: payload[key] for key in (
        "model", "configuration", "selected_multi_operator_groups",
        "auditable_groups", "occupancy_sampled_groups",
        "strict_parallel_groups_any_of_100", "strict_parallel_groups_all_100",
        "occupancy_mae_percentage_points",
    )}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
