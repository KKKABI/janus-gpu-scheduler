#!/usr/bin/env python3
"""Aggregate exact-ready paired §4.8 isolated group measurements."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import sha256_file, write_json_atomic


TIMING_PROCESSES = 5
TRACE_REPLAYS = 10


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.results_root.resolve()
    args.output_dir.mkdir(parents=True)

    cases = {}
    for case in manifest["cases"]:
        case_root = root / case["case_id"]
        process_speedups = []
        process_group_medians = []
        process_solo_medians = defaultdict(list)
        timing_sources = []
        for trial in range(TIMING_PROCESSES):
            path = case_root / "timing" / f"trial_{trial:02d}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model") != case["model"] or payload.get("group") != case["group"]:
                raise RuntimeError(f"{case['case_id']}: timing identity differs")
            if payload.get("profile_sha256") != case["profile_sha256"]:
                raise RuntimeError(f"{case['case_id']}: timing profile SHA differs")
            if payload.get("fx_code_sha256") != case["fx_code_sha256"]:
                raise RuntimeError(f"{case['case_id']}: timing FX identity differs")
            if set(payload.get("correctness", {})) != set(case["group"]):
                raise RuntimeError(f"{case['case_id']}: missing correctness result")
            timing = payload["timing"]
            solo_sum = 0.0
            for op in case["group"]:
                value = float(timing["solo"][op]["median_ms"])
                solo_sum += value
                process_solo_medians[op].append(value)
            group_median = float(timing["group"]["median_ms"])
            process_group_medians.append(group_median)
            process_speedups.append(solo_sum / group_median)
            timing_sources.append(
                {"path": str(path), "sha256": sha256_file(path)}
            )
        overlap_path = case_root / "nsys" / "overlap.json"
        overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
        if overlap.get("model") != case["model"] or overlap.get("group") != case["group"]:
            raise RuntimeError(f"{case['case_id']}: NSYS identity differs")
        if int(overlap.get("replay_count", 0)) != TRACE_REPLAYS:
            raise RuntimeError(f"{case['case_id']}: expected ten NSYS replays")
        per_op_slowdown = {
            op: float(overlap["per_operator_slowdown"][op]["slowdown"])
            for op in case["group"]
        }
        cases[case["case_id"]] = {
            **case,
            "timing_processes": TIMING_PROCESSES,
            "process_speedups": process_speedups,
            "final_group_speedup": statistics.median(process_speedups),
            "group_latency_ms": summarize(process_group_medians),
            "solo_latency_ms": {
                op: summarize(values) for op, values in process_solo_medians.items()
            },
            "strict_overlap_replays": int(overlap["strict_overlap_count"]),
            "trace_replays": int(overlap["replay_count"]),
            "strict_overlap_observed": int(overlap["strict_overlap_count"]) > 0,
            "per_operator_slowdown": per_op_slowdown,
            "average_slowdown": statistics.fmean(per_op_slowdown.values()),
            "worst_slowdown": max(per_op_slowdown.values()),
            "timing_sources": timing_sources,
            "overlap_source": {
                "path": str(overlap_path),
                "sha256": sha256_file(overlap_path),
            },
        }

    pair_rows = []
    for pair in manifest["pairs"]:
        left = cases[pair["case_ids"][0]]
        right = cases[pair["case_ids"][1]]
        if left["ready_signature_sha256"] != right["ready_signature_sha256"]:
            raise RuntimeError(f"{pair['pair_id']}: ready signature differs")
        valid = left["strict_overlap_observed"] and right["strict_overlap_observed"]
        difference = right["average_slowdown"] - left["average_slowdown"]
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "comparison": pair["comparison"],
                "comparison_role": pair["comparison_role"],
                "model": pair["model"],
                "ready_signature_sha256": left["ready_signature_sha256"],
                "left_policy": left["policy"],
                "right_policy": right["policy"],
                "left_group": left["group"],
                "right_group": right["group"],
                "left_resource_class": left["resource_class"],
                "right_resource_class": right["resource_class"],
                "left_strict_overlap_replays": left["strict_overlap_replays"],
                "right_strict_overlap_replays": right["strict_overlap_replays"],
                "valid_paired_interference_result": valid,
                "left_average_slowdown": left["average_slowdown"],
                "right_average_slowdown": right["average_slowdown"],
                "slowdown_difference_right_minus_left": difference,
                "left_group_speedup": left["final_group_speedup"],
                "right_group_speedup": right["final_group_speedup"],
            }
        )

    valid_pairs = [row for row in pair_rows if row["valid_paired_interference_result"]]
    by_comparison = {}
    for comparison in sorted({row["comparison"] for row in pair_rows}):
        rows = [row for row in valid_pairs if row["comparison"] == comparison]
        differences = [row["slowdown_difference_right_minus_left"] for row in rows]
        by_comparison[comparison] = {
            "planned_pairs": sum(row["comparison"] == comparison for row in pair_rows),
            "valid_pairs": len(rows),
            "slowdown_difference_right_minus_left": summarize(differences),
            "right_lower_slowdown_pairs": sum(value < 0 for value in differences),
            "equal_slowdown_pairs": sum(value == 0 for value in differences),
            "right_higher_slowdown_pairs": sum(value > 0 for value in differences),
        }

    by_policy_class = {}
    buckets = defaultdict(list)
    for case in cases.values():
        if case["strict_overlap_observed"]:
            buckets[(case["policy"], case["resource_class"])].append(case)
    for (policy, resource_class), rows in sorted(buckets.items()):
        by_policy_class[f"{policy}:{resource_class}"] = {
            "policy": policy,
            "resource_class": resource_class,
            "group_count": len(rows),
            "group_speedup": summarize([row["final_group_speedup"] for row in rows]),
            "average_slowdown": summarize([row["average_slowdown"] for row in rows]),
        }

    payload = {
        "schema_version": 1,
        "status": "completed",
        "protocol": "janus_4_8_exact_same_ready_paired_isolated_measurement_v1",
        "slowdown_definition": "concurrent OP kernel span / solo OP kernel span - 1",
        "speedup_definition": "sum of solo median CUDA-event times / concurrent group median CUDA-event time",
        "primary_inclusion_rule": "both policy-selected groups show strict full-group kernel overlap in at least one of ten NSYS replays",
        "timing_processes_per_case": TIMING_PROCESSES,
        "nsys_replays_per_case": TRACE_REPLAYS,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "planned_pairs": len(pair_rows),
        "valid_pairs": len(valid_pairs),
        "by_comparison": by_comparison,
        "by_policy_resource_class": by_policy_class,
        "pairs": pair_rows,
        "cases": list(cases.values()),
    }
    write_json_atomic(args.output_dir / "summary.json", payload)
    if pair_rows:
        fields = [
            "pair_id",
            "comparison",
            "comparison_role",
            "model",
            "left_policy",
            "right_policy",
            "left_resource_class",
            "right_resource_class",
            "left_strict_overlap_replays",
            "right_strict_overlap_replays",
            "valid_paired_interference_result",
            "left_average_slowdown",
            "right_average_slowdown",
            "slowdown_difference_right_minus_left",
            "left_group_speedup",
            "right_group_speedup",
        ]
        with (args.output_dir / "pairs.csv").open(
            "x", encoding="utf-8-sig", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(pair_rows)
    print(json.dumps({"planned_pairs": len(pair_rows), "valid_pairs": len(valid_pairs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
