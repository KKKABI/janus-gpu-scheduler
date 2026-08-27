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
PRIMARY_RESOURCE_CLASSES = ("pure_compute", "pure_memory", "mixed_resource")


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


def completion_status(pair_rows: list[dict]) -> dict:
    primary = [
        row
        for row in pair_rows
        if row["comparison_role"] == "primary"
        and row["same_resource_class"]
    ]
    valid = [
        row for row in primary if row["valid_paired_interference_result"]
    ]
    reasons = []
    if not primary:
        reasons.append("no_primary_same_class_pair_was_planned")
    if primary and not valid:
        reasons.append("no_primary_same_class_pair_passed_strict_overlap")
    return {
        "status": "completed" if primary and valid else "inconclusive",
        "primary_planned_pairs": len(primary),
        "primary_valid_pairs": len(valid),
        "inconclusive_reasons": reasons,
    }


def paired_summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["valid_paired_interference_result"]]
    differences = [row["slowdown_difference_right_minus_left"] for row in valid]
    return {
        "planned_pairs": len(rows),
        "valid_pairs": len(valid),
        "left_average_slowdown": summarize(
            [row["left_average_slowdown"] for row in valid]
        ),
        "right_average_slowdown": summarize(
            [row["right_average_slowdown"] for row in valid]
        ),
        "left_group_speedup": summarize(
            [row["left_group_speedup"] for row in valid]
        ),
        "right_group_speedup": summarize(
            [row["right_group_speedup"] for row in valid]
        ),
        "slowdown_difference_right_minus_left": summarize(differences),
        "right_lower_slowdown_pairs": sum(value < 0 for value in differences),
        "equal_slowdown_pairs": sum(value == 0 for value in differences),
        "right_higher_slowdown_pairs": sum(value > 0 for value in differences),
    }


def split_resource_pair_rows(pair_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    same_class = [row for row in pair_rows if row["same_resource_class"]]
    heterogeneous = [row for row in pair_rows if not row["same_resource_class"]]
    if set(row["pair_id"] for row in same_class) & set(
        row["pair_id"] for row in heterogeneous
    ):
        raise AssertionError("same-class and heterogeneous pair tables overlap")
    if any(
        row.get("paired_resource_class") not in PRIMARY_RESOURCE_CLASSES
        for row in same_class
    ):
        raise RuntimeError("unclassified pair reached the same-class table")
    return same_class, heterogeneous


def write_pair_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "pair_id",
        "comparison",
        "comparison_role",
        "analysis_bucket",
        "model",
        "left_policy",
        "right_policy",
        "left_resource_class",
        "right_resource_class",
        "paired_resource_class",
        "resource_class_transition",
        "left_strict_overlap_replays",
        "right_strict_overlap_replays",
        "valid_paired_interference_result",
        "left_average_slowdown",
        "right_average_slowdown",
        "slowdown_difference_right_minus_left",
        "left_group_speedup",
        "right_group_speedup",
    ]
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready_for_isolated_measurement":
        raise RuntimeError(
            f"manifest is not ready: {manifest.get('status')}"
        )
    if int(manifest.get("primary_pair_count", 0)) <= 0:
        raise RuntimeError("manifest has no planned primary same-class pair")
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
            if payload.get("call") != case["call"] or payload.get("mode") != "timing":
                raise RuntimeError(f"{case['case_id']}: timing call/mode differs")
            if payload.get("profile_sha256") != case["profile_sha256"]:
                raise RuntimeError(f"{case['case_id']}: timing profile SHA differs")
            if payload.get("fx_code_sha256") != case["fx_code_sha256"]:
                raise RuntimeError(f"{case['case_id']}: timing FX identity differs")
            if set(payload.get("correctness", {})) != set(case["group"]):
                raise RuntimeError(f"{case['case_id']}: missing correctness result")
            timing = payload["timing"]
            if int(timing.get("repeats", 0)) != 100:
                raise RuntimeError(f"{case['case_id']}: timing repeats differ")
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
        if overlap.get("call") != case["call"]:
            raise RuntimeError(f"{case['case_id']}: NSYS call differs")
        if int(overlap.get("replay_count", 0)) != TRACE_REPLAYS:
            raise RuntimeError(f"{case['case_id']}: expected ten NSYS replays")
        execution_path = case_root / "nsys" / "execution.json"
        sqlite_path = case_root / "nsys" / "full_trace.sqlite"
        if sha256_file(execution_path) != overlap.get("execution_sha256"):
            raise RuntimeError(f"{case['case_id']}: trace execution SHA differs")
        if sha256_file(sqlite_path) != overlap.get("sqlite_sha256"):
            raise RuntimeError(f"{case['case_id']}: trace SQLite SHA differs")
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        if (
            execution.get("model") != case["model"]
            or execution.get("call") != case["call"]
            or execution.get("group") != case["group"]
            or execution.get("mode") != "trace"
            or execution.get("profile_sha256") != case["profile_sha256"]
            or execution.get("fx_code_sha256") != case["fx_code_sha256"]
            or len(execution.get("trace_replays", [])) != TRACE_REPLAYS
        ):
            raise RuntimeError(f"{case['case_id']}: trace execution identity differs")
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
        same_resource_class = (
            left["resource_class"] == right["resource_class"]
        )
        paired_resource_class = (
            left["resource_class"] if same_resource_class else None
        )
        resource_transition = (
            f"{left['resource_class']}->{right['resource_class']}"
        )
        expected_bucket = (
            "same_class_formal"
            if same_resource_class
            else "heterogeneous_exploratory"
        )
        if (
            pair.get("same_resource_class") is not same_resource_class
            or pair.get("paired_resource_class") != paired_resource_class
            or pair.get("resource_class_transition") != resource_transition
            or pair.get("analysis_bucket") != expected_bucket
        ):
            raise RuntimeError(
                f"{pair['pair_id']}: manifest resource classification differs"
            )
        if pair.get("comparison_role") == "primary" and not same_resource_class:
            raise RuntimeError(
                f"{pair['pair_id']}: heterogeneous pair cannot be primary"
            )
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
                "same_resource_class": same_resource_class,
                "paired_resource_class": paired_resource_class,
                "resource_class_transition": resource_transition,
                "analysis_bucket": expected_bucket,
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
    all_same_class_rows, heterogeneous_rows = split_resource_pair_rows(pair_rows)
    by_comparison_same_class = {}
    for comparison in sorted({row["comparison"] for row in all_same_class_rows}):
        rows = [
            row for row in all_same_class_rows
            if row["comparison"] == comparison
        ]
        by_comparison_same_class[comparison] = paired_summary(rows)
    primary_same_class_rows = [
        row
        for row in all_same_class_rows
        if row["comparison_role"] == "primary"
    ]
    by_primary_resource_class = {
        resource_class: paired_summary(
            [
                row for row in primary_same_class_rows
                if row["paired_resource_class"] == resource_class
            ]
        )
        for resource_class in PRIMARY_RESOURCE_CLASSES
    }
    heterogeneous_keys = sorted(
        {
            (row["comparison"], row["resource_class_transition"])
            for row in heterogeneous_rows
        }
    )
    by_heterogeneous_transition = {
        f"{comparison}:{transition}": {
            "comparison": comparison,
            "resource_class_transition": transition,
            **paired_summary(
                [
                    row
                    for row in heterogeneous_rows
                    if row["comparison"] == comparison
                    and row["resource_class_transition"] == transition
                ]
            ),
        }
        for comparison, transition in heterogeneous_keys
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

    completion = completion_status(pair_rows)
    payload = {
        "schema_version": 1,
        "status": completion["status"],
        "protocol": "janus_4_8_exact_same_ready_paired_isolated_measurement_v2",
        "slowdown_definition": "concurrent OP kernel span / solo OP kernel span - 1",
        "speedup_definition": "sum of solo median CUDA-event times / concurrent group median CUDA-event time",
        "primary_inclusion_rule": "both policy-selected groups show strict full-group kernel overlap in at least one of ten NSYS replays",
        "timing_processes_per_case": TIMING_PROCESSES,
        "nsys_replays_per_case": TRACE_REPLAYS,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "planned_pairs": len(pair_rows),
        "valid_pairs": len(valid_pairs),
        **completion,
        "by_comparison_same_class": by_comparison_same_class,
        "primary_same_class_by_resource": by_primary_resource_class,
        "heterogeneous_exploratory_by_transition": by_heterogeneous_transition,
        "by_policy_resource_class": by_policy_class,
        "pairs": pair_rows,
        "cases": list(cases.values()),
    }
    write_json_atomic(args.output_dir / "summary.json", payload)
    write_pair_csv(args.output_dir / "pairs.csv", pair_rows)
    write_pair_csv(
        args.output_dir / "primary_same_class_planned_pairs.csv",
        primary_same_class_rows,
    )
    write_pair_csv(
        args.output_dir / "primary_same_class_valid_pairs.csv",
        [
            row for row in primary_same_class_rows
            if row["valid_paired_interference_result"]
        ],
    )
    write_pair_csv(
        args.output_dir / "heterogeneous_pairs.csv", heterogeneous_rows
    )
    print(json.dumps({
        "status": completion["status"],
        "planned_pairs": len(pair_rows),
        "valid_pairs": len(valid_pairs),
        "primary_planned_pairs": completion["primary_planned_pairs"],
        "primary_valid_pairs": completion["primary_valid_pairs"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
