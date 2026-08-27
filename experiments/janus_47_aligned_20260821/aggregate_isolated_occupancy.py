#!/usr/bin/env python3
"""Aggregate seven-model isolated occupancy and concurrency feasibility."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def summarize(method, model, width, rows):
    auditable = [row for row in rows if row["auditable"]]
    feasible = [row for row in auditable if row["strict_parallel_replays"] > 0]
    return {
        "method": method,
        "model": model,
        "width": width,
        "selected_groups": len(rows),
        "auditable_groups": len(auditable),
        "unmappable_groups": len(rows) - len(auditable),
        "measurement_coverage": len(auditable) / len(rows) if rows else None,
        "observed_concurrent_groups": len(feasible),
        "observed_concurrency_feasibility_rate": (
            len(feasible) / len(auditable)
            if auditable else None
        ),
        "positive_precision_on_auditable_groups": (
            len(feasible) / len(auditable) if auditable else None
        ),
        "positive_precision_conservative_lower_bound": (
            len(feasible) / len(rows) if rows else None
        ),
        "all_1000_replays_concurrent_groups": sum(
            row["strict_parallel_replays"] == row["metrics_replays"] for row in auditable
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repair-json", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input_root.glob("*/occupancy.json"))
    ]
    repair_rows = {}
    for path in args.repair_json:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["groups"]:
            if row["case_id"] in repair_rows:
                raise RuntimeError(f"duplicate repair case: {row['case_id']}")
            if not row["auditable"]:
                raise RuntimeError(f"repair remains unauditable: {row['case_id']}")
            repair_rows[row["case_id"]] = row
    rows_by_method = defaultdict(list)
    rows_by_method_model = defaultdict(list)
    rows_by_method_width = defaultdict(list)
    physical = []
    seen = set()
    for payload in payloads:
        for row in payload["groups"]:
            row = repair_rows.get(row["case_id"], row)
            if row["case_id"] in seen:
                raise RuntimeError(f"duplicate case: {row['case_id']}")
            seen.add(row["case_id"])
            physical.append(row)
            for method in row["selected_for_methods"]:
                rows_by_method[method].append(row)
                rows_by_method_model[(method, row["model"])].append(row)
                rows_by_method_width[(method, row["width"])].append(row)
    summaries = []
    unused_repairs = sorted(set(repair_rows) - seen)
    if unused_repairs:
        raise RuntimeError(f"repair cases absent from base data: {unused_repairs}")
    for method, rows in sorted(rows_by_method.items()):
        summaries.append(summarize(method, "ALL", None, rows))
    for (method, model), rows in sorted(rows_by_method_model.items()):
        summaries.append(summarize(method, model, None, rows))
    for (method, width), rows in sorted(rows_by_method_width.items()):
        summaries.append(summarize(method, "ALL", width, rows))
    args.output_dir.mkdir(parents=True)
    result = {
        "schema_version": 1,
        "protocol": "janus_section_4_7_isolated_gpu_occupancy_aggregate_v1",
        "truth_definition": (
            "A positive group is observed feasible when all OPs have a strict common kernel interval in at least one of 1000 isolated common-start replays."
        ),
        "paper_sm_occupancy_error_status": "not_reproduced",
        "occupancy_boundary": (
            "RTX A5000 Nsight Systems exposes Compute Warps in Flight [Throughput %] "
            "in this capture, which is not the paper's actual SM occupancy definition. "
            "The raw samples remain in each model occupancy.json but are not used to "
            "claim the paper's 4.02% occupancy MAE."
        ),
        "physical_case_count": len(physical),
        "repair_jsons": [str(path.resolve()) for path in args.repair_json],
        "repaired_case_count": len(repair_rows),
        "summaries": summaries,
        "groups": physical,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(summaries[0])
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps([row for row in summaries if row["model"] == "ALL" and row["width"] is None], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
