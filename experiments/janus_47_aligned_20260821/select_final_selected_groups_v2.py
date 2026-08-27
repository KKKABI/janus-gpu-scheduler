#!/usr/bin/env python3
"""Build a Janus 4.7 manifest from final-selected groups, not candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any


MODEL_ALIASES = {
    "GoogLeNet": "GoogLeNet",
    "Inception-v3": "Inception-v3",
    "NASNet": "NASNet",
    "YOLOv8x": "YOLOv8x",
    "ConvNeXt": "ConvNeXt",
    "DeepFM": "DeepFM",
    "BERT": "BERT",
}


def split_ops(value: str) -> list[str]:
    return [part.strip() for part in value.split(" + ") if part.strip()]


def load_original_janus(discovery_root: Path):
    rows = []
    profile_sha = {}
    sources = []
    for path in sorted(discovery_root.glob("*_static_path/scheduler_calls.json")):
        case_dir = path.parent
        identity = json.loads((case_dir / "candidates.json").read_text(encoding="utf-8"))
        model = MODEL_ALIASES[str(identity["model"])]
        if identity["reference_variant"] != "Baseline":
            raise RuntimeError(f"not a baseline path: {case_dir}")
        if not identity["correctness"].get("ok", False):
            raise RuntimeError(f"correctness failed: {model}")
        profile_sha[model] = identity["profile_sha256"]
        calls = json.loads(path.read_text(encoding="utf-8"))
        for call in calls:
            group = list(map(str, call.get("selected_resource", [])))
            if len(group) < 2:
                continue
            rows.append({
                "model": model,
                "group": group,
                "size": len(group),
                "method": "original_janus",
                "call": int(call["call"]),
                "predicted_occupancy": float(call["occ_max"]),
                "source": str(path.resolve()),
            })
        sources.append({"model": model, "method": "original_janus", "path": str(path.resolve())})
    return rows, profile_sha, sources


def load_new_td(newtd_root: Path):
    rows = []
    profile_sha = {}
    sources = []
    for path in sorted(newtd_root.glob("*/calls.csv")):
        case_dir = path.parent
        precision = json.loads((case_dir / "precision.json").read_text(encoding="utf-8"))
        summary = json.loads((case_dir / "artifacts" / "summary.json").read_text(encoding="utf-8"))
        admission_by_call = {
            int(row["call"]): row.get("admission")
            for row in summary.get("selected_admission_trace", [])
        }
        model = MODEL_ALIASES[str(precision["model"])]
        if precision["configuration"] != "NewTD(pair extension)+DRT":
            raise RuntimeError(f"unexpected configuration: {case_dir}")
        if not precision["correctness"].get("ok", False):
            raise RuntimeError(f"correctness failed: {model}")
        profile_sha[model] = precision["profile_sha256"]
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for call in csv.DictReader(stream):
                group = split_ops(call["selected_ops"])
                if len(group) < 2:
                    continue
                rows.append({
                    "model": model,
                    "group": group,
                    "size": len(group),
                    "method": "new_td",
                    "call": int(call["call"]),
                    "predicted_occupancy": float(
                        admission_by_call[int(call["call"])]["initial_utilization"]
                    ),
                    "source": str(path.resolve()),
                })
        sources.append({"model": model, "method": "new_td", "path": str(path.resolve())})
    return rows, profile_sha, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--newtd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original, original_sha, original_sources = load_original_janus(args.discovery_root)
    newtd, newtd_sha, newtd_sources = load_new_td(args.newtd_root)
    if set(original_sha) != set(MODEL_ALIASES) or set(newtd_sha) != set(MODEL_ALIASES):
        raise RuntimeError("seven-model matrix is incomplete")
    if original_sha != newtd_sha:
        differing = [model for model in original_sha if original_sha[model] != newtd_sha[model]]
        raise RuntimeError(f"profile SHA differs between methods: {differing}")

    unique: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    method_counts: dict[tuple[str, str], int] = defaultdict(int)
    for source in original + newtd:
        method_counts[(source["model"], source["method"])] += 1
        key = (source["model"], tuple(source["group"]))
        row = unique.setdefault(key, {
            "model": source["model"],
            "group": source["group"],
            "size": source["size"],
            "selected_for_methods": [],
            "source_occurrences": [],
        })
        if source["method"] not in row["selected_for_methods"]:
            row["selected_for_methods"].append(source["method"])
        row["source_occurrences"].append({
            "method": source["method"],
            "call": source["call"],
            "predicted_occupancy": source["predicted_occupancy"],
            "source": source["source"],
        })

    cases = []
    for index, key in enumerate(sorted(unique), 1):
        row = unique[key]
        first = row["source_occurrences"][0]
        predicted_by_method = {}
        for occurrence in row["source_occurrences"]:
            method = occurrence["method"]
            predicted_by_method[method] = max(
                predicted_by_method.get(method, 0.0),
                float(occurrence["predicted_occupancy"]),
            )
        cases.append({
            "case_id": f"janus4_7_selected_{index:04d}",
            **row,
            "call": first["call"],
            "predicted_occupancy_by_method": predicted_by_method,
            "stratum_population": method_counts[(row["model"], row["selected_for_methods"][0])],
            "sample_weight": 1.0,
            "original_max_concurrent": None,
            "original_any_pair_overlap_ns": None,
        })

    method_summary = []
    for model in sorted(MODEL_ALIASES):
        for method in ("original_janus", "new_td"):
            method_summary.append({
                "model": model,
                "method": method,
                "final_selected_multi_operator_groups": method_counts[(model, method)],
                "isolated_case_count": sum(
                    row["model"] == model and method in row["selected_for_methods"]
                    for row in cases
                ),
            })
    payload = {
        "schema_version": 1,
        "protocol": "janus_section_4_7_final_selected_positive_precision_v1",
        "git_head": "32bf4974994005855896a360c34ba455303f5ff3",
        "source_profile_sha256_by_model": original_sha,
        "controls": {
            "all_lp_forced_to_hp": True,
            "max_ready": 6,
            "final_selected_groups_only": True,
            "rejected_candidates_evaluated": False,
        },
        "sampling": "all unique final-selected multi-operator groups from one canonical path per model and method",
        "primary_truth": "one common-start isolated CUDA Graph replay with strict full-group kernel overlap",
        "sources": original_sources + newtd_sources,
        "method_summary": method_summary,
        "case_count": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "method_summary": method_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
