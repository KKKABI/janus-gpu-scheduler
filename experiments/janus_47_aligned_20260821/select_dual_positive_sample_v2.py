#!/usr/bin/env python3
"""Select held-out positive groups for Janus Section 4.7 precision checks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_MODELS = {
    "GoogLeNet", "Inception-v3", "NASNet", "YOLOv8x",
    "ConvNeXt", "DeepFM", "BERT",
}
EXPECTED_VARIANTS = {"Baseline", "TD+DRT"}


def stable_rank(seed: int, model: str, group: tuple[str, ...], method: str) -> str:
    del method
    material = json.dumps(
        {"seed": seed, "model": model, "group": group},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def new_td_positive(row: dict[str, Any], minimum_overlap_us: float) -> bool:
    if bool(row["static_prediction"]):
        return True
    if int(row["group_size"]) != 2 or not bool(row.get("td_v2_prediction")):
        return False
    overlap_us = float((row.get("td_v2") or {}).get("strict_overlap_duration", 0.0) or 0.0) * 1000.0
    return overlap_us >= minimum_overlap_us


def load_excluded(paths: list[Path]) -> set[tuple[str, tuple[str, ...]]]:
    excluded: set[tuple[str, tuple[str, ...]]] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("cases", []):
            excluded.add((str(row["model"]), tuple(map(str, row["group"]))))
    return excluded


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-model-method", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--minimum-overlap-us", type=float, default=2.0)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.per_model_method < 1:
        raise ValueError("--per-model-method must be positive")

    paths = sorted(args.discovery_root.glob("*/candidates.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    observed = Counter((row["model"], row["reference_variant"]) for row in payloads)
    expected = {(model, variant) for model in EXPECTED_MODELS for variant in EXPECTED_VARIANTS}
    if set(observed) != expected or any(value != 1 for value in observed.values()):
        raise RuntimeError("discovery matrix is incomplete or duplicated")
    heads = {row["git_head"] for row in payloads}
    if len(heads) != 1:
        raise RuntimeError(f"mixed git heads: {sorted(heads)}")
    profile_sha: dict[str, str] = {}
    for payload in payloads:
        if not payload["correctness"].get("ok", False):
            raise RuntimeError(f"discovery correctness failed: {payload['model']}")
        previous = profile_sha.setdefault(payload["model"], payload["profile_sha256"])
        if previous != payload["profile_sha256"]:
            raise RuntimeError(f"profile SHA changed: {payload['model']}")

    excluded = load_excluded(args.exclude_manifest)
    # The isolated replay truth depends on the exact ordered FX group.  Merge
    # repeats seen at multiple scheduler calls/reference paths, retaining every
    # source occurrence for audit.
    unique: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for payload in payloads:
        for source in payload["candidates"]:
            model = str(source["model"])
            group = tuple(map(str, source["operators"]))
            key = (model, group)
            if key in excluded:
                continue
            static = bool(source["static_prediction"])
            new_td = new_td_positive(source, args.minimum_overlap_us)
            overlap_us = float((source.get("td_v2") or {}).get("strict_overlap_duration", 0.0) or 0.0) * 1000.0
            if key not in unique:
                unique[key] = {
                    "model": model,
                    "group": list(group),
                    "size": len(group),
                    "static_prediction": static,
                    "td_v2_prediction": bool(source.get("td_v2_prediction")),
                    "new_td_prediction": new_td,
                    "new_td_predicted_overlap_us": overlap_us,
                    "source_occurrences": [],
                }
            else:
                row = unique[key]
                # A group is a positive if the method admitted it at any
                # observed scheduler state.  Record all states below.
                row["static_prediction"] = row["static_prediction"] or static
                row["td_v2_prediction"] = row["td_v2_prediction"] or bool(source.get("td_v2_prediction"))
                row["new_td_prediction"] = row["new_td_prediction"] or new_td
                row["new_td_predicted_overlap_us"] = max(row["new_td_predicted_overlap_us"], overlap_us)
            unique[key]["source_occurrences"].append({
                "reference_variant": payload["reference_variant"],
                "call": int(source["call"]),
                "candidate_id": source["candidate_id"],
                "ready_signature": source["ready_signature"],
                "static_prediction": static,
                "new_td_prediction": new_td,
            })

    selected: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    method_summary: list[dict[str, Any]] = []
    for model in sorted(EXPECTED_MODELS):
        for method, field in (("original_janus", "static_prediction"), ("new_td", "new_td_prediction")):
            population = [row for row in unique.values() if row["model"] == model and row[field]]
            population.sort(key=lambda row: stable_rank(args.seed, model, tuple(row["group"]), method))
            chosen = population[: args.per_model_method]
            for row in chosen:
                selected[(model, tuple(row["group"]))].add(method)
            method_summary.append({
                "model": model,
                "method": method,
                "positive_population": len(population),
                "sample_count": len(chosen),
                "sample_weight": (len(population) / len(chosen)) if chosen else None,
            })

    summary_by_pair = {(row["model"], row["method"]): row for row in method_summary}
    cases = []
    for index, key in enumerate(sorted(selected), 1):
        row = unique[key]
        methods = sorted(selected[key])
        first = row["source_occurrences"][0]
        method_sampling = {
            method: summary_by_pair[(row["model"], method)] for method in methods
        }
        weights = [item["sample_weight"] for item in method_sampling.values() if item["sample_weight"] is not None]
        cases.append({
            "case_id": f"janus4_7_{index:04d}",
            **row,
            "call": first["call"],
            "selected_for_methods": methods,
            "method_sampling": method_sampling,
            # Compatibility fields used by the existing isolated-group runner.
            "stratum_population": max(item["positive_population"] for item in method_sampling.values()),
            "sample_weight": max(weights) if weights else 0.0,
            "original_max_concurrent": None,
            "original_any_pair_overlap_ns": None,
        })

    manifest = {
        "schema_version": 1,
        "protocol": "janus_section_4_7_dual_positive_precision_hardware_v1",
        "git_head": next(iter(heads)),
        "source_discovery_root": str(args.discovery_root.resolve()),
        "source_profile_sha256_by_model": profile_sha,
        "seed": args.seed,
        "per_model_method": args.per_model_method,
        "minimum_overlap_us_for_td_extension": args.minimum_overlap_us,
        "new_td_rule": "original Static safety path OR TD-v2 pair with predicted strict overlap >= threshold",
        "excluded_manifests": [str(path.resolve()) for path in args.exclude_manifest],
        "excluded_ordered_group_count": len(excluded),
        "sampling_unit": "unique ordered FX operator group",
        "primary_truth": "one common-start isolated CUDA Graph replay with strict full-group kernel overlap",
        "method_summary": method_summary,
        "case_count": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(cases), "method_summary": method_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
