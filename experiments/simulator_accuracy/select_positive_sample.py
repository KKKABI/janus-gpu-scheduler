#!/usr/bin/env python3
"""Select a deterministic, width-stratified sample of simulator positives."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_MODELS = {
    "GoogLeNet",
    "Inception-v3",
    "NASNet",
    "YOLOv8x",
    "ConvNeXt",
    "DeepFM",
    "BERT",
}
EXPECTED_VARIANTS = {"Baseline", "TD+DRT"}


def stable_rank(seed: int, row: dict[str, Any]) -> str:
    material = json.dumps(
        {
            "seed": seed,
            "model": row["model"],
            "ready_signature": row["ready_signature"],
            "operators": row["operators"],
            "static": row["static_prediction"],
            "td": row["td_prediction"],
            "td_v2": row.get("td_v2_prediction"),
            "td_final": row.get("td_final_prediction"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def candidate_key(row: dict[str, Any]) -> tuple:
    return (
        row["model"],
        row["ready_signature"],
        tuple(row["operators"]),
        bool(row["static_prediction"]),
        bool(row["td_prediction"]),
        row.get("td_v2_prediction"),
        row.get("td_final_prediction"),
    )


def stratum_key(
    row: dict[str, Any], prediction_field: str = "td_prediction"
) -> tuple:
    return (
        row["model"],
        bool(row["static_prediction"]),
        bool(row[prediction_field]),
        int(row["group_size"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument(
        "--positive-method",
        choices=("static-td-union", "td-v2", "td-final"),
        default="static-td-union",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        help="exclude every (model, ordered operator group) already sampled",
    )
    parser.add_argument(
        "--td-final-min-overlap-us",
        type=float,
        default=2.0,
        help=(
            "require this much predicted strict full-group overlap in "
            "addition to TD-v2 feasibility"
        ),
    )
    parser.add_argument(
        "--max-group-size",
        type=int,
        default=5,
        help="discard candidate groups wider than this value before sampling",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.per_stratum < 1:
        raise ValueError("--per-stratum must be positive")
    if not 2 <= args.max_group_size <= 5:
        raise ValueError("--max-group-size must be in [2, 5]")

    paths = sorted(args.discovery_root.glob("*/candidates.json"))
    if not paths:
        raise FileNotFoundError("no discovery candidates.json files found")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    observed_pairs = Counter(
        (payload["model"], payload["reference_variant"]) for payload in payloads
    )
    expected_pairs = {
        (model, variant)
        for model in EXPECTED_MODELS
        for variant in EXPECTED_VARIANTS
    }
    if set(observed_pairs) != expected_pairs or any(
        count != 1 for count in observed_pairs.values()
    ):
        missing = sorted(expected_pairs - set(observed_pairs))
        extra = sorted(set(observed_pairs) - expected_pairs)
        raise RuntimeError(
            f"discovery matrix is incomplete: missing={missing}, extra={extra}, "
            f"duplicates={[pair for pair, count in observed_pairs.items() if count != 1]}"
        )

    heads = {payload["git_head"] for payload in payloads}
    if len(heads) != 1:
        raise RuntimeError(f"mixed git heads: {sorted(heads)}")
    profile_sha_by_model: dict[str, str] = {}
    for payload in payloads:
        model = payload["model"]
        sha = payload["profile_sha256"]
        previous = profile_sha_by_model.setdefault(model, sha)
        if previous != sha:
            raise RuntimeError(f"profile SHA changed across paths for {model}")
        if not payload["correctness"].get("ok", False):
            raise RuntimeError(f"discovery correctness failed for {model}")

    # Merge identical ready-set/group predictions visited by both reference
    # paths.  The number of source occurrences remains visible for auditing.
    unique: dict[tuple, dict] = {}
    prediction_field = (
        "td_final_prediction"
        if args.positive_method == "td-final"
        else (
            "td_v2_prediction"
            if args.positive_method == "td-v2"
            else "td_prediction"
        )
    )
    excluded_groups = set()
    if args.exclude_manifest is not None:
        for excluded_path in args.exclude_manifest:
            excluded_payload = json.loads(
                excluded_path.read_text(encoding="utf-8")
            )
            excluded_groups.update(
                (row["model"], tuple(row["group"]))
                for row in excluded_payload.get("cases", [])
            )
    for payload in payloads:
        for source in payload["candidates"]:
            if int(source["group_size"]) > args.max_group_size:
                continue
            if (source["model"], tuple(source["operators"])) in excluded_groups:
                continue
            if args.positive_method == "td-final":
                predicted_overlap_us = float(
                    (source.get("td_v2") or {}).get(
                        "strict_overlap_duration", 0.0
                    )
                    or 0.0
                ) * 1000.0
                source = {
                    **source,
                    "td_final_prediction": bool(
                        source.get("td_v2_prediction")
                        and predicted_overlap_us
                        >= args.td_final_min_overlap_us
                    ),
                    "td_final_min_overlap_us": args.td_final_min_overlap_us,
                    "td_final_predicted_overlap_us": predicted_overlap_us,
                }
                include = bool(source["td_final_prediction"])
            elif args.positive_method == "td-v2":
                include = bool(source.get("td_v2_prediction"))
            else:
                include = bool(
                    source["static_prediction"] or source["td_prediction"]
                )
            if not include:
                continue
            key = candidate_key(source)
            if key not in unique:
                unique[key] = {
                    **source,
                    "source_occurrences": [],
                }
            unique[key]["source_occurrences"].append(
                {
                    "reference_variant": payload["reference_variant"],
                    "call": source["call"],
                    "candidate_id": source["candidate_id"],
                }
            )

    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in unique.values():
        strata[stratum_key(row, prediction_field)].append(row)

    cases = []
    stratum_summaries = []
    case_number = 0
    for key in sorted(strata):
        population_rows = strata[key]
        population_rows.sort(key=lambda row: stable_rank(args.seed, row))
        chosen = population_rows[: min(args.per_stratum, len(population_rows))]
        model, static_pred, selected_pred, width = key
        population = len(population_rows)
        sample_count = len(chosen)
        sample_weight = population / sample_count
        marker = (
            "F"
            if args.positive_method == "td-final"
            else ("V" if args.positive_method == "td-v2" else "T")
        )
        stratum = (
            f"{model}|S{int(static_pred)}{marker}{int(selected_pred)}|K{width}"
        )
        stratum_summaries.append(
            {
                "stratum": stratum,
                "model": model,
                "static_prediction": static_pred,
                "td_prediction": (
                    bool(population_rows[0]["td_prediction"])
                    if args.positive_method != "td-v2"
                    else None
                ),
                "td_v2_prediction": (
                    selected_pred if args.positive_method == "td-v2" else None
                ),
                "td_final_prediction": (
                    selected_pred
                    if args.positive_method == "td-final"
                    else None
                ),
                "group_size": width,
                "population": population,
                "sample_count": sample_count,
                "sample_weight": sample_weight,
            }
        )
        for row in chosen:
            case_number += 1
            cases.append(
                {
                    "case_id": f"sim4_7_{case_number:04d}",
                    "model": model,
                    "call": int(row["call"]),
                    "group": list(row["operators"]),
                    "size": int(row["group_size"]),
                    "static_prediction": static_pred,
                    "td_prediction": bool(row["td_prediction"]),
                    "td_v2_prediction": row.get("td_v2_prediction"),
                    "td_final_prediction": row.get("td_final_prediction"),
                    "td_final_min_overlap_us": row.get(
                        "td_final_min_overlap_us"
                    ),
                    "td_final_predicted_overlap_us": row.get(
                        "td_final_predicted_overlap_us"
                    ),
                    "stratum": stratum,
                    "stratum_population": population,
                    "sample_weight": sample_weight,
                    "source_occurrence_count": len(row["source_occurrences"]),
                    "source_occurrences": row["source_occurrences"],
                    # Compatibility fields for the existing isolated runner.
                    "original_max_concurrent": None,
                    "original_any_pair_overlap_ns": None,
                }
            )

    universe_by_model = {}
    for model in sorted(EXPECTED_MODELS):
        rows = [row for row in unique.values() if row["model"] == model]
        universe_by_model[model] = {
            "positive_union": len(rows),
            "static_positive": sum(row["static_prediction"] for row in rows),
            "td_positive": sum(row["td_prediction"] for row in rows),
            "td_v2_positive": sum(
                bool(row.get("td_v2_prediction")) for row in rows
            ),
            "td_final_positive": sum(
                bool(row.get("td_final_prediction")) for row in rows
            ),
            "sample_count": sum(case["model"] == model for case in cases),
        }
    manifest = {
        "schema_version": 1,
        "protocol": (
            "janus_4_7_td_final_positive_precision_v2"
            if args.positive_method == "td-final"
            else (
                "janus_4_7_td_v2_positive_precision_v1"
                if args.positive_method == "td-v2"
                else "janus_4_7_paired_simulator_positive_precision_v1"
            )
        ),
        "git_head": next(iter(heads)),
        "source_discovery_root": str(args.discovery_root.resolve()),
        "source_profile_sha256_by_model": profile_sha_by_model,
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "positive_method": args.positive_method,
        "excluded_manifests": (
            [str(path.resolve()) for path in args.exclude_manifest]
            if args.exclude_manifest is not None
            else []
        ),
        "excluded_ordered_group_count": len(excluded_groups),
        "td_final_min_overlap_us": (
            args.td_final_min_overlap_us
            if args.positive_method == "td-final"
            else None
        ),
        "td_final_static_fallback": False,
        "max_group_size": args.max_group_size,
        "sampling_unit": (
            "unique (model, ready signature, ordered group, paired labels)"
        ),
        "primary_truth": (
            "one common-start isolated CUDA Graph replay with strict full-group "
            "kernel overlap"
        ),
        "universe_by_model": universe_by_model,
        "strata": stratum_summaries,
        "case_count": len(cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "case_count": len(cases),
                "stratum_count": len(strata),
                "universe_by_model": universe_by_model,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
