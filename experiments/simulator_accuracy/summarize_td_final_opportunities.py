#!/usr/bin/env python3
"""Summarize frozen pair-only TD-final positives on the discovery universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def final_prediction(row: dict, threshold_us: float) -> bool:
    td_v2 = row.get("td_v2") or {}
    overlap_us = float(td_v2.get("strict_overlap_duration", 0.0) or 0.0) * 1000.0
    return bool(row.get("td_v2_prediction") and overlap_us >= threshold_us)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold-us", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    paths = sorted(args.discovery_root.glob("*/candidates.json"))
    if not paths:
        raise FileNotFoundError("no candidates.json files")

    unique: dict[tuple, dict] = {}
    all_width_unique_static: dict[tuple, bool] = {}
    occurrence_count = 0
    duplicate_count = 0
    conflicts: list[dict] = []
    source_files = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_files.append(
            {
                "path": str(path),
                "model": payload["model"],
                "reference_variant": payload["reference_variant"],
                "profile_sha256": payload["profile_sha256"],
                "candidate_count": len(payload["candidates"]),
            }
        )
        for row in payload["candidates"]:
            all_key = (
                row["model"],
                row["ready_signature"],
                tuple(row["operators"]),
            )
            static_label = bool(row["static_prediction"])
            if all_key in all_width_unique_static:
                if all_width_unique_static[all_key] != static_label:
                    raise RuntimeError("Static prediction conflict across duplicate keys")
            else:
                all_width_unique_static[all_key] = static_label
            if int(row["group_size"]) != 2:
                continue
            occurrence_count += 1
            key = (
                row["model"],
                row["ready_signature"],
                tuple(row["operators"]),
            )
            labels = {
                "static": bool(row["static_prediction"]),
                "td_final": final_prediction(row, args.threshold_us),
            }
            if key in unique:
                duplicate_count += 1
                if unique[key]["labels"] != labels:
                    conflicts.append(
                        {
                            "key": [key[0], key[1], list(key[2])],
                            "first": unique[key]["labels"],
                            "later": labels,
                            "later_path": str(path),
                        }
                    )
            else:
                unique[key] = {"model": row["model"], "labels": labels}

    if conflicts:
        raise RuntimeError(f"prediction conflicts for {len(conflicts)} duplicate keys")

    def summarize(rows: list[dict]) -> dict:
        static = sum(row["labels"]["static"] for row in rows)
        final = sum(row["labels"]["td_final"] for row in rows)
        both = sum(
            row["labels"]["static"] and row["labels"]["td_final"]
            for row in rows
        )
        return {
            "unique_pair_candidates": len(rows),
            "static_positive": static,
            "td_final_positive": final,
            "both_positive": both,
            "static_only": static - both,
            "td_final_only": final - both,
            "neither": len(rows) - static - final + both,
            "td_final_vs_static_ratio": final / static if static else None,
            "td_final_vs_static_increase_fraction": (
                (final - static) / static if static else None
            ),
        }

    rows = list(unique.values())
    by_model_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model_rows[row["model"]].append(row)
    overall = summarize(rows)
    static_all_widths = sum(all_width_unique_static.values())
    proposed_total = static_all_widths + overall["td_final_only"]
    output = {
        "schema_version": 1,
        "protocol": "same_universe_pair_only_td_final_opportunities_v1",
        "deduplication_key": ["model", "ready_signature", "ordered_operators"],
        "td_final_rule": {
            "group_size": 2,
            "td_v2_prediction_required": True,
            "minimum_predicted_strict_overlap_us": args.threshold_us,
        },
        "source_root": str(args.discovery_root),
        "source_files": source_files,
        "pair_occurrences": occurrence_count,
        "duplicate_occurrences": duplicate_count,
        "prediction_conflicts": len(conflicts),
        "overall": overall,
        "recommended_union_rule": {
            "definition": (
                "retain every all-width Static positive and add pair-only "
                "TD-final positives that Static rejects"
            ),
            "static_positive_all_widths": static_all_widths,
            "td_final_pair_only_additions": overall["td_final_only"],
            "proposed_positive_total": proposed_total,
            "increase_fraction_vs_static_all_widths": (
                (proposed_total - static_all_widths) / static_all_widths
                if static_all_widths else None
            ),
            "pair_only_union_positive": (
                overall["static_positive"] + overall["td_final_only"]
            ),
            "pair_only_increase_fraction_vs_static_pair": (
                overall["td_final_only"] / overall["static_positive"]
                if overall["static_positive"] else None
            ),
        },
        "by_model": {
            model: summarize(by_model_rows[model])
            for model in sorted(by_model_rows)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    print("sha256", hashlib.sha256(text.encode("utf-8")).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
