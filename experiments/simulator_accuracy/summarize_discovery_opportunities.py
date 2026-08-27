#!/usr/bin/env python3
"""Summarize unique admission opportunities across discovery paths."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


METHODS = ("static_prediction", "td_prediction", "td_v2_prediction")


def logical_key(row: dict[str, Any]) -> tuple:
    return (
        row["model"],
        tuple(row["ready_signature"]),
        tuple(row["operators"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("discovery_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    paths = sorted(args.discovery_root.glob("*/candidates.json"))
    if not paths:
        raise FileNotFoundError("no candidates.json files found")

    unique: dict[tuple, dict[str, Any]] = {}
    occurrence_counts = Counter()
    source_runs = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_runs.append(
            {
                "path": str(path.resolve()),
                "model": payload["model"],
                "reference_variant": payload["reference_variant"],
                "candidate_count": len(payload["candidates"]),
            }
        )
        for row in payload["candidates"]:
            key = logical_key(row)
            target = unique.setdefault(
                key,
                {
                    "model": row["model"],
                    "width": int(row["group_size"]),
                    "occurrences": 0,
                    **{method: False for method in METHODS},
                },
            )
            target["occurrences"] += 1
            occurrence_counts["all"] += 1
            for method in METHODS:
                accepted = bool(row.get(method, False))
                target[method] = target[method] or accepted
                occurrence_counts[method] += int(accepted)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = {"unique_candidates": len(rows)}
        for method in METHODS:
            counts[method] = sum(bool(row[method]) for row in rows)
        counts["static_and_td_v2"] = sum(
            row["static_prediction"] and row["td_v2_prediction"] for row in rows
        )
        counts["static_only_vs_td_v2"] = sum(
            row["static_prediction"] and not row["td_v2_prediction"] for row in rows
        )
        counts["td_v2_only_vs_static"] = sum(
            row["td_v2_prediction"] and not row["static_prediction"] for row in rows
        )
        counts["neither_static_nor_td_v2"] = sum(
            not row["static_prediction"] and not row["td_v2_prediction"]
            for row in rows
        )
        return counts

    rows = list(unique.values())
    by_model_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_width_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model_rows[row["model"]].append(row)
        by_width_rows[row["width"]].append(row)

    output = {
        "protocol": "unique_logical_candidate_opportunities_v1",
        "deduplication_key": ["model", "ready_signature", "ordered_operators"],
        "source_runs": source_runs,
        "occurrence_counts": dict(occurrence_counts),
        "unique": summarize(rows),
        "by_model": {
            model: summarize(model_rows)
            for model, model_rows in sorted(by_model_rows.items())
        },
        "by_width": {
            str(width): summarize(width_rows)
            for width, width_rows in sorted(by_width_rows.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["unique"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
