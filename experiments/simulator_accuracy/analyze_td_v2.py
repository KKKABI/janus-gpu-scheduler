#!/usr/bin/env python3
"""Compare launch-gap variants of TD-v2 against isolated hardware truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CALIBRATION_MODELS = {"GoogLeNet", "Inception-v3", "DeepFM"}
HOLDOUT_MODELS = {"NASNet", "YOLOv8x", "BERT"}


def metrics(rows, prediction_key):
    auditable = [row for row in rows if row["auditable"]]
    predicted = [row for row in auditable if row[prediction_key]]
    true_predicted = [row for row in predicted if row["isolated_strict_parallel"]]
    actual_positive = [row for row in auditable if row["isolated_strict_parallel"]]
    predicted_weight = sum(float(row["sample_weight"]) for row in predicted)
    true_weight = sum(float(row["sample_weight"]) for row in true_predicted)
    actual_weight = sum(float(row["sample_weight"]) for row in actual_positive)
    precision = true_weight / predicted_weight if predicted_weight else None
    recall = true_weight / actual_weight if actual_weight else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "auditable_cases": len(auditable),
        "predicted_positive_cases": len(predicted),
        "true_positive_cases": len(true_predicted),
        "unweighted_precision": (
            len(true_predicted) / len(predicted) if predicted else None
        ),
        "represented_predicted_population": predicted_weight,
        "represented_true_population": true_weight,
        "weighted_precision": precision,
        "weighted_recall_within_current_td_positive_universe": recall,
        "weighted_f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--td-v2-root", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hardware = json.loads(args.hardware.read_text(encoding="utf-8"))
    rows = {row["case_id"]: dict(row) for row in hardware["cases"]}
    predictions = {}
    gaps = None
    for path in sorted(args.td_v2_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if gaps is None:
            gaps = [f"{float(value):.4f}" for value in payload["gaps_ms"]]
        for result in payload["results"]:
            case_id = result["case_id"]
            if case_id in predictions:
                raise RuntimeError(f"duplicate TD-v2 prediction: {case_id}")
            predictions[case_id] = result["gap_results"]
    missing = sorted(set(rows) - set(predictions))
    extra = sorted(set(predictions) - set(rows))
    if missing or extra:
        raise RuntimeError(f"TD-v2/hardware mismatch: missing={missing}, extra={extra}")
    for case_id, row in rows.items():
        for gap, result in predictions[case_id].items():
            row[f"td_v2_gap_{gap}"] = bool(result["strict_parallel"])

    subsets = {
        "all": list(rows.values()),
        "calibration": [
            row for row in rows.values() if row["model"] in CALIBRATION_MODELS
        ],
        "holdout": [
            row for row in rows.values() if row["model"] in HOLDOUT_MODELS
        ],
    }
    tables = {}
    for subset_name, subset in subsets.items():
        tables[subset_name] = {
            gap: metrics(subset, f"td_v2_gap_{gap}") for gap in gaps or []
        }
    by_model = {
        model: {
            gap: metrics(
                [row for row in rows.values() if row["model"] == model],
                f"td_v2_gap_{gap}",
            )
            for gap in gaps or []
        }
        for model in sorted({row["model"] for row in rows.values()})
    }
    by_width = {
        str(width): {
            gap: metrics(
                [row for row in rows.values() if int(row["size"]) == width],
                f"td_v2_gap_{gap}",
            )
            for gap in gaps or []
        }
        for width in sorted({int(row["size"]) for row in rows.values()})
    }
    calibration_candidates = []
    for gap, item in tables["calibration"].items():
        precision = item["unweighted_precision"]
        if precision is not None and precision >= 0.80:
            calibration_candidates.append(
                (item["true_positive_cases"], -float(gap), gap)
            )
    selected_gap = (
        max(calibration_candidates)[2] if calibration_candidates else None
    )
    payload = {
        "schema_version": 1,
        "simulator": "td_v2_nonreserving_launch_ordered_v1",
        "calibration_models": sorted(CALIBRATION_MODELS),
        "holdout_models": sorted(HOLDOUT_MODELS),
        "gap_unit": "milliseconds",
        "gaps": gaps,
        "selection_rule": (
            "among calibration gaps with unweighted positive precision >=80%, "
            "maximize observed true-positive cases; break ties toward the "
            "smaller launch gap"
        ),
        "selected_gap": selected_gap,
        "subsets": tables,
        "by_model": by_model,
        "by_width": by_width,
        "cases": list(rows.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_gap": selected_gap,
                "calibration": (
                    tables["calibration"].get(selected_gap)
                    if selected_gap is not None
                    else None
                ),
                "holdout": (
                    tables["holdout"].get(selected_gap)
                    if selected_gap is not None
                    else None
                ),
                "all": (
                    tables["all"].get(selected_gap)
                    if selected_gap is not None
                    else None
                ),
                "all_gaps": tables["all"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
