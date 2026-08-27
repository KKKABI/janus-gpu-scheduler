#!/usr/bin/env python3
"""Aggregate one-replay hardware truth into Static/TD positive precision."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path


def summarize(rows, prediction_key):
    predicted = [row for row in rows if row[prediction_key]]
    auditable = [row for row in predicted if row["auditable"]]
    true = [row for row in auditable if row["isolated_strict_parallel"]]
    predicted_weight = sum(float(row["sample_weight"]) for row in predicted)
    auditable_weight = sum(float(row["sample_weight"]) for row in auditable)
    true_weight = sum(float(row["sample_weight"]) for row in true)
    return {
        "sampled_positive": len(predicted),
        "auditable_positive": len(auditable),
        "strict_parallel_positive": len(true),
        "audit_coverage": len(auditable) / len(predicted) if predicted else None,
        "unweighted_precision": len(true) / len(auditable) if auditable else None,
        "represented_positive_population": predicted_weight,
        "represented_auditable_population": auditable_weight,
        "represented_true_population": true_weight,
        "stratified_weighted_precision": (
            true_weight / auditable_weight if auditable_weight else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_cases = {row["case_id"]: row for row in manifest["cases"]}

    hardware = {}
    model_payloads = []
    for path in sorted(args.results_root.glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_payloads.append(payload)
        for row in payload["cases"]:
            case_id = row["case_id"]
            if case_id in hardware:
                raise RuntimeError(f"duplicate hardware result: {case_id}")
            hardware[case_id] = row
    missing = sorted(set(manifest_cases) - set(hardware))
    extra = sorted(set(hardware) - set(manifest_cases))
    if missing or extra:
        raise RuntimeError(f"hardware/manifest mismatch: missing={missing}, extra={extra}")

    rows = []
    for case_id, case in manifest_cases.items():
        truth = hardware[case_id]
        if truth["group"] != case["group"] or truth["model"] != case["model"]:
            raise RuntimeError(f"case identity changed: {case_id}")
        rows.append(
            {
                **case,
                "auditable": bool(truth["auditable"]),
                "missing_ops": truth["missing_ops"],
                "strict_overlap_ns": int(truth["strict_overlap_ns"]),
                "isolated_strict_parallel": bool(
                    truth["isolated_strict_parallel"]
                ),
            }
        )

    methods = [
        ("Static", "static_prediction"),
        ("TD", "td_prediction"),
    ]
    if any(row.get("td_v2_prediction") is not None for row in rows):
        methods.append(("TD-v2", "td_v2_prediction"))
    if any(row.get("td_final_prediction") is not None for row in rows):
        methods.append(("TD-final", "td_final_prediction"))
    overall = {name: summarize(rows, key) for name, key in methods}
    by_model = {}
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        by_model[model] = {
            name: summarize(subset, key) for name, key in methods
        }
    by_width = {}
    for width in sorted({row["size"] for row in rows}):
        subset = [row for row in rows if row["size"] == width]
        by_width[str(width)] = {
            name: summarize(subset, key) for name, key in methods
        }
    payload = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "truth_replays_per_group": 1,
        "stable_5_of_5_not_used": True,
        "case_count": len(rows),
        "auditable_cases": sum(row["auditable"] for row in rows),
        "overall": overall,
        "by_model": by_model,
        "by_width": by_width,
        "cases": rows,
    }
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "cases.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fields = [
            "case_id",
            "model",
            "size",
            "static_prediction",
            "td_prediction",
            "td_v2_prediction",
            "td_final_prediction",
            "stratum",
            "sample_weight",
            "auditable",
            "isolated_strict_parallel",
            "strict_overlap_ns",
            "missing_ops",
            "group",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            def csv_value(key):
                value = row.get(key)
                return (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                )

            writer.writerow(
                {key: csv_value(key) for key in fields}
            )

    def pct(value):
        return "—" if value is None else f"{100.0 * value:.2f}%"

    lines = [
        "# Janus 4.7 口径：Static 与 TD 判断器正预测精确率",
        "",
        "主指标只使用一次共同起跑的隔离 CUDA Graph replay；不使用 5/5 稳定组。",
        "",
        "| 判断器 | 抽样正预测 | 可检查 | 严格并行 | 分层加权精确率 | 检查覆盖率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, _ in methods:
        item = overall[name]
        lines.append(
            f"| {name} | {item['sampled_positive']} | "
            f"{item['auditable_positive']} | {item['strict_parallel_positive']} | "
            f"{pct(item['stratified_weighted_precision'])} | "
            f"{pct(item['audit_coverage'])} |"
        )
    lines.extend(
        [
            "",
            "注意：这是判断器正预测精确率，不是 Janus/DRT 最终选择组精确率；未知映射组不进入精确率分母，但必须单独报告覆盖率。",
            "",
        ]
    )
    (args.output_dir / "summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps({"overall": overall}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
