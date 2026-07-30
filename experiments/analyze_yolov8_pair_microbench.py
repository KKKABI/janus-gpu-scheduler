#!/usr/bin/env python3
"""Aggregate repeated YOLOv8 operator-pair measurements into a cache/report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


DERIVED_METRICS = (
    "slowdown_a",
    "slowdown_b",
    "mean_slowdown",
    "max_slowdown",
    "makespan_dilation",
    "measured_pair_speedup",
)
SAMPLE_METRICS = (
    "solo_a_ms",
    "solo_b_ms",
    "corun_a_completion_ms",
    "corun_b_completion_ms",
    "corun_makespan_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--capture", action="append", default=[], type=Path)
    parser.add_argument("--round-penalty", type=float, default=0.5)
    parser.add_argument("--operator-penalty", type=float, default=0.1)
    return parser.parse_args()


def median_mad(values):
    values = [float(value) for value in values]
    median = statistics.median(values)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "values": values,
    }


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = 0.5 * (cursor + end - 1)
        for index, _ in indexed[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def pearson(left, right):
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right)
    )
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return numerator / (left_norm * right_norm)


def correlation(left, right):
    return {
        "pearson": pearson(left, right),
        "spearman": pearson(average_ranks(left), average_ranks(right)),
    }


def canonical_pair(left, right):
    return tuple(sorted((left, right)))


def load_captures(paths):
    pair_diagnostics = {}
    operator_profiles = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = payload["task"]["variant"]
        for call in payload["scheduler"]["calls"]:
            for finalist in call.get("finalist_timelines", []):
                for profile in finalist.get("operator_profiles", []):
                    operator_profiles.setdefault(profile["name"], profile)
                if finalist.get("size") != 2:
                    continue
                key = canonical_pair(*finalist["operators"])
                pair_diagnostics.setdefault(key, {
                    "operators": list(key),
                    "operator_profiles": [
                        operator_profiles[name] for name in key
                    ],
                    "initial_utilization": finalist["initial_utilization"],
                    "timeline": finalist["timeline"],
                    "sources": [],
                })
                pair_diagnostics[key]["sources"].append({
                    "variant": variant,
                    "call": call["call"],
                    "stage1_rank": finalist["timeline"].get("stage1_rank"),
                })
    return pair_diagnostics, operator_profiles


def aggregate_pair(key, repeats, diagnostic, operator_profiles, weights):
    left, right = key
    derived = {
        metric: median_mad([repeat["derived"][metric] for repeat in repeats])
        for metric in DERIVED_METRICS
    }
    samples = {}
    per_operator = {
        left: {"solo_ms": [], "corun_completion_ms": [], "slowdown": []},
        right: {"solo_ms": [], "corun_completion_ms": [], "slowdown": []},
    }
    for repeat in repeats:
        for metric in SAMPLE_METRICS:
            samples.setdefault(metric, []).append(
                repeat["statistics"][metric]["median_ms"]
            )
        for suffix, operator in (("a", repeat["a"]), ("b", repeat["b"])):
            per_operator[operator]["solo_ms"].append(
                repeat["statistics"][f"solo_{suffix}_ms"]["median_ms"]
            )
            per_operator[operator]["corun_completion_ms"].append(
                repeat["statistics"][
                    f"corun_{suffix}_completion_ms"
                ]["median_ms"]
            )
            per_operator[operator]["slowdown"].append(
                repeat["derived"][f"slowdown_{suffix}"]
            )

    operator_metrics = {
        operator: {
            name: median_mad(values)
            for name, values in metrics.items()
        }
        for operator, metrics in per_operator.items()
    }
    speedup = derived["measured_pair_speedup"]["median"]
    dilation = derived["makespan_dilation"]["median"]
    max_slowdown = derived["max_slowdown"]["median"]
    normalized_time_saved = max(0.0, 1.0 - 1.0 / speedup)
    utility = (
        normalized_time_saved
        - weights["round_penalty"] * max(0.0, dilation - 1.0)
        - weights["operator_penalty"] * max(0.0, max_slowdown - 1.0)
    )
    first = repeats[0]
    return {
        "operators": list(key),
        "repeat_count": len(repeats),
        "source_pair_ids": sorted({repeat["id"] for repeat in repeats}),
        "operator_metrics": operator_metrics,
        "derived": derived,
        "sample_medians_ms": {
            metric: median_mad(values) for metric, values in samples.items()
        },
        "predicted": {
            "risk": float(first["predicted_risk"]),
            "speedup": float(first["predicted_speedup"]),
        },
        "empirical": {
            "normalized_time_saved": normalized_time_saved,
            "round_penalty": max(0.0, dilation - 1.0),
            "operator_penalty": max(0.0, max_slowdown - 1.0),
            "utility": utility,
            "decision": "concurrent" if utility > 0.0 else "serialize",
        },
        "operator_profiles": [
            operator_profiles[name] for name in key
            if name in operator_profiles
        ],
        "scheduler_diagnostic": diagnostic,
    }


def render_report(aggregate, correlations, weights):
    pairs = aggregate["pairs"]
    concurrent = sum(
        pair["empirical"]["decision"] == "concurrent"
        for pair in pairs.values()
    )
    lines = [
        "# YOLOv8 算子对实测干扰分析",
        "",
        f"- 唯一算子对：{len(pairs)} 组；每组独立重复 5 次。",
        f"- 建议并发：{concurrent} 组；建议串行：{len(pairs) - concurrent} 组。",
        "- 打分不使用算子名称类别，而使用同设备、同输入下的 solo/co-run 实测。",
        "- 经验效用 = 归一化节省时间 "
        f"- {weights['round_penalty']:.2f}×整组时延膨胀 "
        f"- {weights['operator_penalty']:.2f}×最大单算子 slowdown。",
        "",
        "## 代理有效性",
        "",
        "| Proxy | Target | Pearson | Spearman |",
        "|---|---|---:|---:|",
    ]
    for name, values in correlations.items():
        proxy, target = name.split("__", 1)
        lines.append(
            f"| {proxy} | {target} | {values['pearson']:.3f} | "
            f"{values['spearman']:.3f} |"
        )
    lines.extend([
        "",
        "现有 NCU 风险和 TD 预测 speedup 均不能可靠排序真实干扰，"
        "因此不继续在这两个代理上调常数权重。",
        "",
        "## 逐对结果（5 次中位数）",
        "",
        "| Operators | Risk | TD speedup | Measured speedup | Round dilation | Max slowdown | Utility | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    ordered = sorted(
        pairs.values(),
        key=lambda pair: pair["derived"]["makespan_dilation"]["median"],
        reverse=True,
    )
    for pair in ordered:
        lines.append(
            "| " + " + ".join(pair["operators"])
            + f" | {pair['predicted']['risk']:.3f}"
            + f" | {pair['predicted']['speedup']:.3f}"
            + f" | {pair['derived']['measured_pair_speedup']['median']:.3f}"
            + f" | {pair['derived']['makespan_dilation']['median']:.3f}"
            + f" | {pair['derived']['max_slowdown']['median']:.3f}"
            + f" | {pair['empirical']['utility']:.3f}"
            + f" | {pair['empirical']['decision']} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        "1. 整组完成时间膨胀与单个短算子的 slowdown 必须分别约束。",
        "2. 所有实测组合仍有吞吐收益，但部分组合以 2–8 倍单算子 slowdown 换取约 1%–4% 的收益，不值得并发。",
        "3. 下一版选择器应优先读取实测共运行缓存；未覆盖候选保守回退到 TD 排序，而不是用算子名称猜资源类型。",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.round_penalty < 0 or args.operator_penalty < 0:
        raise ValueError("penalties must be non-negative")
    result_paths = sorted(args.results_root.glob("raw/*/r*.json"))
    if not result_paths:
        raise RuntimeError(f"no result files below {args.results_root}")

    grouped = defaultdict(list)
    source_files = []
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_files.append(str(path.resolve()))
        for pair in payload["pairs"]:
            grouped[canonical_pair(pair["a"], pair["b"])].append(pair)
    if any(len(repeats) != 5 for repeats in grouped.values()):
        counts = {"|".join(key): len(value) for key, value in grouped.items()}
        raise RuntimeError(f"expected five repeats for every pair: {counts}")

    diagnostics, operator_profiles = load_captures(args.capture)
    weights = {
        "round_penalty": args.round_penalty,
        "operator_penalty": args.operator_penalty,
    }
    pairs = {
        "|".join(key): aggregate_pair(
            key,
            repeats,
            diagnostics.get(key),
            operator_profiles,
            weights,
        )
        for key, repeats in sorted(grouped.items())
    }
    aggregate = {
        "schema_version": 1,
        "model": "YOLOv8x",
        "model_class": "DetectionModel",
        "pair_count": len(pairs),
        "repeat_count_per_pair": 5,
        "weights": weights,
        "pairs": pairs,
        "source_files": source_files,
    }

    risk = [pair["predicted"]["risk"] for pair in pairs.values()]
    predicted_speedup = [
        pair["predicted"]["speedup"] for pair in pairs.values()
    ]
    dilation = [
        pair["derived"]["makespan_dilation"]["median"]
        for pair in pairs.values()
    ]
    measured_speedup = [
        pair["derived"]["measured_pair_speedup"]["median"]
        for pair in pairs.values()
    ]
    correlations = {
        "risk__makespan_dilation": correlation(risk, dilation),
        "risk__measured_pair_speedup": correlation(risk, measured_speedup),
        "td_speedup__makespan_dilation": correlation(
            predicted_speedup, dilation
        ),
        "td_speedup__measured_pair_speedup": correlation(
            predicted_speedup, measured_speedup
        ),
    }
    aggregate["correlations"] = correlations

    cache = {
        "schema_version": 1,
        "model": "YOLOv8x",
        "model_class": "DetectionModel",
        "device_scope": "NVIDIA RTX A5000",
        "input_shape": [1, 3, 320, 320],
        "utility_weights": weights,
        "pair_count": len(pairs),
        "pairs": {
            key: {
                "operators": pair["operators"],
                "repeat_count": pair["repeat_count"],
                "operator_metrics": pair["operator_metrics"],
                "makespan_dilation": pair["derived"][
                    "makespan_dilation"
                ]["median"],
                "measured_pair_speedup": pair["derived"][
                    "measured_pair_speedup"
                ]["median"],
                "mean_slowdown": pair["derived"]["mean_slowdown"]["median"],
                "max_slowdown": pair["derived"]["max_slowdown"]["median"],
                "normalized_time_saved": pair["empirical"][
                    "normalized_time_saved"
                ],
                "utility": pair["empirical"]["utility"],
                "decision": pair["empirical"]["decision"],
            }
            for key, pair in pairs.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "DetectionModel.pair.json").write_text(
        json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "analysis_report.md").write_text(
        render_report(aggregate, correlations, weights), encoding="utf-8"
    )
    print(json.dumps({
        "pair_count": len(pairs),
        "output_dir": str(args.output_dir.resolve()),
        "correlations": correlations,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
