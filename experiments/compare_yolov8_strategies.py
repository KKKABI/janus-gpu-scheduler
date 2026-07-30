#!/usr/bin/env python3
"""Compare selected co-run interference using the measured pair cache."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", action="append", required=True, type=Path)
    parser.add_argument("--pair-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def estimate_combo(operators, cache, weights):
    entries = []
    missing = []
    for left, right in itertools.combinations(operators, 2):
        key = "|".join(sorted((left, right)))
        entry = cache.get(key)
        if entry is None:
            missing.append(key)
        else:
            entries.append(entry)
    if missing:
        return {"missing_pairs": missing}

    if len(operators) == 2:
        entry = entries[0]
        speedup = float(entry["measured_pair_speedup"])
        dilation = float(entry["makespan_dilation"])
        mean_slowdown = float(entry["mean_slowdown"])
        max_slowdown = float(entry["max_slowdown"])
    else:
        solo_samples = {operator: [] for operator in operators}
        slowdown_overheads = {operator: [] for operator in operators}
        for entry in entries:
            for operator, metrics in entry["operator_metrics"].items():
                if operator not in solo_samples:
                    continue
                solo_samples[operator].append(
                    float(metrics["solo_ms"]["median"])
                )
                slowdown_overheads[operator].append(max(
                    0.0, float(metrics["slowdown"]["median"]) - 1.0
                ))
        solo = {
            operator: statistics.median(values)
            for operator, values in solo_samples.items()
        }
        slowdowns = {
            operator: 1.0 + sum(slowdown_overheads[operator])
            for operator in operators
        }
        serial = sum(solo.values())
        concurrent = max(
            solo[operator] * slowdowns[operator] for operator in operators
        )
        speedup = serial / concurrent
        dilation = concurrent / max(solo.values())
        mean_slowdown = statistics.fmean(slowdowns.values())
        max_slowdown = max(slowdowns.values())

    normalized_time_saved = max(0.0, 1.0 - 1.0 / speedup)
    utility = (
        normalized_time_saved
        - weights["round_penalty"] * max(0.0, dilation - 1.0)
        - weights["operator_penalty"] * max(0.0, max_slowdown - 1.0)
    )
    return {
        "estimated_speedup": speedup,
        "estimated_makespan_dilation": dilation,
        "estimated_mean_slowdown": mean_slowdown,
        "estimated_max_slowdown": max_slowdown,
        "normalized_time_saved": normalized_time_saved,
        "utility": utility,
    }


def aggregate_run(result, cache, weights):
    selections = []
    missing = []
    for call in result["scheduler"]["calls"]:
        operators = call.get("selected_resource", [])
        if len(operators) < 2:
            continue
        estimate = estimate_combo(operators, cache, weights)
        if "missing_pairs" in estimate:
            missing.extend(estimate["missing_pairs"])
        else:
            selections.append(estimate)
    if missing:
        raise RuntimeError(f"pair cache coverage is incomplete: {sorted(set(missing))}")
    return {
        "process_median_ms": float(result["timing"]["statistics"]["median_ms"]),
        "concurrent_call_count": len(selections),
        "negative_utility_call_count": sum(
            selection["utility"] <= 0.0 for selection in selections
        ),
        "mean_makespan_dilation": statistics.fmean(
            selection["estimated_makespan_dilation"]
            for selection in selections
        ),
        "max_makespan_dilation": max(
            selection["estimated_makespan_dilation"]
            for selection in selections
        ),
        "mean_max_slowdown": statistics.fmean(
            selection["estimated_max_slowdown"]
            for selection in selections
        ),
        "max_slowdown": max(
            selection["estimated_max_slowdown"]
            for selection in selections
        ),
        "mean_utility": statistics.fmean(
            selection["utility"] for selection in selections
        ),
        "selected": selections,
    }


def median_field(runs, field):
    return statistics.median(run[field] for run in runs)


def render_report(comparison):
    variants = comparison["variants"]
    baseline = variants["TD+Janus-no-alpha"]
    empirical = variants["TD+EmpiricalDRT"]
    latency_delta = 100.0 * (
        empirical["median_of_process_medians_ms"]
        / baseline["median_of_process_medians_ms"] - 1.0
    )
    excess_dilation_reduction = 100.0 * (
        1.0
        - (empirical["max_makespan_dilation"] - 1.0)
        / (baseline["max_makespan_dilation"] - 1.0)
    )
    max_slowdown_reduction = 100.0 * (
        1.0 - empirical["max_slowdown"] / baseline["max_slowdown"]
    )
    lines = [
        "# YOLOv8 调度策略正式对照",
        "",
        "| Strategy | Median latency (ms) | Concurrent calls | Negative-utility calls | Mean dilation | Max dilation | Mean max slowdown | Max slowdown | Mean utility |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = (
        "TD+Janus-no-alpha",
        "TD+GuardedDRT",
        "TD+RiskAdjustedDRT",
        "TD+EmpiricalDRT",
    )
    for name in order:
        item = variants[name]
        lines.append(
            f"| {name} | {item['median_of_process_medians_ms']:.3f}"
            f" | {item['concurrent_call_count']:.0f}"
            f" | {item['negative_utility_call_count']:.0f}"
            f" | {item['mean_makespan_dilation']:.3f}"
            f" | {item['max_makespan_dilation']:.3f}"
            f" | {item['mean_max_slowdown']:.3f}"
            f" | {item['max_slowdown']:.3f}"
            f" | {item['mean_utility']:.3f} |"
        )
    lines.extend([
        "",
        "## 结论",
        "",
        f"- 相对 TD+Janus，TD+EmpiricalDRT 的端到端中位时延变化为 {latency_delta:+.2f}%。",
        f"- 最坏整组额外膨胀（dilation−1）降低 {excess_dilation_reduction:.1f}%。",
        f"- 最坏单算子 slowdown 降低 {max_slowdown_reduction:.1f}%。",
        f"- 负经验效用的并发选择从 {baseline['negative_utility_call_count']:.0f} 次降为 {empirical['negative_utility_call_count']:.0f} 次。",
        "- GuardedDRT 与 Janus 的调度序列相同；其时延差异属于同调度测量波动，不能解释成算法收益。",
        "",
        "经验缓存中的算子名称仅作为 FX 节点标识；决策依据是实测 solo/co-run 时延，不使用名称类别规则。",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    cache_payload = json.loads(args.pair_cache.read_text(encoding="utf-8"))
    cache = cache_payload["pairs"]
    weights = cache_payload["utility_weights"]
    result_paths = sorted({
        path.resolve()
        for root in args.result_root
        for path in root.rglob("result.json")
    })
    grouped = {}
    for path in result_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("task", {}).get("model") != "YOLOv8x":
            continue
        grouped.setdefault(result["task"]["variant"], []).append(result)

    required = {
        "TD+Janus-no-alpha",
        "TD+GuardedDRT",
        "TD+RiskAdjustedDRT",
        "TD+EmpiricalDRT",
    }
    if set(grouped) != required:
        raise RuntimeError(
            f"expected variants {sorted(required)}, got {sorted(grouped)}"
        )
    variants = {}
    for name, results in grouped.items():
        if len(results) != 5:
            raise RuntimeError(f"{name} has {len(results)} results, expected 5")
        runs = [aggregate_run(result, cache, weights) for result in results]
        variants[name] = {
            "repeat_count": len(runs),
            "process_medians_ms": [run["process_median_ms"] for run in runs],
            "median_of_process_medians_ms": median_field(
                runs, "process_median_ms"
            ),
            "concurrent_call_count": median_field(
                runs, "concurrent_call_count"
            ),
            "negative_utility_call_count": median_field(
                runs, "negative_utility_call_count"
            ),
            "mean_makespan_dilation": median_field(
                runs, "mean_makespan_dilation"
            ),
            "max_makespan_dilation": median_field(
                runs, "max_makespan_dilation"
            ),
            "mean_max_slowdown": median_field(runs, "mean_max_slowdown"),
            "max_slowdown": median_field(runs, "max_slowdown"),
            "mean_utility": median_field(runs, "mean_utility"),
        }
    comparison = {
        "schema_version": 1,
        "model": "YOLOv8x",
        "pair_profile_count": cache_payload["pair_count"],
        "utility_weights": weights,
        "variants": variants,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "strategy_comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "strategy_comparison.md").write_text(
        render_report(comparison), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
