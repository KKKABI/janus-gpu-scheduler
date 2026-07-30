#!/usr/bin/env python3
"""Extract unique constituent pairs from instrumented scheduler finalists."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def canonical(left, right):
    return tuple(sorted((left, right)))


def pair_interference(left, right):
    products = [
        left_value * right_value
        for left_value, right_value in zip(left["vector"], right["vector"])
    ]
    shared_pressure = 0.5 * max(products) + 0.5 * sum(products) / len(products)
    temporal_overlap = min(left["duration"], right["duration"]) / max(
        left["duration"], right["duration"], 1e-9
    )
    pair_conflict = shared_pressure * temporal_overlap
    static_sums = [
        left["vector"][index] + right["vector"][index]
        for index in range(3)
    ]
    raw_overload = max(0.0, max(static_sums) - 1.0)
    capacity_overload = raw_overload / (1.0 + raw_overload)
    return min(1.0, 0.7 * pair_conflict + 0.3 * capacity_overload)


def main():
    args = parse_args()
    model = None
    profiles = {}
    pairs = {}
    exact = {}
    sources = {}
    for path in args.capture:
        payload = json.loads(path.read_text(encoding="utf-8"))
        current_model = payload["task"]["model"]
        if model is None:
            model = current_model
        elif current_model != model:
            raise RuntimeError(f"captures mix models: {model}, {current_model}")
        variant = payload["task"]["variant"]
        for call in payload["scheduler"]["calls"]:
            for finalist in call.get("finalist_timelines", []):
                for profile in finalist.get("operator_profiles", []):
                    profiles.setdefault(profile["name"], profile)
                operators = finalist.get("operators", [])
                for left, right in itertools.combinations(operators, 2):
                    key = canonical(left, right)
                    pairs[key] = True
                    sources.setdefault(key, []).append({
                        "variant": variant,
                        "call": call["call"],
                        "finalist_size": finalist["size"],
                    })
                if finalist.get("size") == 2:
                    key = canonical(*operators)
                    exact.setdefault(key, finalist["timeline"])

    specs = []
    for index, key in enumerate(sorted(pairs), start=1):
        left, right = key
        timeline = exact.get(key)
        specs.append({
            "id": f"pair_{index:04d}",
            "a": left,
            "b": right,
            "predicted_risk": (
                float(timeline["interference"]["risk"])
                if timeline is not None
                else pair_interference(profiles[left], profiles[right])
            ),
            "predicted_speedup": (
                float(timeline["predicted_speedup"])
                if timeline is not None else 1.0
            ),
            "scheduler_decision": "finalist_constituent",
            "operator_profiles": [profiles[left], profiles[right]],
            "sources": sources[key],
        })

    output = {
        "schema_version": 1,
        "model": model,
        "pair_count": len(specs),
        "pairs": specs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({
        "model": model,
        "pair_count": len(specs),
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
