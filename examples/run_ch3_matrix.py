#!/usr/bin/env python3
"""Resumable chapter-3 experiment matrix; this process never imports torch."""

from __future__ import annotations

import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path
import subprocess
import sys
import time

from build_compatibility_table import build_entry, load_result
from multi_janus_benchmark import atomic_json
from multi_janus_models import MODEL_CHOICES


def pair_slug(left: str, right: str, batch_size: int) -> str:
    ordered = sorted((left, right))
    return f"{ordered[0]}_b{batch_size}__{ordered[1]}_b{batch_size}"


def completed_result(path: Path, expected_models: list[str]) -> bool:
    result_path = path / "result.json"
    if not result_path.exists():
        return False
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return (
        payload.get("models") == expected_models
        and payload.get("overall", {}).get("correctness_ok") is True
    )


def run_benchmark(
    benchmark: Path,
    output: Path,
    models: list[str],
    mode: str,
    iterations: int,
    warmups: int,
    trial: int,
    timeout: int,
    lookup_table: Path | None = None,
) -> None:
    if completed_result(output, models):
        print(f"SKIP complete: {output}", flush=True)
        return
    if output.exists():
        raise RuntimeError(f"incomplete output requires audit, refusing overwrite: {output}")
    command = [
        sys.executable,
        str(benchmark),
        "--models",
        *models,
        "--mode",
        mode,
        "--iterations",
        str(iterations),
        "--warmups",
        str(warmups),
        "--trial",
        str(trial),
        "--timeout",
        str(timeout),
        "--require-mps",
        "--output-dir",
        str(output),
    ]
    if lookup_table is not None:
        command.extend(("--lookup-table", str(lookup_table)))
    print(f"RUN {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def build_matrix_table(
    root: Path,
    models: list[str],
    profile_trials: int,
    batch_size: int,
    minimum_response_gain: float,
    minimum_throughput_gain: float,
    maximum_service_slowdown: float,
) -> Path:
    entries = {}
    for left, right in combinations_with_replacement(models, 2):
        slug = pair_slug(left, right, batch_size)
        sequential = []
        concurrent = []
        for trial in range(profile_trials):
            trial_root = root / "profile" / slug / f"trial_{trial:03d}"
            sequential.append(load_result(trial_root / "sequential" / "result.json"))
            concurrent.append(load_result(trial_root / "concurrent" / "result.json"))
        key, entry = build_entry(
            sequential,
            concurrent,
            minimum_response_gain,
            minimum_throughput_gain,
            maximum_service_slowdown,
        )
        entries[key] = entry
    table = {
        "schema_version": 1,
        "protocol": "multi_janus_pair_compatibility_v1",
        "created_time_ns": time.time_ns(),
        "training_scope": {
            "models": models,
            "batch_size": batch_size,
            "profile_trials": profile_trials,
        },
        "thresholds": {
            "minimum_response_gain": minimum_response_gain,
            "minimum_throughput_gain": minimum_throughput_gain,
            "maximum_service_slowdown": maximum_service_slowdown,
        },
        "entries": entries,
    }
    path = root / "compatibility.json"
    atomic_json(path, table)
    print(f"WROTE {path} entries={len(entries)}", flush=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("solo", "profile", "evaluate", "all"), required=True
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES)
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--profile-trials", type=int, default=3)
    parser.add_argument("--evaluation-trials", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--minimum-response-gain", type=float, default=1.10)
    parser.add_argument("--minimum-throughput-gain", type=float, default=1.05)
    parser.add_argument("--maximum-service-slowdown", type=float, default=1.75)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.iterations <= 0 or args.warmups < 0:
        raise ValueError("invalid batch size, iterations, or warmups")
    if args.profile_trials <= 0 or args.evaluation_trials <= 0:
        raise ValueError("trial counts must be positive")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models contains duplicates")

    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    benchmark = Path(__file__).resolve().with_name("multi_janus_benchmark.py")
    atomic_json(
        root / "last_invocation.json",
        {
            "schema_version": 1,
            "models": args.models,
            "batch_size": args.batch_size,
            "iterations": args.iterations,
            "warmups": args.warmups,
            "profile_trials": args.profile_trials,
            "evaluation_trials": args.evaluation_trials,
        },
    )

    if args.phase in ("solo", "all"):
        for model in args.models:
            run_benchmark(
                benchmark,
                root / "solo" / f"{model}_b{args.batch_size}",
                [model],
                "concurrent",
                args.iterations,
                args.warmups,
                0,
                args.timeout,
            )

    pairs = list(combinations_with_replacement(args.models, 2))
    if args.phase in ("profile", "all"):
        for left, right in pairs:
            slug = pair_slug(left, right, args.batch_size)
            for trial in range(args.profile_trials):
                trial_root = root / "profile" / slug / f"trial_{trial:03d}"
                for mode in ("sequential", "concurrent"):
                    run_benchmark(
                        benchmark,
                        trial_root / mode,
                        [left, right],
                        mode,
                        args.iterations,
                        args.warmups,
                        trial,
                        args.timeout,
                    )
        build_matrix_table(
            root,
            args.models,
            args.profile_trials,
            args.batch_size,
            args.minimum_response_gain,
            args.minimum_throughput_gain,
            args.maximum_service_slowdown,
        )

    if args.phase in ("evaluate", "all"):
        table = root / "compatibility.json"
        if not table.exists():
            raise FileNotFoundError(f"profile table is missing: {table}")
        for left, right in pairs:
            slug = pair_slug(left, right, args.batch_size)
            for trial_offset in range(args.evaluation_trials):
                trial = 100 + trial_offset
                trial_root = root / "evaluation" / slug / f"trial_{trial:03d}"
                for mode in ("sequential", "concurrent", "lookup"):
                    run_benchmark(
                        benchmark,
                        trial_root / mode,
                        [left, right],
                        mode,
                        args.iterations,
                        args.warmups,
                        trial,
                        args.timeout,
                        table if mode == "lookup" else None,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
