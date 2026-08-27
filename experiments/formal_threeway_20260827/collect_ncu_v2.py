#!/usr/bin/env python3
"""Collect 7x3 NCU-v2 profiles, median them, and validate fail-closed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS), str(REPO)]

from build_ncu_median_cache import REPEAT_COUNT, merge_repeated_caches
from common import (
    MODEL_CLASSES,
    MODELS,
    MODEL_SLUGS,
    require_empty_output,
    sha256_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ncu", default="/usr/local/cuda-12.5/bin/ncu")
    parser.add_argument("--repeats", type=int, default=REPEAT_COUNT)
    parser.add_argument("--minimum-duration-coverage", type=float, default=0.50)
    return parser.parse_args()


def assert_idle_gpu() -> None:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout.strip())
    rows = [
        row.strip()
        for row in completed.stdout.splitlines()
        if row.strip() and "No running processes found" not in row
    ]
    if rows:
        raise RuntimeError("GPU has active compute processes: " + "; ".join(rows))


def run_to_files(command, *, env, stdout_path, stderr_path) -> int:
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        return subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        ).returncode


def smoke_cache(model: str, cache_dir: Path, output_dir: Path, args) -> dict:
    env = os.environ.copy()
    for name in (
        "JANUS_ALLOW_LEGACY_NCU",
        "JANUS_NEW_TD_PAIR_EXTENSION",
        "JANUS_NEW_TD_SOLO_ROOT",
        "JANUS_NEW_TD_FINAL_SELECTOR",
        "OPARA_Q3_PROFILE_MAP",
        "OPARA_PAIR_PROFILE_PATH",
    ):
        env.pop(name, None)
    env.update(
        {
            "PYTHONPATH": str(REPO),
            "JANUS_NCU_CACHE_DIR": str(cache_dir),
            "JANUS_REQUIRE_VALID_NCU": "1",
            "JANUS_NCU_REPORT": "1",
            "JANUS_NCU_MIN_DURATION_COVERAGE": str(
                args.minimum_duration_coverage
            ),
        }
    )
    command = [
        args.python,
        str(EXPERIMENTS / "newtd_accuracy" / "run_one_newtd.py"),
        "--model",
        model,
        "--variant",
        "Baseline",
        "--alpha",
        "none",
        "--repeat-index",
        "0",
        "--max-ready",
        "6",
        "--warmup-iterations",
        "1",
        "--timed-iterations",
        "2",
        "--output-dir",
        str(output_dir),
    ]
    stdout_path = output_dir.parent / "smoke.stdout"
    stderr_path = output_dir.parent / "smoke.stderr"
    returncode = run_to_files(
        command,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    if returncode:
        raise RuntimeError(
            f"{model}: fail-closed cache smoke failed; inspect {stderr_path}"
        )
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    report = result.get("ncu_report") or result.get("ncu_profile") or {}
    if not report.get("experimental_valid"):
        raise RuntimeError(f"{model}: invalid NCU report: {report}")
    if result.get("correctness", {}).get("ok") is not True:
        raise RuntimeError(f"{model}: graph output correctness failed")
    return {
        "status": report.get("status"),
        "duration_coverage": report.get("duration_coverage"),
        "mapped_operators": report.get("mapped_operators"),
        "mapped_kernels": report.get("mapped_kernels"),
        "total_kernels": report.get("total_kernels"),
        "cache_sha256": report.get("cache_sha256"),
        "correctness": result["correctness"],
    }


def main() -> int:
    args = parse_args()
    if args.repeats != REPEAT_COUNT:
        raise ValueError(
            f"formal Stage A requires exactly {REPEAT_COUNT} independent NCU repeats"
        )
    if not 0.0 < args.minimum_duration_coverage <= 1.0:
        raise ValueError("minimum duration coverage must be in (0, 1]")
    if os.environ.get("JANUS_ALLOW_LEGACY_NCU"):
        raise RuntimeError("unset JANUS_ALLOW_LEGACY_NCU")

    output = require_empty_output(args.output_dir)
    cache_dir = output / "ncu_cache"
    cache_dir.mkdir()
    raw_root = output / "raw_repeats"
    raw_root.mkdir()
    env = os.environ.copy()
    env.pop("JANUS_ALLOW_LEGACY_NCU", None)
    env["PYTHONUNBUFFERED"] = "1"

    started_all = time.time()
    collection_records = []
    median_records = []
    for model in MODELS:
        repeat_caches = []
        for repeat in range(REPEAT_COUNT):
            assert_idle_gpu()
            repeat_dir = raw_root / MODEL_SLUGS[model] / f"repeat_{repeat}"
            repeat_dir.mkdir(parents=True)
            identity = repeat_dir / "identity.json"
            raw_csv = repeat_dir / "raw.csv"
            ncu_stderr = repeat_dir / "ncu.stderr"
            command = [
                args.ncu,
                "--csv",
                "--page",
                "details",
                "--print-summary",
                "none",
                "--section",
                "SpeedOfLight",
                "--section",
                "LaunchStats",
                "--nvtx",
                "--nvtx-include",
                "regex:JANUS_OP:.*]",
                "--target-processes",
                "all",
                args.python,
                str(HERE / "profile_ncu_target.py"),
                "--model",
                model,
                "--identity-json",
                str(identity),
            ]
            started = time.time()
            returncode = run_to_files(
                command,
                env=env,
                stdout_path=raw_csv,
                stderr_path=ncu_stderr,
            )
            if returncode:
                raise RuntimeError(
                    f"{model}/repeat {repeat}: NCU failed; inspect {ncu_stderr}"
                )
            identity_payload = json.loads(identity.read_text(encoding="utf-8"))
            expected_class = MODEL_CLASSES[model]
            if identity_payload.get("model_class") != expected_class:
                raise RuntimeError(
                    f"{model}: class mismatch: "
                    f"{identity_payload.get('model_class')} != {expected_class}"
                )
            cache_path = repeat_dir / "cache.ncu.v2.json"
            build_log = repeat_dir / "build.log"
            built = subprocess.run(
                [
                    args.python,
                    str(HERE / "build_ncu_v2_cache.py"),
                    "--raw-csv",
                    str(raw_csv),
                    "--identity-json",
                    str(identity),
                    "--output",
                    str(cache_path),
                ],
                cwd=REPO,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            build_log.write_text(built.stdout, encoding="utf-8")
            if built.returncode:
                raise RuntimeError(
                    f"{model}/repeat {repeat}: cache build failed; "
                    f"inspect {build_log}"
                )
            cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_payload.get("schema_version") != 2 or not cache_payload.get(
                "kernels"
            ):
                raise RuntimeError(f"{model}/repeat {repeat}: empty v2 cache")
            repeat_caches.append(cache_path)
            collection_records.append(
                {
                    "model": model,
                    "repeat": repeat,
                    "ncu_seconds": time.time() - started,
                    "identity": str(identity),
                    "identity_sha256": sha256_file(identity),
                    "raw_csv": str(raw_csv),
                    "raw_csv_sha256": sha256_file(raw_csv),
                    "raw_csv_bytes": raw_csv.stat().st_size,
                    "cache": str(cache_path),
                    "cache_sha256": sha256_file(cache_path),
                    "profile_sha256": identity_payload["profile_sha256"],
                    "fx_code_sha256": identity_payload["fx_code_sha256"],
                    "kernel_launches": len(cache_payload["kernels"]),
                }
            )
            write_json_atomic(
                output / "collection_progress.json", collection_records
            )

        final_cache = cache_dir / f"{MODEL_CLASSES[model]}.ncu.v2.json"
        median_records.append(
            merge_repeated_caches(
                model=model,
                cache_paths=repeat_caches,
                output_path=final_cache,
            )
        )
        write_json_atomic(output / "median_progress.json", median_records)

    by_model = {row["model"]: row for row in median_records}
    for model in MODELS:
        assert_idle_gpu()
        smoke_root = output / "smoke" / MODEL_SLUGS[model]
        smoke_root.mkdir(parents=True)
        by_model[model]["smoke"] = smoke_cache(
            model, cache_dir, smoke_root / "run", args
        )
        write_json_atomic(output / "median_progress.json", median_records)

    manifest = {
        "schema_version": 1,
        "status": "completed",
        "protocol": "frozen_seven_model_three_repeat_ncu_v2_median_stage_a",
        "models": list(MODELS),
        "repeat_count": REPEAT_COUNT,
        "raw_collection_count": len(collection_records),
        "aggregation": "strict identity/launch match then per-launch median",
        "minimum_duration_coverage": args.minimum_duration_coverage,
        "cache_dir": str(cache_dir),
        "total_ncu_seconds": sum(row["ncu_seconds"] for row in collection_records),
        "wall_seconds": time.time() - started_all,
        "collection_records": collection_records,
        "median_records": median_records,
        "finished_unix": time.time(),
    }
    write_json_atomic(output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
