#!/usr/bin/env python3
"""Run ten fresh processes for seven models and three frozen policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS), str(REPO)]

from common import (
    DISPLAY_NAMES,
    MODELS,
    MODEL_SLUGS,
    POLICIES,
    POLICY_LABELS,
    require_empty_output,
    sha256_file,
    write_json_atomic,
)


FORMAL_REPEATS = 10


def formal_latency_mean(process_means_ms: list[float]) -> float:
    """One paper-facing latency: arithmetic mean of ten process means."""
    if len(process_means_ms) != FORMAL_REPEATS:
        raise ValueError(
            f"formal latency requires {FORMAL_REPEATS} process means, "
            f"got {len(process_means_ms)}"
        )
    return statistics.fmean(process_means_ms)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ncu-cache-dir", type=Path, required=True)
    parser.add_argument("--solo-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--repeats", type=int, default=FORMAL_REPEATS)
    return parser.parse_args()


def relevant_env(env: dict[str, str]) -> dict[str, str]:
    prefixes = ("JANUS_", "OPARA_")
    return {
        key: value
        for key, value in sorted(env.items())
        if key.startswith(prefixes) or key == "PYTHONPATH"
    }


def policy_environment(
    policy: str, *, ncu_cache: Path, empty_cache: Path, solo_root: Path
) -> dict[str, str]:
    env = os.environ.copy()
    for name in (
        "JANUS_NEW_TD_PAIR_EXTENSION",
        "JANUS_NEW_TD_SOLO_ROOT",
        "JANUS_NEW_TD_MIN_OVERLAP_US",
        "JANUS_NEW_TD_LAUNCH_GAP_MS",
        "JANUS_NEW_TD_FINAL_SELECTOR",
        "JANUS_REQUIRE_VALID_NCU",
        "JANUS_NCU_REPORT",
        "JANUS_NCU_MIN_DURATION_COVERAGE",
        "JANUS_ALLOW_LEGACY_NCU",
        "JANUS_NCU_CACHE_DIR",
        "JANUS_OVERLOAD_WEIGHT",
        "JANUS_TAIL_WEIGHT",
        "JANUS_OCCUPANCY_WEIGHT",
        "OPARA_RECORD_FINALISTS",
        "OPARA_Q3_PROFILE_MAP",
        "OPARA_TD_FINAL_SELECTOR",
        "OPARA_TD_SPEEDUP_GUARD",
        "OPARA_TD_RISK_TRIGGER",
        "OPARA_TD_RISK_PENALTY",
        "OPARA_TD_TIMELINE_SHORTLIST",
        "OPARA_TD_INTERFERENCE_SHORTLIST",
        "OPARA_TD_MAX_EVENTS",
        "OPARA_PAIR_PROFILE_PATH",
        "OPARA_EMPIRICAL_ROUND_PENALTY",
        "OPARA_EMPIRICAL_OPERATOR_PENALTY",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = str(REPO)
    env["JANUS_NCU_CACHE_DIR"] = str(empty_cache)
    env.update(
        {
            "JANUS_OVERLOAD_WEIGHT": "1.0",
            "JANUS_TAIL_WEIGHT": "0.02",
            "JANUS_OCCUPANCY_WEIGHT": "0.005",
            "OPARA_TD_TIMELINE_SHORTLIST": "8",
            "OPARA_TD_INTERFERENCE_SHORTLIST": "12",
            "OPARA_TD_MAX_EVENTS": "100000",
        }
    )
    if policy != "janus":
        env.update(
            {
                "JANUS_NEW_TD_PAIR_EXTENSION": "1",
                "JANUS_NEW_TD_SOLO_ROOT": str(solo_root),
                "JANUS_NEW_TD_MIN_OVERLAP_US": "2.0",
                "JANUS_NEW_TD_LAUNCH_GAP_MS": "0.004096",
                "JANUS_NEW_TD_FINAL_SELECTOR": (
                    "strategy"
                    if policy == "newtd_drt"
                    else "risk_adjusted_interference"
                ),
            }
        )
    if policy == "newtd_ncu_drt":
        env.update(
            {
                "JANUS_NCU_CACHE_DIR": str(ncu_cache),
                "JANUS_REQUIRE_VALID_NCU": "1",
                "JANUS_NCU_REPORT": "1",
                "JANUS_NCU_MIN_DURATION_COVERAGE": "0.50",
            }
        )
    return env


def validate_result(
    payload: dict, *, model: str, policy: str, expected_cache_hashes: dict[str, str]
) -> None:
    if payload.get("status") != "completed":
        raise RuntimeError(f"{model}/{policy}: result is not completed")
    if payload.get("task", {}).get("model") != model:
        raise RuntimeError(f"{model}/{policy}: task identity differs")
    if payload.get("correctness", {}).get("ok") is not True:
        raise RuntimeError(f"{model}/{policy}: output correctness failed")
    params = payload.get("effective_parameters") or {}
    scheduler = payload.get("scheduler", {}).get("summary", {})
    if params.get("max_ready") != 6 or scheduler.get("max_ready") != 6:
        raise RuntimeError(f"{model}/{policy}: max-ready is not six")
    ncu = payload.get("ncu_report") or payload.get("ncu_profile") or {}
    newtd = payload.get("new_td_admission")
    if policy == "janus":
        if newtd is not None or params.get("selection_mode") != "legacy_balance":
            raise RuntimeError(f"{model}: Janus policy identity differs")
        if params.get("time_domain") is not False:
            raise RuntimeError(f"{model}: Janus must use Static admission")
    else:
        if not newtd or newtd.get("mode") != "static_union_frozen_td_pair_v1":
            raise RuntimeError(f"{model}/{policy}: frozen NewTD was not installed")
        if newtd.get("minimum_predicted_overlap_us") != 2.0:
            raise RuntimeError(
                f"{model}/{policy}: NewTD threshold is not frozen at 2.0 us"
            )
        if newtd.get("launch_gap_ms") != 0.004096:
            raise RuntimeError(f"{model}/{policy}: NewTD launch gap differs")
        expected_selector = (
            "strategy" if policy == "newtd_drt" else "risk_adjusted_interference"
        )
        if params.get("final_selector") != expected_selector:
            raise RuntimeError(f"{model}/{policy}: selector identity differs")
    if policy == "newtd_ncu_drt":
        if not ncu.get("experimental_valid"):
            raise RuntimeError(f"{model}: fail-closed NCU report is invalid: {ncu}")
        if ncu.get("cache_sha256") != expected_cache_hashes[model]:
            raise RuntimeError(f"{model}: NCU cache hash differs")
        aggregation = ncu.get("aggregation") or {}
        if (
            aggregation.get("method") != "identity-checked per-launch median"
            or aggregation.get("repeat_count") != 3
        ):
            raise RuntimeError(
                f"{model}: NCU cache is not the formal three-repeat median"
            )
        coverage = ncu.get("duration_coverage")
        if not isinstance(coverage, (int, float)) or not 0.5 <= coverage <= 1.0:
            raise RuntimeError(
                f"{model}: NCU mapping duration coverage is invalid: {coverage}"
            )
    elif ncu.get("experimental_valid"):
        raise RuntimeError(f"{model}/{policy}: non-NCU policy loaded valid NCU data")
    count = payload.get("timing", {}).get("statistics", {}).get("count")
    expected_count = payload.get("model_spec", {}).get("timed_iterations")
    if count != expected_count or not count:
        raise RuntimeError(f"{model}/{policy}: timing sample count differs")


def aggregate(output: Path, repeats: int, verification: dict) -> dict:
    cache_hashes = {
        row["model"]: row["ncu_cache_sha256"]
        for row in verification["records"]
    }
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    results_by_identity: dict[tuple[str, str, int], dict] = {}
    task_records = []
    for result_path in sorted((output / "tasks").glob("**/result.json")):
        policy_payload = json.loads(
            (result_path.parent / "policy.json").read_text(encoding="utf-8")
        )
        policy = policy_payload["policy"]
        model = policy_payload["model"]
        trial = int(policy_payload["trial"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result(
            result,
            model=model,
            policy=policy,
            expected_cache_hashes=cache_hashes,
        )
        results_by_identity[(model, policy, trial)] = result
        process_mean = float(
            result["timing"]["statistics"]["mean_ms"]
        )
        grouped.setdefault((model, policy), []).append((trial, process_mean))
        task_records.append(
            {
                "model": model,
                "policy": policy,
                "trial": trial,
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "process_mean_ms": process_mean,
            }
        )

    aggregates = {}
    for model in MODELS:
        for policy in POLICIES:
            values = sorted(grouped.get((model, policy), []))
            if len(values) != repeats or [trial for trial, _ in values] != list(
                range(repeats)
            ):
                raise RuntimeError(
                    f"{model}/{policy}: expected {repeats} complete trials, got {values}"
                )
            process_means = [value for _, value in values]
            final = formal_latency_mean(process_means)
            aggregates[(model, policy)] = {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "policy": policy,
                "policy_label": POLICY_LABELS[policy],
                "completed_processes": len(process_means),
                "process_means_ms": process_means,
                "final_latency_ms": final,
                "sample_std_process_mean_ms": statistics.stdev(process_means),
            }

    table = []
    drt_speedups = []
    ncu_speedups = []
    ncu_vs_drt = []
    for model in MODELS:
        janus = aggregates[(model, "janus")]["final_latency_ms"]
        drt = aggregates[(model, "newtd_drt")]["final_latency_ms"]
        ncu = aggregates[(model, "newtd_ncu_drt")]["final_latency_ms"]
        row = {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "janus_latency_ms": janus,
            "newtd_drt_latency_ms": drt,
            "newtd_ncu_drt_latency_ms": ncu,
            "newtd_drt_improvement_vs_janus_pct": (janus - drt) / janus * 100.0,
            "newtd_ncu_drt_improvement_vs_janus_pct": (janus - ncu) / janus * 100.0,
            "newtd_ncu_drt_improvement_vs_newtd_drt_pct": (drt - ncu) / drt * 100.0,
        }
        table.append(row)
        drt_speedups.append(janus / drt)
        ncu_speedups.append(janus / ncu)
        ncu_vs_drt.append(drt / ncu)

    def geometric_mean(values):
        return math.exp(statistics.fmean(math.log(value) for value in values))

    def unique_ready_map(result):
        observed = {}
        duplicates = set()
        for call in result["scheduler"]["calls"]:
            key = tuple(call.get("ready_ops", []))
            if key in observed:
                duplicates.add(key)
            else:
                observed[key] = tuple(call.get("selected_resource", []))
        return {
            key: value for key, value in observed.items() if key not in duplicates
        }

    trial_policy_comparisons = []
    for trial in range(repeats):
        for model in MODELS:
            identities = {
                policy: results_by_identity[(model, policy, trial)]
                for policy in POLICIES
            }
            input_hashes = {
                policy: [row["sha256"] for row in result["input_identity"]]
                for policy, result in identities.items()
            }
            if len({tuple(value) for value in input_hashes.values()}) != 1:
                raise RuntimeError(
                    f"{model}/trial{trial}: policy inputs are not byte-identical"
                )
            profiles = {
                result["profile"]["sha256"] for result in identities.values()
            }
            if len(profiles) != 1:
                raise RuntimeError(
                    f"{model}/trial{trial}: policy serial profiles differ"
                )
            maps = {policy: unique_ready_map(result) for policy, result in identities.items()}

            def compare(left, right):
                keys = sorted(set(maps[left]) & set(maps[right]))
                changed = [key for key in keys if maps[left][key] != maps[right][key]]
                changed_records = [
                    {
                        "ready_ops": list(key),
                        "left_selected_resource": list(maps[left][key]),
                        "right_selected_resource": list(maps[right][key]),
                    }
                    for key in changed
                ]
                changed_raw = json.dumps(
                    changed_records,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                return {
                    "left": left,
                    "right": right,
                    "exact_unique_ready_states": len(keys),
                    "changed_selection_states": len(changed),
                    "unchanged_selection_states": len(keys) - len(changed),
                    "changed_selection": bool(changed),
                    "changed_selection_sha256": hashlib.sha256(
                        changed_raw
                    ).hexdigest(),
                    "changed_selection_examples": changed_records[:20],
                }

            ncu = identities["newtd_ncu_drt"]
            ncu_report = ncu.get("ncu_report") or ncu["ncu_profile"]
            trial_policy_comparisons.append(
                {
                    "trial": trial,
                    "model": model,
                    "display_name": DISPLAY_NAMES[model],
                    "input_identity": input_hashes["janus"],
                    "all_policy_inputs_byte_identical": True,
                    "all_policy_profiles_identical": True,
                    "profile_sha256": next(iter(profiles)),
                    "reference_output_sha256": {
                        policy: [
                            row["sha256"]
                            for row in result["reference_output_identity"]
                        ]
                        for policy, result in identities.items()
                    },
                    "output_max_absolute_difference": {
                        policy: result["correctness"][
                            "max_absolute_difference"
                        ]
                        for policy, result in identities.items()
                    },
                    "ncu_report": {
                        "experimental_valid": ncu_report.get(
                            "experimental_valid"
                        ),
                        "status": ncu_report.get("status"),
                        "duration_coverage": ncu_report.get(
                            "duration_coverage"
                        ),
                        "mapped_operators": ncu_report.get(
                            "mapped_operators"
                        ),
                        "mapped_kernels": ncu_report.get(
                            "mapped_kernels"
                        ),
                        "total_kernels": ncu_report.get(
                            "total_kernels"
                        ),
                        "aggregation": ncu_report.get("aggregation"),
                        "selected_concurrent_mean_ncu_coverage": ncu[
                            "scheduler"
                        ]["summary"].get(
                            "selected_interference_mean_ncu_coverage"
                        ),
                    },
                    "comparisons": [
                        compare("newtd_drt", "newtd_ncu_drt"),
                        compare("janus", "newtd_ncu_drt"),
                    ],
                }
            )

    return {
        "schema_version": 1,
        "status": "completed",
        "protocol": "seven_model_three_policy_ten_process_mean_latency_v1",
        "primary_statistic": (
            "arithmetic mean of ten independent process means; one final "
            "latency per model and policy"
        ),
        "models": list(MODELS),
        "policies": [
            {"id": policy, "label": POLICY_LABELS[policy]}
            for policy in POLICIES
        ],
        "formal_table": table,
        "geometric_mean_speedup": {
            "newtd_drt_vs_janus": geometric_mean(drt_speedups),
            "newtd_ncu_drt_vs_janus": geometric_mean(ncu_speedups),
            "newtd_ncu_drt_vs_newtd_drt": geometric_mean(ncu_vs_drt),
        },
        "appendix_aggregates": list(aggregates.values()),
        "trial_policy_comparisons": trial_policy_comparisons,
        "task_records": task_records,
    }


def write_csv_summary(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_cross_policy_audits(output: Path, rows: list[dict]) -> int:
    """Write one canonical comparison and one audit beside each process."""
    written = 0
    for row in rows:
        comparison_path = (
            output
            / "comparisons"
            / f"trial_{row['trial']:02d}"
            / f"{MODEL_SLUGS[row['model']]}.json"
        )
        write_json_atomic(comparison_path, row)
        for policy in POLICIES:
            task_dir = (
                output
                / "tasks"
                / f"trial_{row['trial']:02d}"
                / MODEL_SLUGS[row["model"]]
                / policy
            )
            write_json_atomic(
                task_dir / "cross_policy_audit.json",
                {
                    **row,
                    "current_policy": policy,
                    "canonical_comparison": str(comparison_path),
                },
            )
            written += 1
    return written


def main() -> int:
    args = parse_args()
    if args.repeats != FORMAL_REPEATS:
        raise ValueError(
            f"formal protocol requires exactly {FORMAL_REPEATS} processes"
        )
    output = require_empty_output(args.output_dir)
    ncu_cache = args.ncu_cache_dir.resolve()
    solo_root = args.solo_root.resolve()
    if not ncu_cache.is_dir() or not solo_root.is_dir():
        raise FileNotFoundError("NCU cache directory or solo root is missing")
    expected_python = Path(
        json.loads((EXPERIMENTS / "repro_config.json").read_text(encoding="utf-8"))[
            "environment"
        ]["python_executable"]
    ).resolve()
    if not Path(args.python).resolve().samefile(expected_python):
        raise RuntimeError(
            f"wrong Python: {args.python}; expected {expected_python}"
        )
    git_head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain"], text=True
    )
    if git_status:
        raise RuntimeError("formal worktree must be clean")

    verification_path = output / "asset_verification.json"
    checked = subprocess.run(
        [
            args.python,
            str(HERE / "verify_formal_assets.py"),
            "--ncu-cache-dir",
            str(ncu_cache),
            "--solo-root",
            str(solo_root),
            "--output",
            str(verification_path),
        ],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output / "asset_verification.log").write_text(
        checked.stdout, encoding="utf-8"
    )
    if checked.returncode:
        raise RuntimeError("formal asset verification failed")
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    cache_hashes = {
        row["model"]: row["ncu_cache_sha256"]
        for row in verification["records"]
    }

    empty_cache = output / "empty_ncu_cache"
    empty_cache.mkdir()
    tasks = []
    for trial in range(args.repeats):
        policy_order = list(POLICIES[trial % len(POLICIES) :]) + list(
            POLICIES[: trial % len(POLICIES)]
        )
        model_offset = trial % len(MODELS)
        model_order = list(MODELS[model_offset:]) + list(MODELS[:model_offset])
        for model in model_order:
            for policy in policy_order:
                tasks.append({"trial": trial, "model": model, "policy": policy})
    plan = {
        "schema_version": 1,
        "git_head": git_head,
        "repeats": args.repeats,
        "task_count": len(tasks),
        "ordering": "cyclic policy order and cyclic model order by trial",
        "max_ready": 6,
        "tasks": tasks,
    }
    write_json_atomic(output / "plan.json", plan)
    write_json_atomic(
        output / "run_status.json",
        {"status": "running", "completed": 0, "total": len(tasks)},
    )

    runner = EXPERIMENTS / "newtd_accuracy" / "run_one_newtd.py"
    for index, task in enumerate(tasks, 1):
        model = task["model"]
        policy = task["policy"]
        trial = task["trial"]
        task_dir = (
            output
            / "tasks"
            / f"trial_{trial:02d}"
            / MODEL_SLUGS[model]
            / policy
        )
        task_dir.mkdir(parents=True)
        command = [
            args.python,
            str(runner),
            "--model",
            model,
            "--variant",
            "Baseline" if policy == "janus" else "TD+DRT",
            "--alpha",
            "none",
            "--repeat-index",
            str(trial),
            "--max-ready",
            "6",
            "--output-dir",
            str(task_dir),
        ]
        env = policy_environment(
            policy,
            ncu_cache=ncu_cache,
            empty_cache=empty_cache,
            solo_root=solo_root,
        )
        write_json_atomic(
            task_dir / "policy.json",
            {
                **task,
                "policy_label": POLICY_LABELS[policy],
                "command": command,
                "environment": relevant_env(env),
                "expected_ncu_cache_sha256": (
                    cache_hashes[model] if policy == "newtd_ncu_drt" else None
                ),
            },
        )
        with (task_dir / "stdout.log").open(
            "x", encoding="utf-8"
        ) as stdout, (task_dir / "stderr.log").open(
            "x", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if completed.returncode:
            write_json_atomic(
                output / "run_status.json",
                {
                    "status": "failed",
                    "completed": index - 1,
                    "total": len(tasks),
                    "failed_task": task,
                    "returncode": completed.returncode,
                },
            )
            raise RuntimeError(f"formal task failed: {task}")
        result_path = task_dir / "result.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result(
            payload,
            model=model,
            policy=policy,
            expected_cache_hashes=cache_hashes,
        )
        write_json_atomic(
            output / "run_status.json",
            {"status": "running", "completed": index, "total": len(tasks)},
        )

    summary = aggregate(output, args.repeats, verification)
    summary.update(
        {
            "git_head": git_head,
            "asset_verification": str(verification_path),
            "finished_unix": time.time(),
        }
    )
    write_json_atomic(output / "summary.json", summary)
    write_csv_summary(output / "summary.csv", summary["formal_table"])
    # A child process cannot compare itself with two processes that have not
    # run yet.  After all three finish, put the immutable cross-policy audit
    # beside every process result as well as in comparisons/.
    audit_count = write_cross_policy_audits(
        output, summary["trial_policy_comparisons"]
    )
    if audit_count != len(tasks):
        raise RuntimeError(
            f"expected one cross-policy audit per process: "
            f"{audit_count} != {len(tasks)}"
        )
    write_json_atomic(
        output / "run_status.json",
        {"status": "completed", "completed": len(tasks), "total": len(tasks)},
    )
    (output / "COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
