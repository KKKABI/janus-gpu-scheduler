#!/usr/bin/env python3
"""Discover Static/TD simulator predictions before final candidate scoring.

The protocol mirrors the controls described by Janus Section 4.7:

* every resource-bearing LP operator is forced onto the HP path;
* at most six ready GPU operators are presented to the scheduler;
* when trimming is required, operators with the highest predicted achieved
  occupancy are retained;
* Static and TD are evaluated as paired yes/no predicates on every exact
  multi-operator candidate from the same ready set.

The reference policy only determines which scheduler states are visited.  Its
Janus/DRT final score is never used as a simulator prediction label.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


EXPECTED_HEAD = "32bf4974994005855896a360c34ba455303f5ff3"
MODEL_CHOICES = (
    "GoogLeNet",
    "Inception-v3",
    "NASNet",
    "YOLOv8x",
    "ConvNeXt",
    "DeepFM",
    "BERT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--reference-variant", choices=("Baseline", "TD+DRT"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-ready", type=int, default=6)
    parser.add_argument("--max-group-size", type=int, default=5)
    parser.add_argument("--skip-idle-check", action="store_true")
    parser.add_argument(
        "--solo-profile-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--td-v2-launch-gap-ms", type=float, default=0.004096)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_variant_environment(params: dict[str, Any]) -> None:
    values = {
        "OPARA_TD_FINAL_SELECTOR": params.get("final_selector"),
        "OPARA_TD_SPEEDUP_GUARD": params.get("timeline_speedup_guard"),
        "OPARA_TD_RISK_TRIGGER": params.get("interference_risk_trigger"),
        "OPARA_TD_RISK_PENALTY": params.get("interference_risk_penalty"),
        "OPARA_TD_TIMELINE_SHORTLIST": params.get("td_timeline_shortlist"),
        "OPARA_TD_INTERFERENCE_SHORTLIST": params.get(
            "td_interference_shortlist"
        ),
        "OPARA_TD_MAX_EVENTS": params.get("td_max_events"),
    }
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def static_prediction(resource_model: Any, group: Sequence[Any], now: float) -> dict:
    """Run the repository's original Static all-block placement predicate."""
    model = copy.deepcopy(resource_model)
    model.time_domain = False
    model.update_time(now)
    for operator in group:
        if not operator.kernels:
            return {"prediction": False, "failure_reason": "missing_kernel_profile"}
        if not model.can_apply_launch(operator, now):
            return {
                "prediction": False,
                "failure_reason": "all_profiled_blocks_do_not_fit",
            }
        model.apply_launch(operator, now)
    return {
        "prediction": True,
        "failure_reason": None,
        "predicted_occupancy": float(model.total_utilization()),
    }


def td_prediction(resource_model: Any, group: Sequence[Any], now: float) -> dict:
    """Run the current TD stage-one admission predicate, without DRT scoring."""
    model = copy.deepcopy(resource_model)
    model.time_domain = True
    model.update_time(now)
    metrics = model.evaluate_initial_combo(group, now)
    return {
        "prediction": bool(metrics.get("feasible", False)),
        "failure_reason": metrics.get("failure_reason"),
        "initial_utilization": metrics.get("initial_utilization"),
        "initial_resident_blocks": copy.deepcopy(
            metrics.get("initial_resident_blocks")
        ),
    }


def candidate_row(
    *,
    model: str,
    reference_variant: str,
    call: int,
    ready_ops: Sequence[Any],
    group: Sequence[Any],
    resource_model: Any,
    now: float,
    solo_operators: dict[str, Any] | None = None,
    td_v2_launch_gap_ms: float | None = None,
) -> dict:
    ready_names = [str(operator.name) for operator in ready_ops]
    group_names = [str(operator.name) for operator in group]
    identity = {
        "model": model,
        "reference_variant": reference_variant,
        "call": int(call),
        "ready_ops": ready_names,
        "operators": group_names,
    }
    static = static_prediction(resource_model, group, now)
    td = td_prediction(resource_model, group, now)
    td_v2 = None
    if solo_operators is not None and td_v2_launch_gap_ms is not None:
        from evaluate_td_v2_sample import (
            BLOCK_LIMIT_PER_SM,
            apply_solo_durations,
        )
        from td_v2_simulator import simulate_strict_overlap

        adjusted, missing_solo, duration_scale = apply_solo_durations(
            group, solo_operators
        )
        if missing_solo:
            td_v2 = {
                "prediction": False,
                "failure_reason": "missing_solo_operator_profile",
                "missing_solo_operators": missing_solo,
                "solo_duration_scale": duration_scale,
            }
        else:
            metrics = simulate_strict_overlap(
                adjusted,
                resource_model,
                launch_gap=td_v2_launch_gap_ms,
                kernel_gap=0.0,
                block_limit_per_sm=BLOCK_LIMIT_PER_SM,
            )
            td_v2 = {
                "prediction": bool(metrics.get("strict_parallel", False)),
                "failure_reason": metrics.get("failure_reason"),
                "initial_utilization": metrics.get("initial_utilization"),
                "initial_resident_blocks": metrics.get(
                    "initial_resident_blocks"
                ),
                "strict_overlap_duration": metrics.get(
                    "strict_overlap_duration", 0.0
                ),
                "solo_duration_scale": duration_scale,
            }
    return {
        "candidate_id": stable_hash(identity)[:24],
        **identity,
        "ready_signature": stable_hash(ready_names),
        "group_size": len(group_names),
        "static_prediction": bool(static["prediction"]),
        "td_prediction": bool(td["prediction"]),
        "td_v2_prediction": (
            bool(td_v2["prediction"]) if td_v2 is not None else None
        ),
        "static": static,
        "td": td,
        "td_v2": td_v2,
    }


def enumerate_candidates(
    *,
    model: str,
    reference_variant: str,
    call: int,
    ready_ops: Sequence[Any],
    max_group_size: int,
    resource_model: Any,
    now: float,
    solo_operators: dict[str, Any] | None = None,
    td_v2_launch_gap_ms: float | None = None,
) -> list[dict]:
    rows = []
    upper = min(max_group_size, len(ready_ops))
    for size in range(2, upper + 1):
        for group in combinations(ready_ops, size):
            rows.append(
                candidate_row(
                    model=model,
                    reference_variant=reference_variant,
                    call=call,
                    ready_ops=ready_ops,
                    group=group,
                    resource_model=resource_model,
                    now=now,
                    solo_operators=solo_operators,
                    td_v2_launch_gap_ms=td_v2_launch_gap_ms,
                )
            )
    return rows


def main() -> int:
    args = parse_args()
    if args.max_ready != 6:
        raise ValueError("the paper-aligned protocol requires --max-ready 6")
    if not 2 <= args.max_group_size <= 5:
        raise ValueError("--max-group-size must be in [2, 5]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    repo = Path(__file__).resolve().parents[2]
    experiments = repo / "experiments"
    sys.path[:0] = [str(experiments), str(repo)]
    from harness_common import (
        Task,
        expected_profile_path,
        load_config,
        require_idle_gpu,
    )
    from run_one import (
        compare_outputs,
        load_model_and_inputs,
        seed_everything,
        tensor_leaves,
        variant_parameters,
    )

    git_head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if git_head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected git head: {git_head} != {EXPECTED_HEAD}")
    tracked_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    ).strip()
    if tracked_status:
        raise RuntimeError(f"tracked worktree is dirty:\n{tracked_status}")

    config = load_config()
    expected_python = Path(config["environment"]["python_executable"]).resolve()
    if not Path(sys.executable).resolve().samefile(expected_python):
        raise RuntimeError(
            f"wrong interpreter: {sys.executable}; expected {expected_python}"
        )
    if not args.skip_idle_check:
        require_idle_gpu()

    seed_everything(int(config["measurement"]["seed"]))
    import torch
    from Opara import GraphCapturer, OperatorLauncher
    from Opara.Scheduler import Scheduler as SchedulerClass, get_candidate_stats

    # Match Section 4.7: resource-bearing operators all use the HP path.
    hp_state = {"calls": 0, "resource_operators": 0, "passthrough_operators": 0}

    def force_all_hp(queue, tau=0.5):
        del tau
        hp_state["calls"] += 1
        passthrough = []
        for node in list(queue):
            if getattr(node, "info", None):
                hp_state["resource_operators"] += 1
                continue
            node.is_lowpriority = True
            queue.remove(node)
            passthrough.append(node)
        hp_state["passthrough_operators"] += len(passthrough)
        return passthrough

    OperatorLauncher.pop_lowPriorty_from_queue = force_all_hp

    model, inputs = load_model_and_inputs(args.model, config)
    profile_path = expected_profile_path(model, inputs)
    if not profile_path.is_file():
        raise FileNotFoundError(f"frozen profile is missing: {profile_path}")
    profile_sha256 = sha256_file(profile_path)
    solo_path = None
    solo_payload = None
    solo_operators = None
    if args.solo_profile_root:
        from evaluate_td_v2_sample import load_solo_profiles

        solo_path, solo_payload, solo_operators = load_solo_profiles(
            args.solo_profile_root, args.model, profile_sha256
        )
    with torch.inference_mode():
        reference = [
            tensor.detach().clone() for tensor in tensor_leaves(model(*inputs))
        ]
    torch.cuda.synchronize()

    task = Task(args.model, args.reference_variant, None, 0)
    params = variant_parameters(task, config)
    params["max_ready"] = 6
    os.environ["OPARA_MAX_READY"] = "6"
    apply_variant_environment(params)
    os.environ["OPARA_Q3_PROFILE_MAP"] = str(output_dir / "fx_stream_map.json")
    os.chdir(output_dir)

    original_schedule = SchedulerClass.schedule
    observations: list[dict] = []
    cap_state = {
        "calls": 0,
        "trimmed_calls": 0,
        "raw_ready_max": 0,
        "dropped_operator_count": 0,
    }

    def single_operator_occupancy(scheduler, operator, now):
        model_copy = copy.deepcopy(scheduler.resource_model)
        model_copy.update_time(now)
        if model_copy.time_domain:
            metrics = model_copy.evaluate_initial_combo([operator], now)
            if not metrics.get("feasible"):
                return -1.0
            return float(metrics.get("initial_utilization", 0.0))
        if operator.kernels:
            if not model_copy.can_apply_launch(operator, now):
                return -1.0
            model_copy.apply_launch(operator, now)
        return float(model_copy.total_utilization())

    def observe_and_schedule(scheduler, ready_ops, current_time):
        ready_ops = list(ready_ops)
        cap_state["calls"] += 1
        cap_state["raw_ready_max"] = max(
            cap_state["raw_ready_max"], len(ready_ops)
        )
        if len(ready_ops) > 6:
            ranked = [
                (
                    single_operator_occupancy(
                        scheduler, operator, current_time
                    ),
                    index,
                    operator,
                )
                for index, operator in enumerate(ready_ops)
            ]
            ranked.sort(key=lambda item: (-item[0], item[1]))
            cap_state["trimmed_calls"] += 1
            cap_state["dropped_operator_count"] += len(ready_ops) - 6
            ready_ops = [item[2] for item in ranked[:6]]

        call = cap_state["calls"]
        resource_ops = [operator for operator in ready_ops if operator.kernels]
        observations.extend(
            enumerate_candidates(
                model=args.model,
                reference_variant=args.reference_variant,
                call=call,
                ready_ops=resource_ops,
                max_group_size=args.max_group_size,
                resource_model=scheduler.resource_model,
                now=current_time,
                solo_operators=solo_operators,
                td_v2_launch_gap_ms=(
                    args.td_v2_launch_gap_ms
                    if solo_operators is not None
                    else None
                ),
            )
        )
        return original_schedule(scheduler, ready_ops, current_time)

    SchedulerClass.schedule = observe_and_schedule
    get_candidate_stats(clear=True)
    try:
        runner = GraphCapturer.capturer(
            inputs,
            model,
            copy_outputs=False,
            alpha=params["internal_alpha"],
            selection_mode=params["selection_mode"],
            time_domain=params["time_domain"],
            capture_backend=config["models"][args.model].get(
                "capture_backend", "dynamo_explain"
            ),
        )
    finally:
        SchedulerClass.schedule = original_schedule
    scheduler_calls = get_candidate_stats(clear=True)
    if cap_state["calls"] != len(scheduler_calls):
        raise RuntimeError(
            "observer/scheduler call mismatch: "
            f"{cap_state['calls']} != {len(scheduler_calls)}"
        )
    if hp_state["calls"] <= 0:
        raise RuntimeError("all-HP hook was never called")
    if any(len(row["ready_ops"]) > 6 for row in observations):
        raise RuntimeError("a discovered candidate exceeded the ready cap")

    with torch.inference_mode():
        replay_outputs = runner(*inputs)
    torch.cuda.synchronize()
    correctness = compare_outputs(
        reference,
        replay_outputs,
        float(config["correctness"]["float_rtol"]),
        float(config["correctness"]["float_atol"]),
    )
    if not correctness.get("ok", False):
        raise RuntimeError(f"output correctness failed: {correctness}")

    counts: dict[str, int] = {}
    for row in observations:
        key = (
            f"S{int(row['static_prediction'])}_"
            f"T{int(row['td_prediction'])}_"
            f"V{int(bool(row.get('td_v2_prediction')))}_"
            f"K{row['group_size']}"
        )
        counts[key] = counts.get(key, 0) + 1
    positive_counts = {
        "static": sum(row["static_prediction"] for row in observations),
        "td": sum(row["td_prediction"] for row in observations),
        "td_v2": sum(
            bool(row.get("td_v2_prediction")) for row in observations
        ),
        "union": sum(
            row["static_prediction"] or row["td_prediction"]
            for row in observations
        ),
        "union_static_td_v2": sum(
            row["static_prediction"] or bool(row.get("td_v2_prediction"))
            for row in observations
        ),
    }
    payload = {
        "schema_version": 1,
        "protocol": "janus_4_7_paired_simulator_positive_discovery_v1",
        "prediction_scope": (
            "Static/TD feasibility predicates before Janus/DRT final scoring"
        ),
        "reference_path_role": (
            "visits scheduler states only; final scorer is not a prediction label"
        ),
        "paper_aligned_controls": {
            "all_lp_forced_to_hp": True,
            "max_ready": 6,
            "ready_filter": "highest_predicted_achieved_occupancy",
            "group_sizes": [2, 3, 4, 5],
            "positive_predictions_only_for_hardware_precision": True,
        },
        "model": args.model,
        "reference_variant": args.reference_variant,
        "git_head": git_head,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha256,
        "td_v2_launch_gap_ms": (
            args.td_v2_launch_gap_ms if solo_operators is not None else None
        ),
        "td_v2_duration_source": (
            "solo_operator_cuda_graph_span_v1"
            if solo_operators is not None
            else None
        ),
        "solo_profile_path": str(solo_path) if solo_path else None,
        "solo_profile_target_count": (
            solo_payload.get("target_count") if solo_payload else None
        ),
        "solo_profile_auditable_count": (
            solo_payload.get("auditable_count") if solo_payload else None
        ),
        "effective_parameters": params,
        "input_shapes": [list(value.shape) for value in inputs],
        "input_dtypes": [str(value.dtype) for value in inputs],
        "device": torch.cuda.get_device_name(inputs[0].device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "correctness": correctness,
        "all_hp_hook": hp_state,
        "ready_cap_hook": cap_state,
        "scheduler_call_count": len(scheduler_calls),
        "candidate_count": len(observations),
        "positive_counts": positive_counts,
        "strata": counts,
        "candidates": observations,
    }
    (output_dir / "candidates.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "scheduler_calls.json").write_text(
        json.dumps(json_safe(scheduler_calls), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            json_safe({key: value for key, value in payload.items() if key != "candidates"}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(payload["positive_counts"], indent=2))
    print(json.dumps(payload["strata"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
