#!/usr/bin/env python3
"""Capture one full Janus policy and mark one replay for positive precision.

This entry point implements the Section 4.7-style system protocol used in this
experiment: all LP operators are forced to HP, the ready-set cap is six, and
only the final multi-operator groups selected by the complete policy are later
checked in a hardware trace.  It does not enumerate rejected candidates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


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
    parser.add_argument("--variant", choices=("Baseline", "TD+DRT"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-ready", type=int, default=6)
    parser.add_argument(
        "--new-td-pair-extension",
        action="store_true",
        help=(
            "preserve Static-admitted groups and add only Static-rejected "
            "pairs whose frozen TD simulator predicts at least the requested "
            "strict overlap"
        ),
    )
    parser.add_argument(
        "--solo-profile-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--td-launch-gap-ms", type=float, default=0.004096)
    parser.add_argument("--minimum-predicted-overlap-us", type=float, default=2.0)
    parser.add_argument("--metrics-replays", type=int, default=1)
    parser.add_argument("--skip-idle-check", action="store_true")
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


def apply_variant_environment(params: dict[str, Any]) -> None:
    values = {
        "OPARA_TD_FINAL_SELECTOR": params["final_selector"],
        "OPARA_TD_SPEEDUP_GUARD": params["timeline_speedup_guard"],
        "OPARA_TD_RISK_TRIGGER": params["interference_risk_trigger"],
        "OPARA_TD_RISK_PENALTY": params["interference_risk_penalty"],
        "OPARA_TD_TIMELINE_SHORTLIST": params["td_timeline_shortlist"],
        "OPARA_TD_INTERFERENCE_SHORTLIST": params["td_interference_shortlist"],
        "OPARA_TD_MAX_EVENTS": params["td_max_events"],
    }
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def main() -> int:
    args = parse_args()
    if args.metrics_replays < 1:
        raise ValueError("--metrics-replays must be positive")
    if args.max_ready != 6:
        raise ValueError("the paper-aligned protocol requires --max-ready 6")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    repo = Path(__file__).resolve().parents[2]
    experiments = repo / "experiments"
    sys.path.insert(0, str(experiments))
    sys.path.insert(0, str(repo))
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
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
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

    seed = int(config["measurement"]["seed"])
    seed_everything(seed)
    import torch
    from Opara import GraphCapturer, OperatorLauncher, priority_streams
    from Opara.Scheduler import (
        ResourceModel,
        Scheduler as SchedulerClass,
        get_candidate_stats,
    )

    # Section 4.7 isolates the selected HP groups by making every LP operator HP.
    # Monkeypatching this one classification hook leaves Static/TD feasibility
    # and Janus/DRT final scoring untouched.
    hook_state = {
        "calls": 0,
        "queue_nodes_seen": 0,
        "zero_kernel_nodes_passthrough": 0,
    }

    def force_all_hp(queue, tau=0.5):
        del tau
        hook_state["calls"] += 1
        hook_state["queue_nodes_seen"] += len(queue)
        # Metadata/view nodes with no profiled CUDA kernel are not GPU
        # operators and cannot contribute to co-execution occupancy. Keep all
        # resource-bearing operators on the HP path and pass these nodes through.
        passthrough = []
        for node in list(queue):
            if getattr(node, "info", None):
                continue
            node.is_lowpriority = True
            queue.remove(node)
            passthrough.append(node)
        hook_state["zero_kernel_nodes_passthrough"] += len(passthrough)
        return passthrough

    OperatorLauncher.pop_lowPriorty_from_queue = force_all_hp

    # Section 4.7 keeps at most the six ready GPU operators with the highest
    # predicted achieved occupancy. The repository's OPARA_MAX_READY path uses
    # duration ordering, so apply the paper rule before entering schedule().
    original_schedule = SchedulerClass.schedule
    ready_cap_state = {
        "calls": 0,
        "trimmed_calls": 0,
        "raw_ready_max": 0,
        "dropped_operator_count": 0,
    }
    admission_context = {"active": False, "decisions": {}}
    selected_admission_trace = []

    def single_operator_occupancy(scheduler, operator, current_time):
        model_copy = copy.deepcopy(scheduler.resource_model)
        model_copy.update_time(current_time)
        if model_copy.time_domain:
            metrics = model_copy.evaluate_initial_combo([operator], current_time)
            if not metrics.get("feasible"):
                return -1.0
            return float(metrics.get("initial_utilization", 0.0))
        if operator.kernels:
            if not model_copy.can_apply_launch(operator, current_time):
                return -1.0
            model_copy.apply_launch(operator, current_time)
        return float(model_copy.total_utilization())

    def paper_ready_cap(scheduler, ready_ops, current_time):
        ready_ops = list(ready_ops)
        ready_cap_state["calls"] += 1
        ready_cap_state["raw_ready_max"] = max(
            ready_cap_state["raw_ready_max"], len(ready_ops)
        )
        if len(ready_ops) > 6:
            scored = [
                (
                    single_operator_occupancy(
                        scheduler, operator, current_time
                    ),
                    index,
                    operator,
                )
                for index, operator in enumerate(ready_ops)
            ]
            scored.sort(key=lambda item: (-item[0], item[1]))
            ready_cap_state["trimmed_calls"] += 1
            ready_cap_state["dropped_operator_count"] += len(ready_ops) - 6
            ready_ops = [item[2] for item in scored[:6]]
        if args.new_td_pair_extension:
            admission_context["active"] = True
            admission_context["decisions"] = {}
        try:
            selected = original_schedule(scheduler, ready_ops, current_time)
        finally:
            if args.new_td_pair_extension:
                admission_context["active"] = False
        if args.new_td_pair_extension:
            selected_resource = [
                operator for operator in selected if operator.kernels
            ]
            key = tuple(operator.name for operator in selected_resource)
            decision = admission_context["decisions"].get(key)
            selected_admission_trace.append(
                {
                    "call": ready_cap_state["calls"],
                    "current_time": current_time,
                    "selected_resource": list(key),
                    "selected_size": len(key),
                    "admission": decision,
                }
            )
        return selected

    SchedulerClass.schedule = paper_ready_cap

    model, inputs = load_model_and_inputs(args.model, config)
    profile_path = expected_profile_path(model, inputs)
    configured_profile = config["models"][args.model]["profile_file"]
    if profile_path.name != configured_profile:
        raise RuntimeError(
            f"profile identity mismatch: {profile_path.name} != {configured_profile}"
        )
    if not profile_path.is_file():
        raise FileNotFoundError(f"frozen profile is missing: {profile_path}")
    profile_sha256 = sha256_file(profile_path)

    admission_state = None
    original_evaluate_initial_combo = ResourceModel.evaluate_initial_combo
    if args.new_td_pair_extension:
        if args.variant != "TD+DRT":
            raise ValueError("the new TD pair extension requires --variant TD+DRT")
        if len(args.solo_profile_root) == 0:
            raise ValueError("the new TD pair extension requires --solo-profile-root")
        if args.minimum_predicted_overlap_us <= 0:
            raise ValueError("--minimum-predicted-overlap-us must be positive")

        from evaluate_td_v2_sample import (
            BLOCK_LIMIT_PER_SM,
            apply_solo_durations,
            load_solo_profiles,
        )
        from td_v2_simulator import simulate_strict_overlap

        solo_path, solo_payload, solo_operators = load_solo_profiles(
            args.solo_profile_root, args.model, profile_sha256
        )
        minimum_overlap_ms = args.minimum_predicted_overlap_us / 1000.0
        admission_state = {
            "mode": "static_union_frozen_td_pair_v1",
            "calls": 0,
            "static_accepted": 0,
            "static_rejected": 0,
            "extension_pair_accepted": 0,
            "extension_pair_rejected": 0,
            "wider_extension_blocked": 0,
            "missing_solo_rejected": 0,
            "minimum_predicted_overlap_us": args.minimum_predicted_overlap_us,
            "launch_gap_ms": args.td_launch_gap_ms,
            "block_limit_per_sm": BLOCK_LIMIT_PER_SM,
            "solo_profile_path": str(solo_path),
            "solo_profile_sha256": sha256_file(solo_path),
            "solo_profile_target_count": solo_payload.get("target_count"),
            "solo_profile_auditable_count": solo_payload.get("auditable_count"),
        }

        def static_combo_metrics(resource_model, operators, start_time):
            static_model = copy.deepcopy(resource_model)
            static_model.time_domain = False
            static_model.update_time(start_time)
            for operator in operators:
                if not operator.kernels:
                    continue
                if not static_model.can_apply_launch(operator, start_time):
                    return {"feasible": False, "initial_utilization": -1.0}
                static_model.apply_launch(operator, start_time)
            return {
                "feasible": True,
                "initial_utilization": float(static_model.total_utilization()),
            }

        def evaluate_static_union_td_pair(resource_model, operators, start_time):
            operators = list(operators)
            operator_key = tuple(operator.name for operator in operators)

            def remember(decision):
                if admission_context["active"]:
                    admission_context["decisions"][operator_key] = json_safe(
                        decision
                    )
                return decision

            admission_state["calls"] += 1
            static_metrics = static_combo_metrics(
                resource_model, operators, start_time
            )
            if static_metrics["feasible"]:
                admission_state["static_accepted"] += 1
                return remember({
                    **static_metrics,
                    "failure_reason": None,
                    "admission_source": "static",
                })

            admission_state["static_rejected"] += 1
            if len(operators) != 2:
                admission_state["wider_extension_blocked"] += 1
                return remember({
                    "feasible": False,
                    "initial_utilization": -1.0,
                    "failure_reason": "static_rejected_and_not_pair",
                    "admission_source": None,
                })

            adjusted, missing, duration_scale = apply_solo_durations(
                operators, solo_operators
            )
            if missing:
                admission_state["missing_solo_rejected"] += 1
                admission_state["extension_pair_rejected"] += 1
                return remember({
                    "feasible": False,
                    "initial_utilization": -1.0,
                    "failure_reason": "missing_solo_operator_profile",
                    "missing_solo_operators": missing,
                    "admission_source": None,
                })

            metrics = simulate_strict_overlap(
                adjusted,
                resource_model,
                launch_gap=args.td_launch_gap_ms,
                kernel_gap=0.0,
                block_limit_per_sm=BLOCK_LIMIT_PER_SM,
            )
            overlap_ms = float(metrics.get("strict_overlap_duration", 0.0))
            accepted = bool(metrics.get("strict_parallel")) and overlap_ms >= minimum_overlap_ms
            key = "extension_pair_accepted" if accepted else "extension_pair_rejected"
            admission_state[key] += 1
            return remember({
                "feasible": accepted,
                "initial_utilization": (
                    float(metrics.get("initial_utilization", 0.0))
                    if accepted else -1.0
                ),
                "initial_resident_blocks": metrics.get("initial_resident_blocks"),
                "failure_reason": None if accepted else (
                    metrics.get("failure_reason") or "predicted_overlap_below_threshold"
                ),
                "admission_source": "td_pair_extension" if accepted else None,
                "predicted_strict_overlap_ms": overlap_ms,
                "duration_scale": duration_scale,
            })

        ResourceModel.evaluate_initial_combo = evaluate_static_union_td_pair

    with torch.inference_mode():
        reference = [tensor.detach().clone() for tensor in tensor_leaves(model(*inputs))]
    torch.cuda.synchronize()

    task = Task(args.model, args.variant, None, 0)
    params = variant_parameters(task, config)
    if args.new_td_pair_extension:
        # DRT remains the final ranker.  The shared-timeline selector is an
        # additional TD policy and would confound the admission experiment.
        params["final_selector"] = "strategy"
    params["max_ready"] = 6
    os.environ["OPARA_MAX_READY"] = "6"
    apply_variant_environment(params)
    os.environ["OPARA_Q3_PROFILE_MAP"] = str(output_dir / "fx_stream_map.json")
    os.chdir(output_dir)

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
        ResourceModel.evaluate_initial_combo = original_evaluate_initial_combo
    scheduler_calls = get_candidate_stats(clear=True)
    if not scheduler_calls:
        raise RuntimeError("scheduler produced no calls")
    if hook_state["calls"] <= 0:
        raise RuntimeError("all-HP classification hook was never invoked")
    if ready_cap_state["calls"] <= 0:
        raise RuntimeError("paper ready-cap hook was never invoked")
    if any(int(call.get("ready_used_count", 0)) > 6 for call in scheduler_calls):
        raise RuntimeError("scheduler exceeded the ready-set cap of six")

    replay_outputs = runner(*inputs)
    torch.cuda.synchronize()
    trace_tag = (
        "new_td_pair_drt"
        if args.new_td_pair_extension
        else ("static_janus" if args.variant == "Baseline" else "td_drt")
    )
    # Restrict the high-frequency GPU-metrics capture to the one formal replay.
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"JANUS_PRECISION_REPLAY::{trace_tag}")
    try:
        for _ in range(args.metrics_replays):
            runner._opara_graph.replay()
            torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()

    correctness = compare_outputs(
        reference,
        replay_outputs,
        float(config["correctness"]["float_rtol"]),
        float(config["correctness"]["float_atol"]),
    )
    profile_map = json.loads(
        (output_dir / "fx_stream_map.json").read_text(encoding="utf-8")
    )
    kernel_bearing_names = {
        item["name"]
        for item in profile_map.get("nodes", [])
        if item.get("kernels")
    }
    safe_calls = json_safe(scheduler_calls)
    for call in safe_calls:
        raw_selected = list(call.get("selected_resource") or [])
        gpu_selected = [
            name for name in raw_selected if name in kernel_bearing_names
        ]
        call["selected_gpu_resource"] = gpu_selected
        call["selected_gpu_resource_size"] = len(gpu_selected)
    (output_dir / "scheduler_calls.json").write_text(
        json.dumps(safe_calls, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    selected_groups = [
        call for call in safe_calls
        if int(call.get("selected_gpu_resource_size", 0) or 0) >= 2
    ]
    extension_path = Path(priority_streams.__file__).resolve()
    summary = {
        "schema_version": 1,
        "protocol": "janus_section_4_7_positive_precision_single_trace_v1",
        "paper_aligned_controls": {
            "all_lp_forced_to_hp": True,
            "max_ready": 6,
            "final_selected_groups_only": True,
            "nsys_replays_per_group": 1,
            "rejected_candidates_evaluated": False,
            "kernel_bearing_operators_only": True,
            "ready_filter": "highest_predicted_achieved_occupancy",
            "gpu_metrics_replays": args.metrics_replays,
        },
        "model": args.model,
        "variant": args.variant,
        "configuration": (
            "NewTD(pair extension)+DRT"
            if args.new_td_pair_extension
            else ("Static+Janus" if args.variant == "Baseline" else "TD+DRT")
        ),
        "trace_tag": trace_tag,
        "seed": seed,
        "git_head": git_head,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha256,
        "priority_streams_path": str(extension_path),
        "priority_streams_sha256": sha256_file(extension_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_shapes": [list(value.shape) for value in inputs],
        "input_dtypes": [str(value.dtype) for value in inputs],
        "device": torch.cuda.get_device_name(inputs[0].device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "effective_parameters": params,
        "new_td_admission": admission_state,
        "selected_admission_trace": selected_admission_trace,
        "all_hp_hook": hook_state,
        "ready_cap_hook": ready_cap_state,
        "scheduler_call_count": len(scheduler_calls),
        "kernel_bearing_operator_count": len(kernel_bearing_names),
        "final_multi_operator_group_count": len(selected_groups),
        "correctness": correctness,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
