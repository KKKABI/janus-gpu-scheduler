"""Runtime installer for the frozen Static-union-new-TD pair admission rule."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def install_newtd_pair_admission(
    resource_model_class,
    *,
    model: str,
    profile_sha256: str,
    solo_profile_roots: list[Path],
    launch_gap_ms: float,
    minimum_overlap_us: float,
) -> tuple[Any, dict[str, Any]]:
    from evaluate_td_v2_sample import (
        BLOCK_LIMIT_PER_SM,
        apply_solo_durations,
        load_solo_profiles,
    )
    from td_v2_simulator import simulate_strict_overlap

    if minimum_overlap_us <= 0:
        raise ValueError("minimum_overlap_us must be positive")
    solo_path, solo_payload, solo_operators = load_solo_profiles(
        solo_profile_roots, model, profile_sha256
    )
    minimum_overlap_ms = minimum_overlap_us / 1000.0
    original = resource_model_class.evaluate_initial_combo
    state = {
        "mode": "static_union_frozen_td_pair_v1",
        "calls": 0,
        "static_accepted": 0,
        "static_rejected": 0,
        "extension_pair_accepted": 0,
        "extension_pair_rejected": 0,
        "wider_extension_blocked": 0,
        "missing_solo_rejected": 0,
        "minimum_predicted_overlap_us": minimum_overlap_us,
        "launch_gap_ms": launch_gap_ms,
        "block_limit_per_sm": BLOCK_LIMIT_PER_SM,
        "solo_profile_path": str(solo_path),
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

    def evaluate(resource_model, operators, start_time):
        operators = list(operators)
        state["calls"] += 1
        static_metrics = static_combo_metrics(resource_model, operators, start_time)
        if static_metrics["feasible"]:
            state["static_accepted"] += 1
            return {
                **static_metrics,
                "failure_reason": None,
                "admission_source": "static",
            }
        state["static_rejected"] += 1
        if len(operators) != 2:
            state["wider_extension_blocked"] += 1
            return {
                "feasible": False,
                "initial_utilization": -1.0,
                "failure_reason": "static_rejected_and_not_pair",
                "admission_source": None,
            }
        adjusted, missing, duration_scale = apply_solo_durations(
            operators, solo_operators
        )
        if missing:
            state["missing_solo_rejected"] += 1
            state["extension_pair_rejected"] += 1
            return {
                "feasible": False,
                "initial_utilization": -1.0,
                "failure_reason": "missing_solo_operator_profile",
                "missing_solo_operators": missing,
                "admission_source": None,
            }
        metrics = simulate_strict_overlap(
            adjusted,
            resource_model,
            launch_gap=launch_gap_ms,
            kernel_gap=0.0,
            block_limit_per_sm=BLOCK_LIMIT_PER_SM,
        )
        overlap_ms = float(metrics.get("strict_overlap_duration", 0.0))
        accepted = (
            bool(metrics.get("strict_parallel"))
            and overlap_ms >= minimum_overlap_ms
        )
        state[
            "extension_pair_accepted" if accepted else "extension_pair_rejected"
        ] += 1
        return {
            "feasible": accepted,
            "initial_utilization": (
                float(metrics.get("initial_utilization", 0.0))
                if accepted
                else -1.0
            ),
            "initial_resident_blocks": metrics.get("initial_resident_blocks"),
            "failure_reason": None if accepted else (
                metrics.get("failure_reason")
                or "predicted_overlap_below_threshold"
            ),
            "admission_source": "td_pair_extension" if accepted else None,
            "predicted_strict_overlap_ms": overlap_ms,
            "duration_scale": duration_scale,
        }

    resource_model_class.evaluate_initial_combo = evaluate
    return original, state


def restore_newtd_pair_admission(resource_model_class, original) -> None:
    resource_model_class.evaluate_initial_combo = original
