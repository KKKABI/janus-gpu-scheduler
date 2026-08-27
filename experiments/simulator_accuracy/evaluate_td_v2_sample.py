#!/usr/bin/env python3
"""Revisit sampled scheduler states and evaluate the TD-v2 simulator."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED_HEAD = "32bf4974994005855896a360c34ba455303f5ff3"
# Candidate launch gaps are fixed quantiles measured only on the calibration
# models (GoogLeNet, Inception-v3 and DeepFM): median, p75, p90, p95 and max.
# NASNet, YOLOv8x and BERT remain untouched holdout models.
GAPS_MS = (0.002624, 0.004096, 0.005600, 0.010048, 0.018176)
# RTX A5000 (compute capability 8.6) permits at most 16 resident thread
# blocks per SM.  Shared memory/register/warp checks alone do not encode this
# architectural limit, so pass it explicitly to the simulator.
BLOCK_LIMIT_PER_SM = 16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--discovery-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reference-variant", choices=("Baseline", "TD+DRT"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--solo-profile-root",
        type=Path,
        action="append",
        default=[],
        help="root containing per-model solo operator result.json files",
    )
    return parser.parse_args()


def load_solo_profiles(roots, model, expected_profile_sha):
    matches = []
    for root in roots:
        for path in root.rglob("result.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("model") == model:
                matches.append((path, payload))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one solo profile for {model}, found "
            f"{[str(path) for path, _ in matches]}"
        )
    path, payload = matches[0]
    if payload.get("profile_sha256") != expected_profile_sha:
        raise RuntimeError(
            f"solo/profile SHA mismatch: {payload.get('profile_sha256')} != "
            f"{expected_profile_sha}"
        )
    operators = {}
    for row in payload.get("operators", []):
        if not row.get("auditable"):
            continue
        span_ns = int(row.get("span_duration_ns", 0))
        if span_ns <= 0:
            continue
        operators[row["name"]] = {
            "span_duration_ms": span_ns / 1_000_000.0,
            "active_duration_ms": int(row.get("active_duration_ns", 0))
            / 1_000_000.0,
            "kernel_count": int(row.get("kernel_count", 0)),
        }
    return path, payload, operators


def apply_solo_durations(group, solo_operators):
    adjusted = copy.deepcopy(group)
    missing = []
    factors = {}
    for operator in adjusted:
        solo = solo_operators.get(str(operator.name))
        if solo is None:
            missing.append(str(operator.name))
            continue
        profiled_total = sum(
            max(float(getattr(kernel, "duration", 0.0)), 0.0)
            for kernel in operator.kernels
        )
        if profiled_total <= 0:
            missing.append(str(operator.name))
            continue
        factor = float(solo["span_duration_ms"]) / profiled_total
        for kernel in operator.kernels:
            kernel.duration = max(float(kernel.duration) * factor, 1e-12)
        factors[str(operator.name)] = factor
    return adjusted, missing, factors


def apply_variant_environment(params):
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


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    repo = Path(__file__).resolve().parents[2]
    experiments = repo / "experiments"
    sys.path[:0] = [str(Path(__file__).resolve().parent), str(experiments), str(repo)]
    from td_v2_simulator import simulate_strict_overlap
    from harness_common import Task, load_config, require_idle_gpu
    from run_one import (
        compare_outputs,
        load_model_and_inputs,
        seed_everything,
        tensor_leaves,
        variant_parameters,
    )

    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected head: {head}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["git_head"] != head:
        raise RuntimeError("manifest git head differs")
    expected_profile_sha = manifest["source_profile_sha256_by_model"][args.model]
    solo_path, solo_payload, solo_operators = load_solo_profiles(
        args.solo_profile_root, args.model, expected_profile_sha
    )

    discovery_payloads = []
    candidate_index = {}
    for path in args.discovery_root.glob("*/candidates.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload["model"] == args.model
            and payload["reference_variant"] == args.reference_variant
        ):
            discovery_payloads.append(payload)
            for row in payload["candidates"]:
                candidate_index[row["candidate_id"]] = row
    if len(discovery_payloads) != 1:
        raise RuntimeError(
            f"expected one discovery payload, found {len(discovery_payloads)}"
        )

    targets = []
    for case in manifest["cases"]:
        if case["model"] != args.model:
            continue
        occurrence = next(
            (
                item
                for item in case["source_occurrences"]
                if item["reference_variant"] == args.reference_variant
            ),
            None,
        )
        # Evaluate each sampled case on its first available reference path.
        first = case["source_occurrences"][0]
        if occurrence is None or occurrence != first:
            continue
        source = candidate_index.get(occurrence["candidate_id"])
        if source is None:
            raise RuntimeError(f"missing source candidate for {case['case_id']}")
        targets.append({"case": case, "source": source})
    if not targets:
        print(json.dumps({"model": args.model, "target_count": 0}))
        return 0

    targets_by_call = {}
    for target in targets:
        targets_by_call.setdefault(int(target["source"]["call"]), []).append(target)

    config = load_config()
    expected_python = Path(config["environment"]["python_executable"]).resolve()
    if not Path(sys.executable).resolve().samefile(expected_python):
        raise RuntimeError("wrong Python interpreter")
    require_idle_gpu()
    seed_everything(int(config["measurement"]["seed"]))
    import torch
    from Opara import GraphCapturer, OperatorLauncher
    from Opara.Scheduler import Scheduler as SchedulerClass

    def force_all_hp(queue, tau=0.5):
        del tau
        passthrough = []
        for node in list(queue):
            if getattr(node, "info", None):
                continue
            node.is_lowpriority = True
            queue.remove(node)
            passthrough.append(node)
        return passthrough

    OperatorLauncher.pop_lowPriorty_from_queue = force_all_hp
    model, inputs = load_model_and_inputs(args.model, config)
    with torch.inference_mode():
        reference = [tensor.detach().clone() for tensor in tensor_leaves(model(*inputs))]
    torch.cuda.synchronize()
    task = Task(args.model, args.reference_variant, None, 0)
    params = variant_parameters(task, config)
    params["max_ready"] = 6
    os.environ["OPARA_MAX_READY"] = "6"
    apply_variant_environment(params)

    original_schedule = SchedulerClass.schedule
    call_state = {"call": 0}
    results = []

    def single_occupancy(scheduler, operator, now):
        model_copy = copy.deepcopy(scheduler.resource_model)
        model_copy.update_time(now)
        if model_copy.time_domain:
            metrics = model_copy.evaluate_initial_combo([operator], now)
            return (
                float(metrics.get("initial_utilization", 0.0))
                if metrics.get("feasible")
                else -1.0
            )
        if operator.kernels:
            if not model_copy.can_apply_launch(operator, now):
                return -1.0
            model_copy.apply_launch(operator, now)
        return float(model_copy.total_utilization())

    def wrapped(scheduler, ready_ops, current_time):
        ready_ops = list(ready_ops)
        call_state["call"] += 1
        if len(ready_ops) > 6:
            ranked = [
                (single_occupancy(scheduler, op, current_time), index, op)
                for index, op in enumerate(ready_ops)
            ]
            ranked.sort(key=lambda item: (-item[0], item[1]))
            ready_ops = [item[2] for item in ranked[:6]]
        call = call_state["call"]
        for target in targets_by_call.get(call, []):
            source = target["source"]
            actual_ready = [str(op.name) for op in ready_ops if op.kernels]
            if source["ready_signature"] != __import__(
                "discover_simulator_candidates"
            ).stable_hash(actual_ready):
                raise RuntimeError(
                    f"ready signature changed for {target['case']['case_id']}"
                )
            by_name = {str(op.name): op for op in ready_ops}
            missing = [name for name in target["case"]["group"] if name not in by_name]
            if missing:
                raise RuntimeError(f"target operators not ready: {missing}")
            group = [by_name[name] for name in target["case"]["group"]]
            adjusted_group, missing_solo, duration_scale = apply_solo_durations(
                group, solo_operators
            )
            if missing_solo:
                gap_results = {
                    f"{gap:.4f}": {
                        "feasible": False,
                        "strict_parallel": False,
                        "failure_reason": "missing_solo_operator_profile",
                        "missing_solo_operators": missing_solo,
                    }
                    for gap in GAPS_MS
                }
            else:
                gap_results = {
                    f"{gap:.4f}": simulate_strict_overlap(
                        adjusted_group,
                        scheduler.resource_model,
                        launch_gap=gap,
                        kernel_gap=0.0,
                        block_limit_per_sm=BLOCK_LIMIT_PER_SM,
                    )
                    for gap in GAPS_MS
                }
            results.append(
                {
                    "case_id": target["case"]["case_id"],
                    "model": args.model,
                    "reference_variant": args.reference_variant,
                    "call": call,
                    "group": target["case"]["group"],
                    "solo_duration_scale": duration_scale,
                    "gap_results": gap_results,
                }
            )
        return original_schedule(scheduler, ready_ops, current_time)

    SchedulerClass.schedule = wrapped
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
    if len(results) != len(targets):
        raise RuntimeError(f"evaluated {len(results)} of {len(targets)} targets")
    with torch.inference_mode():
        actual = runner(*inputs)
    torch.cuda.synchronize()
    correctness = compare_outputs(
        reference,
        actual,
        float(config["correctness"]["float_rtol"]),
        float(config["correctness"]["float_atol"]),
    )
    if not correctness.get("ok", False):
        raise RuntimeError(f"output correctness failed: {correctness}")
    payload = {
        "schema_version": 1,
        "simulator": "td_v2_nonreserving_launch_ordered_v1",
        "model": args.model,
        "reference_variant": args.reference_variant,
        "git_head": head,
        "gaps_ms": list(GAPS_MS),
        "block_limit_per_sm": BLOCK_LIMIT_PER_SM,
        "duration_source": "solo_operator_cuda_graph_span_v1",
        "solo_profile_path": str(solo_path),
        "solo_profile_target_count": solo_payload.get("target_count"),
        "solo_profile_auditable_count": solo_payload.get("auditable_count"),
        "target_count": len(targets),
        "correctness": correctness,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"model": args.model, "targets": len(targets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
