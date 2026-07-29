#!/usr/bin/env python3
"""Run and record one Janus configuration in a fresh Python process."""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from harness_common import CONFIG_PATH, REPO_ROOT, Task, expected_profile_path, gpu_snapshot, load_config, require_idle_gpu, runtime_metadata, sha256_file, stats_from_samples, write_json_atomic

sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--alpha", default="none")
    parser.add_argument("--repeat-index", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-ready", default=15, type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_model_and_inputs(name: str, config: dict[str, Any]):
    import torch
    if name == "GoogLeNet":
        import torchvision
        model = torchvision.models.googlenet()
        inputs = (torch.randint(0, 256, (1, 3, 224, 224), dtype=torch.float32, device="cuda:0"),)
    elif name == "Inception-v3":
        import torchvision
        model = torchvision.models.inception_v3(pretrained=True)
        inputs = (torch.randint(0, 256, (1, 3, 299, 299), dtype=torch.float32, device="cuda:0"),)
    elif name == "BERT":
        from transformers import BertModel
        model = BertModel.from_pretrained("/public_0/ZYF/model/bert-base")
        inputs = (torch.randint(0, 30000, (1, 16), dtype=torch.long, device="cuda:0"), torch.ones((1, 16), dtype=torch.long, device="cuda:0"))
    elif name == "NASNet":
        import pretrainedmodels
        model = pretrainedmodels.__dict__["nasnetalarge"](num_classes=1000, pretrained="imagenet")
        inputs = (torch.randint(0, 256, (1, 3, 331, 331), dtype=torch.float32, device="cuda:0"),)
    elif name == "YOLOv8x":
        from ultralytics import YOLO
        model = YOLO("/public_0/ZYF/model/YOLOv8/yolov8x.pt").model
        inputs = (torch.randn((1, 3, 320, 320), device="cuda:0"),)
    elif name == "ConvNeXt":
        import torchvision
        model = torchvision.models.convnext_base(pretrained=False)
        inputs = (torch.randn((1, 3, 224, 224), device="cuda:0"),)
    elif name == "DeepFM":
        sys.path.insert(0, config["source"]["model_reference_snapshot_root"])
        from NCF import DeepFM
        categories = [100 * (index + 1) for index in range(32)]
        model = DeepFM(categories, 16, emb_size=8, hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2])
        inputs = (torch.randint(0, 100, (1, 32), device="cuda:0"), torch.rand((1, 16), device="cuda:0"))
    else:
        raise ValueError(f"unsupported model: {name}")
    return model.to("cuda:0").eval(), inputs


def tensor_leaves(value: Any) -> list[Any]:
    import torch
    if isinstance(value, torch.Tensor): return [value]
    if hasattr(value, "to_tuple") and callable(value.to_tuple): return tensor_leaves(value.to_tuple())
    if isinstance(value, Mapping):
        return [leaf for key in value for leaf in tensor_leaves(value[key])]
    if isinstance(value, (tuple, list)):
        return [leaf for item in value for leaf in tensor_leaves(item)]
    return []


def compare_outputs(reference: list[Any], candidate: Any, rtol: float, atol: float) -> dict[str, Any]:
    import torch
    actual = tensor_leaves(candidate)
    if len(reference) != len(actual): raise AssertionError(f"tensor output count differs: eager={len(reference)}, runner={len(actual)}")
    max_abs = 0.0
    for index, (expected, observed) in enumerate(zip(reference, actual)):
        if expected.shape != observed.shape: raise AssertionError(f"output[{index}] shape differs: {expected.shape} != {observed.shape}")
        if expected.is_floating_point() or expected.is_complex():
            if not torch.isfinite(expected).all() or not torch.isfinite(observed).all(): raise AssertionError(f"output[{index}] contains NaN/Inf")
            torch.testing.assert_close(observed, expected, rtol=rtol, atol=atol, equal_nan=False)
            if expected.numel(): max_abs = max(max_abs, float((expected - observed).abs().max().item()))
        else:
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    return {"ok": True, "tensor_count": len(reference), "max_absolute_difference": max_abs}


def variant_parameters(task: Task, config: dict[str, Any]) -> dict[str, Any]:
    spec = {item["label"]: item for item in config["variants"]}[task.variant]
    if spec.get("alpha") == "alpha_grid":
        if task.alpha not in config["alpha_grid"]: raise ValueError(f"invalid alpha for {task.variant}: {task.alpha}")
    elif task.alpha is not None: raise ValueError(f"{task.variant} must use alpha=null")
    selection = {
        "legacy_balance": "legacy_balance",
        "cosine": "cosine",
        "min_resource": "min_resource",
        "drt_no_alpha": "static_interference",
        "drt_alpha": "static_interference_alpha",
    }[spec["score"]]
    if "internal_alpha" in spec:
        internal_alpha = float(spec["internal_alpha"])
    elif spec["score"] == "drt_no_alpha":
        internal_alpha = 0.0
    else:
        internal_alpha = float(task.alpha)
    return {
        "selection_mode": selection,
        "time_domain": spec["simulator"] == "td",
        "internal_alpha": internal_alpha,
        "final_selector": spec.get("final_selector", "timeline"),
        "timeline_speedup_guard": (
            float(spec.get("timeline_speedup_guard", 0.9))
            if spec["simulator"] == "td" else None
        ),
        "simulator_semantics": (
            "shared_candidate_timeline_v1"
            if spec["simulator"] == "td"
            else "all_blocks_co_resident"
        ),
        "candidate_score_kind": (
            "initial_occupancy_then_predicted_speedup"
            if spec["simulator"] == "td"
            else "initial_occupancy"
        ),
        "td_timeline_shortlist": (
            int(os.environ.get("OPARA_TD_TIMELINE_SHORTLIST", "8"))
            if spec["simulator"] == "td" else None
        ),
        "td_max_events": (
            int(os.environ.get("OPARA_TD_MAX_EVENTS", "100000"))
            if spec["simulator"] == "td" else None
        ),
    }


def main() -> int:
    args = parse_args(); config = load_config()
    if args.max_ready < 5: raise ValueError(f"--max-ready must be >= 5, got {args.max_ready}")
    os.environ["OPARA_MAX_READY"] = str(args.max_ready)
    alpha = None if args.alpha.lower() == "none" else float(args.alpha)
    task = Task(args.model, args.variant, alpha, args.repeat_index)
    output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"; started = time.time()
    os.chdir(output_dir)
    write_json_atomic(status_path, {"status": "running", "task": task.to_dict(), "started_unix": started})
    try:
        expected_python = Path(config["environment"]["python_executable"]).resolve()
        if not Path(sys.executable).resolve().samefile(expected_python): raise RuntimeError(f"wrong interpreter: {sys.executable}; expected {expected_python}")
        require_idle_gpu(); seed_everything(int(config["measurement"]["seed"]))
        import torch
        from Opara import GraphCapturer
        before_load = gpu_snapshot(); model, inputs = load_model_and_inputs(task.model, config)
        profile = expected_profile_path(model, inputs)
        configured_profile = config["models"][task.model]["profile_file"]
        if profile.name != configured_profile:
            raise RuntimeError(f"profile identity mismatch: runtime={profile.name}; configured={configured_profile}")
        if not profile.is_file():
            raise RuntimeError(f"frozen profile is missing; refusing automatic generation: {profile}")
        with torch.inference_mode(): reference = [tensor.detach().clone() for tensor in tensor_leaves(model(*inputs))]
        params = variant_parameters(task, config)
        params["max_ready"] = args.max_ready
        os.environ["OPARA_TD_FINAL_SELECTOR"] = params["final_selector"]
        if params["timeline_speedup_guard"] is not None:
            os.environ["OPARA_TD_SPEEDUP_GUARD"] = str(
                params["timeline_speedup_guard"]
            )
        capture_backend = config["models"][task.model].get("capture_backend", "dynamo_explain")
        capture_started = time.perf_counter()
        runner = GraphCapturer.capturer(inputs, model, copy_outputs=False, alpha=params["internal_alpha"], selection_mode=params["selection_mode"], time_domain=params["time_domain"], capture_backend=capture_backend)
        capture_build_seconds = time.perf_counter() - capture_started
        from Opara.Scheduler import get_candidate_stats
        scheduler_calls = get_candidate_stats(clear=True)
        total_enumerated = sum(item["enumerated_count"] for item in scheduler_calls)
        total_feasible = sum(item["feasible_count"] for item in scheduler_calls)
        selected_timelines = [
            item["selected_timeline"] for item in scheduler_calls
            if "selected_timeline" in item
        ]
        selected_interference = [
            item["interference"] for item in selected_timelines
            if "interference" in item
        ]
        scheduler_summary = {
            "capture_build_seconds": capture_build_seconds,
            "max_ready": args.max_ready,
            "call_count": len(scheduler_calls),
            "enumerated_count": total_enumerated,
            "feasible_count": total_feasible,
            "pass_rate": total_feasible / total_enumerated if total_enumerated else 0.0,
            "truncated_call_count": sum(
                item["ready_count"] > item["ready_used_count"] for item in scheduler_calls
            ),
            "observed_max_ready_count": max(
                (item["ready_count"] for item in scheduler_calls), default=0
            ),
            "observed_max_ready_used_count": max(
                (item["ready_used_count"] for item in scheduler_calls), default=0
            ),
            "observed_max_resource_ready_count": max(
                (item.get("resource_ready_count", item["ready_count"])
                 for item in scheduler_calls),
                default=0,
            ),
            "observed_max_resource_ready_used_count": max(
                (item.get("resource_ready_used_count", item["ready_used_count"])
                 for item in scheduler_calls),
                default=0,
            ),
            "passthrough_call_count": sum(
                item.get("passthrough_count", 0) > 0
                for item in scheduler_calls
            ),
            "passthrough_node_count": sum(
                item.get("passthrough_count", 0)
                for item in scheduler_calls
            ),
            "single_scoring_candidate_calls": sum(
                item["scoring_candidate_count"] == 1 for item in scheduler_calls
            ),
            "timeline_call_count": len(selected_timelines),
            "selected_timeline_mean_speedup": (
                sum(item["predicted_speedup"] for item in selected_timelines)
                / len(selected_timelines)
                if selected_timelines else None
            ),
            "selected_timeline_mean_average_utilization": (
                sum(item["average_utilization"] for item in selected_timelines)
                / len(selected_timelines)
                if selected_timelines else None
            ),
            "selected_timeline_mean_overlap_fraction": (
                sum(item["overlap_fraction"] for item in selected_timelines)
                / len(selected_timelines)
                if selected_timelines else None
            ),
            "selected_timeline_max_event_count": max(
                (item["event_count"] for item in selected_timelines),
                default=0,
            ),
            "selected_interference_mean_risk": (
                sum(item["risk"] for item in selected_interference)
                / len(selected_interference)
                if selected_interference else None
            ),
            "selected_interference_max_risk": max(
                (item["risk"] for item in selected_interference),
                default=None,
            ),
            "selected_interference_mean_pair_conflict": (
                sum(item["pair_conflict"] for item in selected_interference)
                / len(selected_interference)
                if selected_interference else None
            ),
            "selected_interference_mean_ncu_coverage": (
                sum(item["ncu_coverage"] for item in selected_interference)
                / len(selected_interference)
                if selected_interference else None
            ),
        }
        with torch.no_grad(): candidate = runner(*inputs)
        correctness = compare_outputs(reference, candidate, float(config["correctness"]["float_rtol"]), float(config["correctness"]["float_atol"]))
        spec = config["models"][task.model]
        cache = torch.empty(int(config["measurement"]["cache_buffer_bytes"]), dtype=torch.int8, device="cuda:0")
        for _ in range(int(spec["warmup_iterations"])): cache.zero_(); runner(*inputs)
        torch.cuda.synchronize(); before_timing = gpu_snapshot(); samples = []
        for _ in range(int(spec["timed_iterations"])):
            cache.zero_(); torch.cuda._sleep(int(config["measurement"]["pre_sample_cuda_sleep_cycles"]))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); runner(*inputs); end.record(); end.synchronize(); value = float(start.elapsed_time(end))
            if not math.isfinite(value): raise RuntimeError(f"non-finite timing sample: {value}")
            samples.append(value)
        result = {"schema_version": 1, "status": "completed", "task": task.to_dict(), "effective_parameters": params, "scheduler": {"summary": scheduler_summary, "calls": scheduler_calls}, "model_spec": spec, "profile": {"path": str(profile), "sha256": sha256_file(profile)}, "correctness": correctness, "timing": {"samples_ms": samples, "statistics": stats_from_samples(samples)}, "telemetry": {"before_model_load": before_load, "before_timing": before_timing, "after_timing": gpu_snapshot()}, "runtime": {**runtime_metadata(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}, "started_unix": started, "finished_unix": time.time()}
        write_json_atomic(output_dir / "result.json", result); write_json_atomic(status_path, {"status": "completed", "task": task.to_dict()}); return 0
    except Exception as error:
        failure = {"status": "failed", "task": task.to_dict(), "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(), "started_unix": started, "finished_unix": time.time()}
        try: failure["telemetry"] = gpu_snapshot()
        except Exception as telemetry_error: failure["telemetry_error"] = str(telemetry_error)
        write_json_atomic(output_dir / "failure.json", failure); write_json_atomic(status_path, {"status": "failed", "task": task.to_dict(), "error": str(error)}); print(traceback.format_exc(), file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
