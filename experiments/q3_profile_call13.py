#!/usr/bin/env python3
"""Targeted full-CUDA-Graph replay profile for the GoogLeNet call-13 case."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from harness_common import Task, expected_profile_path, load_config, require_idle_gpu, sha256_file
from run_one import compare_outputs, load_model_and_inputs, seed_everything, tensor_leaves, variant_parameters


FROZEN_BASE_HEAD = "3b2880ad5ca4b78d0385c9dd014ac2f4ab420648"
GOOGLENET_PROFILE_SHA256 = "0d29cfcd359efbf8d0630d9ef8171b0f6cd383fbac8ce27d7c6a1b18b3a1ae14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["Baseline", "TD+Janus"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=5)
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def apply_variant_environment(params: dict) -> None:
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


def assert_call13(variant: str, call13: dict) -> dict:
    expected = {
        "Baseline": {
            "ready": {"x_11", "x_13", "x_17"},
            "selected": {"x_13", "x_17"},
            "enumerated": 7,
            "feasible": 3,
        },
        "TD+Janus": {
            "ready": {"x_11", "x_13", "x_17"},
            "selected": {"x_11", "x_13", "x_17"},
            "enumerated": 7,
            "feasible": 7,
        },
    }[variant]
    actual = {
        "ready": set(call13.get("ready_ops", [])),
        "selected": set(call13.get("selected", [])),
        "enumerated": int(call13.get("enumerated_count", -1)),
        "feasible": int(call13.get("feasible_count", -1)),
    }
    if actual != expected:
        raise RuntimeError(
            "call-13 identity mismatch: "
            + json.dumps(
                {
                    "variant": variant,
                    "expected": json_safe(expected),
                    "actual": json_safe(actual),
                    "call13": json_safe(call13),
                },
                ensure_ascii=False,
            )
        )
    return {
        "passed": True,
        "ready": sorted(actual["ready"]),
        "selected": sorted(actual["selected"]),
        "enumerated": actual["enumerated"],
        "feasible": actual["feasible"],
    }


def main() -> int:
    args = parse_args()
    if args.replays < 3:
        raise ValueError("at least three replays are required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config()
    expected_python = Path(config["environment"]["python_executable"]).resolve()
    if not Path(sys.executable).resolve().samefile(expected_python):
        raise RuntimeError(f"wrong interpreter: {sys.executable}; expected {expected_python}")
    git_head = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    base_check = subprocess.run(
        [
            "git",
            "-C",
            str(REPO_ROOT),
            "merge-base",
            "--is-ancestor",
            FROZEN_BASE_HEAD,
            git_head,
        ],
        check=False,
    )
    if base_check.returncode != 0:
        raise RuntimeError(
            f"frozen base {FROZEN_BASE_HEAD} is not an ancestor of {git_head}"
        )
    tracked_dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet"], check=False
    ).returncode != 0
    index_dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--cached", "--quiet"],
        check=False,
    ).returncode != 0
    if tracked_dirty or index_dirty:
        raise RuntimeError("tracked repository files must be clean before profiling")
    require_idle_gpu()
    seed_everything(int(config["measurement"]["seed"]))

    from Opara import GraphCapturer
    from Opara.Scheduler import get_candidate_stats

    model, inputs = load_model_and_inputs("GoogLeNet", config)
    profile_path = expected_profile_path(model, inputs)
    if not profile_path.is_file():
        raise RuntimeError(f"frozen profile is missing: {profile_path}")
    profile_sha256 = sha256_file(profile_path)
    if profile_sha256 != GOOGLENET_PROFILE_SHA256:
        raise RuntimeError(
            f"profile checksum mismatch: {profile_sha256}; expected {GOOGLENET_PROFILE_SHA256}"
        )
    with torch.inference_mode():
        reference = [tensor.detach().clone() for tensor in tensor_leaves(model(*inputs))]
    torch.cuda.synchronize()

    task = Task("GoogLeNet", args.variant, None, 0)
    params = variant_parameters(task, config)
    params["max_ready"] = 15
    os.environ["OPARA_MAX_READY"] = "15"
    apply_variant_environment(params)
    os.environ["OPARA_Q3_PROFILE_MAP"] = str(output_dir / "fx_stream_map.json")
    os.chdir(output_dir)

    get_candidate_stats(clear=True)
    torch.cuda.nvtx.range_push("Q3_SESSION")
    try:
        runner = GraphCapturer.capturer(
            inputs,
            model,
            copy_outputs=False,
            alpha=params["internal_alpha"],
            selection_mode=params["selection_mode"],
            time_domain=params["time_domain"],
            capture_backend=config["models"]["GoogLeNet"].get("capture_backend", "dynamo_explain"),
        )
        scheduler_calls = get_candidate_stats(clear=True)
        if len(scheduler_calls) < 13:
            raise RuntimeError(f"expected at least 13 scheduler calls, got {len(scheduler_calls)}")
        call13 = scheduler_calls[12]
        call13_assertion = assert_call13(args.variant, call13)

        torch.cuda.nvtx.range_push(f"Q3_VALIDATION::{args.variant}")
        try:
            replay_outputs = runner(*inputs)
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()

        for replay_index in range(args.replays):
            torch.cuda.nvtx.range_push(f"Q3_REPLAY::{args.variant}::{replay_index}")
            try:
                runner._opara_graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
    finally:
        torch.cuda.nvtx.range_pop()

    correctness = compare_outputs(
        reference,
        replay_outputs,
        float(config["correctness"]["float_rtol"]),
        float(config["correctness"]["float_atol"]),
    )
    try:
        runner._opara_graph.debug_dump(str(output_dir / "cuda_graph_debug"))
    except Exception as error:
        (output_dir / "cuda_graph_debug_error.txt").write_text(str(error), encoding="utf-8")

    summary = {
        "schema_version": 1,
        "model": "GoogLeNet",
        "variant": args.variant,
        "replays": args.replays,
        "git_head": git_head,
        "frozen_base_head": FROZEN_BASE_HEAD,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha256,
        "effective_parameters": params,
        "scheduler_call_count": len(scheduler_calls),
        "call13_assertion": call13_assertion,
        "call13": json_safe(call13),
        "correctness": correctness,
    }
    (output_dir / "q3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
