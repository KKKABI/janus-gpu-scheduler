#!/usr/bin/env python3
"""Run one frozen FX inference with an NVTX range around every FX OP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS), str(REPO)]

from common import MODELS, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--identity-json", type=Path, required=True)
    return parser.parse_args()


def trace_model(model, inputs, backend):
    import torch

    if backend == "make_fx":
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.no_grad():
            module = make_fx(model)(*inputs)
    elif backend == "dynamo_explain":
        import torch._dynamo as dynamo

        dynamo.reset()
        with torch.no_grad():
            explanation = dynamo.explain(model)(*inputs)
        graphs = (
            explanation.graphs
            if hasattr(explanation, "graphs")
            else explanation[2]
        )
        if not graphs:
            raise RuntimeError("torch._dynamo.explain returned no graph")
        module = graphs[0]
    else:
        raise ValueError(f"unsupported capture backend: {backend}")
    return module.eval().cuda()


def main() -> int:
    args = parse_args()
    identity_path = args.identity_json.resolve()
    if identity_path.exists():
        raise FileExistsError(identity_path)

    import torch
    from harness_common import expected_profile_path, load_config
    from run_one import (
        compare_outputs,
        load_model_and_inputs,
        seed_everything,
        tensor_leaves,
    )

    config = load_config()
    seed_everything(int(config["measurement"]["seed"]))
    model, inputs = load_model_and_inputs(args.model, config)
    profile_path = expected_profile_path(model, inputs)
    configured = config["models"][args.model]["profile_file"]
    if profile_path.name != configured:
        raise RuntimeError(
            f"profile identity mismatch: runtime={profile_path.name}; "
            f"configured={configured}"
        )
    if not profile_path.is_file():
        raise FileNotFoundError(profile_path)

    with torch.inference_mode():
        eager = [leaf.detach().clone() for leaf in tensor_leaves(model(*inputs))]
    backend = config["models"][args.model].get(
        "capture_backend", "dynamo_explain"
    )
    module = trace_model(model, inputs, backend)
    with torch.inference_mode():
        candidate = module(*inputs)
    correctness = compare_outputs(
        eager,
        candidate,
        float(config["correctness"]["float_rtol"]),
        float(config["correctness"]["float_atol"]),
    )

    identity = {
        "schema_version": 1,
        "requested_model": args.model,
        "model_class": model.__class__.__name__,
        "input_shapes": [list(value.shape) for value in inputs],
        "input_dtypes": [str(value.dtype) for value in inputs],
        "device_name": torch.cuda.get_device_name(inputs[0].device),
        "device_capability": list(
            torch.cuda.get_device_capability(inputs[0].device)
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "capture_backend": backend,
        "fx_code_sha256": hashlib.sha256(
            module.code.encode("utf-8")
        ).hexdigest(),
        "fx_node_names": [node.name for node in module.graph.nodes],
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
        "correctness": correctness,
        "git_head": subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    identity_path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    class NvtxInterpreter(torch.fx.Interpreter):
        def run_node(self, node):
            torch.cuda.nvtx.range_push(f"JANUS_OP:{node.name}")
            try:
                return super().run_node(node)
            finally:
                torch.cuda.nvtx.range_pop()

    with torch.inference_mode():
        for _ in range(3):
            module(*inputs)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push("JANUS_NCU_PROFILE")
        try:
            NvtxInterpreter(module).run(*inputs)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
