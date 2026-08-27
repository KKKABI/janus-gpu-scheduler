#!/usr/bin/env python3
"""Capture and replay sampled FX operators one at a time on one CUDA stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


EXPECTED_HEAD = "32bf4974994005855896a360c34ba455303f5ff3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_tree(value):
    import torch

    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(item) for item in value)
    if isinstance(value, list):
        return [clone_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return value


def copy_tree_(destination, source):
    import torch

    if torch.is_tensor(destination):
        destination.copy_(source)
    elif isinstance(destination, dict):
        for key in destination:
            copy_tree_(destination[key], source[key])
    elif isinstance(destination, (list, tuple)):
        for left, right in zip(destination, source):
            copy_tree_(left, right)


def tensor_leaves(value):
    import torch

    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        return [leaf for key in value for leaf in tensor_leaves(value[key])]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in tensor_leaves(item)]
    return []


def compare_outputs(reference, candidate):
    import torch

    expected = tensor_leaves(reference)
    actual = tensor_leaves(candidate)
    if len(expected) != len(actual):
        raise AssertionError(f"tensor leaves differ: {len(expected)} != {len(actual)}")
    max_abs = 0.0
    max_rel = 0.0
    for left, right in zip(expected, actual):
        torch.testing.assert_close(right, left, rtol=1e-4, atol=1e-5)
        if left.numel():
            delta = (right - left).abs()
            max_abs = max(max_abs, float(delta.max()))
            max_rel = max(
                max_rel,
                float((delta / left.abs().clamp_min(1e-12)).max()),
            )
    return {"tensor_leaves": len(actual), "max_abs": max_abs, "max_rel": max_rel}


class RecordingInterpreter:
    def __init__(self, module, target):
        import torch

        class _Recorder(torch.fx.Interpreter):
            def __init__(inner, graph_module):
                super().__init__(graph_module)
                inner.record = None

            def run_node(inner, node):
                args, kwargs = inner.fetch_args_kwargs_from_env(node)
                result = super(_Recorder, inner).run_node(node)
                if node.name == target:
                    inner.record = {
                        "args": clone_tree(args),
                        "kwargs": clone_tree(kwargs),
                        "output": clone_tree(result),
                    }
                return result

        self.interpreter = _Recorder(module)

    def run(self, *inputs):
        self.interpreter.run(*inputs)
        return self.interpreter.record


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
            result = dynamo.explain(model)(*inputs)
        graphs = getattr(result, "graphs", None) or getattr(result, "graph", None)
        module = graphs[0]
    else:
        raise ValueError(f"unsupported capture backend: {backend}")
    return module.cuda()


def invoke(interpreter, node, args, kwargs):
    return getattr(interpreter, node.op)(node.target, args, kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-mode",
        choices=("sampled", "all-kernel"),
        default="sampled",
    )
    parser.add_argument("--discovery-root", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    repo = Path(__file__).resolve().parents[2]
    experiments = repo / "experiments"
    sys.path[:0] = [str(experiments), str(repo)]
    from harness_common import expected_profile_path, load_config
    from run_one import load_model_and_inputs, seed_everything

    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError(f"unexpected head: {head}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["git_head"] != head:
        raise RuntimeError("manifest git head differs")
    cases = [row for row in manifest["cases"] if row["model"] == args.model]
    targets = sorted({name for case in cases for name in case["group"]})
    target_source = "positive sample manifest"
    target_map_path = None
    if args.target_mode == "all-kernel":
        if args.discovery_root is None:
            raise ValueError("--discovery-root is required for all-kernel mode")
        maps = []
        for candidate_path in args.discovery_root.glob("*/candidates.json"):
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            if (
                payload.get("model") == args.model
                and payload.get("reference_variant") == "Baseline"
            ):
                map_path = candidate_path.parent / "fx_stream_map.json"
                if map_path.is_file():
                    maps.append(map_path)
        if len(maps) != 1:
            raise RuntimeError(
                f"expected one Baseline FX map for {args.model}, found {maps}"
            )
        target_map_path = maps[0]
        target_map = json.loads(target_map_path.read_text(encoding="utf-8"))
        targets = sorted(
            row["name"] for row in target_map.get("nodes", []) if row.get("kernels")
        )
        target_source = "Baseline discovery fx_stream_map kernel-bearing nodes"
    if not targets:
        raise ValueError(f"manifest has no target operators for {args.model}")

    import torch
    from torch.fx import Interpreter

    config = load_config()
    seed_everything(int(config["measurement"]["seed"]))
    model, inputs = load_model_and_inputs(args.model, config)
    profile_path = expected_profile_path(model, inputs)
    profile_sha = sha256_file(profile_path)
    expected_sha = manifest["source_profile_sha256_by_model"][args.model]
    if profile_sha != expected_sha:
        raise RuntimeError(f"profile SHA differs: {profile_sha} != {expected_sha}")
    backend = config["models"][args.model].get("capture_backend", "dynamo_explain")
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    module = trace_model(model, inputs, backend)
    nodes = {node.name: node for node in module.graph.nodes}
    missing = sorted(set(targets) - set(nodes))
    if missing:
        raise RuntimeError(f"target FX nodes are missing: {missing}")
    invoker = Interpreter(module)
    stream = torch.cuda.Stream()
    summaries = []

    for index, name in enumerate(targets, start=1):
        recorder = RecordingInterpreter(module, name)
        try:
            with torch.no_grad():
                record = recorder.run(*inputs)
            torch.cuda.synchronize()
            if record is None:
                raise RuntimeError("target node was not recorded")
            pristine_args = clone_tree(record["args"])
            pristine_kwargs = clone_tree(record["kwargs"])
            with torch.no_grad():
                reference = invoke(
                    invoker,
                    nodes[name],
                    clone_tree(pristine_args),
                    clone_tree(pristine_kwargs),
                )
            torch.cuda.synchronize()
            extraction = compare_outputs(record["output"], reference)

            with torch.no_grad(), torch.cuda.stream(stream):
                invoke(invoker, nodes[name], record["args"], record["kwargs"])
            torch.cuda.synchronize()
            copy_tree_(record["args"], pristine_args)
            copy_tree_(record["kwargs"], pristine_kwargs)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            capture_marker = f"SOLO_FX_CAPTURE::{args.model}::{name}"
            with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
                torch.cuda.nvtx.range_push(capture_marker)
                try:
                    captured = invoke(
                        invoker, nodes[name], record["args"], record["kwargs"]
                    )
                finally:
                    torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            copy_tree_(record["args"], pristine_args)
            copy_tree_(record["kwargs"], pristine_kwargs)
            torch.cuda.synchronize()

            replay_marker = f"SOLO_REPLAY::{args.model}::{name}"
            torch.cuda.nvtx.range_push(replay_marker)
            try:
                graph.replay()
                torch.cuda.synchronize()
            finally:
                torch.cuda.nvtx.range_pop()
            correctness = compare_outputs(reference, captured)
            summaries.append(
                {
                    "name": name,
                    "op": nodes[name].op,
                    "target": str(nodes[name].target),
                    "capture_status": "captured",
                    "capture_marker": capture_marker,
                    "replay_marker": replay_marker,
                    "extraction_correctness": extraction,
                    "correctness": correctness,
                }
            )
            del graph, captured, record, recorder, reference
            status = "captured"
        except (AssertionError, RuntimeError) as error:
            summaries.append(
                {
                    "name": name,
                    "op": nodes[name].op,
                    "target": str(nodes[name].target),
                    "capture_status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            status = "failed"
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(targets),
                    "operator": name,
                    "status": status,
                }
            ),
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "protocol": "sampled_fx_operator_solo_cuda_graph_replay_v1",
        "model": args.model,
        "git_head": head,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha,
        "capture_backend": backend,
        "target_mode": args.target_mode,
        "target_source": target_source,
        "target_map_path": str(target_map_path) if target_map_path else None,
        "target_map_sha256": (
            sha256_file(target_map_path) if target_map_path else None
        ),
        "target_count": len(targets),
        "captured_count": sum(
            row["capture_status"] == "captured" for row in summaries
        ),
        "operators": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
