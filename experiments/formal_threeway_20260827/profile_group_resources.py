#!/usr/bin/env python3
"""Measure selected FX operators alone and concurrently without changing Janus.

The target operators must be an independent group previously observed in one
Janus scheduler ready set.  The script evaluates only their transitive input
closure, then launches the target FX nodes either one at a time or on distinct
CUDA streams.  It deliberately does not import or modify the Janus scheduler.

Modes:

* ``timing``: clean CUDA-event solo/group timing and output equality checks.
* ``trace``: a requested number of marked replays for Nsight Systems auditing.
* ``ncu-solo``: one NVTX range per target OP for per-kernel NCU collection.
* ``ncu-group``: one concurrent NVTX range for NCU Application Range Replay.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=MODELS,
        required=True,
    )
    parser.add_argument("--call", type=int, required=True)
    parser.add_argument("--group", nargs="+", required=True)
    parser.add_argument(
        "--mode",
        choices=("timing", "trace", "ncu-solo", "ncu-group"),
        required=True,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--expected-fx-code-sha256", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--skip-idle-check", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_leaves(value: Any) -> list[Any]:
    import torch

    if torch.is_tensor(value):
        return [value]
    if hasattr(value, "to_tuple") and callable(value.to_tuple):
        return tensor_leaves(value.to_tuple())
    if isinstance(value, Mapping):
        return [leaf for key in value for leaf in tensor_leaves(value[key])]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in tensor_leaves(item)]
    return []


def clone_tensor_leaves(value: Any) -> list[Any]:
    return [leaf.detach().clone() for leaf in tensor_leaves(value)]


def compare_tensor_leaves(reference: list[Any], candidate: Any) -> dict[str, Any]:
    import torch

    actual = tensor_leaves(candidate)
    if len(reference) != len(actual):
        raise AssertionError(
            f"tensor leaf count differs: reference={len(reference)}, actual={len(actual)}"
        )
    max_abs = 0.0
    max_rel = 0.0
    for expected, observed in zip(reference, actual):
        torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-5)
        if expected.numel() and (expected.is_floating_point() or expected.is_complex()):
            delta = (observed - expected).abs()
            max_abs = max(max_abs, float(delta.max().item()))
            max_rel = max(
                max_rel,
                float((delta / expected.abs().clamp_min(1e-12)).max().item()),
            )
    return {
        "tensor_leaves": len(actual),
        "max_abs": max_abs,
        "max_rel": max_rel,
    }


def stats(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("no samples")
    median = statistics.median(samples)
    return {
        "count": len(samples),
        "median_ms": median,
        "mean_ms": statistics.fmean(samples),
        "sample_std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "mad_ms": statistics.median(abs(value - median) for value in samples),
    }


def trace_fx(model: Any, inputs: tuple[Any, ...], backend: str) -> Any:
    import torch
    import torch._dynamo as dynamo

    if backend == "make_fx":
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.no_grad():
            module = make_fx(model)(*inputs)
    elif backend == "dynamo_explain":
        dynamo.reset()
        with torch.no_grad():
            result = dynamo.explain(model)(*inputs)
        if isinstance(result, tuple):
            graphs = result[2]
        else:
            graphs = getattr(result, "graphs", None) or getattr(result, "graph", None)
        if not graphs:
            raise RuntimeError("torch._dynamo.explain returned no FX graph")
        module = graphs[0] if isinstance(graphs, (list, tuple)) else graphs
    else:
        raise ValueError(f"unsupported capture backend: {backend}")
    return module.eval().cuda()


def prepare_target_interpreter(module: Any, inputs: tuple[Any, ...], names: list[str]):
    import torch

    nodes_by_name = {node.name: node for node in module.graph.nodes}
    missing = [name for name in names if name not in nodes_by_name]
    if missing:
        raise RuntimeError(f"target FX nodes are missing: {missing}")
    targets = [nodes_by_name[name] for name in names]
    target_set = set(targets)

    required: set[Any] = set()

    def add_ancestors(node: Any) -> None:
        for parent in node.all_input_nodes:
            if parent in target_set:
                raise RuntimeError(
                    f"target group is not independent: {parent.name} -> {node.name}"
                )
            if parent not in required:
                required.add(parent)
                add_ancestors(parent)

    for target in targets:
        add_ancestors(target)

    interpreter = torch.fx.Interpreter(module)
    interpreter.env = {}
    interpreter.args_iter = iter(inputs)
    with torch.inference_mode():
        for node in module.graph.nodes:
            if node in required:
                interpreter.env[node] = interpreter.run_node(node)
    torch.cuda.synchronize()

    missing_inputs = [
        f"{target.name}:{parent.name}"
        for target in targets
        for parent in target.all_input_nodes
        if parent not in interpreter.env
    ]
    if missing_inputs:
        raise RuntimeError(f"failed to materialize target inputs: {missing_inputs}")
    return interpreter, targets, sorted(node.name for node in required)


def execute_node(interpreter: Any, node: Any) -> Any:
    import torch

    with torch.inference_mode():
        output = interpreter.run_node(node)
        interpreter.env[node] = output
        return output


def solo_once(interpreter: Any, node: Any, stream: Any, marker: str | None = None):
    import torch

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if marker:
        torch.cuda.nvtx.range_push(marker)
    try:
        with torch.cuda.stream(stream):
            start.record(stream)
            output = execute_node(interpreter, node)
            end.record(stream)
        end.synchronize()
    finally:
        if marker:
            torch.cuda.nvtx.range_pop()
    return output, float(start.elapsed_time(end))


def group_once(
    interpreter: Any,
    targets: list[Any],
    streams: list[Any],
    outer_marker: str | None = None,
    op_marker_prefix: str | None = None,
):
    import torch

    coordinator = torch.cuda.current_stream()
    gate = torch.cuda.Event(enable_timing=True)
    group_end = torch.cuda.Event(enable_timing=True)
    op_starts = [torch.cuda.Event(enable_timing=True) for _ in targets]
    op_ends = [torch.cuda.Event(enable_timing=True) for _ in targets]
    outputs: dict[str, Any] = {}

    if outer_marker:
        torch.cuda.nvtx.range_push(outer_marker)
    try:
        gate.record(coordinator)
        for index, (node, stream) in enumerate(zip(targets, streams)):
            marker = f"{op_marker_prefix}:{node.name}" if op_marker_prefix else None
            if marker:
                torch.cuda.nvtx.range_push(marker)
            try:
                with torch.cuda.stream(stream):
                    stream.wait_event(gate)
                    op_starts[index].record(stream)
                    outputs[node.name] = execute_node(interpreter, node)
                    op_ends[index].record(stream)
            finally:
                if marker:
                    torch.cuda.nvtx.range_pop()
        for end in op_ends:
            coordinator.wait_event(end)
        group_end.record(coordinator)
        group_end.synchronize()
    finally:
        if outer_marker:
            torch.cuda.nvtx.range_pop()

    return {
        "outputs": outputs,
        "group_ms": float(gate.elapsed_time(group_end)),
        "per_op_ms": {
            node.name: float(start.elapsed_time(end))
            for node, start, end in zip(targets, op_starts, op_ends)
        },
    }


def capture_solo_graph(interpreter: Any, node: Any, stream: Any):
    """Capture one target FX node as a standalone CUDA Graph."""
    import torch

    # CUDA Graph capture requires prior work on a side stream.
    solo_once(interpreter, node, stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        torch.cuda.nvtx.range_push(f"JANUS_SOLO_GRAPH_CAPTURE_OP:{node.name}")
        try:
            output = execute_node(interpreter, node)
        finally:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return {"graph": graph, "output": output, "stream": stream}


def capture_group_graph(interpreter: Any, targets: list[Any], streams: list[Any]):
    """Capture only the independent target group with multi-stream joins."""
    import torch

    group_once(interpreter, targets, streams)
    torch.cuda.synchronize()
    first_stream = streams[0]
    first_event = torch.cuda.Event()
    join_events = [torch.cuda.Event() for _ in streams]
    graph = torch.cuda.CUDAGraph()
    outputs: dict[str, Any] = {}
    with torch.cuda.graph(graph, stream=first_stream):
        first_event.record(first_stream)
        for stream in streams[1:]:
            stream.wait_event(first_event)
        for index, (node, stream) in enumerate(zip(targets, streams)):
            torch.cuda.nvtx.range_push(f"JANUS_GRAPH_CAPTURE_OP:{node.name}")
            try:
                with torch.cuda.stream(stream):
                    outputs[node.name] = execute_node(interpreter, node)
            finally:
                torch.cuda.nvtx.range_pop()
        for index, stream in enumerate(streams[1:], start=1):
            join_events[index].record(stream)
            first_stream.wait_event(join_events[index])
    torch.cuda.synchronize()
    return {
        "graph": graph,
        "outputs": outputs,
        "stream": first_stream,
    }


def replay_solo_graph(captured: dict[str, Any], marker: str | None = None):
    import torch

    stream = captured["stream"]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if marker:
        torch.cuda.nvtx.range_push(marker)
    try:
        with torch.cuda.stream(stream):
            start.record(stream)
            captured["graph"].replay()
            end.record(stream)
        end.synchronize()
    finally:
        if marker:
            torch.cuda.nvtx.range_pop()
    return captured["output"], float(start.elapsed_time(end))


def replay_group_graph(captured: dict[str, Any], marker: str | None = None):
    import torch

    stream = captured["stream"]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if marker:
        torch.cuda.nvtx.range_push(marker)
    try:
        with torch.cuda.stream(stream):
            start.record(stream)
            captured["graph"].replay()
            end.record(stream)
        end.synchronize()
    finally:
        if marker:
            torch.cuda.nvtx.range_pop()
    return {
        "outputs": captured["outputs"],
        "group_ms": float(start.elapsed_time(end)),
        "per_op_ms": {},
    }


def main() -> int:
    args = parse_args()
    if len(args.group) < 2 or len(args.group) > 5:
        raise ValueError("group must contain two through five operator names")
    if len(set(args.group)) != len(args.group):
        raise ValueError("group contains duplicate names")
    if args.call < 1 or args.warmup < 1 or args.repeats < 1:
        raise ValueError("call, warmup and repeats must be positive")
    output_json = args.output_json.resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_json.exists():
        if not args.mode.startswith("ncu-"):
            raise FileExistsError(f"refusing to overwrite {output_json}")
        # Application Range Replay may relaunch the complete target process for
        # each counter pass.  Reuse is allowed only for the exact same identity.
        previous = json.loads(output_json.read_text(encoding="utf-8"))
        expected = (args.model, args.call, args.group, args.mode)
        observed = (
            previous.get("model"),
            previous.get("call"),
            previous.get("group"),
            previous.get("mode"),
        )
        if observed != expected:
            raise FileExistsError(
                f"NCU replay output identity differs: {observed} != {expected}"
            )

    repo = Path(__file__).resolve().parents[2]
    experiments = repo / "experiments"
    sys.path.insert(0, str(experiments))
    sys.path.insert(0, str(repo))

    from harness_common import expected_profile_path, load_config, require_idle_gpu
    from run_one import load_model_and_inputs, seed_everything

    if not args.skip_idle_check:
        require_idle_gpu()
    config = load_config()
    seed = int(config["measurement"]["seed"])
    seed_everything(seed)

    import torch

    model, inputs = load_model_and_inputs(args.model, config)
    profile_path = expected_profile_path(model, inputs)
    expected_profile_sha = args.expected_profile_sha256
    expected_fx_sha = args.expected_fx_code_sha256
    configured_profile = config["models"][args.model]["profile_file"]
    if profile_path.name != configured_profile:
        raise RuntimeError(
            f"runtime profile name differs: {profile_path.name} != "
            f"{configured_profile}"
        )
    backend = config["models"][args.model].get(
        "capture_backend", "dynamo_explain"
    )
    if not profile_path.is_file():
        raise FileNotFoundError(f"frozen serial profile is missing: {profile_path}")
    profile_sha = sha256_file(profile_path)
    if profile_sha != expected_profile_sha:
        raise RuntimeError(
            f"serial profile SHA mismatch: {profile_sha} != "
            f"{expected_profile_sha}"
        )
    module = trace_fx(model, inputs, backend)
    fx_code_sha = hashlib.sha256(module.code.encode("utf-8")).hexdigest()
    if fx_code_sha != expected_fx_sha:
        raise RuntimeError(
            f"FX code SHA mismatch: {fx_code_sha} != {expected_fx_sha}"
        )
    interpreter, targets, ancestors = prepare_target_interpreter(
        module, inputs, args.group
    )
    streams = [torch.cuda.Stream() for _ in targets]

    # Establish deterministic solo references and warm all target kernels.
    solo_references: dict[str, list[Any]] = {}
    for node, stream in zip(targets, streams):
        output, _ = solo_once(interpreter, node, stream)
        solo_references[node.name] = clone_tensor_leaves(output)
    for _ in range(args.warmup):
        for node, stream in zip(targets, streams):
            solo_once(interpreter, node, stream)
        group_once(interpreter, targets, streams)

    solo_graphs = {
        node.name: capture_solo_graph(interpreter, node, stream)
        for node, stream in zip(targets, streams)
    }
    group_graph = capture_group_graph(interpreter, targets, streams)
    solo_references = {
        name: clone_tensor_leaves(replay_solo_graph(captured)[0])
        for name, captured in solo_graphs.items()
    }
    baseline_group = replay_group_graph(group_graph)
    correctness = {
        name: compare_tensor_leaves(solo_references[name], output)
        for name, output in baseline_group["outputs"].items()
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "offline_resource_recording_only",
        "scheduler_modified": False,
        "model": args.model,
        "call": args.call,
        "group": args.group,
        "mode": args.mode,
        "seed": seed,
        "capture_backend": backend,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha,
        "fx_code_sha256": fx_code_sha,
        "fx_node_count": len(list(module.graph.nodes)),
        "ancestor_count": len(ancestors),
        "input_shapes": [list(value.shape) for value in inputs],
        "input_dtypes": [str(value.dtype) for value in inputs],
        "device": torch.cuda.get_device_name(inputs[0].device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "correctness": correctness,
        "git_head": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }

    group_slug = f"{args.model.replace('-', '_')}_c{args.call}_" + "__".join(args.group)
    if args.mode == "timing":
        solo_samples = {node.name: [] for node in targets}
        for _ in range(args.repeats):
            for node in targets:
                _, elapsed = replay_solo_graph(solo_graphs[node.name])
                solo_samples[node.name].append(elapsed)
        group_samples: list[float] = []
        for _ in range(args.repeats):
            observed = replay_group_graph(group_graph)
            group_samples.append(observed["group_ms"])
        solo_stats = {name: stats(values) for name, values in solo_samples.items()}
        result["timing"] = {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "solo": solo_stats,
            "group": stats(group_samples),
            "per_op_group_timing_source": "Nsight Systems trace, not CUDA events",
        }
    elif args.mode == "trace":
        solo_trace_replays = {node.name: [] for node in targets}
        for node in targets:
            for index in range(args.repeats):
                _, elapsed = replay_solo_graph(
                    solo_graphs[node.name],
                    marker=f"JANUS_SOLO_REPLAY:{node.name}:{index}",
                )
                solo_trace_replays[node.name].append({
                    "index": index,
                    "event_ms": elapsed,
                })
        replays = []
        for index in range(args.repeats):
            observed = replay_group_graph(
                group_graph,
                marker=f"JANUS_GROUP_REPLAY:{group_slug}:{index}",
            )
            replays.append({
                "index": index,
                "group_ms": observed["group_ms"],
                "per_op_ms": observed["per_op_ms"],
            })
        result["solo_trace_replays"] = solo_trace_replays
        result["trace_replays"] = replays
    elif args.mode == "ncu-solo":
        profiled = []
        for node in targets:
            _, elapsed = replay_solo_graph(
                solo_graphs[node.name],
                marker=f"JANUS_NCU_SOLO:{node.name}",
            )
            profiled.append({"operator": node.name, "event_ms": elapsed})
        result["ncu_solo_markers"] = profiled
    elif args.mode == "ncu-group":
        observed = replay_group_graph(
            group_graph,
            marker=f"JANUS_NCU_GROUP:{group_slug}",
        )
        result["ncu_group_marker"] = {
            "name": f"JANUS_NCU_GROUP:{group_slug}",
            "event_group_ms": observed["group_ms"],
            "event_per_op_ms": observed["per_op_ms"],
        }
    else:
        raise AssertionError(args.mode)

    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    message = json.dumps({
        "status": "ok",
        "mode": args.mode,
        "model": args.model,
        "call": args.call,
        "group": args.group,
        "output_json": str(output_json),
    }, ensure_ascii=False)
    print(message, file=sys.stderr if args.mode.startswith("ncu-") else sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
