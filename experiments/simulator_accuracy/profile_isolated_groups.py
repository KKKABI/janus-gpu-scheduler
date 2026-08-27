#!/usr/bin/env python3
"""Capture isolated exact FX groups with a common CUDA Graph fork."""

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
        return
    if isinstance(destination, dict):
        for key in destination:
            copy_tree_(destination[key], source[key])
        return
    if isinstance(destination, (list, tuple)):
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
    def __init__(self, module, targets):
        import torch

        class _Recorder(torch.fx.Interpreter):
            def __init__(inner, graph_module):
                super().__init__(graph_module)
                inner.records = {}

            def run_node(inner, node):
                args, kwargs = inner.fetch_args_kwargs_from_env(node)
                result = super(_Recorder, inner).run_node(node)
                if node.name in targets:
                    inner.records[node.name] = {
                        "args": clone_tree(args),
                        "kwargs": clone_tree(kwargs),
                        "output": clone_tree(result),
                    }
                return result

        self.interpreter = _Recorder(module)

    def run(self, *inputs):
        self.interpreter.run(*inputs)
        return self.interpreter.records


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
    if not cases:
        raise ValueError(f"manifest has no cases for {args.model}")

    import torch

    config = load_config()
    seed_everything(int(config["measurement"]["seed"]))
    model, inputs = load_model_and_inputs(args.model, config)
    profile_path = expected_profile_path(model, inputs)
    profile_sha = sha256_file(profile_path)
    expected_sha = manifest["source_profile_sha256_by_model"][args.model]
    if profile_sha != expected_sha:
        raise RuntimeError(f"profile SHA differs: {profile_sha} != {expected_sha}")
    backend = config["models"][args.model].get("capture_backend", "dynamo_explain")
    # Match the production capture path: materialize lazy model state (notably
    # YOLO anchors/strides and library workspaces) before make_fx/dynamo tracing.
    with torch.no_grad():
        model(*inputs)
    torch.cuda.synchronize()
    module = trace_model(model, inputs, backend)
    nodes = {node.name: node for node in module.graph.nodes}
    unique_targets = {name for case in cases for name in case["group"]}
    missing = sorted(unique_targets - set(nodes))
    if missing:
        raise RuntimeError(f"target FX nodes are missing: {missing}")

    from torch.fx import Interpreter

    invoker = Interpreter(module)
    summaries = []
    for index, case in enumerate(cases, start=1):
        group = list(case["group"])
        summary_base = {
            "case_id": case["case_id"],
            "model": args.model,
            "call": case["call"],
            "group": group,
            "size": len(group),
            "stratum_population": case["stratum_population"],
            "sample_weight": case["sample_weight"],
            "original_max_concurrent": case["original_max_concurrent"],
            "original_any_pair_overlap_ns": case[
                "original_any_pair_overlap_ns"
            ],
        }
        recorder = RecordingInterpreter(module, set(group))
        with torch.no_grad():
            records = recorder.run(*inputs)
        torch.cuda.synchronize()
        if set(records) != set(group):
            raise RuntimeError(f"{case['case_id']}: did not record every target")
        pristine = {
            name: {
                "args": clone_tree(records[name]["args"]),
                "kwargs": clone_tree(records[name]["kwargs"]),
            }
            for name in group
        }
        sequential_references = {}
        extraction_correctness = {}
        extraction_error = None
        try:
            with torch.no_grad():
                for name in group:
                    reference_args = clone_tree(pristine[name]["args"])
                    reference_kwargs = clone_tree(pristine[name]["kwargs"])
                    reference_output = invoke(
                        invoker, nodes[name], reference_args, reference_kwargs
                    )
                    torch.cuda.synchronize()
                    sequential_references[name] = clone_tree(reference_output)
                    extraction_correctness[name] = compare_outputs(
                        records[name]["output"], sequential_references[name]
                    )
        except (AssertionError, RuntimeError) as error:
            extraction_error = f"{name}: {type(error).__name__}: {error}"
        if extraction_error:
            summaries.append(
                {
                    **summary_base,
                    "capture_status": "extraction_failed",
                    "extraction_error": extraction_error,
                    "extraction_correctness": extraction_correctness,
                }
            )
            print(
                json.dumps(
                    {
                        "index": index,
                        "total": len(cases),
                        "case": case["case_id"],
                        "status": "extraction_failed",
                    }
                ),
                flush=True,
            )
            del records, recorder, sequential_references
            torch.cuda.empty_cache()
            continue

        streams = [torch.cuda.Stream() for _ in group]
        # Warm every isolated operator before graph capture so lazy library
        # initialization and autotuning are outside the measured replay.
        with torch.no_grad():
            for name, stream in zip(group, streams):
                record = records[name]
                with torch.cuda.stream(stream):
                    invoke(invoker, nodes[name], record["args"], record["kwargs"])
        torch.cuda.synchronize()
        with torch.no_grad():
            for name in group:
                copy_tree_(records[name]["args"], pristine[name]["args"])
                copy_tree_(records[name]["kwargs"], pristine[name]["kwargs"])
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        fork_event = torch.cuda.Event()
        join_events = [torch.cuda.Event() for _ in streams]
        captured_outputs = {}
        main_stream = streams[0]
        with torch.no_grad(), torch.cuda.graph(graph, stream=main_stream):
            fork_event.record(main_stream)
            for stream in streams[1:]:
                stream.wait_event(fork_event)
            for name, stream in zip(group, streams):
                record = records[name]
                with torch.cuda.stream(stream):
                    torch.cuda.nvtx.range_push(
                        f"ISOLATED_FX_CAPTURE::{case['case_id']}::{name}"
                    )
                    try:
                        captured_outputs[name] = invoke(
                            invoker, nodes[name], record["args"], record["kwargs"]
                        )
                    finally:
                        torch.cuda.nvtx.range_pop()
            for event, stream in zip(join_events[1:], streams[1:]):
                event.record(stream)
            for event in join_events[1:]:
                main_stream.wait_event(event)
        torch.cuda.synchronize()

        # Restore static operands mutated by in-place FX operators before the
        # one measured replay (for example YOLO's silu_ nodes).
        with torch.no_grad():
            for name in group:
                copy_tree_(records[name]["args"], pristine[name]["args"])
                copy_tree_(records[name]["kwargs"], pristine[name]["kwargs"])
        torch.cuda.synchronize()
        marker = f"ISOLATED_REPLAY::{case['case_id']}"
        torch.cuda.nvtx.range_push(marker)
        try:
            graph.replay()
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
        correctness = {
            name: compare_outputs(
                sequential_references[name], captured_outputs[name]
            )
            for name in group
        }
        summaries.append(
            {
                **summary_base,
                "capture_status": "captured",
                "marker": marker,
                "correctness": correctness,
                "extraction_correctness": extraction_correctness,
            }
        )
        print(
            json.dumps(
                {"index": index, "total": len(cases), "case": case["case_id"]}
            ),
            flush=True,
        )
        del graph, captured_outputs, records, recorder, sequential_references
        torch.cuda.empty_cache()

    payload = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "model": args.model,
        "git_head": head,
        "profile_path": str(profile_path),
        "profile_sha256": profile_sha,
        "capture_backend": backend,
        "case_count": len(summaries),
        "cases": summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
