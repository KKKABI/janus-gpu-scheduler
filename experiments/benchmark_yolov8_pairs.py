#!/usr/bin/env python3
"""Measure true solo/co-run interference for selected model FX operators."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness_common import (
    gpu_snapshot,
    require_idle_gpu,
    runtime_metadata,
    stats_from_samples,
    write_json_atomic,
)


PAIR_SPECS = (
    {
        "id": "convolution_arange2",
        "a": "convolution",
        "b": "arange_2",
        "predicted_risk": 0.275076632625323,
        "predicted_speedup": 1.1093413587484604,
        "scheduler_decision": "serialize",
    },
    {
        "id": "mul1_mul3",
        "a": "mul_1",
        "b": "mul_3",
        "predicted_risk": 0.290670,
        "predicted_speedup": 1.067331,
        "scheduler_decision": "serialize",
    },
    {
        "id": "convolution85_convolution94",
        "a": "convolution_85",
        "b": "convolution_94",
        "predicted_risk": 0.22501332099869548,
        "predicted_speedup": 1.021369558484009,
        "scheduler_decision": "serialize",
    },
    {
        "id": "convolution76_convolution97",
        "a": "convolution_76",
        "b": "convolution_97",
        "predicted_risk": 0.22652761109226358,
        "predicted_speedup": 1.1406654178931408,
        "scheduler_decision": "keep_concurrent",
    },
    {
        "id": "silu90_silu96",
        "a": "silu__90",
        "b": "silu__96",
        "predicted_risk": 0.10182918628481166,
        "predicted_speedup": 1.1717791411042944,
        "scheduler_decision": "keep_concurrent",
    },
    {
        "id": "cat13_cat14",
        "a": "cat_13",
        "b": "cat_14",
        "predicted_risk": 0.12737493999569935,
        "predicted_speedup": 1.02320987654321,
        "scheduler_decision": "serialize",
    },
)

# Remaining two-operator combinations selected by TD+Janus-no-alpha in the
# matched YOLO run.  Keeping them in the same benchmark makes it possible to
# evaluate the risk score as a ranking signal rather than only auditing the
# pairs for which the risk gate happened to fire.
PAIR_SPECS += (
    {
        "id": "arange_batchnorm",
        "a": "arange",
        "b": "cudnn_batch_norm",
        "predicted_risk": 0.02241772037782786,
        "predicted_speedup": 1.651470588235294,
        "scheduler_decision": "keep_concurrent",
        "source_call": 2,
    },
    {
        "id": "add19_add21",
        "a": "add_19",
        "b": "add_21",
        "predicted_risk": 0.6195026705706138,
        "predicted_speedup": 1.058684521465144,
        "scheduler_decision": "serialize",
        "source_call": 3,
    },
    {
        "id": "mul3_silu",
        "a": "mul_3",
        "b": "silu_",
        "predicted_risk": 0.051062734943309554,
        "predicted_speedup": 1.4246455834242095,
        "scheduler_decision": "keep_concurrent",
        "source_call": 4,
    },
    {
        "id": "mul1_convolution1",
        "a": "mul_1",
        "b": "convolution_1",
        "predicted_risk": 0.1500028766645983,
        "predicted_speedup": 1.009763256971476,
        "scheduler_decision": "serialize",
        "source_call": 5,
    },
    {
        "id": "silu85_silu91",
        "a": "silu__85",
        "b": "silu__91",
        "predicted_risk": 0.020959497555349496,
        "predicted_speedup": 1.6935369318181819,
        "scheduler_decision": "keep_concurrent",
        "source_call": 319,
    },
    {
        "id": "batchnorm76_batchnorm87",
        "a": "cudnn_batch_norm_76",
        "b": "cudnn_batch_norm_87",
        "predicted_risk": 0.014919560418889062,
        "predicted_speedup": 1.0275259067357514,
        "scheduler_decision": "keep_concurrent",
        "source_call": 357,
    },
    {
        "id": "batchnorm90_batchnorm96",
        "a": "cudnn_batch_norm_90",
        "b": "cudnn_batch_norm_96",
        "predicted_risk": 0.08802936500953924,
        "predicted_speedup": 1.1198945981554675,
        "scheduler_decision": "keep_concurrent",
        "source_call": 402,
    },
    {
        "id": "convolution93_convolution102",
        "a": "convolution_93",
        "b": "convolution_102",
        "predicted_risk": 0.0019478758689617198,
        "predicted_speedup": 1.0180448969339053,
        "scheduler_decision": "keep_concurrent",
        "source_call": 405,
    },
    {
        "id": "sub_add22",
        "a": "sub",
        "b": "add_22",
        "predicted_risk": 0.003110380016854574,
        "predicted_speedup": 1.0532435740514077,
        "scheduler_decision": "keep_concurrent",
        "source_call": 415,
    },
    {
        "id": "add23_sub1",
        "a": "add_23",
        "b": "sub_1",
        "predicted_risk": 0.0568831916661358,
        "predicted_speedup": 1.7780779621767657,
        "scheduler_decision": "keep_concurrent",
        "source_call": 416,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="YOLOv8x",
        choices=("YOLOv8x", "GoogLeNet", "ConvNeXt"),
    )
    parser.add_argument(
        "--capture-backend",
        default="auto",
        choices=("auto", "make_fx", "dynamo_explain"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--inner-repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--pair-ids", nargs="*", default=None)
    parser.add_argument(
        "--specs-json",
        type=Path,
        help="optional JSON list (or {pairs: [...]}) of pair specifications",
    )
    return parser.parse_args()


def load_pair_specs(path=None):
    """Load and validate pair specs without importing the GPU stack."""
    if path is None:
        specs = [dict(spec) for spec in PAIR_SPECS]
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        specs = payload.get("pairs") if isinstance(payload, dict) else payload
        if not isinstance(specs, list):
            raise ValueError("pair specs JSON must be a list or contain 'pairs'")
        specs = [dict(spec) for spec in specs]

    required = {"id", "a", "b", "predicted_risk", "predicted_speedup"}
    seen_ids = set()
    for spec in specs:
        missing = required - set(spec)
        if missing:
            raise ValueError(
                f"pair spec is missing fields {sorted(missing)}: {spec}"
            )
        if spec["id"] in seen_ids:
            raise ValueError(f"duplicate pair id: {spec['id']}")
        if spec["a"] == spec["b"]:
            raise ValueError(f"pair must contain distinct nodes: {spec['id']}")
        spec.setdefault("scheduler_decision", "candidate")
        seen_ids.add(spec["id"])
    return specs


def load_model_and_inputs(name):
    import torch

    if name == "YOLOv8x":
        from ultralytics import YOLO

        model = YOLO("/public_0/ZYF/model/YOLOv8/yolov8x.pt").model
        inputs = (torch.randn((1, 3, 320, 320), device="cuda:0"),)
    elif name == "GoogLeNet":
        import torchvision

        model = torchvision.models.googlenet(init_weights=True)
        inputs = (torch.randint(
            0, 256, (1, 3, 224, 224),
            dtype=torch.float32, device="cuda:0",
        ),)
    elif name == "ConvNeXt":
        import torchvision

        model = torchvision.models.convnext_base(weights=None)
        inputs = (torch.randn((1, 3, 224, 224), device="cuda:0"),)
    else:
        raise ValueError(f"unsupported model: {name}")
    return model.to("cuda:0").eval(), inputs


def capture_model_graph(model, inputs, backend):
    import torch

    if backend == "make_fx":
        from torch.fx.experimental.proxy_tensor import make_fx

        with torch.inference_mode():
            graph_module = make_fx(model)(*inputs)
    elif backend == "dynamo_explain":
        import torch._dynamo as dynamo

        dynamo.reset()
        with torch.inference_mode():
            result = dynamo.explain(model)(*inputs)
        graphs = (
            result[2] if isinstance(result, tuple)
            else getattr(result, "graphs", None) or getattr(result, "graph", None)
        )
        if not graphs:
            raise RuntimeError("dynamo.explain did not return a graph")
        graph_module = graphs[0]
    else:
        raise ValueError(f"unsupported capture backend: {backend}")
    return graph_module.to("cuda:0")


def derive_pair_metrics(solo_a, solo_b, corun_a, corun_b, corun_makespan):
    """Return dimensionless slowdown and overlap metrics from median times."""
    return {
        "slowdown_a": corun_a / solo_a,
        "slowdown_b": corun_b / solo_b,
        "mean_slowdown": 0.5 * (corun_a / solo_a + corun_b / solo_b),
        "max_slowdown": max(corun_a / solo_a, corun_b / solo_b),
        "makespan_dilation": corun_makespan / max(solo_a, solo_b),
        "measured_pair_speedup": (solo_a + solo_b) / corun_makespan,
    }


def clone_tree(value):
    import torch
    from torch.utils._pytree import tree_map

    def clone_leaf(leaf):
        if not isinstance(leaf, torch.Tensor):
            return leaf
        clone = torch.empty_strided(
            leaf.size(),
            leaf.stride(),
            dtype=leaf.dtype,
            device=leaf.device,
        )
        clone.copy_(leaf)
        return clone

    return tree_map(clone_leaf, value)


def record_operator_inputs(graph_module, inputs, target_names):
    import torch
    from torch.fx import Interpreter

    class RecordingInterpreter(Interpreter):
        def __init__(self, module):
            super().__init__(module)
            self.records = {}

        def run_node(self, node):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            if node.name in target_names and node.name not in self.records:
                if node.op not in {
                        "call_function", "call_method", "call_module"}:
                    raise RuntimeError(
                        f"unsupported target node kind: {node.name}={node.op}"
                    )
                frozen_args, frozen_kwargs = clone_tree((args, kwargs))
                self.records[node.name] = {
                    "name": node.name,
                    "op": node.op,
                    "target": node.target,
                    "target_text": str(node.target),
                    "module": (
                        self.module.get_submodule(node.target)
                        if node.op == "call_module" else None
                    ),
                    "args": frozen_args,
                    "kwargs": frozen_kwargs,
                }
            return getattr(self, node.op)(node.target, args, kwargs)

    interpreter = RecordingInterpreter(graph_module)
    with torch.inference_mode():
        interpreter.run(*inputs)
    torch.cuda.synchronize()
    missing = sorted(set(target_names) - set(interpreter.records))
    if missing:
        raise RuntimeError(f"target FX nodes not found: {missing}")
    return interpreter.records


def invoke(record, args, kwargs):
    if record["op"] == "call_function":
        return record["target"](*args, **kwargs)
    if record["op"] == "call_method":
        return getattr(args[0], record["target"])(*args[1:], **kwargs)
    if record["op"] == "call_module":
        return record["module"](*args, **kwargs)
    raise RuntimeError(f"unsupported target node kind: {record['op']}")


def capture_repeated(record, stream, inner_repeats, warmup):
    import torch

    warm_args, warm_kwargs = clone_tree((record["args"], record["kwargs"]))
    with torch.cuda.stream(stream), torch.inference_mode():
        for _ in range(warmup):
            invoke(record, warm_args, warm_kwargs)
    stream.synchronize()

    args, kwargs = clone_tree((record["args"], record["kwargs"]))
    graph = torch.cuda.CUDAGraph()
    output = None
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=stream), torch.inference_mode():
        for _ in range(inner_repeats):
            output = invoke(record, args, kwargs)
    torch.cuda.synchronize()
    return {
        "graph": graph,
        "stream": stream,
        "output": output,
        "args": args,
        "kwargs": kwargs,
    }


def measure_solo(captured, inner_repeats, cache, sleep_cycles):
    import torch

    torch.cuda.synchronize()
    cache.zero_()
    torch.cuda._sleep(sleep_cycles)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = captured["stream"]
    with torch.cuda.stream(stream):
        start.record(stream)
        captured["graph"].replay()
        end.record(stream)
    end.synchronize()
    return float(start.elapsed_time(end)) / inner_repeats


def measure_corun(left, right, inner_repeats, cache, sleep_cycles):
    import torch

    torch.cuda.synchronize()
    cache.zero_()
    torch.cuda._sleep(sleep_cycles)
    origin = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end_left = torch.cuda.Event(enable_timing=True)
    end_right = torch.cuda.Event(enable_timing=True)
    finish = torch.cuda.Event(enable_timing=True)

    start.record(origin)
    left["stream"].wait_event(start)
    right["stream"].wait_event(start)
    with torch.cuda.stream(left["stream"]):
        left["graph"].replay()
        end_left.record(left["stream"])
    with torch.cuda.stream(right["stream"]):
        right["graph"].replay()
        end_right.record(right["stream"])
    origin.wait_event(end_left)
    origin.wait_event(end_right)
    finish.record(origin)
    finish.synchronize()
    return (
        float(start.elapsed_time(end_left)) / inner_repeats,
        float(start.elapsed_time(end_right)) / inner_repeats,
        float(start.elapsed_time(finish)) / inner_repeats,
    )


def benchmark_pair(spec, records, args, cache, rng):
    import torch

    left_stream = torch.cuda.Stream()
    right_stream = torch.cuda.Stream()
    left = capture_repeated(
        records[spec["a"]], left_stream, args.inner_repeats, args.warmup
    )
    right = capture_repeated(
        records[spec["b"]], right_stream, args.inner_repeats, args.warmup
    )

    samples = {
        "solo_a_ms": [],
        "solo_b_ms": [],
        "corun_a_completion_ms": [],
        "corun_b_completion_ms": [],
        "corun_makespan_ms": [],
    }
    sleep_cycles = 1_000_000
    for _ in range(args.samples):
        modes = ["solo_a", "solo_b", "corun"]
        rng.shuffle(modes)
        for mode in modes:
            if mode == "solo_a":
                samples["solo_a_ms"].append(measure_solo(
                    left, args.inner_repeats, cache, sleep_cycles
                ))
            elif mode == "solo_b":
                samples["solo_b_ms"].append(measure_solo(
                    right, args.inner_repeats, cache, sleep_cycles
                ))
            else:
                time_a, time_b, makespan = measure_corun(
                    left, right, args.inner_repeats, cache, sleep_cycles
                )
                samples["corun_a_completion_ms"].append(time_a)
                samples["corun_b_completion_ms"].append(time_b)
                samples["corun_makespan_ms"].append(makespan)

    statistics = {
        name: stats_from_samples(values)
        for name, values in samples.items()
    }
    medians = {
        name: item["median_ms"] for name, item in statistics.items()
    }
    derived = derive_pair_metrics(
        medians["solo_a_ms"],
        medians["solo_b_ms"],
        medians["corun_a_completion_ms"],
        medians["corun_b_completion_ms"],
        medians["corun_makespan_ms"],
    )
    predicted_saved = 1.0 - 1.0 / spec["predicted_speedup"]
    derived["predicted_normalized_time_saved"] = predicted_saved
    derived["predicted_risk_adjusted_utility"] = (
        predicted_saved - 0.5 * spec["predicted_risk"]
    )

    result = {
        **spec,
        "node_a_target": records[spec["a"]]["target_text"],
        "node_b_target": records[spec["b"]]["target_text"],
        "samples": samples,
        "statistics": statistics,
        "derived": derived,
    }
    del left, right
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.samples < 3:
        raise ValueError("--samples must be >= 3")
    if args.inner_repeats < 1 or args.warmup < 1:
        raise ValueError("--inner-repeats and --warmup must be >= 1")

    available_specs = load_pair_specs(args.specs_json)
    selected_specs = list(available_specs)
    if args.pair_ids:
        requested = set(args.pair_ids)
        selected_specs = [
            spec for spec in available_specs if spec["id"] in requested
        ]
        missing = requested - {spec["id"] for spec in selected_specs}
        if missing:
            raise ValueError(f"unknown pair ids: {sorted(missing)}")

    require_idle_gpu()
    started = time.time()
    before = gpu_snapshot()

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    benchmark_specs = list(selected_specs)
    rng.shuffle(benchmark_specs)

    model, inputs = load_model_and_inputs(args.model)
    with torch.inference_mode():
        model(*inputs)  # initialize model-side caches before graph capture
    capture_backend = args.capture_backend
    if capture_backend == "auto":
        capture_backend = (
            "make_fx" if args.model == "YOLOv8x" else "dynamo_explain"
        )
    graph_module = capture_model_graph(model, inputs, capture_backend)

    target_names = {
        spec[key] for spec in selected_specs for key in ("a", "b")
    }
    records = record_operator_inputs(graph_module, inputs, target_names)
    cache = torch.empty(4 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    pair_results = [
        benchmark_pair(spec, records, args, cache, rng)
        for spec in benchmark_specs
    ]
    pair_results.sort(key=lambda pair: pair["id"])
    torch.cuda.synchronize()

    result = {
        "schema_version": 1,
        "status": "completed",
        "model": args.model,
        "model_class": model.__class__.__name__,
        "input_shapes": [list(tensor.shape) for tensor in inputs],
        "method": {
            "capture": "one CUDA graph per FX operator",
            "capture_backend": capture_backend,
            "execution": "two independent CUDA streams",
            "sample_order": "seeded shuffle of solo_a, solo_b, corun",
            "samples": args.samples,
            "inner_repeats_per_graph": args.inner_repeats,
            "warmup": args.warmup,
            "cache_buffer_bytes": int(cache.numel()),
            "seed": args.seed,
            "pair_order": [spec["id"] for spec in benchmark_specs],
        },
        "pairs": pair_results,
        "runtime": {
            **runtime_metadata(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "telemetry": {
            "before": before,
            "after": gpu_snapshot(),
        },
        "started_unix": started,
        "finished_unix": time.time(),
    }
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "pairs": [
            {
                "id": pair["id"],
                "decision": pair["scheduler_decision"],
                **pair["derived"],
            }
            for pair in pair_results
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
