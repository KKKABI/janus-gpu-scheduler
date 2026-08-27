#!/usr/bin/env python3
"""Auditable multi-process, multi-Janus benchmark for thesis chapter 3.

The parent never imports torch.  Each child owns one CUDA context and one
captured Janus graph.  Raw per-request arrival, queue, GPU-event, service, and
completion timings are retained instead of being reduced to one mean inside
the worker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Iterable

from multi_janus_models import MODEL_CHOICES


SCHEMA_VERSION = 2


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def stats(values: Iterable[float]) -> dict:
    data = [float(value) for value in values]
    if not data:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p95": percentile(data, 0.95),
        "p99": percentile(data, 0.99),
        "std": statistics.pstdev(data),
        "min": min(data),
        "max": max(data),
    }


def pair_key(models: list[str], batch_sizes: list[int]) -> str:
    if len(models) != len(batch_sizes):
        raise ValueError("models and batch_sizes differ")
    return "__".join(
        sorted(f"{model}:b{batch}" for model, batch in zip(models, batch_sizes))
    )


def load_lookup_decision(path: Path, models, batch_sizes) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = pair_key(list(models), list(batch_sizes))
    entry = payload.get("entries", {}).get(key)
    if entry is None:
        return {
            "key": key,
            "found": False,
            "allow_concurrent": False,
            "reason": "missing_entry_fail_closed",
        }
    return {
        "key": key,
        "found": True,
        "allow_concurrent": bool(entry["allow_concurrent"]),
        "reason": entry.get("reason", "table_entry"),
        "entry": entry,
    }


def collect_provenance(repo_root: Path) -> dict:
    def run(command):
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "output": completed.stdout.strip(),
        }

    return {
        "git_head": run(["git", "rev-parse", "HEAD"]),
        "git_status": run(["git", "status", "--short"]),
        "nvidia_smi": run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "compute_processes": run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "mps_processes": run(["pgrep", "-a", "nvidia-cuda-mps"]),
        "mps_pipe_directory": os.getenv("CUDA_MPS_PIPE_DIRECTORY"),
        "mps_log_directory": os.getenv("CUDA_MPS_LOG_DIRECTORY"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }


def wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 5_000_000:
            time.sleep(min(remaining / 1e9 / 2.0, 0.005))


def run_client(args) -> int:
    import torch

    from multi_janus_models import (
        clone_tensor_leaves,
        compare_outputs,
        load_model,
    )

    process_start_ns = time.monotonic_ns()
    output_dir = Path(args.output_dir).resolve()
    control_dir = output_dir / "control"
    log_identity = f"client_{args.client_id}_{args.model}"
    ready_path = control_dir / f"ready_{args.client_id}.json"
    result_path = output_dir / "clients" / f"{log_identity}.json"
    start_path = control_dir / "start.json"

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(0)
    model_load_start_ns = time.monotonic_ns()
    model, inputs = load_model(args.model, args.batch_size)
    model_load_end_ns = time.monotonic_ns()

    with torch.inference_mode():
        eager_reference = clone_tensor_leaves(model(*inputs))
    torch.cuda.synchronize()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Opara import GraphCapturer

    graph_build_start_ns = time.monotonic_ns()
    runner = GraphCapturer.capturer(
        inputs, model, copy_outputs=False, sm_fraction=args.sm_fraction
    )
    with torch.inference_mode():
        first_graph_output = runner(*inputs)
    torch.cuda.synchronize()
    correctness = compare_outputs(eager_reference, first_graph_output)
    graph_build_end_ns = time.monotonic_ns()

    with torch.inference_mode():
        for _ in range(args.warmups):
            runner(*inputs)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    ready_ns = time.monotonic_ns()
    preparation = {
        "process_to_ready_ms": (ready_ns - process_start_ns) / 1e6,
        "model_load_ms": (model_load_end_ns - model_load_start_ns) / 1e6,
        "graph_build_and_first_replay_ms": (
            graph_build_end_ns - graph_build_start_ns
        ) / 1e6,
    }
    atomic_json(
        ready_path,
        {
            "client_id": args.client_id,
            "pid": os.getpid(),
            "model": args.model,
            "batch_size": args.batch_size,
            "correctness": correctness,
            "preparation": preparation,
            "memory_allocated_bytes": torch.cuda.memory_allocated(),
            "memory_reserved_bytes": torch.cuda.memory_reserved(),
        },
    )

    while not start_path.exists():
        time.sleep(0.01)
    start_payload = json.loads(start_path.read_text(encoding="utf-8"))
    scheduled_start_ns = int(start_payload["scheduled_start_monotonic_ns"])
    wait_until(scheduled_start_ns)

    sequential = args.effective_mode == "sequential"
    if sequential and (args.turn_read_fd is None or args.turn_write_fd is None):
        raise RuntimeError("sequential client is missing its in-memory token pipe")

    samples = []
    with torch.inference_mode():
        for iteration in range(args.iterations):
            request_ready_ns = time.monotonic_ns()
            if sequential:
                token = os.read(args.turn_read_fd, 1)
                if token != b"1":
                    raise RuntimeError(f"invalid or closed turn token: {token!r}")

            service_start_ns = time.monotonic_ns()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            runner(*inputs)
            end_event.record()
            end_event.synchronize()
            service_end_ns = time.monotonic_ns()
            gpu_event_ms = float(start_event.elapsed_time(end_event))

            is_final_turn = (
                iteration == args.iterations - 1
                and args.client_id == args.client_count - 1
            )
            if sequential and not is_final_turn:
                os.write(args.turn_write_fd, b"1")

            samples.append(
                {
                    "iteration": iteration,
                    "request_ready_monotonic_ns": request_ready_ns,
                    "service_start_monotonic_ns": service_start_ns,
                    "finish_monotonic_ns": service_end_ns,
                    "queue_ms": (service_start_ns - request_ready_ns) / 1e6,
                    "host_service_ms": (service_end_ns - service_start_ns) / 1e6,
                    "response_ms": (service_end_ns - request_ready_ns) / 1e6,
                    "gpu_event_ms": gpu_event_ms,
                }
            )

    if sequential:
        os.close(args.turn_read_fd)
        os.close(args.turn_write_fd)
    result = {
        "schema_version": SCHEMA_VERSION,
        "client_id": args.client_id,
        "pid": os.getpid(),
        "model": args.model,
        "batch_size": args.batch_size,
        "mode_effective": args.effective_mode,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "correctness": correctness,
        "preparation": preparation,
        "memory_allocated_bytes": torch.cuda.memory_allocated(),
        "memory_reserved_bytes": torch.cuda.memory_reserved(),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "samples": samples,
        "summary": {
            "queue_ms": stats(sample["queue_ms"] for sample in samples),
            "host_service_ms": stats(
                sample["host_service_ms"] for sample in samples
            ),
            "response_ms": stats(
                sample["response_ms"] for sample in samples
            ),
            "gpu_event_ms": stats(
                sample["gpu_event_ms"] for sample in samples
            ),
        },
    }
    atomic_json(result_path, result)
    return 0


def wait_for_ready(control_dir: Path, processes, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    expected = len(processes)
    while time.monotonic() < deadline:
        ready = list(control_dir.glob("ready_*.json"))
        if len(ready) == expected:
            return
        failed = [
            process.returncode
            for process in processes
            if process.poll() is not None
        ]
        if failed:
            raise RuntimeError(f"client failed before ready: {failed}")
        time.sleep(0.2)
    raise TimeoutError(f"only {len(list(control_dir.glob('ready_*.json')))} clients ready")


def aggregate(output_dir: Path, metadata: dict) -> dict:
    clients = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "clients").glob("client_*.json"))
    ]
    if not clients:
        raise RuntimeError("no client results")
    samples = [sample for client in clients for sample in client["samples"]]
    starts = [sample["request_ready_monotonic_ns"] for sample in samples]
    finishes = [sample["finish_monotonic_ns"] for sample in samples]
    window_seconds = (max(finishes) - min(starts)) / 1e9
    if window_seconds <= 0:
        raise RuntimeError(f"invalid measurement window: {window_seconds}")
    correctness_ok = all(
        int(client["correctness"].get("tensor_leaves", 0)) > 0
        for client in clients
    )
    result = {
        **metadata,
        "clients": clients,
        "overall": {
            "request_count": len(samples),
            "measurement_window_seconds": window_seconds,
            "throughput_requests_per_second": len(samples) / window_seconds,
            "correctness_ok": correctness_ok,
            "queue_ms": stats(sample["queue_ms"] for sample in samples),
            "host_service_ms": stats(
                sample["host_service_ms"] for sample in samples
            ),
            "response_ms": stats(sample["response_ms"] for sample in samples),
            "gpu_event_ms": stats(sample["gpu_event_ms"] for sample in samples),
        },
    }
    return result


def run_parent(args) -> int:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory must not already exist: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    control_dir = output_dir / "control"
    clients_dir = output_dir / "clients"
    logs_dir = output_dir / "logs"
    control_dir.mkdir()
    clients_dir.mkdir()
    logs_dir.mkdir()

    batch_sizes = args.batch_sizes or [1] * len(args.models)
    if len(batch_sizes) == 1 and len(args.models) > 1:
        batch_sizes = batch_sizes * len(args.models)
    if len(batch_sizes) != len(args.models):
        raise ValueError("--batch-sizes must have one value or one per model")

    lookup = None
    effective_mode = args.mode
    if args.mode == "lookup":
        if len(args.models) != 2:
            raise ValueError("lookup policy currently requires exactly two clients")
        if not args.lookup_table:
            raise ValueError("lookup policy requires --lookup-table")
        lookup = load_lookup_decision(
            Path(args.lookup_table), args.models, batch_sizes
        )
        effective_mode = (
            "concurrent" if lookup["allow_concurrent"] else "sequential"
        )

    repo_root = Path(__file__).resolve().parents[1]
    provenance = collect_provenance(repo_root)
    if args.require_mps and not provenance["mps_processes"]["output"]:
        raise RuntimeError("--require-mps was set but no MPS process is visible")

    processes = []
    log_streams = []
    turn_pipes = (
        [os.pipe() for _ in args.models]
        if effective_mode == "sequential"
        else []
    )
    for client_id, (model, batch_size) in enumerate(
        zip(args.models, batch_sizes)
    ):
        log_path = logs_dir / f"client_{client_id}_{model}.log"
        stream = log_path.open("w", encoding="utf-8")
        log_streams.append(stream)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--client-id",
            str(client_id),
            "--client-count",
            str(len(args.models)),
            "--model",
            model,
            "--batch-size",
            str(batch_size),
            "--effective-mode",
            effective_mode,
            "--iterations",
            str(args.iterations),
            "--warmups",
            str(args.warmups),
            "--sm-fraction",
            str(args.sm_fraction),
            "--output-dir",
            str(output_dir),
        ]
        passed_fds = ()
        if turn_pipes:
            read_fd = turn_pipes[client_id][0]
            write_fd = turn_pipes[(client_id + 1) % len(turn_pipes)][1]
            command.extend(
                (
                    "--turn-read-fd",
                    str(read_fd),
                    "--turn-write-fd",
                    str(write_fd),
                )
            )
            passed_fds = (read_fd, write_fd)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["JANUS_DEBUG_OUTPUT_PATH"] = str(
            logs_dir / f"client_{client_id}_kernel_names.txt"
        )
        environment["JANUS_OPERATOR_OUTPUT_PATH"] = str(
            logs_dir / f"client_{client_id}_operator_debug.txt"
        )
        environment["JANUS_GRAPH_OUTPUT_PATH"] = str(
            logs_dir / f"client_{client_id}_graph_debug.txt"
        )
        processes.append(
            subprocess.Popen(
                command,
                cwd=repo_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                pass_fds=passed_fds,
            )
        )

    if turn_pipes:
        os.write(turn_pipes[0][1], b"1")
        for read_fd, write_fd in turn_pipes:
            os.close(read_fd)
            os.close(write_fd)

    try:
        wait_for_ready(control_dir, processes, args.timeout)
        scheduled_start_ns = time.monotonic_ns() + int(args.start_delay * 1e9)
        atomic_json(
            control_dir / "start.json",
            {
                "scheduled_start_monotonic_ns": scheduled_start_ns,
                "created_wall_time_ns": time.time_ns(),
            },
        )
        deadline = time.monotonic() + args.timeout
        for process in processes:
            remaining = max(1.0, deadline - time.monotonic())
            process.wait(timeout=remaining)
        returncodes = [process.returncode for process in processes]
        if any(code != 0 for code in returncodes):
            raise RuntimeError(f"client return codes: {returncodes}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for stream in log_streams:
            stream.close()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "protocol": "multi_janus_closed_loop_v2_pipe_token",
        "mode_requested": args.mode,
        "mode_effective": effective_mode,
        "lookup_decision": lookup,
        "models": args.models,
        "batch_sizes": batch_sizes,
        "iterations_per_client": args.iterations,
        "warmups": args.warmups,
        "sm_fraction_internal_simulator_only": args.sm_fraction,
        "trial": args.trial,
        "provenance": provenance,
    }
    result = aggregate(output_dir, metadata)
    atomic_json(output_dir / "result.json", result)
    print(json.dumps({
        "output_dir": str(output_dir),
        "mode_requested": args.mode,
        "mode_effective": effective_mode,
        "models": args.models,
        "throughput_requests_per_second": result["overall"]["throughput_requests_per_second"],
        "mean_response_ms": result["overall"]["response_ms"]["mean"],
        "p95_response_ms": result["overall"]["response_ms"]["p95"],
        "correctness_ok": result["overall"]["correctness_ok"],
    }, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--models", nargs="+", choices=MODEL_CHOICES)
    result.add_argument("--batch-sizes", nargs="+", type=int)
    result.add_argument(
        "--mode", choices=("sequential", "concurrent", "lookup")
    )
    result.add_argument("--lookup-table")
    result.add_argument("--iterations", type=int, default=100)
    result.add_argument("--warmups", type=int, default=30)
    result.add_argument("--sm-fraction", type=float, default=1.0)
    result.add_argument("--trial", type=int, default=0)
    result.add_argument("--require-mps", action="store_true")
    result.add_argument("--start-delay", type=float, default=1.0)
    result.add_argument("--timeout", type=int, default=900)
    result.add_argument("--output-dir")

    result.add_argument("--client-id", type=int)
    result.add_argument("--client-count", type=int)
    result.add_argument("--model", choices=MODEL_CHOICES)
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--effective-mode", choices=("sequential", "concurrent"))
    result.add_argument("--turn-read-fd", type=int)
    result.add_argument("--turn-write-fd", type=int)
    return result


def validate_args(args) -> None:
    if args.client_id is not None:
        required = (
            args.client_count,
            args.model,
            args.effective_mode,
            args.output_dir,
        )
        if any(value is None for value in required):
            raise ValueError("client mode is missing an internal argument")
    else:
        if not args.models or not args.mode or not args.output_dir:
            raise ValueError("parent mode requires --models, --mode, --output-dir")
    if args.iterations <= 0 or args.warmups < 0:
        raise ValueError("invalid iterations/warmups")
    if not 0 < args.sm_fraction <= 1:
        raise ValueError("sm_fraction must be in (0,1]")


def main() -> int:
    args = parser().parse_args()
    validate_args(args)
    if args.client_id is not None:
        return run_client(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
