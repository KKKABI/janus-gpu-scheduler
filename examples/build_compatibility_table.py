#!/usr/bin/env python3
"""Build the graph-pair admission lookup table from frozen profiling trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from multi_janus_benchmark import atomic_json, pair_key


def load_result(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("overall", {}).get("correctness_ok", False):
        raise ValueError(f"correctness gate failed: {path}")
    return payload


def client_gpu_medians(payload: dict) -> dict:
    return {
        int(client["client_id"]): {
            "identity": (client["model"], int(client["batch_size"])),
            "gpu_event_ms": float(client["summary"]["gpu_event_ms"]["median"]),
        }
        for client in payload["clients"]
    }


def build_entry(
    sequential: list[dict],
    concurrent: list[dict],
    minimum_response_gain: float,
    minimum_throughput_gain: float,
    maximum_service_slowdown: float,
) -> tuple[str, dict]:
    if not sequential or len(sequential) != len(concurrent):
        raise ValueError("paired sequential/concurrent trials are required")
    models = list(sequential[0]["models"])
    batch_sizes = list(sequential[0]["batch_sizes"])
    key = pair_key(models, batch_sizes)
    response_gains = []
    throughput_gains = []
    service_slowdowns = []
    trials = []
    for serial, parallel in zip(sequential, concurrent):
        if pair_key(serial["models"], serial["batch_sizes"]) != key:
            raise ValueError("sequential trial identity differs")
        if pair_key(parallel["models"], parallel["batch_sizes"]) != key:
            raise ValueError("concurrent trial identity differs")
        response_gain = (
            serial["overall"]["response_ms"]["mean"]
            / parallel["overall"]["response_ms"]["mean"]
        )
        throughput_gain = (
            parallel["overall"]["throughput_requests_per_second"]
            / serial["overall"]["throughput_requests_per_second"]
        )
        serial_gpu = client_gpu_medians(serial)
        parallel_gpu = client_gpu_medians(parallel)
        if serial_gpu.keys() != parallel_gpu.keys():
            raise ValueError("sequential/concurrent client ids differ")
        per_client_slowdowns = []
        for client_id in serial_gpu:
            if serial_gpu[client_id]["identity"] != parallel_gpu[client_id]["identity"]:
                raise ValueError("sequential/concurrent client identity differs")
            per_client_slowdowns.append(
                parallel_gpu[client_id]["gpu_event_ms"]
                / serial_gpu[client_id]["gpu_event_ms"]
            )
        maximum = max(per_client_slowdowns)
        response_gains.append(response_gain)
        throughput_gains.append(throughput_gain)
        service_slowdowns.append(maximum)
        trials.append({
            "trial": serial.get("trial"),
            "response_gain": response_gain,
            "throughput_gain": throughput_gain,
            "maximum_service_slowdown": maximum,
        })

    response_gain = statistics.median(response_gains)
    throughput_gain = statistics.median(throughput_gains)
    service_slowdown = statistics.median(service_slowdowns)
    gates = {
        "response_gain": response_gain >= minimum_response_gain,
        "throughput_gain": throughput_gain >= minimum_throughput_gain,
        "service_slowdown": service_slowdown <= maximum_service_slowdown,
    }
    allow = all(gates.values())
    reason = "all_gates_pass" if allow else "gate_failed:" + ",".join(
        name for name, passed in gates.items() if not passed
    )
    return key, {
        "models": models,
        "batch_sizes": batch_sizes,
        "allow_concurrent": allow,
        "reason": reason,
        "median_response_gain": response_gain,
        "median_throughput_gain": throughput_gain,
        "median_maximum_service_slowdown": service_slowdown,
        "gates": gates,
        "trials": trials,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequential", nargs="+", type=Path, required=True)
    parser.add_argument("--concurrent", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-response-gain", type=float, default=1.10)
    parser.add_argument("--minimum-throughput-gain", type=float, default=1.05)
    parser.add_argument("--maximum-service-slowdown", type=float, default=1.75)
    args = parser.parse_args()

    sequential = [load_result(path) for path in args.sequential]
    concurrent = [load_result(path) for path in args.concurrent]
    key, entry = build_entry(
        sequential,
        concurrent,
        args.minimum_response_gain,
        args.minimum_throughput_gain,
        args.maximum_service_slowdown,
    )
    payload = {
        "schema_version": 1,
        "protocol": "multi_janus_pair_compatibility_v1",
        "thresholds": {
            "minimum_response_gain": args.minimum_response_gain,
            "minimum_throughput_gain": args.minimum_throughput_gain,
            "maximum_service_slowdown": args.maximum_service_slowdown,
        },
        "entries": {key: entry},
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
