#!/usr/bin/env python3
"""Create compact, auditable CSV/JSON summaries from chapter-3 raw results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

from multi_janus_benchmark import atomic_json


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def client_gpu(payload: dict) -> dict[int, float]:
    return {
        int(client["client_id"]): float(client["summary"]["gpu_event_ms"]["median"])
        for client in payload["clients"]
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["no_rows"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile_rows(root: Path) -> list[dict]:
    rows = []
    for sequential_path in sorted(root.glob("profile/*/trial_*/sequential/result.json")):
        trial_root = sequential_path.parents[1]
        concurrent_path = trial_root / "concurrent" / "result.json"
        if not concurrent_path.exists():
            continue
        sequential = load(sequential_path)
        concurrent = load(concurrent_path)
        serial_gpu = client_gpu(sequential)
        parallel_gpu = client_gpu(concurrent)
        slowdowns = [parallel_gpu[key] / serial_gpu[key] for key in serial_gpu]
        rows.append(
            {
                "pair": trial_root.parent.name,
                "trial": sequential.get("trial"),
                "correctness": sequential["overall"]["correctness_ok"]
                and concurrent["overall"]["correctness_ok"],
                "sequential_mean_response_ms": sequential["overall"]["response_ms"]["mean"],
                "concurrent_mean_response_ms": concurrent["overall"]["response_ms"]["mean"],
                "response_gain": sequential["overall"]["response_ms"]["mean"]
                / concurrent["overall"]["response_ms"]["mean"],
                "sequential_throughput_rps": sequential["overall"]["throughput_requests_per_second"],
                "concurrent_throughput_rps": concurrent["overall"]["throughput_requests_per_second"],
                "throughput_gain": concurrent["overall"]["throughput_requests_per_second"]
                / sequential["overall"]["throughput_requests_per_second"],
                "maximum_gpu_service_slowdown": max(slowdowns),
                "client0_gpu_slowdown": slowdowns[0],
                "client1_gpu_slowdown": slowdowns[1],
            }
        )
    return rows


def solo_rows(root: Path) -> list[dict]:
    rows = []
    for result_path in sorted(root.glob("solo/*/result.json")):
        payload = load(result_path)
        client = payload["clients"][0]
        rows.append(
            {
                "model": client["model"],
                "batch_size": client["batch_size"],
                "correctness": payload["overall"]["correctness_ok"],
                "mean_response_ms": payload["overall"]["response_ms"]["mean"],
                "p95_response_ms": payload["overall"]["response_ms"]["p95"],
                "median_gpu_event_ms": client["summary"]["gpu_event_ms"]["median"],
                "throughput_rps": payload["overall"]["throughput_requests_per_second"],
                "offline_process_to_ready_ms": client.get("preparation", {}).get(
                    "process_to_ready_ms"
                ),
                "offline_graph_build_ms": client.get("preparation", {}).get(
                    "graph_build_and_first_replay_ms"
                ),
                "max_memory_reserved_bytes": client["max_memory_reserved_bytes"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--solo-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    profile = profile_rows(args.root)
    solo = solo_rows(args.solo_root or args.root)
    write_csv(args.output_dir / "profile_pairs.csv", profile)
    write_csv(args.output_dir / "solo_models.csv", solo)

    table_path = args.root / "compatibility.json"
    table = load(table_path) if table_path.exists() else None
    aggregate = {
        "profile_complete_pair_trials": len(profile),
        "profile_correct_pair_trials": sum(bool(row["correctness"]) for row in profile),
        "solo_models": len(solo),
        "compatibility_table_present": table is not None,
        "allowed_pairs": (
            sum(entry["allow_concurrent"] for entry in table["entries"].values())
            if table
            else None
        ),
        "total_table_pairs": len(table["entries"]) if table else None,
        "median_response_gain": (
            statistics.median(row["response_gain"] for row in profile)
            if profile
            else None
        ),
        "median_throughput_gain": (
            statistics.median(row["throughput_gain"] for row in profile)
            if profile
            else None
        ),
    }
    atomic_json(args.output_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
