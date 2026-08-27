#!/usr/bin/env python3
"""Build one exact-identity, per-launch median NCU-v2 cache."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import sha256_file, write_json_atomic


REPEAT_COUNT = 3
DESCRIPTOR_FIELDS = (
    "launch_id",
    "op_name",
    "name",
    "grid_size",
    "block_size",
)
METRIC_FIELDS = (
    "mem_thru",
    "dram_thru",
    "l2_thru",
    "comp_thru",
    "dur_ns",
)
REQUIRED_IDENTITY_FIELDS = (
    "requested_model",
    "model_class",
    "input_shapes",
    "input_dtypes",
    "device_name",
    "device_capability",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "capture_backend",
    "fx_code_sha256",
    "fx_node_names",
    "profile_path",
    "profile_sha256",
    "git_head",
)


def relative_range(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0.0:
        return 0.0 if max(values) == min(values) else math.inf
    return (max(values) - min(values)) / abs(median)


def launch_descriptor(row: dict) -> tuple:
    missing = [field for field in DESCRIPTOR_FIELDS if field not in row]
    if missing:
        raise ValueError(f"launch descriptor is incomplete: {missing}")
    descriptor = tuple(row[field] for field in DESCRIPTOR_FIELDS)
    launch_id, op_name, name, grid_size, block_size = descriptor
    if not isinstance(launch_id, int) or isinstance(launch_id, bool):
        raise ValueError(f"invalid launch_id: {launch_id!r}")
    if not isinstance(op_name, str) or not op_name:
        raise ValueError(f"launch {launch_id}: missing OP identity")
    if not isinstance(name, str) or not name:
        raise ValueError(f"launch {launch_id}: missing kernel identity")
    if int(grid_size) <= 0 or int(block_size) <= 0:
        raise ValueError(f"launch {launch_id}: invalid launch geometry")
    return descriptor


def p95(values: list[float]) -> float | None:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    return finite[min(len(finite) - 1, math.ceil(0.95 * len(finite)) - 1)]


def merge_repeated_caches(
    *, model: str, cache_paths: list[Path], output_path: Path
) -> dict:
    if len(cache_paths) != REPEAT_COUNT:
        raise ValueError(
            f"{model}: expected exactly {REPEAT_COUNT} caches, got "
            f"{len(cache_paths)}"
        )
    if output_path.exists():
        raise FileExistsError(output_path)
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in cache_paths
    ]
    if any(payload.get("schema_version") != 2 for payload in payloads):
        raise ValueError(f"{model}: every source cache must use schema v2")

    identity = payloads[0].get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{model}: identity is missing")
    missing = [
        field
        for field in REQUIRED_IDENTITY_FIELDS
        if identity.get(field) in (None, "", [])
    ]
    if missing:
        raise ValueError(f"{model}: incomplete identity fields: {missing}")
    if identity.get("requested_model") != model:
        raise ValueError(f"{model}: requested-model identity differs")
    identity_signature = {
        field: identity[field] for field in REQUIRED_IDENTITY_FIELDS
    }
    for repeat, payload in enumerate(payloads[1:], start=1):
        observed = payload.get("identity")
        if not isinstance(observed, dict) or {
            field: observed.get(field) for field in REQUIRED_IDENTITY_FIELDS
        } != identity_signature:
            raise ValueError(f"{model}: repeat {repeat} identity differs")

    launch_maps = []
    for repeat, payload in enumerate(payloads):
        launches = payload.get("kernels")
        if not isinstance(launches, list) or not launches:
            raise ValueError(f"{model}: repeat {repeat} has no launches")
        mapping = {}
        for row in launches:
            descriptor = launch_descriptor(row)
            launch_id = descriptor[0]
            if launch_id in mapping:
                raise ValueError(
                    f"{model}: repeat {repeat} duplicates launch {launch_id}"
                )
            mapping[launch_id] = row
        launch_maps.append(mapping)

    launch_ids = set(launch_maps[0])
    if any(set(mapping) != launch_ids for mapping in launch_maps[1:]):
        raise ValueError(f"{model}: launch sets differ across repeats")

    merged_launches = []
    metric_ranges = {metric: [] for metric in METRIC_FIELDS}
    for launch_id in sorted(launch_ids):
        rows = [mapping[launch_id] for mapping in launch_maps]
        descriptor = launch_descriptor(rows[0])
        if any(launch_descriptor(row) != descriptor for row in rows[1:]):
            raise ValueError(
                f"{model}: launch {launch_id} identity/geometry differs"
            )
        merged = dict(zip(DESCRIPTOR_FIELDS, descriptor))
        merged["metrics"] = {}
        for metric in METRIC_FIELDS:
            values = []
            for repeat, row in enumerate(rows):
                value = (row.get("metrics") or {}).get(metric)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"{model}: repeat {repeat}, launch {launch_id} "
                        f"has invalid {metric}: {value!r}"
                    )
                values.append(float(value))
            merged["metrics"][metric] = statistics.median(values)
            metric_ranges[metric].append(relative_range(values))
        merged_launches.append(merged)

    output = {
        "schema_version": 2,
        "identity": identity,
        "kernels": merged_launches,
        "aggregation": {
            "method": "identity-checked per-launch median",
            "repeat_count": REPEAT_COUNT,
            "source_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in cache_paths
            ],
        },
    }
    write_json_atomic(output_path, output)
    return {
        "model": model,
        "model_class": identity["model_class"],
        "repeat_count": REPEAT_COUNT,
        "kernel_launches": len(merged_launches),
        "profile_sha256": identity["profile_sha256"],
        "fx_code_sha256": identity["fx_code_sha256"],
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "kernel_launch_p95_relative_range": {
            metric: p95(values) for metric, values in metric_ranges.items()
        },
        "infinite_relative_range_count": {
            metric: sum(not math.isfinite(value) for value in values)
            for metric, values in metric_ranges.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("caches", type=Path, nargs=REPEAT_COUNT)
    args = parser.parse_args()
    report = merge_repeated_caches(
        model=args.model,
        cache_paths=[path.resolve() for path in args.caches],
        output_path=args.output.resolve(),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
