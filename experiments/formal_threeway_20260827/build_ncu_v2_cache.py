#!/usr/bin/env python3
"""Convert raw, per-launch NCU CSV into an identity-bound v2 cache."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
import re


def number(value):
    text = str(value or "0").replace(",", "").strip()
    return float(text) if text else 0.0


def duration_ns(value, unit):
    scale = {
        "second": 1e9,
        "msecond": 1e6,
        "usecond": 1e3,
        "nsecond": 1.0,
        "s": 1e9,
        "ms": 1e6,
        "us": 1e3,
        "ns": 1.0,
    }.get(str(unit).strip(), 1.0)
    return number(value) * scale


def read_rows(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = next(
        index for index, line in enumerate(lines)
        if "Kernel Name" in line and "Metric Name" in line and "ID" in line
    )
    return csv.DictReader(lines[header:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--identity-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    launches = OrderedDict()
    for row in read_rows(args.raw_csv):
        launch_id = row.get("ID", "")
        kernel_name = row.get("Kernel Name", "")
        metric_name = row.get("Metric Name", "")
        if not launch_id or not kernel_name or not metric_name:
            continue
        nvtx_text = " ".join(
            value for key, value in row.items()
            if key and "Push/Pop_Range" in key and value
        )
        op_match = re.search(r"JANUS_OP:([A-Za-z0-9_]+)", nvtx_text)
        launch = launches.setdefault(launch_id, {
            "launch_id": int(launch_id),
            "op_name": op_match.group(1) if op_match else None,
            "name": kernel_name,
            "grid_size": 0,
            "block_size": 0,
            "metrics": {},
        })
        value = row.get("Metric Value", "0")
        unit = row.get("Metric Unit", "")
        if metric_name == "Grid Size":
            launch["grid_size"] = int(number(value))
        elif metric_name == "Block Size":
            launch["block_size"] = int(number(value))
        elif metric_name == "Duration":
            launch["metrics"]["dur_ns"] = duration_ns(value, unit)
        elif metric_name == "DRAM Throughput":
            launch["metrics"]["dram_thru"] = number(value)
        elif metric_name == "L2 Cache Throughput":
            launch["metrics"]["l2_thru"] = number(value)
        elif metric_name in {"Compute (SM) Throughput", "Compute (SM) [%]"}:
            launch["metrics"]["comp_thru"] = number(value)
        elif metric_name in {"Memory [%]", "Memory Throughput"}:
            launch["metrics"]["mem_thru"] = number(value)

    kernels = sorted(launches.values(), key=lambda item: item["launch_id"])
    incomplete = [
        item["launch_id"] for item in kernels
        if not item["grid_size"] or not item["block_size"]
        or not {"dram_thru", "l2_thru", "comp_thru"}.issubset(item["metrics"])
    ]
    if incomplete:
        raise RuntimeError(
            f"{len(incomplete)} launches lack geometry/throughput metrics; first IDs={incomplete[:10]}"
        )
    missing_op = [item["launch_id"] for item in kernels if not item["op_name"]]
    if missing_op:
        raise RuntimeError(
            f"{len(missing_op)} launches lack JANUS_OP NVTX identity; first IDs={missing_op[:10]}"
        )

    payload = {
        "schema_version": 2,
        "identity": json.loads(args.identity_json.read_text(encoding="utf-8")),
        "kernels": kernels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "kernel_launches": len(kernels),
        "schema_version": 2,
    }))


if __name__ == "__main__":
    main()
