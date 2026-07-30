#!/usr/bin/env python3
"""Build an Opara NCU JSON cache from an exported Nsight Compute report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Opara.ncu_profiler import parse_ncu_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model-class", required=True)
    parser.add_argument(
        "--ncu",
        type=Path,
        default=Path("/usr/local/cuda-12.5/bin/ncu"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = args.report.resolve()
    if not report.is_file():
        raise FileNotFoundError(report)

    completed = subprocess.run(
        [
            str(args.ncu),
            "--import",
            str(report),
            "--csv",
            "--print-summary",
            "per-kernel",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = parse_ncu_csv(completed.stdout)
    if not data:
        raise RuntimeError(f"no supported NCU metrics found in {report}")

    cache_path = REPO_ROOT / "Opara" / "ncu_result" / f"{args.model_class}.ncu.json"
    cache_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    nonzero = sum(
        1 for entry in data.values()
        if entry["dram_thru"] > 0 or entry["mem_thru"] > 0
    )
    print(json.dumps({
        "cache": str(cache_path),
        "kernel_count": len(data),
        "nonzero_memory_kernel_count": nonzero,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
