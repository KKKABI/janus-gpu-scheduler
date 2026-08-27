#!/usr/bin/env python3
"""CPU-only, auditable preflight for the three formal Bash entrypoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from common import sha256_file, write_json_atomic


SCRIPT_NAMES = (
    "run_stage_a_profiles.sh",
    "run_stage_b_latency.sh",
    "run_stage_c_same_ready.sh",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        raise ValueError("--expected-commit must be a lowercase 40-character SHA")
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != args.expected_commit:
        raise RuntimeError(
            f"formal commit mismatch: actual={actual}, "
            f"expected={args.expected_commit}"
        )
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    if status:
        raise RuntimeError("formal repository is dirty")

    script_root = repo / "experiments" / "formal_threeway_20260827"
    scripts = [script_root / name for name in SCRIPT_NAMES]
    if any(not path.is_file() for path in scripts):
        raise FileNotFoundError("one or more formal stage entrypoints are missing")
    checked = subprocess.run(
        ["bash", "-n", *map(str, scripts)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if checked.returncode:
        raise RuntimeError("bash -n failed:\n" + checked.stdout)
    bash_version = subprocess.check_output(
        ["bash", "--version"], text=True
    ).splitlines()[0]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "gpu_used": False,
        "repo": str(repo),
        "expected_commit": args.expected_commit,
        "actual_commit": actual,
        "bash_version": bash_version,
        "command": ["bash", "-n", *map(str, scripts)],
        "scripts": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in scripts
        ],
    }
    write_json_atomic(output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
