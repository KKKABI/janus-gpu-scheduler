#!/usr/bin/env python3
"""Expand, execute and aggregate the frozen Janus benchmark matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness_common import PRIMARY_VARIANTS, REPO_ROOT, aggregate_results, expand_tasks, git_metadata, gpu_snapshot, load_config, runtime_metadata, task_id, validate_run_id, verify_manifest, write_json_atomic


def parse_args() -> argparse.Namespace:
    config = load_config()
    primary_models = [name for name, spec in config["models"].items() if spec.get("role") == "primary"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=primary_models)
    parser.add_argument("--variants", nargs="+", default=list(PRIMARY_VARIANTS))
    parser.add_argument("--repeats", type=int, default=int(config["measurement"]["independent_process_repeats"]))
    parser.add_argument("--run-id")
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "experiments" / "results")
    parser.add_argument("--dry-run", action="store_true", help="print the deterministic plan; write nothing")
    parser.add_argument("--preflight", action="store_true", help="verify config/assets; do not run GPU inference")
    return parser.parse_args()


def manifest_paths() -> list[Path]:
    base = REPO_ROOT / "experiments"
    return [base / "profile_manifest.sha256", base / "model_asset_manifest.sha256", base / "model_reference_manifest.sha256"]


def preflight() -> dict[str, Any]:
    config = load_config(); expected = Path(config["environment"]["python_executable"]).resolve()
    if not Path(sys.executable).resolve().samefile(expected): raise RuntimeError(f"wrong interpreter: {sys.executable}; expected {expected}")
    checks = {path.name: verify_manifest(path) for path in manifest_paths()}
    git = git_metadata()
    if git["status_porcelain"]: raise RuntimeError("experiment worktree must be clean before a run")
    return {"schema_version": 1, "runtime": runtime_metadata(), "config": config, "manifest_checks": checks}


def default_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{git_metadata()['commit'][:8]}"


def run_task(task, run_root: Path) -> int:
    directory = run_root / "tasks" / task_id(task); directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, str(REPO_ROOT / "experiments" / "run_one.py"), "--model", task.model, "--variant", task.variant, "--alpha", "none" if task.alpha is None else str(task.alpha), "--repeat-index", str(task.repeat_index), "--output-dir", str(directory)]
    write_json_atomic(directory / "invocation.json", {"command": command, "task": task.to_dict()})
    with (directory / "stdout.log").open("x", encoding="utf-8") as stdout, (directory / "stderr.log").open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=False)
    write_json_atomic(directory / "exit.json", {"returncode": completed.returncode})
    return completed.returncode


def main() -> int:
    args = parse_args(); config = load_config(); tasks = expand_tasks(config, args.models, args.variants, args.repeats)
    plan = {"schema_version": 1, "models": args.models, "variants": args.variants, "repeats": args.repeats, "task_count": len(tasks), "tasks": [task.to_dict() for task in tasks]}
    if args.dry_run:
        json.dump(plan, sys.stdout, ensure_ascii=False, indent=2); sys.stdout.write("\n"); return 0
    audit = preflight()
    if args.preflight:
        print(json.dumps({"status": "ok", "checked_manifests": list(audit["manifest_checks"])}, indent=2)); return 0
    run_id = validate_run_id(args.run_id or default_run_id()); run_root = args.results_root.resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=False); write_json_atomic(run_root / "plan.json", plan); write_json_atomic(run_root / "audit.json", audit)
    write_json_atomic(run_root / "run_status.json", {"status": "running", "completed": 0, "total": len(tasks)})
    failures = []
    for index, task in enumerate(tasks, start=1):
        returncode = run_task(task, run_root)
        if returncode: failures.append({"task": task.to_dict(), "returncode": returncode})
        write_json_atomic(run_root / "run_status.json", {"status": "running", "completed": index, "total": len(tasks), "failures": failures})
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((run_root / "tasks").glob("*/result.json"))]
    summary = {"schema_version": 1, "status": "failed" if failures else "completed", "planned_tasks": len(tasks), "completed_tasks": len(records), "failed_tasks": failures, "aggregates": aggregate_results(records), "finished_gpu": gpu_snapshot()}
    write_json_atomic(run_root / "summary.json", summary); write_json_atomic(run_root / "run_status.json", summary)
    return 1 if failures else 0


if __name__ == "__main__": raise SystemExit(main())
