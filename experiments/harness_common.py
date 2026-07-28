"""Pure helpers for the reproducible Janus benchmark harness."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "experiments" / "repro_config.json"
PRIMARY_VARIANTS = ("Baseline", "TD+Janus", "Static+Cos", "TD+Cos", "Static+DRT", "TD+DRT")


@dataclass(frozen=True)
class Task:
    model: str
    variant: str
    alpha: float | None
    repeat_index: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["repeat_number"] = self.repeat_index + 1
        data["task_id"] = task_id(self)
        return data


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


def task_id(task: Task) -> str:
    alpha = "none" if task.alpha is None else str(task.alpha).replace(".", "p")
    return f"r{task.repeat_index + 1:02d}__{slug(task.model)}__{slug(task.variant)}__a-{alpha}"


def expand_tasks(config: dict[str, Any], models: Iterable[str], variants: Iterable[str], repeats: int) -> list[Task]:
    model_names, variant_names = list(models), list(variants)
    specs = {item["label"]: item for item in config["variants"]}
    unknown_models = sorted(set(model_names) - set(config["models"]))
    unknown_variants = sorted(set(variant_names) - set(specs))
    if unknown_models or unknown_variants:
        raise ValueError(f"unknown models={unknown_models}, variants={unknown_variants}")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    all_tasks: list[Task] = []
    base_seed = int(config["measurement"]["seed"])
    for repeat_index in range(repeats):
        tasks: list[Task] = []
        for model in model_names:
            for variant in variant_names:
                alpha_spec = specs[variant].get("alpha")
                alphas = config["alpha_grid"] if alpha_spec == "alpha_grid" else [None]
                tasks.extend(Task(model, variant, alpha, repeat_index) for alpha in alphas)
        random.Random(base_seed + repeat_index).shuffle(tasks)
        all_tasks.extend(tasks)
    return all_tasks


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected, name = line.split(maxsplit=1)
            entries.append((expected, name.strip()))
    return entries


def verify_manifest(path: Path, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    checks = []
    for expected, name in read_manifest(path):
        target = Path(name)
        if not target.is_absolute():
            target = repo_root / target
        actual = sha256_file(target) if target.is_file() else None
        checks.append({"path": str(target), "expected_sha256": expected, "actual_sha256": actual, "ok": actual == expected})
    failed = [item["path"] for item in checks if not item["ok"]]
    if failed:
        raise RuntimeError(f"checksum verification failed: {failed}")
    return checks


def command_output(args: list[str], cwd: Path = REPO_ROOT, check: bool = True) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if check and result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {args!r}: {result.stderr.strip()}")
    return result.stdout.strip()


def git_metadata(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"], repo_root),
        "branch": command_output(["git", "branch", "--show-current"], repo_root),
        "status_porcelain": command_output(["git", "status", "--porcelain=v1"], repo_root),
        "worktree": str(repo_root),
    }


def gpu_snapshot() -> dict[str, Any]:
    fields = ["timestamp", "name", "uuid", "driver_version", "temperature.gpu", "power.draw", "clocks.sm", "clocks.mem", "memory.used", "memory.total", "utilization.gpu"]
    raw = command_output(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"])
    rows = [dict(zip(fields, [value.strip() for value in line.split(",")])) for line in raw.splitlines()]
    return {"fields": fields, "rows": rows, "raw": raw}


def gpu_compute_processes() -> list[str]:
    output = command_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], check=False)
    return [line.strip() for line in output.splitlines() if line.strip()]


def require_idle_gpu() -> None:
    processes = gpu_compute_processes()
    if processes:
        raise RuntimeError(f"GPU has competing compute processes: {processes}")


def expected_profile_path(model: Any, inputs: tuple[Any, ...]) -> Path:
    stem = model.__class__.__name__ + "".join("_" + str(tensor.shape) for tensor in inputs)
    return REPO_ROOT / "Opara" / "profile_result" / f"{stem}.pt.trace.json"


def stats_from_samples(samples: list[float]) -> dict[str, float | int]:
    if not samples:
        raise ValueError("no timing samples")
    median = statistics.median(samples)
    return {"count": len(samples), "median_ms": median, "mean_ms": statistics.mean(samples), "sample_std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0, "mad_ms": statistics.median(abs(value - median) for value in samples), "min_ms": min(samples), "max_ms": max(samples)}


def aggregate_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float | None], list[float]] = {}
    for record in records:
        key = (record["task"]["model"], record["task"]["variant"], record["task"]["alpha"])
        grouped.setdefault(key, []).append(float(record["timing"]["statistics"]["median_ms"]))
    summary = []
    for (model, variant, alpha), medians in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else item[0][2])):
        stats = stats_from_samples(medians)
        summary.append({"model": model, "variant": variant, "alpha": alpha, "process_medians_ms": medians, "median_of_process_medians_ms": stats["median_ms"], "mean_of_process_medians_ms": stats["mean_ms"], "sample_std_of_process_medians_ms": stats["sample_std_ms"], "mad_of_process_medians_ms": stats["mad_ms"], "completed_repeats": len(medians)})
    return summary


def runtime_metadata() -> dict[str, Any]:
    return {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(), "python_executable": sys.executable, "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"), "conda_prefix": os.environ.get("CONDA_PREFIX"), "git": git_metadata(), "gpu": gpu_snapshot()}


def validate_run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError("run_id may contain only letters, digits, dot, underscore and dash")
    return value
