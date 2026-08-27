#!/usr/bin/env python3
"""Read-only structural verification for formal NCU and NewTD assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS), str(REPO)]

from common import (
    DISPLAY_NAMES,
    MODEL_CLASSES,
    MODELS,
    sha256_file,
    write_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ncu-cache-dir", type=Path, required=True)
    parser.add_argument("--solo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cache_dir = args.ncu_cache_dir.resolve()
    solo_root = args.solo_root.resolve()
    if not cache_dir.is_dir() or not solo_root.is_dir():
        raise FileNotFoundError("NCU cache directory or solo root is missing")

    from harness_common import load_config

    config = load_config()
    records = []
    solo_candidates = []
    for path in solo_root.rglob("result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("model") in MODELS:
            solo_candidates.append((path, payload))

    for model in MODELS:
        class_name = MODEL_CLASSES[model]
        profile = REPO / "Opara" / "profile_result" / config["models"][model][
            "profile_file"
        ]
        if not profile.is_file():
            raise FileNotFoundError(f"{model}: profile missing: {profile}")
        profile_sha = sha256_file(profile)

        cache = cache_dir / f"{class_name}.ncu.v2.json"
        if not cache.is_file():
            raise FileNotFoundError(f"{model}: cache missing: {cache}")
        cache_payload = json.loads(cache.read_text(encoding="utf-8"))
        identity = cache_payload.get("identity") or {}
        if cache_payload.get("schema_version") != 2:
            raise RuntimeError(f"{model}: cache is not schema v2")
        aggregation = cache_payload.get("aggregation") or {}
        if (
            aggregation.get("method") != "identity-checked per-launch median"
            or aggregation.get("repeat_count") != 3
            or len(aggregation.get("source_files", [])) != 3
        ):
            raise RuntimeError(
                f"{model}: formal cache is not a three-repeat median: "
                f"{aggregation}"
            )
        for source in aggregation["source_files"]:
            source_path = Path(source.get("path", ""))
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"{model}: median source cache is missing: {source_path}"
                )
            if sha256_file(source_path) != source.get("sha256"):
                raise RuntimeError(
                    f"{model}: median source cache SHA differs: {source_path}"
                )
        if identity.get("model_class") != class_name:
            raise RuntimeError(f"{model}: NCU model class mismatch")
        if identity.get("profile_sha256") != profile_sha:
            raise RuntimeError(
                f"{model}: NCU/profile SHA mismatch: "
                f"{identity.get('profile_sha256')} != {profile_sha}"
            )
        if not cache_payload.get("kernels") or any(
            not row.get("op_name") for row in cache_payload["kernels"]
        ):
            raise RuntimeError(f"{model}: NCU cache has missing OP identities")

        solo = [
            (path, payload)
            for path, payload in solo_candidates
            if payload.get("model") == model
        ]
        if len(solo) != 1:
            raise RuntimeError(
                f"{model}: expected one solo result, got "
                f"{[str(path) for path, _ in solo]}"
            )
        solo_path, solo_payload = solo[0]
        if solo_payload.get("profile_sha256") != profile_sha:
            raise RuntimeError(f"{model}: solo/profile SHA mismatch")
        auditable = [
            row
            for row in solo_payload.get("operators", [])
            if row.get("auditable") and int(row.get("span_duration_ns", 0)) > 0
        ]
        if not auditable:
            raise RuntimeError(f"{model}: solo profile has no auditable OP")
        records.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "model_class": class_name,
                "profile": str(profile),
                "profile_sha256": profile_sha,
                "ncu_cache": str(cache),
                "ncu_cache_sha256": sha256_file(cache),
                "ncu_kernel_launches": len(cache_payload["kernels"]),
                "ncu_aggregation": aggregation,
                "ncu_fx_code_sha256": identity.get("fx_code_sha256"),
                "solo_result": str(solo_path),
                "solo_result_sha256": sha256_file(solo_path),
                "solo_auditable_operators": len(auditable),
            }
        )

    legacy = sorted(cache_dir.glob("*.ncu.json"))
    if legacy:
        raise RuntimeError(f"formal cache directory contains legacy caches: {legacy}")
    payload = {
        "schema_version": 1,
        "status": "structurally_valid",
        "important_boundary": (
            "runtime/GPU/FX identity is checked again by each fail-closed "
            "NewTD+NCU-DRT process"
        ),
        "records": records,
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"status": payload["status"], "models": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
