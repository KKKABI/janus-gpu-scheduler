#!/usr/bin/env python3
"""Create one profile-SHA-checked NewTD solo root for the formal runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS), str(REPO)]

from common import MODELS, MODEL_SLUGS, require_empty_output, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output = require_empty_output(args.output_root)

    from harness_common import load_config

    config = load_config()
    wanted = [model for model in MODELS if model != "YOLOv8x"]
    matches = {model: [] for model in wanted}
    for path in source.rglob("result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = payload.get("model")
        if model in matches:
            matches[model].append((path, payload))

    records = []
    for model in wanted:
        candidates = matches[model]
        if len(candidates) != 1:
            raise RuntimeError(
                f"expected one historical solo result for {model}, got "
                f"{[str(path) for path, _ in candidates]}"
            )
        path, payload = candidates[0]
        profile = REPO / "Opara" / "profile_result" / config["models"][model][
            "profile_file"
        ]
        expected_sha = sha256_file(profile)
        if payload.get("profile_sha256") != expected_sha:
            raise RuntimeError(
                f"{model}: solo/profile SHA mismatch: "
                f"{payload.get('profile_sha256')} != {expected_sha}"
            )
        target = output / MODEL_SLUGS[model] / "result.json"
        target.parent.mkdir(parents=True)
        shutil.copy2(path, target)
        records.append(
            {
                "model": model,
                "source": str(path),
                "source_sha256": sha256_file(path),
                "copied_to": str(target),
                "profile_sha256": expected_sha,
                "auditable_count": payload.get("auditable_count"),
            }
        )
    (output / "prepared_sources.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "awaiting YOLOv8x BackboneWrapper solo capture",
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
