#!/usr/bin/env python3
"""Build an all-FX target manifest for the frozen YOLOv8x backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
REPO = EXPERIMENTS.parent
sys.path[:0] = [str(HERE), str(EXPERIMENTS), str(REPO)]

from common import sha256_file
from profile_ncu_target import trace_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    import torch
    from harness_common import expected_profile_path, load_config
    from run_one import load_model_and_inputs, seed_everything

    config = load_config()
    seed_everything(int(config["measurement"]["seed"]))
    model, inputs = load_model_and_inputs("YOLOv8x", config)
    if model.__class__.__name__ != "BackboneWrapper":
        raise RuntimeError(
            f"formal YOLO identity must be BackboneWrapper, got "
            f"{model.__class__.__name__}"
        )
    profile = expected_profile_path(model, inputs)
    configured = config["models"]["YOLOv8x"]["profile_file"]
    if profile.name != configured or not profile.is_file():
        raise RuntimeError(
            f"YOLO profile mismatch: runtime={profile}; configured={configured}"
        )
    backend = config["models"]["YOLOv8x"].get(
        "capture_backend", "dynamo_explain"
    )
    with torch.inference_mode():
        model(*inputs)
    module = trace_model(model, inputs, backend)
    targets = [
        node.name
        for node in module.graph.nodes
        if node.op not in {"placeholder", "get_attr", "output"}
    ]
    if not targets:
        raise RuntimeError("YOLO BackboneWrapper FX graph has no target nodes")
    head = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    payload = {
        "schema_version": 1,
        "git_head": head,
        "purpose": "all FX nodes for NewTD solo-duration lookup",
        "model_scope": "YOLOv8x BackboneWrapper only",
        "source_profile_sha256_by_model": {
            "YOLOv8x": sha256_file(profile)
        },
        "fx_code_sha256": hashlib.sha256(
            module.code.encode("utf-8")
        ).hexdigest(),
        "target_count": len(targets),
        "cases": [
            {
                "case_id": "yolov8x_backbone_all_fx",
                "model": "YOLOv8x",
                "group": targets,
            }
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
