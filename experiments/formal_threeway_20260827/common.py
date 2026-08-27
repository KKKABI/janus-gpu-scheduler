"""Shared identities for the frozen seven-model formal experiment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


MODELS = (
    "GoogLeNet",
    "Inception-v3",
    "NASNet",
    "YOLOv8x",
    "ConvNeXt",
    "DeepFM",
    "BERT",
)

MODEL_SLUGS = {
    "GoogLeNet": "googlenet",
    "Inception-v3": "inception_v3",
    "NASNet": "nasnet",
    "YOLOv8x": "yolov8x_backbone",
    "ConvNeXt": "convnext",
    "DeepFM": "deepfm",
    "BERT": "bert",
}

MODEL_CLASSES = {
    "GoogLeNet": "GoogLeNet",
    "Inception-v3": "Inception3",
    "NASNet": "NASNetALarge",
    "YOLOv8x": "BackboneWrapper",
    "ConvNeXt": "ConvNeXt",
    "DeepFM": "DeepFM",
    "BERT": "BertModel",
}

DISPLAY_NAMES = {
    **{model: model for model in MODELS},
    "YOLOv8x": "YOLOv8x BackboneWrapper",
}

POLICIES = (
    "janus",
    "newtd_drt",
    "newtd_ncu_drt",
)

POLICY_LABELS = {
    "janus": "Original Janus",
    "newtd_drt": "NewTD+DRT",
    "newtd_ncu_drt": "NewTD+NCU-DRT",
}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_models(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    unknown = sorted(set(values) - set(MODELS))
    if unknown:
        raise ValueError(f"unsupported models: {unknown}")
    return tuple(values)


def require_empty_output(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"refusing to reuse output directory: {resolved}")
    resolved.mkdir(parents=True)
    return resolved
