#!/usr/bin/env python3
"""Seven-model registry shared by the multi-Janus chapter-3 experiments."""

from __future__ import annotations

import os
import sys


MODEL_CHOICES = (
    "GoogLeNet",
    "Inception3",
    "NASNetALarge",
    "ConvNeXt",
    "DeepFM",
    "BertModel",
    "YOLOv8x",
)


def _require_path(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def load_model(name: str, batch_size: int = 1):
    """Load the same seven batch-1 model wrappers used by the prior chapters."""
    import torch

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    torch.manual_seed(20260827)
    torch.cuda.manual_seed_all(20260827)

    if name == "GoogLeNet":
        import torchvision

        model = torchvision.models.googlenet(weights=None).eval().cuda()
        inputs = (torch.randn(batch_size, 3, 224, 224, device="cuda"),)
    elif name == "Inception3":
        import torchvision

        model = torchvision.models.inception_v3(weights=None).eval().cuda()
        inputs = (torch.randn(batch_size, 3, 299, 299, device="cuda"),)
    elif name == "NASNetALarge":
        import pretrainedmodels

        model = pretrainedmodels.__dict__["nasnetalarge"](
            num_classes=1000, pretrained=None
        ).eval().cuda()
        inputs = (torch.randn(batch_size, 3, 331, 331, device="cuda"),)
    elif name == "ConvNeXt":
        import torchvision

        model = torchvision.models.convnext_base(weights=None).eval().cuda()
        inputs = (torch.randn(batch_size, 3, 224, 224, device="cuda"),)
    elif name == "DeepFM":
        # NCF.py lives beside the experiment entry point in examples/.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from NCF import DeepFM

        cate = [100 * (index + 1) for index in range(32)]
        model = DeepFM(
            cate,
            16,
            emb_size=8,
            hid_dims=[256, 128],
            num_classes=1,
            dropout=[0.2, 0.2],
        ).eval().cuda()
        inputs = (
            torch.stack(
                [
                    torch.randint(0, cardinality, (batch_size,), device="cuda")
                    for cardinality in cate
                ],
                dim=1,
            ),
            torch.rand(batch_size, 16, device="cuda"),
        )
    elif name == "BertModel":
        from transformers import BertModel

        model_path = _require_path(
            os.getenv("JANUS_BERT_MODEL_PATH", "/public_0/ZYF/model/bert-base"),
            "BERT checkpoint",
        )
        model = BertModel.from_pretrained(model_path).eval().cuda()
        inputs = (
            torch.randint(
                0, 30000, (batch_size, 16), dtype=torch.long, device="cuda"
            ),
            torch.ones((batch_size, 16), dtype=torch.long, device="cuda"),
        )
    elif name == "YOLOv8x":
        import torch
        from ultralytics import YOLO

        checkpoint = _require_path(
            os.getenv(
                "JANUS_YOLO_MODEL_PATH", "/public_0/ZYF/model/YOLOv8/yolov8x.pt"
            ),
            "YOLOv8x checkpoint",
        )
        inner = YOLO(checkpoint).model.eval().cuda()

        class BackboneWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.backbone = model.model[0]

            def forward(self, value):
                return self.backbone(value)

        model = BackboneWrapper(inner).eval().cuda()
        inputs = (torch.randn(batch_size, 3, 320, 320, device="cuda"),)
    else:
        raise ValueError(f"unknown model: {name}")
    return model, inputs


def tensor_leaves(value):
    """Flatten tensor-bearing PyTorch outputs without depending on one model type."""
    import torch

    if torch.is_tensor(value):
        return [value]
    if hasattr(value, "to_tuple") and callable(value.to_tuple):
        return tensor_leaves(value.to_tuple())
    if isinstance(value, dict):
        leaves = []
        for key in sorted(value):
            leaves.extend(tensor_leaves(value[key]))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for item in value:
            leaves.extend(tensor_leaves(item))
        return leaves
    return []


def clone_tensor_leaves(value):
    return [tensor.detach().clone() for tensor in tensor_leaves(value)]


def compare_outputs(reference_leaves, actual):
    """Fail closed when a captured graph changes the model output."""
    import torch

    observed = tensor_leaves(actual)
    if len(reference_leaves) != len(observed):
        raise AssertionError(
            f"tensor leaf count differs: eager={len(reference_leaves)} "
            f"graph={len(observed)}"
        )
    max_abs = 0.0
    max_rel = 0.0
    for expected, graph in zip(reference_leaves, observed):
        torch.testing.assert_close(graph, expected, rtol=1e-4, atol=1e-5)
        difference = (graph - expected).abs()
        if difference.numel():
            max_abs = max(max_abs, float(difference.max().item()))
            denominator = expected.abs().clamp_min(1e-8)
            max_rel = max(
                max_rel, float((difference / denominator).max().item())
            )
    return {
        "tensor_leaves": len(reference_leaves),
        "max_abs": max_abs,
        "max_rel": max_rel,
    }
