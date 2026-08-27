"""Small model wrappers shared by the frozen formal experiment runners."""

from __future__ import annotations


def build_yolov8x_backbone():
    """Return the exact YOLOv8x BackboneWrapper used by the NCU-v2 assets.

    The complete Ultralytics DetectionModel did not pass the established
    output-consistency boundary.  Formal seven-model tables therefore use and
    label this first-stage backbone wrapper explicitly.
    """
    import torch
    from ultralytics import YOLO

    class BackboneWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.backbone = model.model[0]

        def forward(self, value):
            return self.backbone(value)

    checkpoint = "/public_0/ZYF/model/YOLOv8/yolov8x.pt"
    inner = YOLO(checkpoint).model.eval()
    model = BackboneWrapper(inner)
    inputs = (torch.randn((1, 3, 320, 320), device="cuda:0"),)
    return model.to("cuda:0").eval(), inputs
