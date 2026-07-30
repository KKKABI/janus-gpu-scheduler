#!/usr/bin/env python3
"""Small, deterministic inference target for Nsight Compute.

Run this program under ``ncu --profile-from-start off``.  Model loading and
warm-up happen before ``cudaProfilerStart`` so the report contains exactly one
measured inference.
"""

from __future__ import annotations

import argparse
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("yolov8x",), required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.model == "yolov8x":
        from ultralytics import YOLO

        model = YOLO("/public_0/ZYF/model/YOLOv8/yolov8x.pt").model
        inputs = (torch.randn((1, 3, 320, 320), device="cuda:0"),)
    else:  # pragma: no cover - argparse validates the choice
        raise ValueError(f"unsupported model: {args.model}")

    model = model.to("cuda:0").eval()
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(*inputs)
        torch.cuda.synchronize()

        torch.cuda.cudart().cudaProfilerStart()
        model(*inputs)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
