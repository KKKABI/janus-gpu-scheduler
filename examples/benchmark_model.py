"""Run one Janus model at one batch size in an isolated process."""

import argparse
import gc
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Opara import GraphCapturer


def build_googlenet(batch_size):
    import torchvision
    model = torchvision.models.googlenet().eval().cuda()
    inputs = (torch.randint(
        0, 256, (batch_size, 3, 224, 224),
        dtype=torch.float32, device="cuda"),)
    return model, inputs


def build_inception(batch_size):
    import torchvision
    model = torchvision.models.inception_v3(pretrained=True).eval().cuda()
    inputs = (torch.randint(
        0, 256, (batch_size, 3, 299, 299),
        dtype=torch.float32, device="cuda"),)
    return model, inputs


def build_nasnet(batch_size):
    import pretrainedmodels
    model = pretrainedmodels.__dict__["nasnetalarge"](
        num_classes=1000, pretrained="imagenet").eval().cuda()
    inputs = (torch.randint(
        0, 256, (batch_size, 3, 331, 331),
        dtype=torch.float32, device="cuda"),)
    return model, inputs


def build_deepfm(batch_size):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from NCF import DeepFM
    cate_fea_nuniqs = [100 * (i + 1) for i in range(32)]
    nume_fea_size = 16
    model = DeepFM(
        cate_fea_nuniqs,
        nume_fea_size,
        emb_size=8,
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
    ).eval().cuda()
    sparse = torch.randint(
        0, 100, (batch_size, len(cate_fea_nuniqs)), device="cuda")
    dense = torch.rand(batch_size, nume_fea_size, device="cuda")
    return model, (sparse, dense)


def build_bert(batch_size):
    bert_model_path = "/public_0/ZYF/model/bert-base"
    sys.path.append(bert_model_path)
    from transformers import BertModel
    model = BertModel.from_pretrained(bert_model_path).eval().cuda()
    input_ids = torch.randint(
        0, 30000, (batch_size, 16), dtype=torch.long, device="cuda")
    attention_mask = torch.ones(
        (batch_size, 16), dtype=torch.long, device="cuda")
    return model, (input_ids, attention_mask)


class BackboneWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.model[0]

    def forward(self, x):
        return self.backbone(x)


def build_yolo(batch_size):
    from ultralytics import YOLO
    yolo = YOLO("/public_0/ZYF/model/YOLOv8/yolov8x.pt").model.eval().cuda()
    model = BackboneWrapper(yolo).eval().cuda()
    inputs = (torch.randn(batch_size, 3, 320, 320, device="cuda"),)
    return model, inputs


def build_convnext(batch_size):
    import torchvision
    model = torchvision.models.convnext_base(
        pretrained=False).eval().cuda()
    inputs = (torch.randn(batch_size, 3, 224, 224, device="cuda"),)
    return model, inputs


BUILDERS = {
    "googlenet": build_googlenet,
    "inception": build_inception,
    "nasnet": build_nasnet,
    "deepfm": build_deepfm,
    "bert": build_bert,
    "yolo": build_yolo,
    "convnext": build_convnext,
}


def benchmark(runner, inputs, iterations):
    cache = torch.empty(
        int(4 * (1024 ** 2)), dtype=torch.int8, device="cuda")
    times = []
    for _ in range(iterations):
        cache.zero_()
        torch.cuda._sleep(1_000_000)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner(*inputs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.mean(times)), float(np.std(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=BUILDERS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--tune-iterations", type=int, default=100)
    parser.add_argument("--tune-repeats", type=int, default=5)
    args = parser.parse_args()

    iterations = args.iterations
    if iterations is None:
        iterations = 100 if args.model == "yolo" else 300

    result = {
        "model": args.model,
        "batch_size": args.batch_size,
        "selection_mode": os.getenv("JANUS_SELECTION_MODE", "default"),
        "iterations": iterations,
    }
    try:
        model, inputs = BUILDERS[args.model](args.batch_size)
        if args.autotune:
            runner = GraphCapturer.autotune_capturer(
                inputs,
                model,
                iterations=args.tune_iterations,
                repeats=args.tune_repeats,
            )
            result["selected_schedule"] = runner.janus_schedule
            result["tuning_measurements"] = runner.janus_tuning_measurements
        else:
            runner = GraphCapturer.capturer(inputs, model)
        mean_ms, std_ms = benchmark(runner, inputs, iterations)
        result.update({
            "status": "ok",
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        })
    except Exception as exc:
        result.update({
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False))
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("RESULT_JSON=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
