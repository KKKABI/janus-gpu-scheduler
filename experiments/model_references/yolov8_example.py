import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from Opara import GraphCapturer
from googlenet_example import run_torch_model, run_sequence_graph, run_parallel_graph

# 显式加载本地权重
from ultralytics import YOLO
local_model_path = '/public_0/ZYF/model/YOLOv8/yolov8x.pt'
yolo_model = YOLO(local_model_path).model  # 直接取 .model 避免 hub 下载
yolo_model = yolo_model.eval().cuda()

def flush_cache():
    cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')
    cache.zero_()

if __name__ == '__main__':
    warm_ups = 10
    iterations = 100
    img_size = 320

    # 输入模拟图像
    x = torch.randn(1, 3, img_size, img_size, device='cuda')

    inputs = (x,)

    # PyTorch 原始推理
    print("\n===== PyTorch原始模型性能测试 =====")
    with torch.no_grad():
        torch_out = run_torch_model(yolo_model, inputs, iterations, warm_ups)

    # CUDA Graph 顺序执行
    print("\n===== CUDA Graph顺序执行性能测试 =====")
    with torch.no_grad():
        run_sequence_graph(yolo_model, inputs, iterations, warm_ups, 0, iterations)

    # Opara 并行执行 feature map
    print("\n===== Opara并行执行性能测试 =====")
    with torch.no_grad():
        # 只捕获 backbone 输出
        class BackboneWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.backbone = model.model[0]  # yolov8n.model[0] 是 backbone
            def forward(self, x):
                return self.backbone(x)

        backbone_model = BackboneWrapper(yolo_model).eval().cuda()
        Opara = GraphCapturer.capturer(inputs, backbone_model)
        opara_out = run_parallel_graph(Opara, inputs, iterations, warm_ups, 0, iterations)

    # 输出一致性检查
    print("\n===== 输出检查 =====")
    torch_feat = backbone_model(x).detach().cpu()
    if isinstance(opara_out, (list, tuple)):
        opara_feat = opara_out[0].detach().cpu()
    else:
        opara_feat = opara_out.detach().cpu()

    print("PyTorch 输出 shape:", torch_feat.shape)
    print("Opara 输出 shape  :", opara_feat.shape)
    diff = torch.abs(torch_feat - opara_feat)
    print(f"最大绝对误差: {diff.max():.6f}, 平均绝对误差: {diff.mean():.6f}")
