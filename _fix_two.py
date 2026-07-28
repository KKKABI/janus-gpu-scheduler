"""Fix YOLOv8x and ConvNeXt with correct model configs"""
import io, sys, os, numpy as np
from contextlib import redirect_stdout
sys.path.insert(0, '.')

import torch
torch.cuda.empty_cache()

model_name = sys.argv[1]
out_file = sys.argv[2]

if model_name == 'YOLOv8x':
    from ultralytics import YOLO
    yolo = YOLO('/public_0/ZYF/model/YOLOv8/yolov8x.pt').model.eval().cuda()
    # _BackboneWrapper from benchmark_all.py
    class _BackboneWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model.model
        def forward(self, x):
            y = self.model(x)
            return y if isinstance(y, (list, tuple)) else [y]
    model = _BackboneWrapper(yolo).eval().cuda()
    inputs = (torch.randn(1, 3, 320, 320, device='cuda:0'),)
elif model_name == 'ConvNeXt':
    import torchvision
    model = torchvision.models.convnext_base(weights=None).eval().cuda()
    inputs = (torch.randn(1, 3, 224, 224, device='cuda:0'),)

from Opara import GraphCapturer
cache = torch.empty(int(4*(1024**2)), dtype=torch.int8, device='cuda')

# Baseline
with redirect_stdout(io.StringIO()):
    runner = GraphCapturer.capturer(inputs, model, copy_outputs=False)
for _ in range(50): cache.zero_(); runner(*inputs)
torch.cuda.synchronize()
times=[]
for _ in range(100):
    cache.zero_()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();runner(*inputs);e.record();torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
row = f"{model_name},Baseline,0.9,{np.median(times):.4f},{np.min(times):.4f},{np.max(times):.4f}"
with open(out_file,'a') as f: f.write(row+'\n')
print(f"Baseline: {np.median(times):.4f}", file=sys.stderr)

# Cosine + MinRes
for sname, smode in [('Cosine','cosine'),('MinRes','min_resource')]:
    for alpha in [0.9,0.8,0.5,0.2]:
        with redirect_stdout(io.StringIO()):
            runner = GraphCapturer.capturer(inputs, model, copy_outputs=False,
                alpha=alpha, selection_mode=smode, time_domain=True)
        for _ in range(50): cache.zero_(); runner(*inputs)
        torch.cuda.synchronize()
        times=[]
        for _ in range(100):
            cache.zero_()
            s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            s.record();runner(*inputs);e.record();torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        row = f"{model_name},{sname},{alpha},{np.median(times):.4f},{np.min(times):.4f},{np.max(times):.4f}"
        with open(out_file,'a') as f: f.write(row+'\n')
        print(f"  {sname} a={alpha}: {np.median(times):.4f}", file=sys.stderr)
