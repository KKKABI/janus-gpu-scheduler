"""Benchmark helper — called by benchmark_full.sh
Usage: python _bench_helper.py <model_name> <strategy> <alpha>
"""
import io, sys, os, numpy as np
sys.path.insert(0, '.')
from contextlib import redirect_stdout

import torch
torch.cuda.empty_cache()

# ── Model loading ──
model_name = sys.argv[1]

if model_name == 'GoogLeNet':
    import torchvision
    model = torchvision.models.googlenet(weights=None).eval().cuda()
    inputs = (torch.randn(1,3,224,224,device='cuda:0'),)
elif model_name == 'Inception-v3':
    import torchvision
    model = torchvision.models.inception_v3(weights=None).eval().cuda()
    inputs = (torch.randn(1,3,299,299,device='cuda:0'),)
elif model_name == 'ConvNeXt':
    import torchvision
    model = torchvision.models.convnext_tiny(weights=None).eval().cuda()
    inputs = (torch.randn(1,3,224,224,device='cuda:0'),)
elif model_name == 'DeepFM':
    sys.path.insert(0, '/public_0/LYX/janus/examples')
    from NCF import DeepFM
    cate = [100*(i+1) for i in range(32)]
    model = DeepFM(cate, 16, emb_size=8, hid_dims=[256,128], num_classes=1, dropout=[0.2,0.2]).eval().cuda()
    inputs = (torch.randint(0,100,(1,32),device='cuda'), torch.rand(1,16,device='cuda'))
elif model_name == 'BERT':
    sys.path.insert(0, '/public_0/ZYF/model/bert-base')
    from transformers import BertModel
    model = BertModel.from_pretrained('/public_0/ZYF/model/bert-base').eval().cuda()
    inputs = (torch.randint(0,30522,(1,16),device='cuda'),)
elif model_name == 'NASNet':
    import pretrainedmodels
    model = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet').eval().cuda()
    inputs = (torch.randn(1,3,331,331,device='cuda:0'),)
elif model_name == 'YOLOv8x':
    from ultralytics import YOLO
    yolo = YOLO('/public_0/ZYF/model/YOLOv8/yolov8x.pt').model.eval().cuda()
    class W(torch.nn.Module):
        def __init__(self,m): super().__init__(); self.m = m
        def forward(self,x): return self.m(x)
    model = W(yolo).eval().cuda()
    inputs = (torch.randn(1,3,640,640,device='cuda:0'),)
else:
    raise ValueError(f'Unknown model: {model_name}')

# ── Capturer params ──
strategy = sys.argv[2]
alpha = float(sys.argv[3])

if strategy == 'Baseline':
    from Opara import GraphCapturer
    with redirect_stdout(io.StringIO()):
        runner = GraphCapturer.capturer(inputs, model, copy_outputs=False)
elif strategy == 'DRT':
    from Opara import GraphCapturer
    with redirect_stdout(io.StringIO()):
        runner = GraphCapturer.capturer(inputs, model, copy_outputs=False,
            alpha=alpha, selection_mode='static_interference', time_domain=True)
else:
    # Cosine, MinRes
    mode_map = {'Cosine': 'cosine', 'MinRes': 'min_resource'}
    from Opara import GraphCapturer
    with redirect_stdout(io.StringIO()):
        runner = GraphCapturer.capturer(inputs, model, copy_outputs=False,
            alpha=alpha, selection_mode=mode_map[strategy], time_domain=True)

# ── Measurement ──
cache = torch.empty(int(4*(1024**2)), dtype=torch.int8, device='cuda')
for _ in range(50):
    cache.zero_()
    runner(*inputs)
torch.cuda.synchronize()

times = []
for _ in range(100):
    cache.zero_()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); runner(*inputs); e.record()
    torch.cuda.synchronize()
    times.append(s.elapsed_time(e))

print(f"{np.median(times):.4f},{np.min(times):.4f},{np.max(times):.4f}")
