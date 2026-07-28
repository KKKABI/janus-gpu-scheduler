"""Benchmark ONE model, ALL strategies in this worktree (direction B: Cosine, MinRes)"""
import io, sys, os, json, numpy as np
from contextlib import redirect_stdout
sys.path.insert(0, '.')

model_name = sys.argv[1]
out_file = sys.argv[2]  # append results here

import torch
torch.cuda.empty_cache()

# ── Model loading ──
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
    raise ValueError(f'Unknown: {model_name}')

# ── Test strategies ──
from Opara import GraphCapturer

cache = torch.empty(int(4*(1024**2)), dtype=torch.int8, device='cuda')

strategies = [
    ('Cosine', 'cosine'),
    ('MinRes', 'min_resource'),
]

results = []
for sname, smode in strategies:
    for alpha in [0.9, 0.8, 0.5, 0.2]:
        print(f"  {sname} α={alpha} ...", file=sys.stderr)
        with redirect_stdout(io.StringIO()):
            runner = GraphCapturer.capturer(inputs, model, copy_outputs=False,
                alpha=alpha, selection_mode=smode, time_domain=True)

        # warmup
        for _ in range(50):
            cache.zero_()
            runner(*inputs)
        torch.cuda.synchronize()

        # measure
        times = []
        for _ in range(100):
            cache.zero_()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); runner(*inputs); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

        row = f"{model_name},{sname},{alpha},{np.median(times):.4f},{np.min(times):.4f},{np.max(times):.4f}"
        results.append(row)
        print(f"    => {np.median(times):.4f} ms", file=sys.stderr)

# Write results
with open(out_file, 'a') as f:
    for r in results:
        f.write(r + '\n')

print(f"Done {model_name}: {len(results)} rows -> {out_file}", file=sys.stderr)
