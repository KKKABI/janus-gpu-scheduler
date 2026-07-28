#!/bin/bash
PYTHON=/home/lyx/.conda/envs/opara/bin/python
export PYTHONUNBUFFERED=1

echo "===== 1/3 Janus Baseline (original) ====="
cd /public_0/LYX/janus_original_baseline
$PYTHON -c "
import sys; sys.path.insert(0,'.')
import torch, torchvision, numpy as np
from Opara import GraphCapturer
model = torchvision.models.googlenet(weights=None).eval().cuda()
inputs = (torch.randn(1,3,224,224,device='cuda:0'),)
runner = GraphCapturer.capturer(inputs, model, copy_outputs=False)
cache=torch.empty(int(4*(1024**2)),dtype=torch.int8,device='cuda')
for _ in range(50):  # 充分预热
    cache.zero_()
    runner(*inputs)
torch.cuda.synchronize()
times=[];
for _ in range(100):
    cache.zero_()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();runner(*inputs);e.record();torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
print(f'Baseline: median={np.median(times):.4f} min={np.min(times):.4f} max={np.max(times):.4f}')
"

echo ""
echo "===== 2/3 Cosine (方向B, α=0.9) ====="
cd /public_0/LYX/janus
$PYTHON -c "
import sys; sys.path.insert(0,'.')
import torch, torchvision, numpy as np
from Opara import GraphCapturer
model = torchvision.models.googlenet(weights=None).eval().cuda()
inputs = (torch.randn(1,3,224,224,device='cuda:0'),)
runner = GraphCapturer.capturer(inputs, model, copy_outputs=False, alpha=0.9, selection_mode='cosine', time_domain=True)
cache=torch.empty(int(4*(1024**2)),dtype=torch.int8,device='cuda')
for _ in range(50):
    cache.zero_()
    runner(*inputs)
torch.cuda.synchronize()
times=[];
for _ in range(100):
    cache.zero_()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();runner(*inputs);e.record();torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
print(f'Cosine α=0.9: median={np.median(times):.4f} min={np.min(times):.4f} max={np.max(times):.4f}')
" 2>/dev/null

echo ""
echo "===== 3/3 DRT (v2.1, α=0.9) ====="
cd /public_0/LYX/janus_static_interference
$PYTHON -c "
import sys; sys.path.insert(0,'.')
import torch, torchvision, numpy as np
from Opara import GraphCapturer
model = torchvision.models.googlenet(weights=None).eval().cuda()
inputs = (torch.randn(1,3,224,224,device='cuda:0'),)
runner = GraphCapturer.capturer(inputs, model, copy_outputs=False, alpha=0.9, selection_mode='static_interference', time_domain=True)
cache=torch.empty(int(4*(1024**2)),dtype=torch.int8,device='cuda')
for _ in range(50):
    cache.zero_()
    runner(*inputs)
torch.cuda.synchronize()
times=[];
for _ in range(100):
    cache.zero_()
    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    s.record();runner(*inputs);e.record();torch.cuda.synchronize()
    times.append(s.elapsed_time(e))
print(f'DRT α=0.9: median={np.median(times):.4f} min={np.min(times):.4f} max={np.max(times):.4f}')
" 2>/dev/null

echo ""
echo "===== DONE ====="
