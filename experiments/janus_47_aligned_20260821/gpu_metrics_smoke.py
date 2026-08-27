#!/usr/bin/env python3
import torch

torch.manual_seed(20260821)
left = torch.randn(2048, 2048, device="cuda")
right = torch.randn(2048, 2048, device="cuda")
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("JANUS_47_GPU_METRICS_SMOKE")
for _ in range(20):
    left = left @ right
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
print(float(left[0, 0]))
