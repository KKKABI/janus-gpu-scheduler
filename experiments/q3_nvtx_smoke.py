#!/usr/bin/env python3
import torch

x = torch.ones(1024, device="cuda")
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("Q3_SESSION")
try:
    for index in range(3):
        torch.cuda.nvtx.range_push(f"SMOKE::{index}")
        try:
            x = x * 2
            torch.cuda.synchronize()
        finally:
            torch.cuda.nvtx.range_pop()
finally:
    torch.cuda.nvtx.range_pop()
print(float(x[0].item()))
