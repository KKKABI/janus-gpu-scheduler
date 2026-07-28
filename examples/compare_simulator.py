"""
Compare Static vs Time-domain simulator: how many combos pass feasibility?
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from Opara import GraphCapturer

print("=" * 70)
print("  SIMULATOR COMPARISON: Static vs Time-domain")
print("=" * 70)

model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False).eval().cuda()
inputs = (torch.randn(1, 3, 224, 224, device='cuda:0'),)

for td in [False, True]:
    label = "TIME-DOMAIN" if td else "STATIC"
    print(f"\n{'─'*40}")
    print(f"  Mode: {label}")
    print(f"{'─'*40}")

    runner = GraphCapturer.capturer(
        inputs, model, copy_outputs=False,
        alpha=0.9, selection_mode='max_occupancy', time_domain=td
    )
    # Trigger graph capture (runs schedule() internally)
    try:
        runner(*inputs)
    except:
        pass

print("\n" + "=" * 70)
print("Compare the [DIAG] lines above:")
print("  enumerated = total combos tried")
print("  feasible   = passed can_launch (score >= 0)")
print("=" * 70)
