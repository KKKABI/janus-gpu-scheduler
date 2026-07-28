"""Quick GoogLeNet DRT(static_interference) strategy across alphas"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torchvision, numpy as np
from Opara import GraphCapturer
from Opara.Scheduler import _CANDIDATE_STATS
import io
from contextlib import redirect_stdout

model = torchvision.models.googlenet(weights=None).eval().cuda()
inputs = (torch.randn(1, 3, 224, 224, device='cuda:0'),)

for alpha in [0.9, 0.8, 0.5, 0.2]:
    _CANDIDATE_STATS.clear()
    f = io.StringIO()
    with redirect_stdout(f):
        runner = GraphCapturer.capturer(
            inputs, model, copy_outputs=False,
            alpha=alpha, selection_mode='static_interference', time_domain=True
        )
    cands = [s.get(f'a={alpha}', 0) for s in _CANDIDATE_STATS]
    total = len(cands)
    avg = sum(cands) / total if total else 0
    single = sum(1 for c in cands if c <= 1)
    print(f"DRT α={alpha}: calls={total} avg_cand={avg:.1f} only1={single}/{total} ({100*single/total:.0f}%) min={min(cands)} max={max(cands)}")
    _CANDIDATE_STATS.clear()

    for _ in range(5): runner(*inputs)
    torch.cuda.synchronize()
    times = []
    for _ in range(100):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); runner(*inputs); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    print(f"  latency: median={np.median(times):.4f} min={np.min(times):.4f} max={np.max(times):.4f}")
