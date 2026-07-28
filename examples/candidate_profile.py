"""
Compare Static vs Time-domain across alpha values for multiple models.
Reports: avg candidate count, % single-candidate, min/max, latency
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision
import numpy as np
from Opara import GraphCapturer
from Opara.Scheduler import _CANDIDATE_STATS, dump_candidate_stats
import io
from contextlib import redirect_stdout


def run_and_collect(model, inputs, td_label, alpha, selection_mode, time_domain):
    from Opara.Scheduler import _CANDIDATE_STATS, dump_candidate_stats
    _CANDIDATE_STATS.clear()

    f = io.StringIO()
    with redirect_stdout(f):
        runner = GraphCapturer.capturer(
            inputs, model, copy_outputs=False,
            alpha=alpha, selection_mode=selection_mode, time_domain=time_domain
        )

    total_calls = len(_CANDIDATE_STATS)
    if total_calls == 0:
        _CANDIDATE_STATS.clear()
        return {'alpha': alpha, 'avg_cand': 0, 'pct_single': 0,
                'min_cand': 0, 'max_cand': 0, 'latency': 0, 'calls': 0}

    alpha_key = f'a={alpha}'
    cands = [s.get(alpha_key, 0) for s in _CANDIDATE_STATS]
    avg_cand = sum(cands) / total_calls
    min_cand = min(cands)
    max_cand = max(cands)
    single_count = sum(1 for c in cands if c <= 1)
    pct_single = single_count / total_calls * 100
    _CANDIDATE_STATS.clear()

    for _ in range(5):
        runner(*inputs)
    torch.cuda.synchronize()

    times = []
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runner(*inputs)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    latency = np.median(times)

    return {
        'alpha': alpha, 'avg_cand': avg_cand,
        'min_cand': min_cand, 'max_cand': max_cand,
        'pct_single': pct_single, 'latency': latency, 'calls': total_calls
    }


def test_model(model_name, model, inputs):
    print(f"\n{'#'*60}")
    print(f"#  {model_name}")
    print(f"{'#'*60}")

    alphas = [0.9, 0.8, 0.5, 0.2]

    for mode_label, td in [("Static", False), ("Time-domain", True)]:
        print(f"\n  {mode_label}")
        print(f"  {'α':>6}  {'avg':>6}  {'仅1%':>6}  {'min':>5}  {'max':>5}  {'延迟ms':>8}  {'轮次':>6}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*6}")

        for alpha in alphas:
            result = run_and_collect(model, inputs, f"{model_name}-{mode_label}",
                                     alpha, selection_mode='cosine', time_domain=td)
            if result['calls'] == 0:
                print(f"  (failed)")
                continue
            print(f"  {result['alpha']:>6.1f}  {result['avg_cand']:>6.1f}  "
                  f"{result['pct_single']:>5.0f}%  {result['min_cand']:>5}  "
                  f"{result['max_cand']:>5}  {result['latency']:>8.3f}  "
                  f"{result['calls']:>6}")


def main():
    # GoogLeNet
    print("Loading GoogLeNet...")
    model = torchvision.models.googlenet(weights=None).eval().cuda()
    inputs = (torch.randn(1, 3, 224, 224, device='cuda:0'),)
    test_model("GoogLeNet", model, inputs)

    # ConvNeXt-Tiny
    print("\nLoading ConvNeXt-Tiny...")
    model = torchvision.models.convnext_tiny(weights=None).eval().cuda()
    inputs = (torch.randn(1, 3, 224, 224, device='cuda:0'),)
    test_model("ConvNeXt-Tiny", model, inputs)

    # ResNet50
    print("\nLoading ResNet50...")
    model = torchvision.models.resnet50(weights=None).eval().cuda()
    inputs = (torch.randn(1, 3, 224, 224, device='cuda:0'),)
    test_model("ResNet50", model, inputs)


if __name__ == '__main__':
    main()
