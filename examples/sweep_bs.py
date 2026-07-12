"""Batch size sweep for multiple models. Usage: python sweep_bs.py <bs> [model]
    Models: convnext, googlenet, inception_v3, bert (default: all)
"""
import torch, torchvision, numpy as np, sys, os
from transformers import BertModel, BertConfig
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Opara import GraphCapturer

def flush_cache(cache):
    cache.zero_()

def run_opara(opara, inputs, iters=20, warm=3):
    cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')
    times = []
    for _ in range(iters):
        flush_cache(cache)
        torch.cuda._sleep(1_000_000)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        opara(*inputs)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return np.mean(times[warm:]), np.std(times[warm:])

MODELS = {
    'convnext':     lambda bs: (torch.randn(bs, 3, 224, 224, device='cuda'), torchvision.models.convnext_base(pretrained=False).eval().cuda()),
    'googlenet':    lambda bs: (torch.randn(bs, 3, 224, 224, device='cuda'), torchvision.models.googlenet(pretrained=False).eval().cuda()),
    'inception_v3': lambda bs: (torch.randn(bs, 3, 299, 299, device='cuda'), torchvision.models.inception_v3(pretrained=False).eval().cuda()),
    'bert':         lambda bs: ((torch.randint(0, 30522, (bs, 16), device='cuda'), torch.randint(0, 2, (bs, 16), device='cuda')),
                                  BertModel(BertConfig()).eval().cuda()),
}

if __name__ == '__main__':
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    target = sys.argv[2] if len(sys.argv) > 2 else None
    to_run = [target] if target else list(MODELS.keys())
    for name in to_run:
        print(f"\n=== {name} bs={bs} ===")
        try:
            inputs, model = MODELS[name](bs)
            if not isinstance(inputs, (list, tuple)):
                inputs = (inputs,)
            opara = GraphCapturer.capturer(inputs, model)
            mean, std = run_opara(opara, inputs)
            print(f"{name:>14s} bs={bs}: {mean:.4f} ms ± {std:.4f}")
        except Exception as e:
            print(f"{name:>14s} bs={bs}: ERROR - {e}")
        torch.cuda.empty_cache()
