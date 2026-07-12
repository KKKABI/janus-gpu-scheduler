"""Quick ConvNeXt batch size sweep for baseline vs direction B comparison."""
import torch, torchvision, numpy as np, sys, os
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

if __name__ == '__main__':
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print(f"\n=== ConvNeXt bs={bs} ===")
    x = torch.randn(bs, 3, 224, 224, device="cuda")
    model = torchvision.models.convnext_base(pretrained=False).eval().cuda()
    opara = GraphCapturer.capturer((x,), model)
    mean, std = run_opara(opara, (x,))
    print(f"Opara bs={bs}: {mean:.4f} ms ± {std:.4f}")
