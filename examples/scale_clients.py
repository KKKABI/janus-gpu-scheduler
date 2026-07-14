#!/usr/bin/env python3
"""
Multi-client scalability test — N clients on one GPU.

Usage:
  python scale_clients.py --model mobilenetv2 --clients 2 --iterations 100 --warmups 30
  python scale_clients.py --model mobilenetv2 --clients 2,4,6,8
"""

import os, sys, json, time, subprocess, argparse
import numpy as np

READY_DIR  = '/tmp/opara_scale_ready'
START_FILE = '/tmp/opara_scale_start'
SCRIPT     = os.path.abspath(__file__)

MODEL_REGISTRY = {
    'mobilenetv2': {
        'module': 'torchvision.models',
        'factory': 'mobilenet_v2',
        'input_shape': (1, 3, 224, 224),
    },
    'resnet50': {
        'module': 'torchvision.models',
        'factory': 'resnet50',
        'input_shape': (1, 3, 224, 224),
    },
    'convnext': {
        'module': 'torchvision.models',
        'factory': 'convnext_base',
        'input_shape': (1, 3, 224, 224),
    },
    'googlenet': {
        'module': 'torchvision.models',
        'factory': 'googlenet',
        'input_shape': (1, 3, 224, 224),
    },
    'nasnet': {
        'module': 'pretrainedmodels',
        'factory': 'nasnetalarge',
        'input_shape': (1, 3, 331, 331),
        'extra_kwargs': {'num_classes': 1000, 'pretrained': 'imagenet'},
    },
}


def build_model(name, device):
    import torch
    spec = MODEL_REGISTRY[name]
    x = torch.randint(0, 256, spec['input_shape'], dtype=torch.float32, device=device)
    mod = __import__(spec['module'], fromlist=[spec['factory']])
    factory = getattr(mod, spec['factory'])
    extra_kwargs = spec.get('extra_kwargs', {})
    if spec['module'] == 'torchvision.models':
        model = factory(weights=None, **extra_kwargs).to(device).eval()
    else:
        model = factory(**extra_kwargs).to(device).eval()
    return model, (x,)


def run_client(client_id, model_name, iterations, warm_ups, sm_fraction=1.0):
    import torch

    device = torch.device('cuda:0')
    cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')
    def flush_cache():
        cache.zero_()

    label = f"[C{client_id}:{model_name}]"
    print(f"{label} Loading model ...", flush=True)
    model, inputs = build_model(model_name, device)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Opara import GraphCapturer
    print(f"{label} Capturing Opara CUDA Graph (sm_fraction={sm_fraction}) ...", flush=True)
    runner = GraphCapturer.capturer(inputs, model, sm_fraction=sm_fraction)
    print(f"{label} Captured.", flush=True)

    print(f"{label} Warming up ({warm_ups} iters) ...", flush=True)
    with torch.no_grad():
        for _ in range(warm_ups):
            runner(*inputs)
    torch.cuda.synchronize()

    os.makedirs(READY_DIR, exist_ok=True)
    with open(os.path.join(READY_DIR, f'client_{client_id}'), 'w') as f:
        f.write(str(os.getpid()))
    print(f"{label} Ready, waiting for start ...", flush=True)

    while not os.path.exists(START_FILE):
        time.sleep(0.05)

    print(f"{label} Running ({iterations} iters) ...", flush=True)

    times = []
    with torch.no_grad():
        for i in range(iterations):
            flush_cache()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            runner(*inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    tp = 1000.0 / avg_ms if avg_ms > 0 else 0

    result = {'client_id': client_id, 'model': model_name,
              'avg_ms': avg_ms, 'std_ms': std_ms,
              'throughput_ips': tp, 'iterations': iterations}
    with open(os.path.join(READY_DIR, f'result_{client_id}'), 'w') as f:
        json.dump(result, f)
    print(f"{label} Done. avg={avg_ms:.2f}ms, tp={tp:.2f} iter/s", flush=True)


def cleanup():
    import shutil
    if os.path.exists(START_FILE):
        os.remove(START_FILE)
    if os.path.exists(READY_DIR):
        shutil.rmtree(READY_DIR, ignore_errors=True)


def run_scale(model_name, n_clients, iterations, warm_ups, sm_fraction=1.0):
    cleanup()
    os.makedirs(READY_DIR, exist_ok=True)

    print(f"\nLaunching {n_clients} x {model_name} (sm_fraction={sm_fraction}) ...")
    procs, log_files = [], []
    for cid in range(n_clients):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        cmd = [sys.executable, SCRIPT, '--client', str(cid),
               '--model', model_name,
               '--iterations', str(iterations), '--warmups', str(warm_ups),
               '--sm-fraction', str(sm_fraction)]
        lf = open(os.path.join(READY_DIR, f'log_{cid}.txt'), 'w')
        log_files.append(lf)
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                       env=env, text=True))

    for cid in range(n_clients):
        while not os.path.exists(os.path.join(READY_DIR, f'client_{cid}')):
            time.sleep(0.1)

    time.sleep(0.5)
    with open(START_FILE, 'w') as f:
        f.write('go')

    for cid, p in enumerate(procs):
        p.wait(timeout=600)
        log_files[cid].close()

    results = []
    for cid in range(n_clients):
        rf = os.path.join(READY_DIR, f'result_{cid}')
        if os.path.exists(rf):
            with open(rf) as f:
                results.append(json.load(f))
        else:
            print(f"[ERROR] Client {cid} no result. Log:")
            lf = os.path.join(READY_DIR, f'log_{cid}.txt')
            if os.path.exists(lf):
                with open(lf) as f:
                    for line in f.readlines()[-20:]:
                        print(f"  {line.rstrip()}")

    cleanup()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client', type=int)
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--clients', type=str, default='2',
                        help='comma-separated list, e.g. 2,4,6,8')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=30)
    parser.add_argument('--sm-fraction', type=float, default=1.0,
                        help='SM fraction for multi-tenant (e.g. 0.5 for 2 concurrent)')
    args = parser.parse_args()

    if args.client is not None:
        run_client(args.client, args.model, args.iterations, args.warmups, args.sm_fraction)
        return

    client_counts = [int(x) for x in args.clients.split(',')]
    ITER, WARM = args.iterations, args.warmups
    SM_FRAC = args.sm_fraction

    print("=" * 75)
    print(f"  Opara Scale Test: {args.model}")
    print(f"  Clients: {client_counts}  |  Iters: {ITER}  |  Warmups: {WARM}")
    print("=" * 75)

    all_results = {}
    wall_times = {}

    for n in client_counts:
        t0 = time.time()
        results = run_scale(args.model, n, ITER, WARM, SM_FRAC)
        wall = time.time() - t0
        all_results[n] = results
        wall_times[n] = wall

        lats = [r['avg_ms'] for r in results]
        tps  = [r['throughput_ips'] for r in results]
        total_ms = max(lats)  # 并发总延迟 = 最慢客户端
        print(f"\n  N={n}  wall={wall:.1f}s  "
              f"latency: min={min(lats):.3f} max={max(lats):.3f} avg={np.mean(lats):.3f}ms  "
              f"total={total_ms:.3f}ms  "
              f"aggregate_tp={sum(tps):.1f} iter/s")

    # ── summary ──
    n0 = client_counts[0]
    single_tp = all_results[n0][0]['throughput_ips']  # single-client throughput as baseline

    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    print(f"  {'N':<6} {'Wall':>8} {'Min':>8} {'Max':>8} {'Avg':>8} {'Total':>9} {'Agg TP':>10}  "
          f"{'Speedup':>8}  {'Efficiency':>10}")
    print("  " + "-" * 84)

    for n in client_counts:
        results = all_results[n]
        lats = [r['avg_ms'] for r in results]
        total_ms = max(lats)  # 并发总延迟 = 最慢客户端单次推理
        agg_tp = sum(r['throughput_ips'] for r in results)
        speedup = agg_tp / single_tp
        eff = speedup / n

        print(f"  {n:<6} {wall_times[n]:>7.1f}s {min(lats):>7.3f}ms {max(lats):>7.3f}ms "
              f"{np.mean(lats):>7.3f}ms {total_ms:>7.3f}ms {agg_tp:>9.1f}  "
              f"{speedup:>7.2f}x  {eff:>9.1%}")

    # Per-client latency degradation
    print(f"\n  Per-client latency degradation (vs N={n0} single client):")
    ref_lat = all_results[n0][0]['avg_ms']
    for n in client_counts:
        results = all_results[n]
        avg_lat = np.mean([r['avg_ms'] for r in results])
        print(f"    N={n}: avg={avg_lat:.3f}ms  "
              f"slowdown={avg_lat / ref_lat:.2f}x")

    print("=" * 75)


if __name__ == '__main__':
    main()
