#!/usr/bin/env python3
"""
Multi-client comparison across model pairs.
  A = MobileNetV2, B = ResNet50
  Pairs: A+A, A+B, B+B
  Modes: pytorch_sequential, opara_sequential, opara_concurrent

Usage:
  python multi_client_compare.py
  python multi_client_compare.py --iterations 100 --warmups 30
"""

import os, sys, json, time, fcntl, shutil, argparse, subprocess
import numpy as np

READY_DIR   = '/tmp/opara_multi_ready'
START_FILE  = '/tmp/opara_multi_start'
LOCK_FILE   = '/tmp/opara_seq_lock'
SCRIPT      = os.path.abspath(__file__)

MODES = ['pytorch_sequential', 'opara_sequential', 'opara_concurrent']

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

MODEL_PAIRS = [
    ('mobilenetv2', 'mobilenetv2'),
    ('googlenet', 'googlenet'),
    ('mobilenetv2', 'resnet50'),
    ('resnet50', 'resnet50'),
    ('nasnet', 'mobilenetv2'),
    ('nasnet', 'nasnet'),
]

PRETTY_MODE = {
    'pytorch_sequential': 'Native PyTorch Sequential',
    'opara_sequential':   'Opara Sequential (lock gated)',
    'opara_concurrent':   'Opara Concurrent (free run)',
}


# ── client process ──────────────────────────────────────────────────────────

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


def run_client(client_id, model_name, mode, iterations, warm_ups, sm_fraction=1.0):
    import torch

    device = torch.device('cuda:0')
    cache = torch.empty(int(4 * (1024 ** 2)), dtype=torch.int8, device='cuda')
    def flush_cache():
        cache.zero_()

    model_label = f"[C{client_id}:{model_name}]"
    print(f"{model_label} Loading model ...", flush=True)
    model, inputs = build_model(model_name, device)

    use_opara = mode.startswith('opara')
    use_lock  = mode.endswith('sequential')

    if use_opara:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Opara import GraphCapturer
        print(f"{model_label} Capturing Opara CUDA Graph ...", flush=True)
        runner = GraphCapturer.capturer(inputs, model, sm_fraction=sm_fraction)
        print(f"{model_label} Captured.", flush=True)
    else:
        runner = model
        print(f"{model_label} Using native PyTorch.", flush=True)

    print(f"{model_label} Warming up ({warm_ups} iters) ...", flush=True)
    with torch.no_grad():
        for _ in range(warm_ups):
            runner(*inputs)
    torch.cuda.synchronize()

    os.makedirs(READY_DIR, exist_ok=True)
    with open(os.path.join(READY_DIR, f'client_{client_id}'), 'w') as f:
        f.write(str(os.getpid()))
    print(f"{model_label} Ready, waiting for start ...", flush=True)

    while not os.path.exists(START_FILE):
        time.sleep(0.05)

    print(f"{model_label} Running ({mode}, {iterations} iters) ...", flush=True)

    lock_fd = open(LOCK_FILE, 'w') if use_lock else None

    times = []
    with torch.no_grad():
        for i in range(iterations):
            flush_cache()

            if lock_fd:
                fcntl.lockf(lock_fd, fcntl.LOCK_EX)

            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            runner(*inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

            if lock_fd:
                torch.cuda.synchronize()
                fcntl.lockf(lock_fd, fcntl.LOCK_UN)

    if lock_fd:
        lock_fd.close()

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    tp = 1000.0 / avg_ms if avg_ms > 0 else 0

    result = {'client_id': client_id, 'model': model_name, 'mode': mode,
              'avg_ms': avg_ms, 'std_ms': std_ms,
              'throughput_ips': tp, 'iterations': iterations}
    with open(os.path.join(READY_DIR, f'result_{client_id}'), 'w') as f:
        json.dump(result, f)
    print(f"{model_label} Done. avg={avg_ms:.2f}ms, tp={tp:.2f} iter/s", flush=True)


# ── orchestrator ────────────────────────────────────────────────────────────

def cleanup():
    for f in [START_FILE, LOCK_FILE]:
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists(READY_DIR):
        shutil.rmtree(READY_DIR, ignore_errors=True)


def run_experiment(mode, model_a, model_b, iterations, warm_ups, sm_fraction=1.0):
    cleanup()
    os.makedirs(READY_DIR, exist_ok=True)

    models = [model_a, model_b]
    procs, log_files = [], []
    for cid in range(2):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'
        cmd = [sys.executable, SCRIPT, '--client', str(cid),
               '--model', models[cid],
               '--mode', mode, '--iterations', str(iterations), '--warmups', str(warm_ups),
               '--sm-fraction', str(sm_fraction)]
        lf = open(os.path.join(READY_DIR, f'log_{cid}.txt'), 'w')
        log_files.append(lf)
        procs.append(subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                       env=env, text=True))

    for cid in range(2):
        while not os.path.exists(os.path.join(READY_DIR, f'client_{cid}')):
            time.sleep(0.1)

    time.sleep(1.0)
    with open(START_FILE, 'w') as f:
        f.write('go')

    for cid, p in enumerate(procs):
        p.wait(timeout=600)
        log_files[cid].close()

    results = []
    for cid in range(2):
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


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--client', type=int)
    parser.add_argument('--model', type=str, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=30)
    parser.add_argument('--sm-fraction', type=float, default=1.0,
                        help='SM fraction for multi-tenant (e.g. 0.5 for 2 concurrent)')
    args = parser.parse_args()

    if args.client is not None:
        run_client(args.client, args.model, args.mode, args.iterations, args.warmups, args.sm_fraction)
        return

    ITER, WARM = args.iterations, args.warmups
    SM_FRAC = args.sm_fraction

    print("=" * 70)
    print("  Opara Multi-Client: Model-Pair Matrix")
    print(f"  A = MobileNetV2  |  B = ResNet50  |  Iters: {ITER}  |  Warmups: {WARM}")
    print("=" * 70)

    all_results = {}

    for pair_idx, (ma, mb) in enumerate(MODEL_PAIRS):
        pair_label = f"{ma}+{mb}"
        all_results[pair_label] = {}

        for mode_idx, mode in enumerate(MODES):
            label = f"[pair {pair_idx+1}/{len(MODEL_PAIRS)}  mode {mode_idx+1}/{len(MODES)}]"
            print(f"\n{label} {pair_label} — {PRETTY_MODE[mode]} ...")
            t0 = time.time()
            all_results[pair_label][mode] = run_experiment(mode, ma, mb, ITER, WARM, SM_FRAC)
            print(f"      wall-clock: {time.time() - t0:.1f}s")

    # ── report ──
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for pair_label in all_results:
        print(f"\n── Pair: {pair_label} ──")
        for mode in MODES:
            results = all_results[pair_label][mode]
            print(f"  {PRETTY_MODE[mode]}:")
            total_tp = 0
            for r in sorted(results, key=lambda x: x['client_id']):
                print(f"    Client {r['client_id']} ({r['model']}): {r['avg_ms']:.2f} ms +/- {r['std_ms']:.2f}  "
                      f"({r['throughput_ips']:.2f} iter/s)")
                total_tp += r['throughput_ips']
            print(f"    → aggregate: {total_tp:.2f} iter/s")

    # ── summary matrix ──
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    header = f"  {'Pair':<22} {'Mode':<22} {'C0 (ms)':>8} {'C1 (ms)':>8} {'Total (ms)':>10} {'Agg TP':>10}  {'Speedup':>8}"
    print(header)
    print("  " + "-" * 88)

    for pair_label in all_results:
        pytorch_lat = max(r['avg_ms'] for r in all_results[pair_label]['pytorch_sequential'])
        pytorch_tp  = sum(r['throughput_ips'] for r in all_results[pair_label]['pytorch_sequential'])
        first = True
        for mode in MODES:
            results = sorted(all_results[pair_label][mode], key=lambda x: x['client_id'])
            lats   = [r['avg_ms'] for r in results]
            tp     = sum(r['throughput_ips'] for r in results)
            # sequential: wall-clock = sum of both clients; concurrent: max
            if 'concurrent' in mode:
                total_ms = max(lats)
            else:
                total_ms = sum(lats)
            speedup = tp / pytorch_tp if pytorch_tp > 0 else 0
            label = pair_label if first else ""
            mode_short = {'pytorch_sequential': 'pytorch_seq', 'opara_sequential': 'opara_seq',
                          'opara_concurrent': 'opara_conc'}[mode]
            print(f"  {label:<22} {mode_short:<22} {lats[0]:>8.2f} {lats[1]:>8.2f} {total_ms:>10.2f} {tp:>10.1f}  {speedup:>7.2f}x")
            first = False
        print()

    # ── cross-pair comparison ──
    print("  CROSS-PAIR: Opara Concurrent vs Opara Sequential (latency & throughput)")
    print("-" * 90)

    header = f"  {'Pair':<22} {'Seq Total':>10} {'Conc Total':>10} {'Seq TP':>10} {'Conc TP':>10}  {'TP Gain':>8}  {'Lat Gain':>8}"
    print(header)
    print("  " + "-" * 88)
    for pair_label in all_results:
        seq_results = sorted(all_results[pair_label]['opara_sequential'], key=lambda x: x['client_id'])
        conc_results = sorted(all_results[pair_label]['opara_concurrent'], key=lambda x: x['client_id'])

        seq_total_ms = sum(r['avg_ms'] for r in seq_results)
        conc_total_ms = max(r['avg_ms'] for r in conc_results)
        seq_tp = sum(r['throughput_ips'] for r in seq_results)
        conc_tp = sum(r['throughput_ips'] for r in conc_results)

        tp_gain  = conc_tp / seq_tp if seq_tp > 0 else 0
        lat_gain = seq_total_ms / conc_total_ms if conc_total_ms > 0 else 0

        print(f"  {pair_label:<22} {seq_total_ms:>10.2f} {conc_total_ms:>10.2f} {seq_tp:>10.1f} {conc_tp:>10.1f}  {tp_gain:>7.2f}x  {lat_gain:>7.2f}x")

    print("=" * 90)


if __name__ == '__main__':
    main()
