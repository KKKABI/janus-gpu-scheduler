#!/usr/bin/env python3
"""Test CSS weight coefficients and tau thresholds separately for Opara scheduling.

CSS = w_time * S_time + w_res * S_res
Operators with CSS < tau and not on critical path -> low priority.

Experiment 1 — CSS weights (tau fixed at 0.5):
  w_time:w_res = 1:0, 0.75:0.25, 0.5:0.5, 0.3:0.7, 0:1

Experiment 2 — tau thresholds (CSS fixed at 0.5:0.5):
  tau = 0, 0.25, 0.5, 0.75, 1

Runs bert_example.py, inceptionv3_example.py, deepfm_example.py for each config.
Saves full stdout/stderr logs and structured JSON results.
"""

import os
import sys
import subprocess
import re
import json
import time
from datetime import datetime

REPO_ROOT = '/public_0/LYX/janus'
OPLAUNCHER = os.path.join(REPO_ROOT, 'Opara', 'OperatorLauncher.py')
RECORDS_DIR = os.path.join(REPO_ROOT, 'css_test_records')

# ── Experiment 1: vary CSS weights, tau fixed ──
WEIGHT_CONFIGS = [
    (1.0, 0.0, "w1.0_r0.0"),
    (0.75, 0.25, "w0.75_r0.25"),
    (0.5, 0.5, "w0.5_r0.5"),
    (0.3, 0.7, "w0.3_r0.7"),
    (0.0, 1.0, "w0.0_r1.0"),
]
FIXED_TAU_FOR_WEIGHTS = 0.5

# ── Experiment 2: vary tau, CSS weights fixed ──
TAU_CONFIGS = [0.0, 0.25, 0.5, 0.75, 1.0]
FIXED_WEIGHT_FOR_TAU = (0.5, 0.5)

EXAMPLES = ['bert_example.py', 'inceptionv3_example.py', 'deepfm_example.py']

INPUT_SPECS = {
    'bert_example.py': {
        'batch_size': 1, 'seq_length': 16,
        'warm_ups': 10, 'iterations': 300,
        'inputs': [
            {'name': 'input_ids', 'dtype': 'torch.long', 'shape': [1, 16],
             'generator': 'torch.randint(0, 30000, ...)'},
            {'name': 'attention_mask', 'dtype': 'torch.long', 'shape': [1, 16],
             'generator': 'torch.ones(...)'},
        ],
        'model': 'BertModel (bert-base)',
    },
    'inceptionv3_example.py': {
        'batch_size': 1, 'input_size': 299,
        'warm_ups': 100, 'iterations': 300,
        'inputs': [
            {'name': 'x', 'dtype': 'torch.float32', 'shape': [1, 3, 299, 299],
             'generator': 'torch.randint(low=0, high=256, ...)'},
        ],
        'model': 'inception_v3 (torchvision)',
    },
    'deepfm_example.py': {
        'batch_size': 1, 'num_cate_features': 32, 'nume_fea_size': 16,
        'warm_ups': 100, 'iterations': 3000,
        'inputs': [
            {'name': 'X_sparse', 'dtype': 'torch.long', 'shape': [1, 32],
             'generator': 'torch.randint(0, 100, ...)'},
            {'name': 'X_dense', 'dtype': 'torch.float32', 'shape': [1, 16],
             'generator': 'torch.rand(...)'},
        ],
        'model': 'DeepFM',
    },
}


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def set_css_params(w_time, w_res, tau):
    content = read_file(OPLAUNCHER)

    # Replace CSS weight line
    pattern_css = r'CSS = [\d.]+ \* S_time \+ [\d.]+ \* S_res'
    new_css = f'CSS = {w_time} * S_time + {w_res} * S_res'
    m = re.search(pattern_css, content)
    if m:
        content = content[:m.start()] + new_css + content[m.end():]
    else:
        raise RuntimeError(f"Cannot find CSS line in {OPLAUNCHER}")

    # Replace tau default value
    pattern_tau = r'def pop_lowPriorty_from_queue\(queue, tau=[\d.]+\)'
    new_tau = f'def pop_lowPriorty_from_queue(queue, tau={tau})'
    m = re.search(pattern_tau, content)
    if m:
        content = content[:m.start()] + new_tau + content[m.end():]
    else:
        raise RuntimeError(f"Cannot find tau default in {OPLAUNCHER}")

    write_file(OPLAUNCHER, content)


def parse_timing(stdout):
    times = {}
    for line in stdout.split('\n'):
        m = re.match(r'Time of Opara:\s+([\d.]+)\s+ms\s+std:\s+([\d.]+)', line)
        if m:
            times['opara_time_ms'] = float(m.group(1))
            times['opara_std_ms'] = float(m.group(2))
        m = re.match(r'Time of native PyTorch:\s+([\d.]+)\s+ms\s+std:\s+([\d.]+)', line)
        if m:
            times['torch_time_ms'] = float(m.group(1))
            times['torch_std_ms'] = float(m.group(2))
        m = re.match(r'Time of sequential CUDA Graph:\s+([\d.]+)\s+ms\s+std:\s+([\d.]+)', line)
        if m:
            times['seq_time_ms'] = float(m.group(1))
            times['seq_std_ms'] = float(m.group(2))
        m = re.search(r'output of PyTorch == output of Opara:\s*(True|False)', line)
        if m:
            times['correctness_ok'] = (m.group(1) == 'True')
        m = re.search(r'Absolute difference:\s*([\d.e+\-]+)', line)
        if m:
            times['abs_diff'] = float(m.group(1))
    return times


def run_example(script_name):
    script_path = os.path.join(REPO_ROOT, 'examples', script_name)
    t0 = time.time()
    try:
        result = subprocess.run(
            ['python', script_path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour per model
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}
        )
        elapsed = time.time() - t0
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'elapsed_sec': round(elapsed, 1),
            'timed_out': False,
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        return {
            'stdout': e.stdout or '',
            'stderr': (e.stderr or '') + f'\n[TIMEOUT] Killed after {elapsed:.0f}s (>1 hour)',
            'returncode': -1,
            'elapsed_sec': round(elapsed, 1),
            'timed_out': True,
        }


def save_run(tag, rec, parsed, experiment, vary_param, vary_value):
    """Save stdout/stderr to files and return a structured record dict."""
    with open(os.path.join(RECORDS_DIR, f'{tag}_stdout.txt'), 'w') as f:
        f.write(rec['stdout'])
    with open(os.path.join(RECORDS_DIR, f'{tag}_stderr.txt'), 'w') as f:
        f.write(rec['stderr'])
    return {
        'tag': tag,
        'experiment': experiment,
        'vary_param': vary_param,
        'vary_value': vary_value,
        'returncode': rec['returncode'],
        'elapsed_sec': rec['elapsed_sec'],
        'timed_out': rec.get('timed_out', False),
        'parsed': parsed,
        'stdout_file': f'{tag}_stdout.txt',
        'stderr_file': f'{tag}_stderr.txt',
    }


def print_pivot_table(runs, experiment, vary_param, vary_values, row_labels, examples):
    """Print a pivot table: rows=configs, cols=models, cells=Opara time (ms)."""
    print(f"\n{'='*110}")
    print(f"Experiment: {experiment}")
    print(f"Opara Time (ms) — lower is better")
    print(f"{'='*110}")
    header = f"{vary_param:<20}"
    for ex in examples:
        header += f"  {ex:<35}"
    print(header)
    print("-" * len(header))
    for vv, rl in zip(vary_values, row_labels):
        row = f"{rl:<20}"
        for ex in examples:
            found = [r for r in runs
                     if r['experiment'] == experiment
                     and r['vary_value'] == vv
                     and r['tag'].endswith(ex.replace('.py', ''))]
            if found:
                t = found[0]['parsed'].get('opara_time_ms')
                s = found[0]['parsed'].get('opara_std_ms', 0)
                if t is not None:
                    row += f"  {t:<10.4f} ms (std:{s:.2f})     "
                else:
                    row += f"  {'FAIL':<35}"
            else:
                row += f"  {'N/A':<35}"
        print(row)


def flush_json(all_runs):
    """Write results to JSON immediately — safe to call after every run."""
    json_path = os.path.join(RECORDS_DIR, 'full_results.json')
    with open(json_path, 'w') as f:
        json.dump({
            'test_time': datetime.now().isoformat(),
            'repo_root': REPO_ROOT,
            'experiment_1_css_weights': {'fixed_tau': FIXED_TAU_FOR_WEIGHTS, 'configs': [[w, r] for w, r, _ in WEIGHT_CONFIGS]},
            'experiment_2_tau': {'fixed_css': f'{FIXED_WEIGHT_FOR_TAU[0]}:{FIXED_WEIGHT_FOR_TAU[1]}', 'configs': TAU_CONFIGS},
            'input_specs': INPUT_SPECS,
            'completed': len(all_runs),
            'runs': all_runs,
        }, f, indent=2, default=str)


def main():
    os.makedirs(RECORDS_DIR, exist_ok=True)
    original = read_file(OPLAUNCHER)
    all_runs = []
    total = len(WEIGHT_CONFIGS) * len(EXAMPLES) + len(TAU_CONFIGS) * len(EXAMPLES)
    n = 0

    try:
        # ═══════════════════════════════════════════════════════════
        # Experiment 1: vary CSS weights, tau fixed at 0.5
        # ═══════════════════════════════════════════════════════════
        for w_time, w_res, coe_label in WEIGHT_CONFIGS:
            print(f"\n{'='*70}")
            print(f"[EXP1 - CSS weights] w_time:w_res = {w_time}:{w_res}  |  tau = {FIXED_TAU_FOR_WEIGHTS} (fixed)")
            print(f"{'='*70}")

            set_css_params(w_time, w_res, FIXED_TAU_FOR_WEIGHTS)

            for example in EXAMPLES:
                n += 1
                tag = f"EXP1_{coe_label}_{example.replace('.py','')}"
                print(f"  [{n}/{total}] {example}...", flush=True)
                rec = run_example(example)
                parsed = parse_timing(rec['stdout'] + '\n' + rec['stderr'])
                run_record = save_run(tag, rec, parsed, 'CSS_weights', 'w_time:w_res', f'{w_time}:{w_res}')
                run_record['w_time'] = w_time
                run_record['w_res'] = w_res
                run_record['tau'] = FIXED_TAU_FOR_WEIGHTS
                run_record['example'] = example
                all_runs.append(run_record)
                flush_json(all_runs)  # incremental save

                if rec.get('timed_out'):
                    print(f"    TIMEOUT (>1 hour, skipped)")
                else:
                    ot = parsed.get('opara_time_ms')
                    if ot is not None:
                        print(f"    Opara: {ot:.4f} ms  (std: {parsed.get('opara_std_ms', 0):.4f})")
                    else:
                        print(f"    FAILED (returncode={rec['returncode']})")

        # ═══════════════════════════════════════════════════════════
        # Experiment 2: vary tau, CSS weights fixed at 0.5:0.5
        # ═══════════════════════════════════════════════════════════
        w_fixed, r_fixed = FIXED_WEIGHT_FOR_TAU
        for tau in TAU_CONFIGS:
            print(f"\n{'='*70}")
            print(f"[EXP2 - tau] tau = {tau}  |  w_time:w_res = {w_fixed}:{r_fixed} (fixed)")
            print(f"{'='*70}")

            set_css_params(w_fixed, r_fixed, tau)

            for example in EXAMPLES:
                n += 1
                tag = f"EXP2_tau{tau}_{example.replace('.py','')}"
                print(f"  [{n}/{total}] {example}...", flush=True)
                rec = run_example(example)
                parsed = parse_timing(rec['stdout'] + '\n' + rec['stderr'])
                run_record = save_run(tag, rec, parsed, 'tau', 'tau', tau)
                run_record['w_time'] = w_fixed
                run_record['w_res'] = r_fixed
                run_record['tau'] = tau
                run_record['example'] = example
                all_runs.append(run_record)
                flush_json(all_runs)  # incremental save

                if rec.get('timed_out'):
                    print(f"    TIMEOUT (>1 hour, skipped)")
                else:
                    ot = parsed.get('opara_time_ms')
                    if ot is not None:
                        print(f"    Opara: {ot:.4f} ms  (std: {parsed.get('opara_std_ms', 0):.4f})")
                    else:
                        print(f"    FAILED (returncode={rec['returncode']})")

        # ── Summary ──
        print_pivot_table(all_runs, 'CSS_weights', 'w_time:w_res',
                          [f'{w}:{r}' for w, r, _ in WEIGHT_CONFIGS],
                          [f'{w}:{r}' for w, r, _ in WEIGHT_CONFIGS],
                          EXAMPLES)

        print_pivot_table(all_runs, 'tau', 'tau',
                          TAU_CONFIGS,
                          [f'tau={t}' for t in TAU_CONFIGS],
                          EXAMPLES)

        # DeepFM speedup pivot for CSS weights
        deepfm_weight_runs = [r for r in all_runs
                              if r['experiment'] == 'CSS_weights'
                              and r['example'] == 'deepfm_example.py']
        if deepfm_weight_runs:
            print(f"\n{'='*110}")
            print("DeepFM Speedup: Opara / Sequential CUDA Graph (CSS weights experiment)")
            print(f"{'='*110}")
            print(f"{'w_time:w_res':<20}  Opara(ms)    Seq(ms)      Speedup")
            print("-" * 65)
            for w_time, w_res, coe_label in WEIGHT_CONFIGS:
                found = [r for r in deepfm_weight_runs
                         if r['w_time'] == w_time and r['w_res'] == w_res]
                if found:
                    p = found[0]['parsed']
                    ot, st = p.get('opara_time_ms'), p.get('seq_time_ms')
                    if ot and st and st > 0:
                        print(f"  {w_time}:{w_res:<18} {ot:<12.4f} {st:<12.4f} {ot/st:.4f}")

            print(f"\n{'='*110}")
            print("DeepFM Speedup: Opara / Sequential CUDA Graph (tau experiment)")
            print(f"{'='*110}")
            print(f"{'tau':<20}  Opara(ms)    Seq(ms)      Speedup")
            print("-" * 65)
            for tau in TAU_CONFIGS:
                found = [r for r in all_runs
                         if r['experiment'] == 'tau'
                         and r['example'] == 'deepfm_example.py'
                         and r['tau'] == tau]
                if found:
                    p = found[0]['parsed']
                    ot, st = p.get('opara_time_ms'), p.get('seq_time_ms')
                    if ot and st and st > 0:
                        print(f"  tau={tau:<15} {ot:<12.4f} {st:<12.4f} {ot/st:.4f}")

        print(f"\nAll {len(all_runs)} runs saved incrementally to {RECORDS_DIR}/full_results.json")

    finally:
        write_file(OPLAUNCHER, original)
        print(f"Restored original {OPLAUNCHER}")


if __name__ == '__main__':
    main()
