#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: run_occupancy_case.sh MODEL VARIANT CASE_SLUG" >&2
  exit 64
fi
model=$1
requested_variant=$2
case_slug=$3

repo=/public_0/LYX/janus_simulator_accuracy_20260820
tool=$repo/experiments/newtd_accuracy
profile=$tool/profile_selected_groups_gpu_metrics_20260821.py
out_root=${JANUS_OCCUPANCY_OUT_ROOT:?JANUS_OCCUPANCY_OUT_ROOT must be set}
out=$out_root/$case_slug
python_bin=/home/lyx/.conda/envs/opara/bin/python
nsys_bin=/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys
solo=/public_0/LYX/janus_simulator_accuracy_outputs_20260820/solo_operator_all_kernel_v1

variant=$requested_variant
extra=()
if [[ $requested_variant == NewTD+DRT ]]; then
  variant=TD+DRT
  extra=(
    --new-td-pair-extension
    --solo-profile-root "$solo"
    --td-launch-gap-ms 0.004096
    --minimum-predicted-overlap-us 2.0
  )
fi

exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo "GPU experiment lock is busy" >&2; exit 75; }
test ! -e "$out"
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
[[ -z $active ]] || { echo "GPU has active process: $active" >&2; exit 76; }

mkdir -p "$out"
cd "$repo"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset JANUS_ALLOW_LEGACY_NCU || true

"$nsys_bin" profile \
  --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node \
  --gpu-metrics-device=0 \
  --gpu-metrics-frequency=200000 \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --show-output=false \
  --export=sqlite --force-overwrite=false \
  -o "$out/full_trace" \
  "$python_bin" "$profile" \
    --model "$model" --variant "$variant" \
    --max-ready 6 --skip-idle-check \
    --metrics-replays 100 \
    "${extra[@]}" \
    --output-dir "$out/artifacts" \
  >"$out/profile.stdout" 2>"$out/profile.stderr"

test -s "$out/full_trace.nsys-rep"
test -s "$out/full_trace.sqlite"
test -s "$out/artifacts/summary.json"
test -s "$out/artifacts/scheduler_calls.json"

"$python_bin" - "$out/full_trace.sqlite" "$out/artifacts/summary.json" <<'PY'
import json, sqlite3, sys
db=sqlite3.connect(sys.argv[1])
summary=json.load(open(sys.argv[2], encoding='utf-8'))
tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if 'GPU_METRICS' not in tables:
    raise RuntimeError('GPU_METRICS table is missing')
samples=db.execute('SELECT count(*) FROM GPU_METRICS').fetchone()[0]
kernels=db.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
if samples <= 0 or kernels <= 0:
    raise RuntimeError(f'empty trace: samples={samples} kernels={kernels}')
print(json.dumps({
    'model': summary['model'],
    'configuration': summary['configuration'],
    'selected_groups': summary['final_multi_operator_group_count'],
    'gpu_metric_rows': samples,
    'kernel_rows': kernels,
}, ensure_ascii=False))
PY
