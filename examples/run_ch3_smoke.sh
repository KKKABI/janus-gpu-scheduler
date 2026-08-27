#!/usr/bin/env bash
set -Eeuo pipefail

repo=${1:-/public_0/LYX/janus_multijanus_ch3_20260827}
out=${2:-/public_0/LYX/janus_multijanus_ch3_outputs/smoke_20260827}
python_bin=${PYTHON_BIN:-/home/lyx/.conda/envs/opara/bin/python}
lock_file=/tmp/janus_gpu0.lock
mps_pipe=/tmp/lyx-mps-ch3-20260827
mps_log=/tmp/lyx-mps-log-ch3-20260827

if [[ -e "$out" ]]; then
  echo "output already exists: $out" >&2
  exit 2
fi

exec 9>"$lock_file"
flock -n 9 || { echo "GPU experiment lock is busy" >&2; exit 3; }

if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; then
  echo "GPU has an existing compute process" >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY="$mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$mps_log"
mkdir -p "$mps_pipe" "$mps_log"

cleanup() {
  printf 'quit\n' | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup EXIT
nvidia-cuda-mps-control -d
sleep 1
pgrep -a nvidia-cuda-mps

mkdir -p "$(dirname "$out")"
"$python_bin" "$repo/examples/multi_janus_benchmark.py" \
  --models GoogLeNet GoogLeNet \
  --mode sequential --iterations 10 --warmups 3 --require-mps \
  --output-dir "$out/sequential"
"$python_bin" "$repo/examples/multi_janus_benchmark.py" \
  --models GoogLeNet GoogLeNet \
  --mode concurrent --iterations 10 --warmups 3 --require-mps \
  --output-dir "$out/concurrent"
"$python_bin" "$repo/examples/build_compatibility_table.py" \
  --sequential "$out/sequential/result.json" \
  --concurrent "$out/concurrent/result.json" \
  --output "$out/compatibility.json"
"$python_bin" "$repo/examples/multi_janus_benchmark.py" \
  --models GoogLeNet GoogLeNet \
  --mode lookup --lookup-table "$out/compatibility.json" \
  --iterations 10 --warmups 3 --require-mps \
  --output-dir "$out/lookup"

"$python_bin" - "$out" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for mode in ("sequential", "concurrent", "lookup"):
    payload = json.loads((root / mode / "result.json").read_text())
    assert payload["overall"]["correctness_ok"]
    assert payload["overall"]["request_count"] == 20
print("CH3_SMOKE_OK")
PY
