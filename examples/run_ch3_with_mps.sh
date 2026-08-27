#!/usr/bin/env bash
set -Eeuo pipefail

repo=${1:?repository path is required}
shift
python_bin=${PYTHON_BIN:-/home/lyx/.conda/envs/opara/bin/python}
lock_file=/tmp/janus_gpu0.lock
run_identity="${USER:-user}-ch3-$$"
mps_pipe="/tmp/${run_identity}-mps-pipe"
mps_log="/tmp/${run_identity}-mps-log"

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

"$python_bin" "$repo/examples/run_ch3_matrix.py" "$@"
