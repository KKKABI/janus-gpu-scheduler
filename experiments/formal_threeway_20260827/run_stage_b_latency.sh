#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
DEFAULT_REPO=$(realpath "$SCRIPT_DIR/../..")
REPO=$(realpath "${JANUS_FORMAL_REPO:-$DEFAULT_REPO}")
EXPECTED_COMMIT=${JANUS_FORMAL_EXPECTED_COMMIT:?set JANUS_FORMAL_EXPECTED_COMMIT to the reviewed 40-character commit}
if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo 'JANUS_FORMAL_EXPECTED_COMMIT must be a lowercase 40-character SHA' >&2
  exit 78
fi
ACTUAL_COMMIT=$(git -C "$REPO" rev-parse HEAD)
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "formal commit mismatch: actual=$ACTUAL_COMMIT expected=$EXPECTED_COMMIT" >&2
  exit 78
fi
EXPECTED_SCRIPT=$(realpath "$REPO/experiments/formal_threeway_20260827/$(basename "$SCRIPT_PATH")")
if [[ "$SCRIPT_PATH" != "$EXPECTED_SCRIPT" ]]; then
  echo "entry script is outside the frozen repository: $SCRIPT_PATH" >&2
  exit 79
fi
OUT=${JANUS_STAGE_B_OUT:?JANUS_STAGE_B_OUT must be set to a new directory}
CACHE=${JANUS_FORMAL_NCU_CACHE_DIR:?JANUS_FORMAL_NCU_CACHE_DIR must be set}
SOLO=${JANUS_FORMAL_SOLO_ROOT:?JANUS_FORMAL_SOLO_ROOT must be set}
PY=${JANUS_FORMAL_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
DIR=$REPO/experiments/formal_threeway_20260827

test ! -e "$OUT"
exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo 'GPU experiment lock is busy' >&2; exit 75; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; then
  echo 'GPU has an active compute process' >&2
  exit 76
fi

nvidia-smi -q > "${OUT}.nvidia_smi_start.txt"
env -u JANUS_ALLOW_LEGACY_NCU \
  "$PY" "$DIR/run_threeway_latency.py" \
    --output-dir "$OUT" \
    --ncu-cache-dir "$CACHE" \
    --solo-root "$SOLO" \
    --python "$PY" --repeats 10 \
  >"${OUT}.stdout" 2>"${OUT}.stderr"
nvidia-smi -q > "$OUT/nvidia_smi_end.txt"
