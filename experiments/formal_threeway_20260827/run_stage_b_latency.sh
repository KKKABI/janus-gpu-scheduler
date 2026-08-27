#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${JANUS_FORMAL_REPO:-/public_0/LYX/janus_release_newtd_ncu_20260827}
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
