#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${JANUS_SIM_REPO:-/public_0/LYX/janus_simulator_accuracy_20260820}
MANIFEST=${JANUS_SIM_MANIFEST:?JANUS_SIM_MANIFEST must be set}
OUT=${JANUS_SOLO_PROFILE_OUT:?JANUS_SOLO_PROFILE_OUT must be set}
PY=${JANUS_SIM_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
NSYS=${JANUS_SIM_NSYS:-/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys}
DIR=$REPO/experiments/simulator_accuracy
DISCOVERY=${JANUS_SIM_DISCOVERY:-}
TARGET_MODE=${JANUS_SOLO_TARGET_MODE:-sampled}

test ! -e "$OUT"
mkdir -p "$OUT"
exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo 'GPU experiment lock is busy' >&2; exit 75; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo 'GPU has an active compute process' >&2
  exit 76
fi

read -r -a models <<< "${JANUS_SOLO_MODELS:-GoogLeNet Inception-v3 NASNet YOLOv8x DeepFM BERT}"
for model in "${models[@]}"; do
  slug=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  model_out=$OUT/$slug
  mkdir "$model_out"
  profile_args=(
    --manifest "$MANIFEST"
    --model "$model"
    --output-dir "$model_out/artifacts"
    --target-mode "$TARGET_MODE"
  )
  if [[ -n "$DISCOVERY" ]]; then
    profile_args+=(--discovery-root "$DISCOVERY")
  fi
  "$NSYS" profile \
    --trace=cuda,nvtx,cudnn,cublas --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --show-output=false --export=sqlite \
    --force-overwrite=true -o "$model_out/full_trace" \
    "$PY" "$DIR/profile_solo_operators.py" \
      "${profile_args[@]}" \
    >"$model_out/profile.stdout" 2>"$model_out/profile.stderr"
  test -s "$model_out/full_trace.sqlite"
  test -s "$model_out/artifacts/summary.json"
  "$PY" "$DIR/analyze_solo_operator_trace.py" \
    --sqlite "$model_out/full_trace.sqlite" \
    --summary "$model_out/artifacts/summary.json" \
    --output "$model_out/result.json"
done
