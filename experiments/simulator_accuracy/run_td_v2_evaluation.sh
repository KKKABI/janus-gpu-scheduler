#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${JANUS_SIM_REPO:-/public_0/LYX/janus_simulator_accuracy_20260820}
BASE=${JANUS_SIM_BASE:-/public_0/LYX/janus_simulator_accuracy_outputs_20260820}
OUT=${JANUS_TDV2_OUT:-$BASE/td_v2_evaluation_v1}
MANIFEST=${JANUS_SIM_MANIFEST:-$BASE/positive_sample_v1.json}
DISCOVERY=${JANUS_SIM_DISCOVERY:-$BASE/discovery_v1}
SOLO_ROOTS=${JANUS_TDV2_SOLO_ROOTS:?JANUS_TDV2_SOLO_ROOTS must be set}
PY=${JANUS_SIM_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
SCRIPT=$REPO/experiments/simulator_accuracy/evaluate_td_v2_sample.py

test ! -e "$OUT"
mkdir -p "$OUT"
exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo 'GPU experiment lock is busy' >&2; exit 75; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo 'GPU has an active compute process' >&2
  exit 76
fi

models=(GoogLeNet Inception-v3 NASNet YOLOv8x DeepFM BERT)
variants=(Baseline TD+DRT)
solo_args=()
for root in $SOLO_ROOTS; do
  solo_args+=(--solo-profile-root "$root")
done
for model in "${models[@]}"; do
  slug=$(printf '%s' "$model" | tr '[:upper:]+' '[:lower:]_' | tr -cd '[:alnum:]_-')
  for variant in "${variants[@]}"; do
    if [[ "$variant" == Baseline ]]; then path=static; else path=td; fi
    target=$OUT/${slug}_${path}.json
    "$PY" "$SCRIPT" --manifest "$MANIFEST" --discovery-root "$DISCOVERY" \
      --model "$model" --reference-variant "$variant" --output "$target" \
      "${solo_args[@]}"
  done
done
