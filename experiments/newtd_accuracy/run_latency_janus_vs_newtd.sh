#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/public_0/LYX/janus_simulator_accuracy_20260820
TOOL=$REPO/experiments/newtd_accuracy
PY=/home/lyx/.conda/envs/opara/bin/python
OUTROOT=${JANUS_NEW_TD_LATENCY_OUTROOT:-/public_0/LYX/janus_td_crossmodel_outputs_20260821/latency_newtd_threshold5_diagnostic_v1}
SOLO=/public_0/LYX/janus_simulator_accuracy_outputs_20260820/solo_operator_all_kernel_v1

test ! -e "$OUTROOT"
mkdir -p "$OUTROOT"
exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo "GPU experiment lock is busy" >&2; exit 75; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; then
  echo "GPU has an active compute process" >&2
  exit 76
fi

git -C "$REPO" rev-parse HEAD > "$OUTROOT/git_head.txt"
find "$TOOL" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$OUTROOT/tool_sha256.txt"
nvidia-smi -q > "$OUTROOT/nvidia_smi_start.txt"

models=(GoogLeNet Inception-v3 NASNet YOLOv8x ConvNeXt DeepFM BERT)
slugs=(googlenet inception_v3 nasnet yolov8x convnext deepfm bert)

run_janus() {
  local model=$1 slug=$2
  echo "START Janus model=$model utc=$(date -u +%FT%TZ)"
  "$PY" "$REPO/experiments/run_one.py" \
    --model "$model" --variant Baseline --alpha none \
    --repeat-index 0 --max-ready 6 \
    --output-dir "$OUTROOT/${slug}_janus" \
    >"$OUTROOT/${slug}_janus.log" 2>&1
  echo "DONE Janus model=$model utc=$(date -u +%FT%TZ)"
}

run_newtd() {
  local model=$1 slug=$2
  echo "START NewTD+DRT model=$model utc=$(date -u +%FT%TZ)"
  env \
    PYTHONPATH="$REPO" \
    JANUS_NEW_TD_PAIR_EXTENSION=1 \
    JANUS_NEW_TD_SOLO_ROOT="$SOLO" \
    JANUS_NEW_TD_MIN_OVERLAP_US=5.0 \
    JANUS_NEW_TD_LAUNCH_GAP_MS=0.004096 \
    "$PY" "$TOOL/run_one_newtd.py" \
      --model "$model" --variant TD+DRT --alpha none \
      --repeat-index 0 --max-ready 6 \
      --output-dir "$OUTROOT/${slug}_newtd_drt" \
      >"$OUTROOT/${slug}_newtd_drt.log" 2>&1
  echo "DONE NewTD+DRT model=$model utc=$(date -u +%FT%TZ)"
}

for index in "${!models[@]}"; do
  model=${models[$index]}
  slug=${slugs[$index]}
  if (( index % 2 == 0 )); then
    run_janus "$model" "$slug"
    run_newtd "$model" "$slug"
  else
    run_newtd "$model" "$slug"
    run_janus "$model" "$slug"
  fi
done

nvidia-smi -q > "$OUTROOT/nvidia_smi_end.txt"
touch "$OUTROOT/COMPLETE"
