#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/public_0/LYX/janus_simulator_accuracy_20260820
TOOL=$REPO/experiments/newtd_accuracy
OUTROOT=${JANUS_NEW_TD_FORMAL_OUTROOT:-/public_0/LYX/janus_td_crossmodel_outputs_20260821/final_selected_newtd_formal_v2}
SOLO=/public_0/LYX/janus_simulator_accuracy_outputs_20260820/solo_operator_all_kernel_v1

test ! -e "$OUTROOT"
mkdir -p "$OUTROOT"
git -C "$REPO" rev-parse HEAD > "$OUTROOT/git_head.txt"
find "$TOOL" -maxdepth 1 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$OUTROOT/tool_sha256.txt"
nvidia-smi -q > "$OUTROOT/nvidia_smi_start.txt"

export JANUS_PRECISION_REPO="$REPO"
export JANUS_PRECISION_TOOL_DIR="$TOOL"
export JANUS_PRECISION_OUT_ROOT="$OUTROOT"
export JANUS_NEW_TD_SOLO_ROOT="$SOLO"
export JANUS_NEW_TD_LAUNCH_GAP_MS=${JANUS_NEW_TD_LAUNCH_GAP_MS:-0.004096}
export JANUS_NEW_TD_MIN_OVERLAP_US=${JANUS_NEW_TD_MIN_OVERLAP_US:-2.0}

models=(
  GoogLeNet
  Inception-v3
  NASNet
  YOLOv8x
  ConvNeXt
  DeepFM
  BERT
)
slugs=(
  googlenet
  inception_v3
  nasnet
  yolov8x
  convnext
  deepfm
  bert
)

for index in "${!models[@]}"; do
  model=${models[$index]}
  slug=${slugs[$index]}
  echo "START model=$model slug=$slug utc=$(date -u +%FT%TZ)"
  "$TOOL/run_precision_case.sh" "$model" NewTD+DRT "${slug}_newtd_drt"
  echo "DONE model=$model slug=$slug utc=$(date -u +%FT%TZ)"
done

nvidia-smi -q > "$OUTROOT/nvidia_smi_end.txt"
touch "$OUTROOT/COMPLETE"
