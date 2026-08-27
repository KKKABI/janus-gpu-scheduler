#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${JANUS_FORMAL_REPO:-/public_0/LYX/janus_release_newtd_ncu_20260827}
OUT=${JANUS_STAGE_A_OUT:?JANUS_STAGE_A_OUT must be set to a new directory}
OLD_SOLO=${JANUS_FROZEN_SOLO_ROOT:?set JANUS_FROZEN_SOLO_ROOT to the six-model frozen solo root}
PY=${JANUS_FORMAL_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
NCU=${JANUS_FORMAL_NCU:-/usr/local/cuda-12.5/bin/ncu}
NSYS=${JANUS_FORMAL_NSYS:-/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys}
DIR=$REPO/experiments/formal_threeway_20260827

test ! -e "$OUT"
mkdir -p "$OUT"
exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo 'GPU experiment lock is busy' >&2; exit 75; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | grep -q '[0-9]'; then
  echo 'GPU has an active compute process' >&2
  exit 76
fi

git -C "$REPO" rev-parse HEAD > "$OUT/git_head.txt"
git -C "$REPO" status --porcelain > "$OUT/git_status.txt"
test ! -s "$OUT/git_status.txt" || {
  echo 'formal worktree must be clean' >&2
  exit 77
}
nvidia-smi -q > "$OUT/nvidia_smi_start.txt"

env -u JANUS_ALLOW_LEGACY_NCU \
  "$PY" "$DIR/collect_ncu_v2.py" \
    --output-dir "$OUT/ncu" \
    --python "$PY" --ncu "$NCU" \
    --repeats 3 \
    --minimum-duration-coverage 0.50 \
  >"$OUT/collect_ncu.stdout" 2>"$OUT/collect_ncu.stderr"

"$PY" "$DIR/prepare_solo_root.py" \
  --source-root "$OLD_SOLO" \
  --output-root "$OUT/solo_operator_profiles" \
  >"$OUT/prepare_solo.stdout" 2>"$OUT/prepare_solo.stderr"

"$PY" "$DIR/build_yolo_solo_manifest.py" \
  --output "$OUT/yolo_backbone_solo_manifest.json" \
  >"$OUT/build_yolo_solo_manifest.stdout" \
  2>"$OUT/build_yolo_solo_manifest.stderr"

YOLO_OUT=$OUT/solo_operator_profiles/yolov8x_backbone
mkdir -p "$YOLO_OUT"
"$NSYS" profile \
  --trace=cuda,nvtx,cudnn,cublas --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --show-output=false --export=sqlite \
  --force-overwrite=true -o "$YOLO_OUT/full_trace" \
  "$PY" "$REPO/experiments/simulator_accuracy/profile_solo_operators.py" \
    --manifest "$OUT/yolo_backbone_solo_manifest.json" \
    --model YOLOv8x --target-mode sampled \
    --output-dir "$YOLO_OUT/artifacts" \
  >"$YOLO_OUT/profile.stdout" 2>"$YOLO_OUT/profile.stderr"

test -s "$YOLO_OUT/full_trace.sqlite"
test -s "$YOLO_OUT/artifacts/summary.json"
"$PY" "$REPO/experiments/simulator_accuracy/analyze_solo_operator_trace.py" \
  --sqlite "$YOLO_OUT/full_trace.sqlite" \
  --summary "$YOLO_OUT/artifacts/summary.json" \
  --output "$YOLO_OUT/result.json" \
  >"$YOLO_OUT/analyze.stdout" 2>"$YOLO_OUT/analyze.stderr"

"$PY" "$DIR/verify_formal_assets.py" \
  --ncu-cache-dir "$OUT/ncu/ncu_cache" \
  --solo-root "$OUT/solo_operator_profiles" \
  --output "$OUT/asset_verification.json"

nvidia-smi -q > "$OUT/nvidia_smi_end.txt"
touch "$OUT/COMPLETE"
