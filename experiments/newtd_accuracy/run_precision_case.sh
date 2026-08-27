#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: run_precision_case.sh MODEL VARIANT CASE_SLUG" >&2
  exit 64
fi
MODEL=$1
REQUESTED_VARIANT=$2
CASE_SLUG=$3

VARIANT=$REQUESTED_VARIANT
EXTRA_PROFILE_ARGS=()
if [ "$REQUESTED_VARIANT" = "NewTD+DRT" ]; then
  VARIANT=TD+DRT
  SOLO_ROOT=${JANUS_NEW_TD_SOLO_ROOT:?JANUS_NEW_TD_SOLO_ROOT must be set}
  EXTRA_PROFILE_ARGS=(
    --new-td-pair-extension
    --solo-profile-root "$SOLO_ROOT"
    --td-launch-gap-ms "${JANUS_NEW_TD_LAUNCH_GAP_MS:-0.004096}"
    --minimum-predicted-overlap-us "${JANUS_NEW_TD_MIN_OVERLAP_US:-2.0}"
  )
fi

REPO=${JANUS_PRECISION_REPO:-/public_0/LYX/janus_precision7_single_20260816}
OUT_ROOT=${JANUS_PRECISION_OUT_ROOT:?JANUS_PRECISION_OUT_ROOT must be set}
PY=${JANUS_PRECISION_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
NSYS=${JANUS_PRECISION_NSYS:-/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys}
TOOL_DIR=${JANUS_PRECISION_TOOL_DIR:-$REPO/experiments/janus_precision7}
PROFILE=$TOOL_DIR/profile_selected_groups.py
ANALYZE=$TOOL_DIR/analyze_single_replay.py
OUT=$OUT_ROOT/$CASE_SLUG

exec 9>/tmp/janus_precision7_gpu0.lock
flock -n 9 || { echo "GPU experiment lock is busy" >&2; exit 75; }
test ! -e "$OUT"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU has an active compute process" >&2
  exit 76
fi

mkdir -p "$OUT"
cd "$REPO"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
unset JANUS_ALLOW_LEGACY_NCU || true

"$NSYS" profile \
  --trace=cuda,nvtx,cudnn,cublas \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node --show-output=false \
  --export=sqlite --force-overwrite=true \
  -o "$OUT/full_trace" \
  "$PY" "$PROFILE" \
    --model "$MODEL" --variant "$VARIANT" \
    --max-ready 6 --skip-idle-check \
    "${EXTRA_PROFILE_ARGS[@]}" \
    --output-dir "$OUT/artifacts" \
  >"$OUT/profile.stdout" 2>"$OUT/profile.stderr"

test -s "$OUT/full_trace.nsys-rep"
test -s "$OUT/full_trace.sqlite"
test -s "$OUT/artifacts/fx_stream_map.json"
test -s "$OUT/artifacts/scheduler_calls.json"
test -s "$OUT/artifacts/summary.json"

"$PY" "$ANALYZE" \
  --sqlite "$OUT/full_trace.sqlite" \
  --artifacts-dir "$OUT/artifacts" \
  --output-json "$OUT/precision.json" \
  --output-csv "$OUT/calls.csv" \
  >"$OUT/analyze.stdout" 2>"$OUT/analyze.stderr"

"$PY" - "$OUT/precision.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d['correctness']['ok']
print(json.dumps({
  'model':d['model'], 'configuration':d['configuration'],
  'planned':d['planned_multi_operator_groups'],
  'auditable':d['auditable_groups'],
  'actual':d['actual_full_group_concurrent_groups'],
  'precision':d['paper_like_positive_precision'],
  'coverage':d['audit_coverage'],
}, ensure_ascii=False))
PY
