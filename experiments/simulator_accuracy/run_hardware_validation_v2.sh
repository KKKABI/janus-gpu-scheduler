#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${JANUS_SIM_REPO:-/public_0/LYX/janus_simulator_accuracy_20260820}
MANIFEST=${JANUS_SIM_MANIFEST:?JANUS_SIM_MANIFEST must be set}
OUT=${JANUS_SIM_VALIDATION_OUT:?JANUS_SIM_VALIDATION_OUT must be set}
PY=${JANUS_SIM_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
NSYS=${JANUS_SIM_NSYS:-/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys}
DIR=$REPO/experiments/simulator_accuracy

test ! -e "$OUT"
mkdir -p "$OUT"
exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo 'GPU experiment lock is busy' >&2; exit 75; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo 'GPU has an active compute process' >&2
  exit 76
fi

cp "$MANIFEST" "$OUT/manifest.json"
sha256sum "$MANIFEST" "$DIR/profile_isolated_groups.py" \
  "$DIR/analyze_isolated_trace_v2.py" >"$OUT/input_sha256.txt"
git -C "$REPO" rev-parse HEAD >"$OUT/git_head.txt"
nvidia-smi -q >"$OUT/nvidia_smi_start.txt"

models=(GoogLeNet Inception-v3 NASNet YOLOv8x ConvNeXt DeepFM BERT)
for model in "${models[@]}"; do
  count=$(
    "$PY" - "$MANIFEST" "$model" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
print(sum(row['model']==sys.argv[2] for row in p['cases']))
PY
  )
  if [[ "$count" == 0 ]]; then
    echo "skip $model: no sampled positive groups"
    continue
  fi
  slug=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  model_out=$OUT/$slug
  mkdir "$model_out"
  start_epoch=$(date +%s)
  "$NSYS" profile \
    --trace=cuda,nvtx,cudnn,cublas --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --show-output=false --export=sqlite \
    --force-overwrite=true -o "$model_out/full_trace" \
    "$PY" "$DIR/profile_isolated_groups.py" \
      --manifest "$MANIFEST" --model "$model" \
      --output-dir "$model_out/artifacts" \
    >"$model_out/profile.stdout" 2>"$model_out/profile.stderr"
  test -s "$model_out/full_trace.nsys-rep"
  test -s "$model_out/full_trace.sqlite"
  test -s "$model_out/artifacts/summary.json"
  "$PY" "$DIR/analyze_isolated_trace_v2.py" \
    --rep "$model_out/full_trace.nsys-rep" \
    --sqlite "$model_out/full_trace.sqlite" \
    --summary "$model_out/artifacts/summary.json" \
    --output "$model_out/result_v2.json"
  end_epoch=$(date +%s)
  printf '{"model":"%s","case_count":%s,"elapsed_seconds":%s}\n' \
    "$model" "$count" "$((end_epoch-start_epoch))" >"$model_out/timing.json"
done

nvidia-smi -q >"$OUT/nvidia_smi_end.txt"
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"$OUT/COMPLETE"
