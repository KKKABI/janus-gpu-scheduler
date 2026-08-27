#!/usr/bin/env bash
set -Eeuo pipefail

repo=/public_0/LYX/janus_simulator_accuracy_20260820
manifest=${JANUS47_MANIFEST:-/public_0/LYX/janus_47_aligned_outputs_20260821/final_selected_sample_v2/manifest.json}
out=${JANUS47_OUT:-/public_0/LYX/janus_47_aligned_outputs_20260821/final_selected_occupancy_v1}
python_bin=/home/lyx/.conda/envs/opara/bin/python
nsys_bin=/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys
profile=$repo/experiments/janus_47_aligned_20260821/profile_isolated_groups_gpu_metrics.py
models=(GoogLeNet Inception-v3 NASNet YOLOv8x ConvNeXt DeepFM BERT)

test ! -e "$out"
mkdir -p "$out"
exec 9>/tmp/janus_gpu0.lock
flock -n 9 || { echo "GPU experiment lock is busy" >&2; exit 75; }
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')
[[ -z $active ]] || { echo "GPU has active process: $active" >&2; exit 76; }
cp "$manifest" "$out/manifest.json"

for model in "${models[@]}"; do
  count=$("$python_bin" - "$manifest" "$model" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
print(sum(row['model']==sys.argv[2] for row in p['cases']))
PY
  )
  if [[ $count == 0 ]]; then
    echo "skip $model: no final-selected multi-operator groups"
    continue
  fi
  slug=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
  case_out=$out/$slug
  mkdir "$case_out"
  start_epoch=$(date +%s)
  "$nsys_bin" profile \
    --trace=cuda,nvtx \
    --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node \
    --gpu-metrics-device=0 \
    --gpu-metrics-frequency=10000 \
    --capture-range=cudaProfilerApi \
    --capture-range-end=repeat \
    --show-output=false \
    --export=sqlite --force-overwrite=false \
    -o "$case_out/full_trace" \
    "$python_bin" "$profile" \
      --manifest "$manifest" --model "$model" \
      --metrics-replays 1000 \
      --output-dir "$case_out/artifacts" \
    >"$case_out/profile.stdout" 2>"$case_out/profile.stderr"
  test -s "$case_out/artifacts/summary.json"
  "$python_bin" - "$case_out" <<'PY'
import json,pathlib,sqlite3,sys
root=pathlib.Path(sys.argv[1])
summary=json.load(open(root/'artifacts'/'summary.json',encoding='utf-8'))
captured=sum(row.get('capture_status')=='captured' for row in summary['cases'])
sqlites=sorted(root.glob('full_trace.*.sqlite'))
reports=sorted(root.glob('full_trace.*.nsys-rep'))
assert len(sqlites)==captured, (len(sqlites),captured)
assert len(reports)==captured, (len(reports),captured)
for path in sqlites:
    db=sqlite3.connect(path)
    tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert 'GPU_METRICS' in tables, path
    assert db.execute('SELECT count(*) FROM GPU_METRICS').fetchone()[0] > 0, path
PY
  end_epoch=$(date +%s)
  printf '{"model":"%s","case_count":%s,"elapsed_seconds":%s}\n' \
    "$model" "$count" "$((end_epoch-start_epoch))" >"$case_out/timing.json"
done

touch "$out/COMPLETE"
