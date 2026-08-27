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
OUT=${JANUS_STAGE_C_OUT:?JANUS_STAGE_C_OUT must be set to a new directory}
LATENCY=${JANUS_STAGE_B_ROOT:?JANUS_STAGE_B_ROOT must point to completed stage B}
CACHE=${JANUS_FORMAL_NCU_CACHE_DIR:?JANUS_FORMAL_NCU_CACHE_DIR must be set}
PY=${JANUS_FORMAL_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
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
test -f "$LATENCY/COMPLETE"
test -f "$LATENCY/summary.json"
test ! -n "$(git -C "$REPO" status --porcelain)" || {
  echo 'formal worktree must be clean' >&2
  exit 77
}

git -C "$REPO" rev-parse HEAD > "$OUT/git_head.txt"
nvidia-smi -q > "$OUT/nvidia_smi_start.txt"
"$PY" "$DIR/select_same_ready_pairs.py" \
  --latency-root "$LATENCY" --ncu-cache-dir "$CACHE" \
  --output "$OUT/manifest.json" \
  >"$OUT/select.stdout" 2>"$OUT/select.stderr"

# Audit the bounded manifest before starting any target-group timing.
if ! "$PY" - "$OUT/manifest.json" "$OUT/INCONCLUSIVE.json" <<'PY'
import json,sys
manifest=json.load(open(sys.argv[1],encoding='utf-8'))
if manifest.get('status')!='ready_for_isolated_measurement' or int(manifest.get('primary_pair_count',0))<=0:
    json.dump({
        'status':'inconclusive',
        'reason':'no_planned_primary_same_class_pair',
        'manifest_status':manifest.get('status'),
        'primary_pair_count':manifest.get('primary_pair_count',0),
    },open(sys.argv[2],'x',encoding='utf-8'),ensure_ascii=False,indent=2)
    sys.exit(1)
PY
then
  echo 'Stage C is inconclusive before timing: no primary same-class pair' >&2
  exit 80
fi

# Five independent timing processes.  Pair-side order is reversed cyclically
# so one policy is not always measured first.
"$PY" - "$OUT/manifest.json" <<'PY' > "$OUT/timing_plan.tsv"
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
by_id={row['case_id']:row for row in p['cases']}
for trial in range(5):
    for pair_index,pair in enumerate(p['pairs']):
        ids=list(pair['case_ids'])
        if (trial+pair_index)%2:
            ids.reverse()
        for case_id in ids:
            row=by_id[case_id]
            print(trial,case_id,row['model'],row['call'],row['profile_sha256'],row['fx_code_sha256'],'|'.join(row['group']),sep='\t')
PY

while IFS=$'\t' read -r trial case_id model call profile_sha fx_sha group_csv; do
  test -n "$case_id" || continue
  IFS='|' read -r -a group <<< "$group_csv"
  case_out=$OUT/results/$case_id/timing
  mkdir -p "$case_out"
  "$PY" "$DIR/profile_group_resources.py" \
    --model "$model" --call "$call" --group "${group[@]}" \
    --expected-profile-sha256 "$profile_sha" \
    --expected-fx-code-sha256 "$fx_sha" \
    --mode timing --warmup 10 --repeats 100 --skip-idle-check \
    --output-json "$case_out/trial_$(printf '%02d' "$trial").json" \
    >"$case_out/trial_$(printf '%02d' "$trial").stdout" \
    2>"$case_out/trial_$(printf '%02d' "$trial").stderr"
done < "$OUT/timing_plan.tsv"

"$PY" - "$OUT/manifest.json" <<'PY' > "$OUT/trace_plan.tsv"
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
for pair_index,pair in enumerate(p['pairs']):
    ids=list(pair['case_ids'])
    if pair_index%2:
        ids.reverse()
    by_id={row['case_id']:row for row in p['cases']}
    for case_id in ids:
        row=by_id[case_id]
        print(case_id,row['model'],row['call'],row['profile_sha256'],row['fx_code_sha256'],'|'.join(row['group']),sep='\t')
PY

while IFS=$'\t' read -r case_id model call profile_sha fx_sha group_csv; do
  test -n "$case_id" || continue
  IFS='|' read -r -a group <<< "$group_csv"
  case_out=$OUT/results/$case_id/nsys
  mkdir -p "$case_out"
  "$NSYS" profile \
    --trace=cuda,nvtx,cudnn,cublas --sample=none --cpuctxsw=none \
    --cuda-graph-trace=node --show-output=false --export=sqlite \
    --force-overwrite=true -o "$case_out/full_trace" \
    "$PY" "$DIR/profile_group_resources.py" \
      --model "$model" --call "$call" --group "${group[@]}" \
      --expected-profile-sha256 "$profile_sha" \
      --expected-fx-code-sha256 "$fx_sha" \
      --mode trace --warmup 10 --repeats 10 --skip-idle-check \
      --output-json "$case_out/execution.json" \
    >"$case_out/profile.stdout" 2>"$case_out/profile.stderr"
  test -s "$case_out/full_trace.sqlite"
  "$PY" "$DIR/analyze_nsys_group.py" \
    --sqlite "$case_out/full_trace.sqlite" \
    --execution-json "$case_out/execution.json" \
    --output-json "$case_out/overlap.json" \
    >"$case_out/analyze.stdout" 2>"$case_out/analyze.stderr"
done < "$OUT/trace_plan.tsv"

"$PY" "$DIR/aggregate_same_ready.py" \
  --manifest "$OUT/manifest.json" \
  --results-root "$OUT/results" \
  --output-dir "$OUT/analysis" \
  >"$OUT/aggregate.stdout" 2>"$OUT/aggregate.stderr"
nvidia-smi -q > "$OUT/nvidia_smi_end.txt"
if ! "$PY" - "$OUT/manifest.json" "$OUT/analysis/summary.json" \
  "$OUT/COMPLETE" "$OUT/INCONCLUSIVE.json" "$ACTUAL_COMMIT" <<'PY'
import hashlib,json,sys
manifest_path,summary_path,complete_path,inconclusive_path,git_head=sys.argv[1:]
manifest=json.load(open(manifest_path,encoding='utf-8'))
summary=json.load(open(summary_path,encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()
record={
    'status':summary.get('status'),
    'git_head':git_head,
    'manifest_sha256':sha(manifest_path),
    'summary_sha256':sha(summary_path),
    'primary_planned_pairs':summary.get('primary_planned_pairs',0),
    'primary_valid_pairs':summary.get('primary_valid_pairs',0),
    'inconclusive_reasons':summary.get('inconclusive_reasons',[]),
}
if summary.get('status')!='completed' or int(summary.get('primary_planned_pairs',0))<=0 or int(summary.get('primary_valid_pairs',0))<=0:
    json.dump(record,open(inconclusive_path,'x',encoding='utf-8'),ensure_ascii=False,indent=2)
    sys.exit(1)
json.dump(record,open(complete_path,'x',encoding='utf-8'),ensure_ascii=False,indent=2)
PY
then
  echo 'Stage C finished without a valid primary pair; marked INCONCLUSIVE' >&2
  exit 81
fi
