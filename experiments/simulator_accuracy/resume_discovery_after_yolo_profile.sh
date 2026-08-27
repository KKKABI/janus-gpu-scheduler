#!/usr/bin/env bash
set -Eeuo pipefail

REPO=${JANUS_SIM_REPO:-/public_0/LYX/janus_simulator_accuracy_20260820}
OUT=${JANUS_SIM_DISCOVERY_OUT:-/public_0/LYX/janus_simulator_accuracy_outputs_20260820/discovery_v1}
PY=${JANUS_SIM_PYTHON:-/home/lyx/.conda/envs/opara/bin/python}
SCRIPT=$REPO/experiments/simulator_accuracy/discover_simulator_candidates.py

test -d "$OUT"
test ! -e "$OUT/aggregate.json"
completed=$(find "$OUT" -mindepth 2 -maxdepth 2 -name summary.json | wc -l)
if [[ "$completed" != 6 ]]; then
  echo "expected six completed pre-YOLO runs, found $completed" >&2
  exit 2
fi

models=(YOLOv8x ConvNeXt DeepFM BERT)
variants=(Baseline TD+DRT)
for model in "${models[@]}"; do
  model_slug=$(printf '%s' "$model" | tr '[:upper:]+' '[:lower:]_' | tr -cd '[:alnum:]_-')
  for variant in "${variants[@]}"; do
    if [[ "$variant" == "Baseline" ]]; then
      variant_slug=static_path_retry1
    else
      variant_slug=td_path_retry1
    fi
    target=$OUT/${model_slug}_${variant_slug}
    "$PY" "$SCRIPT" \
      --model "$model" \
      --reference-variant "$variant" \
      --max-ready 6 \
      --max-group-size 5 \
      --output-dir "$target"
  done
done

"$PY" - "$OUT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "model": payload["model"],
        "reference_variant": payload["reference_variant"],
        "scheduler_calls": payload["scheduler_call_count"],
        "candidates": payload["candidate_count"],
        **payload["positive_counts"],
        "strata": payload["strata"],
    })
if len(rows) != 14:
    raise SystemExit(f"expected 14 completed runs, found {len(rows)}")
summary = {
    "schema_version": 1,
    "protocol": "janus_4_7_paired_simulator_positive_discovery_v1",
    "run_count": len(rows),
    "runs": rows,
}
(root / "aggregate.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
