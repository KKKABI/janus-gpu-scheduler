#!/bin/bash
PYTHON=/home/lyx/.conda/envs/opara/bin/python
export PYTHONUNBUFFERED=1
HELPER=/public_0/LYX/janus/_bench_helper.py
OUTFILE=/tmp/full_benchmark.txt
> $OUTFILE

echo "Model,Strategy,Alpha,Median,Min,Max" | tee -a $OUTFILE

STRATEGIES=(
  "Baseline|/public_0/LYX/janus_original_baseline"
  "Cosine|/public_0/LYX/janus"
  "MinRes|/public_0/LYX/janus"
  "DRT|/public_0/LYX/janus_static_interference"
)

MODELS="GoogLeNet DeepFM Inception-v3 BERT NASNet YOLOv8x ConvNeXt"

for MODEL in $MODELS; do
  for ENTRY in "${STRATEGIES[@]}"; do
    SNAME="${ENTRY%%|*}"
    SWT="${ENTRY##*|}"

    if [ "$SNAME" = "Baseline" ]; then
      ALPHAS="0.9"  # baseline ignores alpha
    else
      ALPHAS="0.9 0.8 0.5 0.2"
    fi

    for ALPHA in $ALPHAS; do
      echo ">>> $MODEL $SNAME α=$ALPHA" >&2
      RESULT=$(cd "$SWT" && $PYTHON "$HELPER" "$MODEL" "$SNAME" "$ALPHA" 2>/dev/null | tail -1)
      if [ -n "$RESULT" ]; then
        echo "$MODEL,$SNAME,$ALPHA,$RESULT" | tee -a $OUTFILE
      else
        echo "$MODEL,$SNAME,$ALPHA,FAILED" | tee -a $OUTFILE
      fi
    done
  done
done

echo "===== DONE =====" | tee -a $OUTFILE
