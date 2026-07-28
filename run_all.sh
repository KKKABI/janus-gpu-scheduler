#!/bin/bash
# Run all models through all 3 worktrees, one model at a time
PY=/home/lyx/.conda/envs/opara/bin/python
OUT=/tmp/full_bench_v3.txt
> $OUT
echo "Model,Strategy,Alpha,Median,Min,Max" > $OUT

MODELS="GoogLeNet DeepFM Inception-v3 BERT NASNet YOLOv8x ConvNeXt"

for M in $MODELS; do
  echo "========== $M ==========" >&2

  echo "  [Baseline]" >&2
  cd /public_0/LYX/janus_original_baseline && $PY /public_0/LYX/janus/_bench_baseline.py $M $OUT 2>>/tmp/bench_err.txt

  echo "  [Cosine + MinRes]" >&2
  cd /public_0/LYX/janus && $PY /public_0/LYX/janus/_bench_dirB.py $M $OUT 2>>/tmp/bench_err.txt

  echo "  [DRT]" >&2
  cd /public_0/LYX/janus_static_interference && $PY /public_0/LYX/janus/_bench_drt.py $M $OUT 2>>/tmp/bench_err.txt

  echo "  Done $M" >&2
done

echo "===== ALL DONE =====" >&2
cat $OUT
