#!/bin/bash
PY=/home/lyx/.conda/envs/opara/bin/python
OUT=/tmp/full_bench_v3.txt
# Append to existing file (already has header + GoogLeNet 13 rows + DeepFM Baseline)
# Run remaining models first, DeepFM last

for M in Inception-v3 BERT NASNet YOLOv8x ConvNeXt DeepFM; do
  echo "========== $M ==========" >&2

  echo "  [Baseline]" >&2
  cd /public_0/LYX/janus_original_baseline && $PY /public_0/LYX/janus/_bench_baseline.py $M $OUT 2>>/tmp/bench_err3.txt

  echo "  [Cosine + MinRes]" >&2
  cd /public_0/LYX/janus && $PY /public_0/LYX/janus/_bench_dirB.py $M $OUT 2>>/tmp/bench_err3.txt

  echo "  [DRT]" >&2
  cd /public_0/LYX/janus_static_interference && $PY /public_0/LYX/janus/_bench_drt.py $M $OUT 2>>/tmp/bench_err3.txt

  echo "  Done $M" >&2
done

echo "===== ALL DONE =====" >&2
cat $OUT
