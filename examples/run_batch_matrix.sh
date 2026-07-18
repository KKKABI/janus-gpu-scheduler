#!/usr/bin/env bash
set +e

models=(googlenet inception nasnet deepfm bert yolo convnext)
batch_sizes=(4 8 16)
python_bin="${JANUS_PYTHON:-python}"
timeout_s="${JANUS_MODEL_TIMEOUT:-1800}"

for batch_size in "${batch_sizes[@]}"; do
    for model in "${models[@]}"; do
        echo "MATRIX_START model=${model} batch_size=${batch_size} mode=${JANUS_SELECTION_MODE:-default}"
        timeout "${timeout_s}" "${python_bin}" examples/benchmark_model.py \
            --model "${model}" --batch-size "${batch_size}"
        rc=$?
        echo "MATRIX_END model=${model} batch_size=${batch_size} rc=${rc}"
    done
done
