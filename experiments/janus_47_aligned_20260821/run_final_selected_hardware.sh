#!/usr/bin/env bash
set -Eeuo pipefail

export JANUS_SIM_REPO=/public_0/LYX/janus_simulator_accuracy_20260820
export JANUS_SIM_MANIFEST=/public_0/LYX/janus_47_aligned_outputs_20260821/final_selected_sample_v1/manifest.json
export JANUS_SIM_VALIDATION_OUT=/public_0/LYX/janus_47_aligned_outputs_20260821/final_selected_hardware_v1
export JANUS_SIM_PYTHON=/home/lyx/.conda/envs/opara/bin/python
export JANUS_SIM_NSYS=/opt/nvidia/nsight-systems/2024.3.1/target-linux-x64/nsys

exec bash /public_0/LYX/janus_simulator_accuracy_20260820/experiments/simulator_accuracy/run_hardware_validation_v2.sh
