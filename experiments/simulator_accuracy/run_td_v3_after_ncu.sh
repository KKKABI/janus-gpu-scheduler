#!/usr/bin/env bash
set -Eeuo pipefail

NCU_PID_FILE=/public_0/LYX/janus_ncu_stability_outputs_20260820/formal_3repeat_v1.pid
while [[ -f "$NCU_PID_FILE" ]] && kill -0 "$(cat "$NCU_PID_FILE")" 2>/dev/null; do
  sleep 15
done

export JANUS_SIM_REPO=/public_0/LYX/janus_simulator_accuracy_20260820
export JANUS_SIM_MANIFEST=/public_0/LYX/janus_simulator_accuracy_outputs_20260820/td_final_v3_pair_holdout_sample_500_v2.json
export JANUS_SIM_VALIDATION_OUT=/public_0/LYX/janus_simulator_accuracy_outputs_20260820/td_final_v3_pair_holdout_hardware_500_v1
exec bash /public_0/LYX/janus_simulator_accuracy_20260820/experiments/simulator_accuracy/run_hardware_validation_v2.sh
