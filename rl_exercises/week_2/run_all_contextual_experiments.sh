#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/results/contextual_logs"

mkdir -p "${LOG_DIR}"

exercises=(
  "w2_contextual_tilt_only"
  "w2_contextual_friction_only"
  "w2_contextual_joint_variation"
  "w2_contextual_policy_tilt_only"
  "w2_contextual_policy_friction_only"
  "w2_contextual_policy_joint_variation"
  "w2_non_contextual_value_tilt_only"
  "w2_non_contextual_value_friction_only"
  "w2_non_contextual_value_joint_variation"
  "w2_non_contextual_policy_tilt_only"
  "w2_non_contextual_policy_friction_only"
  "w2_non_contextual_policy_joint_variation"
)

for exercise in "${exercises[@]}"; do
  log_file="${LOG_DIR}/${exercise}.log"
  echo "Running ${exercise}"
  (
    cd "${ROOT_DIR}"
    python rl_exercises/week_2/contextual_train_agent.py "+exercise=${exercise}"
  ) 2>&1 | tee "${log_file}"
done
