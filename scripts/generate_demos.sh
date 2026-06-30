#!/usr/bin/env bash
# Generate the simulated demonstration dataset with domain randomization.
# Wrapper around sim2real_soarm.data.record.
#
# Usage: scripts/generate_demos.sh [NUM_EPISODES] [extra record.py flags]
set -euo pipefail

NUM_EPISODES="${1:-500}"; shift || true
OUT="${OUT:-recordings/sim_redcubes_bluecup}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
uv run python -m sim2real_soarm.data.record \
  --num-episodes "${NUM_EPISODES}" \
  --out "${OUT}" \
  "$@"
