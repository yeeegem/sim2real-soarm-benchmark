#!/usr/bin/env bash
# Train LeRobot ACT on the simulated dataset. ACT consumes the front+wrist
# cameras and the 6-DOF state directly (no camera renaming needed, unlike
# SmolVLA), and the dataset is schema-identical to the real one, so the
# resulting checkpoint deploys zero-shot via soarm_eval.run.
#
# Usage: scripts/train_act.sh [extra lerobot-train flags]
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-recordings/sim_redcubes_bluecup}"
DATASET_REPO_ID="${DATASET_REPO_ID:-sim/sim_redcubes_bluecup}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/act_sim}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

uv run lerobot-train \
  --policy.type=act \
  --policy.push_to_hub=false \
  --policy.chunk_size=100 \
  --policy.n_action_steps=20 \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq=5000 \
  --log_freq=200 \
  --policy.device=cuda \
  --seed=42 \
  --output_dir="${OUTPUT_DIR}" \
  "$@"

echo "Done. Evaluate on the real arm with:"
echo "  uv run python -m sim2real_soarm.soarm_eval.run \\"
echo "    --checkpoint ${OUTPUT_DIR}/checkpoints/last/pretrained_model --tier A"
