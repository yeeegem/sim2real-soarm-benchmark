#!/usr/bin/env bash
# (Scaffold, not run in the first iteration.) Finetune SmolVLA on the simulated
# dataset, mirroring the sibling smolvla-soarm-benchmark recipe. SmolVLA expects
# camera1/camera2 (+ an empty 3rd), so the front/wrist cameras are renamed; the
# resulting checkpoint is scored by the same soarm_eval.run (its loader
# dispatches SmolVLA automatically).
#
# Usage: scripts/train_smolvla.sh [extra lerobot-train flags]
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-recordings/sim_redcubes_bluecup}"
DATASET_REPO_ID="${DATASET_REPO_ID:-sim/sim_redcubes_bluecup}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/smolvla_sim}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"

uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --policy.empty_cameras=1 \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq=2000 \
  --policy.device=cuda \
  --seed=42 \
  --output_dir="${OUTPUT_DIR}" \
  "$@"

echo "Done. Evaluate on the real arm with:"
echo "  uv run python -m sim2real_soarm.soarm_eval.run \\"
echo "    --checkpoint ${OUTPUT_DIR}/checkpoints/last/pretrained_model --tier A"
