"""Tiered, operator-scored evaluation of a sim-trained policy on the SO-ARM101.

Runs ``num_trials_per_tier`` episodic rollouts of one checkpoint for one tier,
prompts the operator to score each trial, and appends results to a CSV with the
*same schema* as the two sibling repos -- so a sim-to-real result drops straight
into the same comparison table. Resumes automatically (already-logged trials for
the tier are skipped).

Works with either an ACT or a SmolVLA checkpoint (``load_policy`` dispatches on
the checkpoint's config type).

Usage::

    uv run python -m sim2real_soarm.soarm_eval.run \\
        --checkpoint runs/act_sim/checkpoints/last/pretrained_model \\
        --config configs/eval_real.yaml --tier A [--trials 30]

Aggregate CSV(s) with ``python -m sim2real_soarm.soarm_eval.metrics``.
On-arm runs are executed by the operator, not in CI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

from sim2real_soarm.soarm_eval.harness import EvalHarness, load_results
from sim2real_soarm.soarm_eval.infer import connect_robot, load_policy, read_pose

DEFAULT_CHECKPOINT = "runs/act_sim/checkpoints/last/pretrained_model"
DEFAULT_CONFIG = "configs/eval_real.yaml"


def _set_override(cfg: dict, dotted_key: str, value: str) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(value)


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load the YAML eval config and apply ``key.nested=value`` CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        key, value = ov.split("=", 1)
        _set_override(cfg, key, value)
    return cfg


def resolve_task(cfg: dict) -> str:
    """Use the configured instruction, or read it from the dataset metadata."""
    configured = cfg.get("eval", {}).get("task")
    if configured:
        return str(configured)
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    ds = cfg["dataset"]
    meta = LeRobotDatasetMetadata(ds["repo_id"], root=ds.get("root"))
    task = str(meta.tasks.index[0])
    print(f"Using task instruction from dataset metadata: {task!r}")
    return task


def resolve_init_pose(cfg: dict, robot, motor_names: list[str]) -> np.ndarray:
    configured = cfg.get("infer", {}).get("init_pose")
    if configured is not None:
        return np.asarray(list(configured), dtype=np.float32)
    print("infer.init_pose is null: capturing the arm's current pose as the init pose.")
    return read_pose(robot, motor_names)


def dataset_features(cfg: dict) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    ds = cfg["dataset"]
    meta = LeRobotDatasetMetadata(ds["repo_id"], root=ds.get("root"))
    return meta.features


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Tiered sim-to-real evaluation on SO-ARM101.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to an ACT or SmolVLA pretrained_model/ dir.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to eval config YAML.")
    parser.add_argument("--tier", required=True, choices=["A", "B", "C"], help="Evaluation tier.")
    parser.add_argument("--trials", type=int, default=None,
                        help="Trials for this tier (default: eval.num_trials_per_tier).")
    parser.add_argument("--output", default=None,
                        help="Output dir for results.csv (default: <checkpoint>/../../eval).")
    parser.add_argument("--debug-frames", action="store_true",
                        help="Dump the first trial's camera frames (as the policy sees them).")
    args, overrides = parser.parse_known_args(argv)

    cfg = load_config(args.config, overrides or None)
    num_trials = args.trials if args.trials is not None else int(cfg["eval"]["num_trials_per_tier"])

    checkpoint = Path(args.checkpoint)
    run_dir = checkpoint.parent.parent.parent  # <run>/checkpoints/<step>/pretrained_model -> <run>
    output_dir = Path(args.output) if args.output else run_dir / "eval"

    existing = [r for r in load_results(output_dir / "results.csv") if r.tier == args.tier]
    start_idx = len(existing)
    if start_idx >= num_trials:
        print(f"Tier {args.tier} already has {start_idx}/{num_trials} trials. Nothing to do.")
        return
    if start_idx > 0:
        print(f"Resuming tier {args.tier} at trial {start_idx + 1}/{num_trials}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    task = resolve_task(cfg)
    features = dataset_features(cfg)
    policy, preprocessor, postprocessor = load_policy(checkpoint, device)
    robot, motor_names = connect_robot(cfg)

    try:
        init_pose = resolve_init_pose(cfg, robot, motor_names)
        harness = EvalHarness(
            policy, preprocessor, postprocessor, features, task,
            robot, motor_names, device, init_pose,
            fps=float(cfg["dataset"]["fps"]),
            max_steps=int(cfg["eval"]["max_episode_steps"]),
            dump_frames_dir=(output_dir / "debug_frames") if args.debug_frames else None,
        )
        results = harness.run_tier(args.tier, num_trials, output_dir, start_idx=start_idx)
        print(f"\nSaved {len(results)} trial(s) to {output_dir / 'results.csv'}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
