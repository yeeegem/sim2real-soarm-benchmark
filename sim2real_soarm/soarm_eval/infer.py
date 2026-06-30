"""Real-robot inference for a sim-trained policy on the SO-ARM101.

Ported from the sibling ``smolvla-soarm-benchmark/soarm_eval/infer.py`` so the
arm is driven, homed, and scored identically -- the results are therefore
directly comparable across all three repos. The only change is ``load_policy``,
which now dispatches on the checkpoint's policy type (ACT or SmolVLA) instead of
hard-coding SmolVLA, since this repo trains ACT first.

Both ACT and SmolVLA expose ``select_action`` and manage their own action-chunk
queue, so the control loop (``run_episode``) is the same plain per-tick
``get_observation -> predict_action -> send_action`` at ``fps``. We reuse
LeRobot's canonical inference helpers (``build_dataset_frame``,
``predict_action``, ``make_robot_action``).
"""

from __future__ import annotations

import collections
import enum
import json
import select
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import build_dataset_frame
from lerobot.policies.utils import make_robot_action
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action


# ---------------------------------------------------------------------------
# Policy construction (ACT or SmolVLA)
# ---------------------------------------------------------------------------

def _policy_class(checkpoint_dir: str):
    """Pick the LeRobot policy class from the checkpoint's config.json type."""
    ptype = json.loads((Path(checkpoint_dir) / "config.json").read_text()).get("type", "").lower()
    if ptype == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        return ACTPolicy
    if ptype == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAPolicy
    raise ValueError(f"Unsupported policy type {ptype!r} in {checkpoint_dir}/config.json")


def load_policy(checkpoint_dir: str | Path, device: torch.device):
    """Load a sim-trained ACT or SmolVLA checkpoint and its pre/post processors.

    ``checkpoint_dir`` is a LeRobot ``pretrained_model/`` directory (the format
    written by ``lerobot-train``). Normalization and the front/wrist camera
    remapping are baked into the saved processors. Returns
    ``(policy, preprocessor, postprocessor)``.
    """
    from lerobot.policies.factory import make_pre_post_processors

    checkpoint_dir = str(checkpoint_dir)
    Policy = _policy_class(checkpoint_dir)
    policy = Policy.from_pretrained(checkpoint_dir)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        checkpoint_dir,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor


# ---------------------------------------------------------------------------
# Arm pose helpers (shared with the eval harness)
# ---------------------------------------------------------------------------

def read_pose(robot: SOFollower, motor_names: list[str]) -> np.ndarray:
    """Return the arm's current joint positions in *motor_names* order."""
    obs = robot.get_observation()
    return np.array([float(obs[f"{m}.pos"]) for m in motor_names], dtype=np.float32)


def move_to_init(
    robot: SOFollower,
    motor_names: list[str],
    init_pose: np.ndarray,
    fps: float,
    settle_steps: int = 45,
) -> None:
    """Drive the arm to a fixed init pose and hold it until it settles."""
    step_duration = 1.0 / fps
    action_dict = {f"{m}.pos": float(v) for m, v in zip(motor_names, init_pose)}
    for _ in range(settle_steps):
        step_t0 = time.perf_counter()
        robot.send_action(action_dict)
        remaining = step_duration - (time.perf_counter() - step_t0)
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# Control loop
# ---------------------------------------------------------------------------

class StopReason(str, enum.Enum):
    """Why an episode ended."""

    OPERATOR = "operator"   # operator pressed Enter (task done or clearly failed)
    TIMEOUT = "timeout"     # hit max_steps without operator stop


def _enter_pressed() -> bool:
    """Non-blocking check for a newline on stdin (operator pressed Enter)."""
    if not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


def _dump_obs_frames(observation_frame: dict, dump_dir: Path) -> None:
    """Save each camera image from a built obs frame to PNG for inspection."""
    from PIL import Image

    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    for key, value in observation_frame.items():
        if not key.startswith(f"{OBS_STR}.images."):
            continue
        cam = key.removeprefix(f"{OBS_STR}.images.")
        arr = np.asarray(value)
        path = dump_dir / f"{cam}.png"
        Image.fromarray(arr).save(path)
        print(f"  dumped {key} {arr.shape} {arr.dtype} -> {path}")


def run_episode(
    policy,
    preprocessor,
    postprocessor,
    features: dict,
    task: str,
    robot: SOFollower,
    motor_names: list[str],
    device: torch.device,
    fps: float,
    max_steps: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    dump_frames_dir: Path | None = None,
) -> tuple[StopReason, float]:
    """Run one episode and return ``(stop_reason, duration_s)``.

    Each control tick: capture an observation, build the LeRobot obs frame,
    ``predict_action`` (preprocess -> ``policy.select_action`` -> postprocess) to
    one denormalised 6-DOF action, send it, paced to ``1/fps``. Works for any
    LeRobot policy that exposes ``select_action`` (ACT, SmolVLA). The episode
    ends on *should_stop* (OPERATOR) or after *max_steps* (TIMEOUT).
    """
    if should_stop is None:
        should_stop = _enter_pressed

    step_duration = 1.0 / fps
    use_amp = bool(getattr(policy.config, "use_amp", False))
    robot_type = getattr(robot, "robot_type", None)

    policy.reset()  # drop any queued actions from the previous trial
    latencies: collections.deque = collections.deque(maxlen=100)
    t_start = time.perf_counter()
    step_count = 0

    while True:
        step_t0 = time.perf_counter()

        obs = robot.get_observation()
        observation_frame = build_dataset_frame(features, obs, prefix=OBS_STR)

        if dump_frames_dir is not None and step_count == 0:
            print(f"Dumping first-tick camera frames to {dump_frames_dir}:")
            _dump_obs_frames(observation_frame, dump_frames_dir)

        infer_t0 = time.perf_counter()
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )
        latency_ms = (time.perf_counter() - infer_t0) * 1000
        latencies.append(latency_ms)
        arr = list(latencies)
        print(
            f"inference {latency_ms:.1f} ms | mean {np.mean(arr):.1f} "
            f"| p95 {np.percentile(arr, 95):.1f} | max {np.max(arr):.1f}"
        )

        robot_action = make_robot_action(action_values, features)
        robot.send_action(robot_action)

        step_count += 1
        if should_stop():
            return StopReason.OPERATOR, time.perf_counter() - t_start
        if max_steps is not None and step_count >= max_steps:
            return StopReason.TIMEOUT, time.perf_counter() - t_start

        remaining = step_duration - (time.perf_counter() - step_t0)
        if remaining > 0:
            time.sleep(remaining)


# ---------------------------------------------------------------------------
# Robot setup
# ---------------------------------------------------------------------------

def connect_robot(cfg: dict) -> tuple[SOFollower, list[str]]:
    """Connect to the SO-ARM101 follower and set the Feetech acceleration ramp.

    *cfg* is the parsed ``configs/eval_real.yaml`` dict. Returns the connected
    robot and its motor names (in state-vector order).
    """
    infer = cfg["infer"]
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=cam_cfg["path"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
            fourcc=cam_cfg.get("fourcc"),
            backend=cam_cfg.get("backend", "auto"),
        )
        for name, cam_cfg in infer["cameras"].items()
    }
    robot_cfg = SOFollowerRobotConfig(
        port=infer["robot_port"],
        id=infer["robot_id"],
        cameras=cameras,
    )
    robot = SOFollower(robot_cfg)
    robot.connect()
    motor_names = list(robot.bus.motors.keys())
    print(f"Robot connected. Motors: {motor_names}")

    accel = infer.get("motor_acceleration")
    if accel is not None:
        for motor in motor_names:
            robot.bus.write("Acceleration", motor, int(accel))
        print(f"Wrote Acceleration={int(accel)} to all {len(motor_names)} motors.")

    return robot, motor_names
