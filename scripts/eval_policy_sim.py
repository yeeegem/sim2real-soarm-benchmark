"""Closed-loop sim evaluation of a trained policy -- the decisive diagnostic for
zero-shot sim-to-real failures.

The scored benchmark is the real-arm ``soarm_eval`` harness. But when a sim-trained
policy misses the cube on the real arm, the first question is *where* the failure
lives. Running the checkpoint **closed-loop in the exact sim it trained on** (same
LeRobot inference path as the real harness: ``build_dataset_frame`` ->
``predict_action``) separates three cases:

  * **Ignores vision** (causal confusion): the gripper drives to a fixed point and
    misses cubes even in sim -> large closest-reach to the chosen cube.
  * **Mode collapse**: vision works (reaches the *chosen* cube precisely across
    randomized positions) but the policy always picks the same side -> pick-side
    distribution is one-sided, |P(left)-0.5| ~ 0.5.
  * **Visual/camera gap**: works well in sim but still fails on the real arm ->
    the sim vision doesn't transfer; fix DR + camera calibration (Stage 2).

Grasp oracle: recording used a *weld* grasp fired by the scripted expert at a
fixed step (``scene.attach``); a learned policy has no such hook, and the arm is
non-colliding with the cubes by design, so it could never physically pick a cube
in sim. To still get an end-to-end signal that reflects *positioning*, we weld the
nearest cube to the gripper the moment the policy closes the gripper while the
fingertip (``tcp``) is within ``--grasp-radius`` of it, and release it when the
gripper opens. So a "success" means the policy positioned on a cube, closed,
carried it to the cup, and opened -- with the unreliable frictional pinch
abstracted away. Which cube it welds is recorded as the picked side.

Usage::

    MUJOCO_GL=egl uv run python scripts/eval_policy_sim.py \\
        --checkpoint runs/act_sim/checkpoints/last/pretrained_model \\
        --episodes 40 [--video rollout.mp4 --video-episodes 4]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


def _to_vec(action_values) -> np.ndarray:
    """Coerce predict_action's output (torch tensor or array) to a 1-D float array."""
    try:
        import torch

        if isinstance(action_values, torch.Tensor):
            return action_values.detach().to("cpu").float().numpy().reshape(-1)
    except ImportError:
        pass
    return np.asarray(action_values, dtype=np.float32).reshape(-1)


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Write frames to mp4 (preferred) or gif, whichever the backend supports."""
    import imageio.v3 as iio

    path = Path(path)
    try:
        iio.imwrite(path, frames, fps=fps)
        print(f"Saved rollout video -> {path}  ({len(frames)} frames)")
    except Exception as e:  # noqa: BLE001 - fall back to a gif if ffmpeg is missing
        gif = path.with_suffix(".gif")
        iio.imwrite(gif, frames, duration=1000 / fps, loop=0)
        print(f"mp4 write failed ({e}); saved gif -> {gif}  ({len(frames)} frames)")


def run(
    checkpoint: str,
    episodes: int,
    max_steps: int,
    seed: int,
    task: str | None,
    grasp_radius: float,
    grip_closed_below: float,
    n_substeps: int,
    video: str | None,
    video_episodes: int,
) -> dict:
    import torch
    from lerobot.datasets.feature_utils import build_dataset_frame
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action

    from sim2real_soarm.data.record import DEFAULT_TASK, MOTOR_NAMES, build_features
    from sim2real_soarm.sim.expert import sample_layout
    from sim2real_soarm.sim.scene import Scene
    from sim2real_soarm.soarm_eval.infer import load_policy

    task = task or DEFAULT_TASK
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | checkpoint: {checkpoint}")
    print(f"Task: {task!r}")

    policy, preprocessor, postprocessor = load_policy(checkpoint, device)
    robot_type = "so_follower"
    features = build_features()

    scene = Scene()
    rng = np.random.default_rng(seed)

    rows = []
    n_success = 0
    picked = {"left": 0, "right": 0}     # which cube the grasp oracle welded
    landed = {"left": 0, "right": 0}     # which cube ended in the cup
    video_frames: list[np.ndarray] = []

    for ep in range(episodes):
        # Alternate which side is forced closer, so cube layouts span both sides
        # symmetrically (the policy has no target input -- it picks freely).
        layout = sample_layout(scene.cfg, rng, target="left" if ep % 2 == 0 else "right")
        scene.reset(layout)
        policy.reset()

        cube_xy = {
            "left": np.array(layout.cube_left_xy, float),
            "right": np.array(layout.cube_right_xy, float),
        }
        attached: str | None = None
        picked_side: str | None = None
        reach = {"left": np.inf, "right": np.inf}   # closest fingertip-xy approach
        record_this = video is not None and ep < video_episodes

        for _ in range(max_steps):
            state = scene.get_state().astype(np.float32)
            obs = {name: float(v) for name, v in zip(MOTOR_NAMES, state)}
            obs["front"] = scene.render("front")
            obs["wrist"] = scene.render("wrist")
            if record_this:
                video_frames.append(np.concatenate([obs["front"], obs["wrist"]], axis=1))
            frame = build_dataset_frame(features, obs, prefix=OBS_STR)

            action = _to_vec(
                predict_action(
                    observation=frame, policy=policy, device=device,
                    preprocessor=preprocessor, postprocessor=postprocessor,
                    use_amp=bool(getattr(policy.config, "use_amp", False)),
                    task=task, robot_type=robot_type,
                )
            )
            scene.step(action, n_substeps=n_substeps)

            tcp = scene.tcp_xpos()
            for side in ("left", "right"):
                reach[side] = min(reach[side],
                                  float(np.hypot(tcp[0] - cube_xy[side][0],
                                                 tcp[1] - cube_xy[side][1])))

            # Proximity grasp oracle: weld nearest cube on close, release on open.
            grip = action[5] if action.size >= 6 else state[5]
            if attached is None and grip < grip_closed_below:
                for side in ("left", "right"):
                    if float(np.linalg.norm(tcp - scene.body_xpos(f"cube_{side}"))) < grasp_radius:
                        scene.attach(side)
                        attached = picked_side = side
                        break
            elif attached is not None and grip >= grip_closed_below:
                scene.detach(attached)
                attached = None

        last = scene.get_state().astype(np.float32)
        for _ in range(40):
            scene.step(last, n_substeps=n_substeps)
        landed_side = next((c for c in ("left", "right") if scene.cube_in_cup(c)), None)
        success = landed_side is not None
        n_success += int(success)
        if picked_side:
            picked[picked_side] += 1
        if landed_side:
            landed[landed_side] += 1

        nearest = min(reach["left"], reach["right"])
        rows.append({"nearest_cm": 100 * nearest, "picked": picked_side,
                     "landed": landed_side, "success": success})
        print(f"ep {ep + 1:>3}/{episodes} "
              f"| reach L={100 * reach['left']:5.1f} R={100 * reach['right']:5.1f} cm "
              f"| picked={picked_side or '-':<5} | success={success} "
              f"(landed={landed_side})", flush=True)

    if video and video_frames:
        # Each tick advances ~n_substeps*dt (~1/30 s at the default 17), so 30 fps
        # plays back close to real time.
        _save_video(video_frames, Path(video), fps=30)
    scene.close()

    # -- aggregate diagnostics ----------------------------------------------
    succ = n_success / max(episodes, 1)
    med_nearest = float(np.median([r["nearest_cm"] for r in rows]))
    n_picked = picked["left"] + picked["right"]
    p_left_pick = picked["left"] / max(n_picked, 1)
    p_left_land = landed["left"] / max(n_success, 1)

    print("\n==================== sim closed-loop diagnostic ====================")
    print(f"episodes             : {episodes}")
    print(f"success rate         : {succ:.2f}  ({n_success}/{episodes})")
    print(f"pick side (grasped)  : left={picked['left']} right={picked['right']} "
          f"| |P(left)-0.5|={abs(p_left_pick - 0.5):.2f}")
    print(f"land side (in cup)   : left={landed['left']} right={landed['right']} "
          f"| |P(left)-0.5|={abs(p_left_land - 0.5):.2f}")
    print(f"reach to chosen cube : median={med_nearest:.1f} cm  "
          f"(small => vision locates the chosen cube; large => ignores vision)")
    print("--------------------------------------------------------------------")
    one_sided = abs(p_left_pick - 0.5) > 0.35 and n_picked >= 4
    if med_nearest > 4.0:
        print("READ: gripper does NOT reach the chosen cube even in sim")
        print("      -> causal confusion (ignores vision). Prioritize Stage 1.")
    elif one_sided:
        print("READ: vision WORKS (reaches the chosen cube precisely) but the")
        print("      policy MODE-COLLAPSES to one side -> the mode-balance failure.")
        print("      If the real arm also misses entirely, a visual/camera gap is")
        print("      stacked on top. Do Stage 1 (balance) + Stage 2 (camera).")
    else:
        print("READ: vision works and both modes are used in sim. If the real arm")
        print("      still misses, the gap is visual/camera -> prioritize Stage 2.")
    print("====================================================================")

    return {"success_rate": succ, "median_reach_cm": med_nearest,
            "picked": picked, "landed": landed}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",
                   default="runs/act_sim/checkpoints/last/pretrained_model",
                   help="ACT or SmolVLA pretrained_model/ dir.")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=250,
                   help="policy ticks per episode (recorded demos are ~175).")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--task", default=None,
                   help="language instruction (default: the recorder's DEFAULT_TASK).")
    p.add_argument("--grasp-radius", type=float, default=0.035,
                   help="fingertip-to-cube distance (m) that triggers the grasp oracle.")
    p.add_argument("--grip-closed-below", type=float, default=30.0,
                   help="gripper command (RANGE_0_100) below which counts as 'closed'.")
    p.add_argument("--n-substeps", type=int, default=17,
                   help="MuJoCo substeps per tick (17 ~= 30 fps, matches recording).")
    p.add_argument("--video", default=None,
                   help="write a front|wrist rollout video here (mp4, or gif fallback).")
    p.add_argument("--video-episodes", type=int, default=4,
                   help="how many of the first episodes to record into --video.")
    args = p.parse_args(argv)

    run(checkpoint=args.checkpoint, episodes=args.episodes, max_steps=args.max_steps,
        seed=args.seed, task=args.task, grasp_radius=args.grasp_radius,
        grip_closed_below=args.grip_closed_below, n_substeps=args.n_substeps,
        video=args.video, video_episodes=args.video_episodes)


if __name__ == "__main__":
    main()
