"""Interactively position the wrist-camera mount in the MuJoCo viewer.

You can't drag a body-attached geom in the plain viewer, so this nudges the
mount's pose with the keyboard, updates it live, and prints the exact
gripper-local pos/quat to paste into configs/scene.yaml (cameras.wrist.mount).

    uv run python scripts/tune_mount.py            # arm at a grasp pose

Keys (translate 2 mm, rotate 5 deg; all in the gripper's local frame):
    translate  x: q/a   y: w/s   z: e/d
    rotate     x: r/f   y: t/g   z: y/h
    step size  +/-      print pose: p      quit: Esc
"""

from __future__ import annotations

import os

import numpy as np

os.environ["MUJOCO_GL"] = "glfw"  # windowed viewer

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from sim2real_soarm.sim import ik, kinematics as K  # noqa: E402
from sim2real_soarm.sim.scene import Layout, Scene  # noqa: E402


def _grasp_pose(scene: Scene):
    """Pose the arm with its gripper at the left cube so the wrist is visible."""
    tcp = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    home = K.state_to_ctrl(np.array(scene.cfg["init_pose"]["state"]))
    sc = mujoco.MjData(scene.model)
    sc.qpos[:6] = home
    mujoco.mj_forward(scene.model, sc)
    cu = scene.cfg["cubes"]
    cx = sum(cu["x_range"]) / 2
    cy = sum(cu["left_y_range"]) / 2
    q5 = ik.solve_ik(scene.model, sc, tcp, np.array([cx, cy, 0.02]), None,
                     ori_weight=1e-6, damping=0.05)
    scene.set_arm_state(np.append(K.qpos_to_state(np.append(q5, 0.0))[:5], 42.0))


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose", choices=["home", "grasp"], default="home",
                    help="arm pose while tuning (default: the configured init/home pose)")
    args = ap.parse_args()

    scene = Scene(make_renderer=False)
    cu, cp = scene.cfg["cubes"], scene.cfg["cup"]

    def mid(r):
        return (r[0] + r[1]) / 2

    # reset() puts the arm at the configured init_pose (home).
    scene.reset(Layout((mid(cu["x_range"]), mid(cu["left_y_range"])),
                       (mid(cu["x_range"]), mid(cu["right_y_range"])),
                       (mid(cp["x_range"]), mid(cp["y_range"])), target="left"))
    if args.pose == "grasp":
        _grasp_pose(scene)  # reach a cube, so the wrist is lower / easy to see
    m, d = scene.model, scene.data

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_cam_mount_body")
    if bid < 0:
        raise SystemExit("wrist_cam_mount_body not found")

    state = {"step_t": 0.002, "step_r": np.deg2rad(5.0)}

    def show():
        p = np.round(m.body_pos[bid], 4).tolist()
        q = np.round(m.body_quat[bid], 4).tolist()
        print(f"  mount.pos: {p}   mount.quat: {q}")

    def rotate_local(axis):
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, np.asarray(axis, float), state["step_r"])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, m.body_quat[bid], dq)  # local-frame rotation
        m.body_quat[bid] = out

    def key_cb(keycode):
        try:
            c = chr(keycode).lower()
        except ValueError:
            return
        t = state["step_t"]
        moves = {
            "q": ("t", 0, +t), "a": ("t", 0, -t),
            "w": ("t", 1, +t), "s": ("t", 1, -t),
            "e": ("t", 2, +t), "d": ("t", 2, -t),
        }
        rots = {"r": [1, 0, 0], "f": [-1, 0, 0], "t": [0, 1, 0], "g": [0, -1, 0],
                "y": [0, 0, 1], "h": [0, 0, -1]}
        if c in moves:
            _, i, dv = moves[c]
            m.body_pos[bid, i] += dv
            show()
        elif c in rots:
            rotate_local(rots[c])
            show()
        elif c == "p":
            show()
        elif c in ("+", "="):
            state["step_t"] *= 2; state["step_r"] *= 2
            print(f"  step: {state['step_t']*1000:.1f} mm / {np.rad2deg(state['step_r']):.1f} deg")
        elif c == "-":
            state["step_t"] /= 2; state["step_r"] /= 2
            print(f"  step: {state['step_t']*1000:.1f} mm / {np.rad2deg(state['step_r']):.1f} deg")

    print(__doc__)
    print("Starting pose:")
    show()
    with mujoco.viewer.launch_passive(m, d, key_callback=key_cb) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(m, d)
            viewer.sync()


if __name__ == "__main__":
    main()
