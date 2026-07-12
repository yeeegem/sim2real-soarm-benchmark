"""Interactively position the wrist camera OR its mount in the MuJoCo viewer.

You can't drag a body-attached part in the plain viewer, so this nudges the
target's pose with the keyboard, live, and prints the exact gripper-local
pos/quat (and fovy for the camera) to paste into configs/scene.yaml.

    uv run python scripts/tune_wrist.py --target camera   # frame the wrist cam
    uv run python scripts/tune_wrist.py --target mount     # place the mount STL

For --target camera the viewer opens looking THROUGH the wrist camera, so as you
nudge you see its framing change (match it to your real wrist camera). For
--target mount it opens in a free orbit view so you can see the mount clamp.

Keys (translate 2 mm, rotate 5 deg, in the gripper's local frame):
    translate  x: q/a   y: w/s   z: e/d
    rotate     x: r/f   y: t/g   z: y/h
    fovy (camera only): z/x        step size: +/-
    print pose: p        quit: Esc
"""

from __future__ import annotations

import argparse
import os

import numpy as np

os.environ["MUJOCO_GL"] = "glfw"  # windowed viewer

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from sim2real_soarm.sim import ik, kinematics as K  # noqa: E402
from sim2real_soarm.sim.scene import Layout, Scene  # noqa: E402


def _grasp_pose(scene: Scene):
    """Pose the arm with its gripper at the left cube so the wrist is close."""
    tcp = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    home = K.state_to_ctrl(np.array(scene.cfg["init_pose"]["state"]))
    sc = mujoco.MjData(scene.model)
    sc.qpos[:6] = home
    mujoco.mj_forward(scene.model, sc)
    cu = scene.cfg["cubes"]
    cx, cy = sum(cu["x_range"]) / 2, sum(cu["left_y_range"]) / 2
    q5 = ik.solve_ik(scene.model, sc, tcp, np.array([cx, cy, 0.02]), None,
                     ori_weight=1e-6, damping=0.05)
    scene.set_arm_state(np.append(K.qpos_to_state(np.append(q5, 0.0))[:5], 42.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["camera", "mount"], default="camera")
    ap.add_argument("--pose", choices=["home", "grasp"], default="home",
                    help="arm pose while tuning (default: configured init/home pose)")
    args = ap.parse_args()

    scene = Scene(make_renderer=False)
    m, d = scene.model, scene.data
    cu, cp = scene.cfg["cubes"], scene.cfg["cup"]

    def mid(r):
        return (r[0] + r[1]) / 2

    scene.reset(Layout((mid(cu["x_range"]), mid(cu["left_y_range"])),
                       (mid(cu["x_range"]), mid(cu["right_y_range"])),
                       (mid(cp["x_range"]), mid(cp["y_range"])), target="left"))
    if args.pose == "grasp":
        _grasp_pose(scene)

    if args.target == "mount":
        idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wrist_cam_mount_body")
        pos_arr, quat_arr, has_fovy, label = m.body_pos, m.body_quat, False, "cameras.wrist.mount"
    else:
        idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        pos_arr, quat_arr, has_fovy, label = m.cam_pos, m.cam_quat, True, "cameras.wrist"
    if idx < 0:
        raise SystemExit(f"{args.target} not found")

    state = {"step_t": 0.002, "step_r": np.deg2rad(5.0)}

    def show():
        p = np.round(pos_arr[idx], 4).tolist()
        q = np.round(quat_arr[idx], 4).tolist()
        extra = f"   fovy: {float(m.cam_fovy[idx]):.1f}" if has_fovy else ""
        print(f"  {label}:  pos: {p}   quat: {q}{extra}")

    def rotate_local(axis):
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, np.asarray(axis, float), state["step_r"])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, quat_arr[idx], dq)  # local-frame rotation
        quat_arr[idx] = out

    def key_cb(keycode):
        try:
            c = chr(keycode).lower()
        except ValueError:
            return
        t = state["step_t"]
        moves = {"q": (0, +t), "a": (0, -t), "w": (1, +t), "s": (1, -t),
                 "e": (2, +t), "d": (2, -t)}
        rots = {"r": [1, 0, 0], "f": [-1, 0, 0], "t": [0, 1, 0], "g": [0, -1, 0],
                "y": [0, 0, 1], "h": [0, 0, -1]}
        if c in moves:
            i, dv = moves[c]
            pos_arr[idx, i] += dv
            show()
        elif c in rots:
            rotate_local(rots[c])
            show()
        elif has_fovy and c in ("z", "x"):
            m.cam_fovy[idx] += 2.0 if c == "x" else -2.0
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
        if args.target == "camera":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = idx  # look THROUGH the wrist camera
        while viewer.is_running():
            mujoco.mj_forward(m, d)
            viewer.sync()


if __name__ == "__main__":
    main()
