"""Inspect the MuJoCo scene: launch the interactive viewer, or render the
front/wrist/overview cameras to PNGs.

Interactive (orbit with the mouse; in the viewer's right panel you can switch
the active camera to 'front'/'wrist' to see exactly what the policy sees)::

    uv run python scripts/view_scene.py --interactive

Render stills to ./scene_views/ (works headless)::

    uv run python scripts/view_scene.py            # front.png, wrist.png, overview.png

Add --pose grasp to pose the arm reaching a cube instead of at the home pose.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from sim2real_soarm.sim.scene import Layout, Scene


def _layout_from_cfg(cfg) -> Layout:
    """A representative layout at the midpoints of the configured ranges, so the
    preview always reflects configs/scene.yaml (cube/cup placement)."""
    def mid(rng):
        return (rng[0] + rng[1]) / 2

    cu, cp = cfg["cubes"], cfg["cup"]
    return Layout(
        cube_left_xy=(mid(cu["x_range"]), mid(cu["left_y_range"])),
        cube_right_xy=(mid(cu["x_range"]), mid(cu["right_y_range"])),
        cup_xy=(mid(cp["x_range"]), mid(cp["y_range"])),
        target="left",
    )


def _pose_grasp(scene: Scene):
    """Pose the arm with its gripper at the left cube (so the wrist cam frames it)."""
    import mujoco

    from sim2real_soarm.sim import ik, kinematics as K

    tcp = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    home = K.state_to_ctrl(np.array(scene.cfg["init_pose"]["state"]))
    sc = mujoco.MjData(scene.model)
    sc.qpos[:6] = home
    mujoco.mj_forward(scene.model, sc)
    q5 = ik.solve_ik(scene.model, sc, tcp, np.array([0.25, 0.10, 0.02]), None,
                     ori_weight=1e-6, damping=0.05)
    scene.set_arm_state(np.append(K.qpos_to_state(np.append(q5, 0.0))[:5], 42.0))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interactive", action="store_true",
                   help="launch the interactive GLFW viewer instead of rendering")
    p.add_argument("--pose", choices=["home", "grasp"], default="home")
    p.add_argument("--out", default="scene_views")
    args = p.parse_args(argv)

    if args.interactive:
        # GLFW windowed backend for the live viewer.
        os.environ["MUJOCO_GL"] = "glfw"
        import mujoco.viewer

        scene = Scene(make_renderer=False)
        scene.reset(_layout_from_cfg(scene.cfg))
        if args.pose == "grasp":
            _pose_grasp(scene)
        print("Opening viewer. In the right panel, expand 'Rendering' -> 'Camera' "
              "and pick 'front' or 'wrist' to see the policy's views. Close the "
              "window to exit.")
        mujoco.viewer.launch(scene.model, scene.data)
        return

    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v3 as iio

    scene = Scene()
    scene.reset(_layout_from_cfg(scene.cfg))
    if args.pose == "grasp":
        _pose_grasp(scene)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for cam in ("front", "wrist"):
        iio.imwrite(out / f"{cam}.png", scene.render(cam))
    # Free overview camera (no 'camera=' -> default orbit view).
    scene._renderer.update_scene(scene.data)
    iio.imwrite(out / "overview.png", scene._renderer.render())
    scene.close()

    # Print the configured camera poses so they are easy to tweak.
    cam = scene.cfg["cameras"]
    print(f"Wrote {out}/front.png, wrist.png, overview.png")
    print("\nConfigured camera poses (configs/scene.yaml -> cameras):")
    print(f"  front: pos={cam['front']['pos']} lookat={cam['front']['lookat']} fovy={cam['front']['fovy']}")
    print(f"  wrist: pos(local)={cam['wrist']['pos']} lookat_local={cam['wrist']['lookat_local']} fovy={cam['wrist']['fovy']}")


if __name__ == "__main__":
    main()
