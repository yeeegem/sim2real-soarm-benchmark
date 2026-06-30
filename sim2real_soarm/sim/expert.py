"""Scripted IK pick-and-place expert that seeds the bimodal distribution.

Given a :class:`~sim2real_soarm.sim.scene.Layout`, the expert plans a feed-forward
joint-space setpoint trajectory (in LeRobot units) by solving IK at a handful of
key poses and interpolating between them. A separate runner (the recorder)
applies the setpoints to the scene at 30 Hz and logs observations, so the expert
stays pure planning and is deterministic given a seed.

Bimodality: the *target* cube (left/right) is chosen by the layout sampler 50/50,
so a policy trained on the demos sees both modes in balance -- the whole point of
the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2real_soarm.sim import ik, kinematics as K
from sim2real_soarm.sim.scene import Layout, Scene

CONTROL_HZ = 30.0


@dataclass
class ExpertConfig:
    grip_open: float = 42.0          # RANGE_0_100 (matches dataset open band)
    grip_closed: float = 4.0         # squeezes a 3 cm cube via actuator force
    pregrasp_z: float = 0.075
    grasp_z: float = 0.020
    lift_z: float = 0.12
    transport_z: float = 0.13
    place_z: float = 0.06
    retreat_z: float = 0.14
    # Per-segment durations in 30 Hz control steps.
    reach: int = 28
    descend: int = 16
    grip_dwell: int = 12
    lift: int = 16
    transport: int = 32
    place: int = 16
    release_dwell: int = 12
    retreat: int = 12
    ret_home: int = 30


@dataclass
class Plan:
    """A planned episode: setpoints (T, 6) in LeRobot units + grasp events."""

    setpoints: np.ndarray
    target: str
    attach_step: int   # step at which to weld the target cube to the gripper
    detach_step: int   # step at which to release it (over the cup)


class ScriptedExpert:
    def __init__(self, scene: Scene, cfg: ExpertConfig | None = None):
        self.scene = scene
        self.cfg = cfg or ExpertConfig()
        import mujoco

        self._mj = mujoco
        self._sid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self._home = np.array(scene.cfg["init_pose"]["state"], float)

    # -- IK at a world pose, warm-started from a scratch state ----------------

    def _ik_arm(self, scratch, pos) -> np.ndarray:
        # Position-only IK: the long gripper cannot hold a strict vertical
        # approach to a low cube, and the weld grasp makes orientation moot.
        return ik.solve_ik(
            self.scene.model, scratch, self._sid, np.asarray(pos, float),
            target_mat=None, ori_weight=1e-6, damping=0.05,
        )

    def plan(self, layout: Layout) -> Plan:
        c = self.cfg
        cube_xy = layout.cube_left_xy if layout.target == "left" else layout.cube_right_xy
        cx, cy = cube_xy
        ux, uy = layout.cup_xy

        # A dedicated scratch MjData (NOT a deepcopy, which corrupts MjData) is
        # re-seeded at the home pose before each solve, so every keyframe picks
        # the same clean IK branch the standalone solver finds, instead of
        # drifting into contorted at-limit solutions.
        scratch = self._mj.MjData(self.scene.model)
        home_ctrl = K.state_to_ctrl(self._home)

        def arm_state(pos, grip):
            scratch.qpos[:] = 0.0
            scratch.qpos[:6] = home_ctrl
            self._mj.mj_forward(self.scene.model, scratch)
            q5 = self._ik_arm(scratch, pos)
            arm_deg = K.qpos_to_state(np.append(q5, 0.0))[:5]
            return np.append(arm_deg, grip).astype(np.float32)

        # Key setpoints (arm via position-only IK, gripper explicit). The weld
        # grasp makes orientation irrelevant, and position-only IK is the only
        # thing that reaches a low cube with this long gripper.
        home = self._home.copy()
        pregrasp = arm_state((cx, cy, c.pregrasp_z), c.grip_open)
        grasp = arm_state((cx, cy, c.grasp_z), c.grip_open)
        grasp_closed = grasp.copy()
        grasp_closed[5] = c.grip_closed
        lift = arm_state((cx, cy, c.lift_z), c.grip_closed)
        transport = arm_state((ux, uy, c.transport_z), c.grip_closed)
        place = arm_state((ux, uy, c.place_z), c.grip_closed)
        release = place.copy()
        release[5] = c.grip_open
        retreat = arm_state((ux, uy, c.retreat_z), c.grip_open)

        # Stitch segments: (target_state, n_steps). Dwell = repeat same state.
        segs = [
            (pregrasp, c.reach),
            (grasp, c.descend),
            (grasp_closed, c.grip_dwell),
            (lift, c.lift),
            (transport, c.transport),
            (place, c.place),
            (release, c.release_dwell),
            (retreat, c.retreat),
            (home, c.ret_home),
        ]
        traj = _stitch(home, segs)
        # Weld the cube once the gripper has descended (start of grip dwell);
        # release it at the start of the release dwell, over the cup.
        attach_step = c.reach + c.descend
        detach_step = c.reach + c.descend + c.grip_dwell + c.lift + c.transport + c.place
        return Plan(traj, layout.target, attach_step, detach_step)


def _stitch(start: np.ndarray, segments) -> np.ndarray:
    """Linear interpolation through (target, n_steps) waypoints from ``start``."""
    out = []
    cur = start.copy()
    for target, n in segments:
        for k in range(1, n + 1):
            out.append(cur + (target - cur) * (k / n))
        cur = target.copy()
    return np.asarray(out, dtype=np.float32)


# -- layout sampling (50/50 bimodal) -----------------------------------------

def sample_layout(cfg: dict, rng: np.random.Generator) -> Layout:
    """Sample cube/cup positions and a 50/50 left/right pick target."""
    cu, cp = cfg["cubes"], cfg["cup"]
    lx, ly = rng.uniform(*cu["x_range"]), rng.uniform(*cu["left_y_range"])
    rx, ry = rng.uniform(*cu["x_range"]), rng.uniform(*cu["right_y_range"])
    ux, uy = rng.uniform(*cp["x_range"]), rng.uniform(*cp["y_range"])
    target = "left" if rng.random() < 0.5 else "right"
    return Layout((lx, ly), (rx, ry), (ux, uy), target=target)
