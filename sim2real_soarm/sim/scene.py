"""MuJoCo pick-and-place scene: SO-ARM101 + gray-blue table, white walls, two
red cubes, a procedural wall-ring blue cup, and front/wrist cameras.

The arm comes from the vendored ``assets/so101/so101.xml``; everything else is
added with the MuJoCo ``MjSpec`` model-editing API from ``configs/scene.yaml``,
so the procedural cup (a bottom disk + N thin box wall segments forming a real
interior cavity) and the camera placement are all config-driven -- no XML string
templating.

Coordinate frame: arm base at the origin, table top at z=0, the arm reaches
toward +x. Left cube = +y, right cube = -y (a fixed labelling for the
mode-balance metric).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from sim2real_soarm.sim import kinematics as K

_REPO = Path(__file__).resolve().parents[2]
_SO101 = _REPO / "assets" / "so101" / "so101.xml"
_SCENE_CFG = _REPO / "configs" / "scene.yaml"


def _quat_lookat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """wxyz quat orienting a MuJoCo camera at ``pos`` to look at ``target``.

    MuJoCo cameras look along their local -Z with +Y up.
    """
    import mujoco

    pos = np.asarray(pos, float)
    target = np.asarray(target, float)
    z = pos - target  # local +z points away from the target
    n = np.linalg.norm(z)
    if n < 1e-9:
        raise ValueError("camera pos coincides with target")
    z /= n
    up = np.asarray(up, float)
    if abs(np.dot(z, up)) > 0.999:  # looking straight up/down -> pick another up
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1).reshape(9)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


@dataclass
class Layout:
    """One episode's object placement and which cube is the pick target."""

    cube_left_xy: tuple[float, float]
    cube_right_xy: tuple[float, float]
    cup_xy: tuple[float, float]
    target: str  # "left" or "right"


class Scene:
    """Loads the composed model and exposes reset / step / render helpers."""

    def __init__(self, cfg: dict | None = None, render_size: tuple[int, int] | None = None,
                 make_renderer: bool = True):
        import mujoco

        self.mj = mujoco
        self.cfg = cfg or yaml.safe_load(_SCENE_CFG.read_text())
        self.model = self._build()
        K.apply_reachable_ranges(self.model)
        self._apply_collision_masks()
        self._recolor_arm()
        self.data = mujoco.MjData(self.model)

        # qpos addresses of each free-body joint (cubes, cup) for fast posing.
        self._free_qadr = {}
        for name in ("cube_left", "cube_right", "cup"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jadr = self.model.body_jntadr[bid]
            self._free_qadr[name] = self.model.jnt_qposadr[jadr]

        # Render options that hide site markers (the tcp/gripperframe sites would
        # otherwise show as little spheres in the images).
        self._scene_opt = mujoco.MjvOption()
        self._scene_opt.sitegroup[:] = 0

        # Offscreen renderer (EGL). Skipped for the interactive GLFW viewer,
        # which would conflict with an EGL context in the same process.
        self._renderer = None
        if make_renderer:
            h = render_size[0] if render_size else self.cfg["cameras"]["height"]
            w = render_size[1] if render_size else self.cfg["cameras"]["width"]
            if "MUJOCO_GL" not in os.environ:
                os.environ["MUJOCO_GL"] = "egl"
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)

    # -- model construction --------------------------------------------------

    def _build(self):
        mj = self.mj
        spec = mj.MjSpec.from_file(str(_SO101))
        wb = spec.worldbody
        c = self.cfg

        # Table: a box whose top face is at z=0.
        t = c["table"]
        wb.add_geom(
            name="table", type=mj.mjtGeom.mjGEOM_BOX,
            pos=[t["center_x"], 0.0, -t["thickness"] / 2],
            size=[t["half_x"], t["half_y"], t["thickness"] / 2],
            rgba=t["rgba"], friction=[1.0, 0.02, 0.001],
        )

        # White walls: front (far +x edge) + left/right (the near side is open).
        wcfg = c["walls"]
        wh, wt = wcfg["height"], wcfg["thickness"]
        wb.add_geom(
            name="wall_front", type=mj.mjtGeom.mjGEOM_BOX,
            pos=[wcfg["front_x"], 0.0, wh / 2],
            size=[wt, t["half_y"], wh / 2], rgba=wcfg["rgba"], contype=0, conaffinity=0,
        )
        for sgn, tag in ((1, "left"), (-1, "right")):
            wb.add_geom(
                name=f"wall_{tag}", type=mj.mjtGeom.mjGEOM_BOX,
                pos=[t["center_x"], sgn * wcfg["side_y"], wh / 2],
                size=[t["half_x"], wt, wh / 2], rgba=wcfg["rgba"], contype=0, conaffinity=0,
            )

        # Lights (type 1 == directional in mjtLightType; spotlights would be 0).
        for i, L in enumerate(c["lights"]):
            wb.add_light(name=f"light{i}", pos=L["pos"], dir=L["dir"], type=0)

        # Cubes (free bodies). Initial pose overwritten every reset().
        cu = c["cubes"]
        for name, y0 in (("cube_left", 0.10), ("cube_right", -0.10)):
            b = wb.add_body(name=name, pos=[0.25, y0, cu["z"]])
            b.add_freejoint()
            b.add_geom(
                type=mj.mjtGeom.mjGEOM_BOX, size=[cu["size"]] * 3, mass=cu["mass"],
                rgba=cu["rgba"], friction=cu["friction"], condim=4,
                solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
            )

        self._add_cup(wb)

        # Cameras.
        cam = c["cameras"]
        f = cam["front"]
        wb.add_camera(
            name="front", pos=f["pos"], quat=_quat_lookat(f["pos"], f["lookat"]),
            fovy=f["fovy"], resolution=[cam["width"], cam["height"]],
        )
        wcam = cam["wrist"]
        parent = spec.body(wcam["parent_body"])
        wpos = np.asarray(wcam["pos"], float)
        # Prefer an explicit local quat (community/official camera definitions
        # give one); otherwise aim it via a local look-at point.
        if "quat" in wcam:
            wquat = np.asarray(wcam["quat"], float)
        else:
            wquat = _quat_lookat(wpos, wcam["lookat_local"])
        parent.add_camera(
            name="wrist", pos=wpos, quat=wquat,
            fovy=wcam["fovy"], resolution=[cam["width"], cam["height"]],
        )
        # Real white printed hex-nut camera mount (official SO101 STL, mm -> m)
        # attached to the gripper, with the black camera module at the lens.
        # Its transform (mount.pos/quat in the gripper local frame) is config-
        # driven so it can be dialled in to clamp the wrist correctly.
        # Wrap the mount mesh in its own child body so its pose is authored
        # cleanly (MuJoCo recenters mesh *geoms* to their CoM, but not body
        # frames) -- this makes scripts/tune_mount.py's printed body pose paste
        # straight back into mount.pos/quat.
        mcfg = wcam.get("mount", {})
        spec.add_mesh(name="wrist_cam_mount_mesh", file="wrist_cam_mount.stl",
                      scale=[0.001, 0.001, 0.001])
        mount_body = parent.add_body(
            name="wrist_cam_mount_body",
            pos=mcfg.get("pos", [0.0, 0.045, 0.0]),
            quat=mcfg.get("quat", [0.7071, -0.7071, 0.0, 0.0]),
        )
        mount_body.add_geom(
            name="wrist_cam_mount", type=mj.mjtGeom.mjGEOM_MESH,
            meshname="wrist_cam_mount_mesh",
            rgba=[0.90, 0.90, 0.92, 1.0], contype=0, conaffinity=0, group=2,
        )
        parent.add_geom(
            name="wrist_cam_module", type=mj.mjtGeom.mjGEOM_BOX,
            pos=wpos.tolist(), quat=wquat.tolist(),
            size=[0.012, 0.012, 0.008], rgba=[0.05, 0.05, 0.05, 1.0],
            contype=0, conaffinity=0, group=2,
        )

        # Two grasp-reference sites, both at the fingertip depth (z=-0.105) so
        # the whole gripper stays above the table (a point between the finger
        # bases would drive the long fingers ~65 mm through the table).
        #
        # On the real SO-ARM101 only the moving jaw actuates; the fixed jaw's
        # inner face is stationary at local x=-0.0079. So we model a real grasp:
        #  - `tcp` (x=0.015): the IK/approach target. With the cube here the
        #    fixed jaw has ~8 mm clearance, so it won't hit the cube on descent.
        #  - `grasp_snug` (x=0.007): where the moving jaw pushes the cube -- its
        #    left face against the fixed jaw. attach() welds the cube here, so at
        #    grasp the cube moves over to the fixed jaw and both claws touch it.
        spec.body("gripper").add_site(name="tcp", pos=[0.015, 0.0, -0.105])
        spec.body("gripper").add_site(name="grasp_snug", pos=[0.007, 0.0, -0.105])

        # Inactive welds used to model a reliable grasp (see attach/detach):
        # frictional grasping of a 3 cm cube is unreliable with this gripper's
        # bulky meshes, so the expert welds the cube to the gripper at grasp.
        for cube in ("cube_left", "cube_right"):
            eq = spec.add_equality(
                type=mj.mjtEq.mjEQ_WELD, name=f"weld_{cube}",
                name1="gripper", name2=cube, active=False,
            )
            eq.objtype = mj.mjtObj.mjOBJ_BODY

        return spec.compile()

    def _apply_collision_masks(self):
        """Filter contacts so the slim fingers can reach low cubes without the
        bulky arm/gripper meshes false-colliding with the table.

        Using symmetric contype==conaffinity masks (two geoms collide iff the
        masks share a bit), with one bit per desired contact edge:
            table-cube, table-cup, cube-cup, cube/cup-cube, finger-cube.
        The arm (except the fingers) is given mask 0 -> it collides with nothing,
        so it can dip to table height; cubes still rest on the table and drop
        into the cup, and the fingers still grasp the cubes.
        """
        mj, m = self.mj, self.model

        def bid(n):
            return mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n)

        TABLE, CUBE, CUP, NONE = 0b011, 0b111, 0b110, 0
        # Grasping is done by a weld (see attach/detach), not by finger friction,
        # so the whole arm is non-colliding -- it can dip to table height and
        # never knocks the cubes. Cubes still rest on the table, collide with
        # each other, and drop into the cup.
        cubes = {bid("cube_left"), bid("cube_right")}
        cup = bid("cup")
        arm = {bid(n) for n in
               ("base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper",
                "moving_jaw_so101_v1")}
        for gi in range(m.ngeom):
            b = m.geom_bodyid[gi]
            gname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gi)
            if gname == "table":  # table geom lives on the worldbody
                mask = TABLE
            elif b in cubes:
                mask = CUBE
            elif b == cup:
                mask = CUP
            elif b in arm:
                mask = NONE
            else:
                continue  # walls etc. already non-colliding
            m.geom_contype[gi] = mask
            m.geom_conaffinity[gi] = mask

    def _recolor_arm(self):
        """Recolour the vendored MJCF's printed parts (yellow) to the configured
        arm colour, matching the real white-printed SO-ARM101. Dark motor
        materials are left untouched."""
        m = self.model
        rgba = self.cfg.get("robot", {}).get("arm_rgba")
        if rgba is None:
            return
        for i in range(m.nmat):
            r, g, b = m.mat_rgba[i, :3]
            if r > 0.6 and g > 0.5 and b < 0.4:  # the yellow printed-part material
                m.mat_rgba[i, :3] = rgba[:3]

    def _add_cup(self, wb):
        mj = self.mj
        cp = self.cfg["cup"]
        b = wb.add_body(name="cup", pos=[0.27, 0.0, 0.0])
        b.add_freejoint()
        inner, wallt, h = cp["inner_radius"], cp["wall_thickness"], cp["height"]
        bt = cp["bottom_thickness"]
        outer = inner + wallt
        # Bottom disk (heavy + wide base -> stays put but tippable).
        b.add_geom(
            name="cup_bottom", type=mj.mjtGeom.mjGEOM_CYLINDER,
            pos=[0, 0, bt / 2], size=[outer, bt / 2, 0], mass=cp["bottom_mass"],
            rgba=cp["rgba"], friction=cp["friction"], condim=3,
        )
        # Wall ring: N thin boxes around a circle, slight tangential overlap.
        n = int(cp["n_segments"])
        r_wall = inner + wallt / 2
        chord_half = r_wall * math.tan(math.pi / n)
        for i in range(n):
            th = 2 * math.pi * i / n
            b.add_geom(
                name=f"cup_wall_{i}", type=mj.mjtGeom.mjGEOM_BOX,
                pos=[r_wall * math.cos(th), r_wall * math.sin(th), bt + h / 2],
                quat=_z_quat(th), size=[wallt / 2, chord_half, h / 2],
                mass=cp["wall_mass"], rgba=cp["rgba"], friction=cp["friction"], condim=3,
            )

    # -- episode control -----------------------------------------------------

    def _set_free_pose(self, name: str, xy, z: float, yaw: float = 0.0):
        adr = self._free_qadr[name]
        self.data.qpos[adr : adr + 3] = [xy[0], xy[1], z]
        self.data.qpos[adr + 3 : adr + 7] = _z_quat(yaw)

    def set_arm_state(self, state: np.ndarray, settle: bool = False):
        """Set the arm to a LeRobot-unit state (instant) and its actuator target."""
        ctrl = K.state_to_ctrl(state)
        self.data.qpos[:6] = ctrl
        self.data.qvel[:6] = 0.0
        self.data.ctrl[:6] = ctrl
        self.mj.mj_forward(self.model, self.data)

    def reset(self, layout: Layout, init_state: np.ndarray | None = None):
        self.mj.mj_resetData(self.model, self.data)
        cu = self.cfg["cubes"]
        self._set_free_pose("cube_left", layout.cube_left_xy, cu["z"])
        self._set_free_pose("cube_right", layout.cube_right_xy, cu["z"])
        self._set_free_pose("cup", layout.cup_xy, 0.0)
        state = init_state if init_state is not None else np.array(self.cfg["init_pose"]["state"], float)
        self.set_arm_state(state)
        self.mj.mj_forward(self.model, self.data)

    def step(self, ctrl_state: np.ndarray, n_substeps: int = 1):
        """Command the arm with a LeRobot-unit target and advance the sim."""
        self.data.ctrl[:6] = K.state_to_ctrl(ctrl_state)
        for _ in range(n_substeps):
            self.mj.mj_step(self.model, self.data)

    def get_state(self) -> np.ndarray:
        """Current arm state in LeRobot units (degrees + RANGE_0_100 gripper)."""
        return K.qpos_to_state(self.data.qpos[:6])

    def attach(self, cube: str):
        """Weld ``cube`` ('left'/'right') to the gripper for a reliable, precise
        grasp. The cube is first snapped to the grasp point (the ``tcp`` site,
        between the fingers) so the grasp is exact regardless of position-control
        tracking lag, then welded at that relative pose."""
        mj, m, d = self.mj, self.model, self.data
        eid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_EQUALITY, f"weld_cube_{cube}")
        g = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper")
        cb = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"cube_{cube}")

        # Snap the cube onto the snug grasp point (against the fixed jaw) so it
        # moves over to the fixed jaw as the moving jaw closes -- a precise,
        # realistic grasp.
        snug = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "grasp_snug")
        adr = self._free_qadr[f"cube_{cube}"]
        d.qpos[adr : adr + 3] = d.site_xpos[snug]
        dof = m.body_dofadr[cb]
        d.qvel[dof : dof + 6] = 0.0
        mj.mj_forward(m, d)

        R1 = d.xmat[g].reshape(3, 3)
        relpos = R1.T @ (d.xpos[cb] - d.xpos[g])
        q1 = np.zeros(4)
        q1c = np.zeros(4)
        relq = np.zeros(4)
        mj.mju_mat2Quat(q1, d.xmat[g])
        mj.mju_negQuat(q1c, q1)
        mj.mju_mulQuat(relq, q1c, d.xquat[cb])
        m.eq_data[eid, 0:3] = 0.0          # anchor (unused with relpose)
        m.eq_data[eid, 3:6] = relpos       # relative position
        m.eq_data[eid, 6:10] = relq        # relative orientation
        m.eq_data[eid, 10] = 1.0           # torquescale
        d.eq_active[eid] = 1

    def detach(self, cube: str):
        """Release a welded cube."""
        eid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_EQUALITY, f"weld_cube_{cube}")
        self.data.eq_active[eid] = 0

    def render(self, camera: str) -> np.ndarray:
        self._renderer.update_scene(self.data, camera=camera, scene_option=self._scene_opt)
        return self._renderer.render()

    def close(self):
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- geometry helpers used by the expert / success check -----------------

    def body_xpos(self, name: str) -> np.ndarray:
        bid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def ee_xpos(self) -> np.ndarray:
        sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, "gripperframe")
        return self.data.site_xpos[sid].copy()

    def tcp_xpos(self) -> np.ndarray:
        sid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_SITE, "tcp")
        return self.data.site_xpos[sid].copy()

    def cube_in_cup(self, cube: str) -> bool:
        """True if the named cube ('left'/'right') rests inside the cup cavity."""
        cube_p = self.body_xpos(f"cube_{cube}")
        cup_p = self.body_xpos("cup")
        cp = self.cfg["cup"]
        radial = math.hypot(cube_p[0] - cup_p[0], cube_p[1] - cup_p[1])
        z_rel = cube_p[2] - cup_p[2]
        inside_xy = radial < (cp["inner_radius"] - self.cfg["cubes"]["size"])
        inside_z = self.cfg["cubes"]["size"] < z_rel < cp["height"]
        return bool(inside_xy and inside_z)


def _z_quat(yaw: float) -> np.ndarray:
    """wxyz quat for a rotation of ``yaw`` radians about +Z."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
