"""Smoke tests for the composed MuJoCo scene."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from sim2real_soarm.sim.scene import Layout, Scene  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    s = Scene()
    yield s
    s.close()


def test_builds_with_expected_dofs(scene):
    # 6 arm joints + 3 free bodies * 7 = 27 qpos; 2 cameras; 6 actuators.
    assert scene.model.nq == 6 + 3 * 7
    assert scene.model.ncam == 2
    assert scene.model.nu == 6


def test_reset_places_objects(scene):
    lay = Layout((0.25, 0.10), (0.25, -0.10), (0.27, 0.0), target="left")
    scene.reset(lay)
    np.testing.assert_allclose(scene.body_xpos("cube_left")[:2], [0.25, 0.10], atol=1e-3)
    np.testing.assert_allclose(scene.body_xpos("cube_right")[:2], [0.25, -0.10], atol=1e-3)
    np.testing.assert_allclose(scene.body_xpos("cup")[:2], [0.27, 0.0], atol=1e-3)


def test_render_shapes(scene):
    lay = Layout((0.25, 0.10), (0.25, -0.10), (0.27, 0.0), target="left")
    scene.reset(lay)
    for cam in ("front", "wrist"):
        img = scene.render(cam)
        assert img.shape == (480, 640, 3)
        assert img.dtype == np.uint8


def test_cube_in_cup_detection(scene):
    lay = Layout((0.25, 0.10), (0.25, -0.10), (0.27, 0.0), target="left")
    scene.reset(lay)
    assert not scene.cube_in_cup("left")
    # Teleport the left cube into the cup cavity and re-evaluate.
    adr = scene._free_qadr["cube_left"]
    scene.data.qpos[adr : adr + 3] = [0.27, 0.0, 0.03]
    scene.mj.mj_forward(scene.model, scene.data)
    assert scene.cube_in_cup("left")


def test_state_round_trips_through_scene(scene):
    lay = Layout((0.25, 0.10), (0.25, -0.10), (0.27, 0.0), target="left")
    target = np.array([-12.0, -25.0, 20.0, 70.0, 2.0, 40.0])
    scene.reset(lay, init_state=target)
    np.testing.assert_allclose(scene.get_state(), target, atol=0.5)
