"""Domain randomization changes visuals/dynamics without breaking the sim."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from sim2real_soarm.sim.randomization import DomainRandomizer  # noqa: E402
from sim2real_soarm.sim.scene import Layout, Scene  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    s = Scene()
    yield s
    s.close()


def test_dr_varies_colors_and_camera(scene):
    dr = DomainRandomizer(scene)
    lay = Layout((0.25, 0.10), (0.25, -0.10), (0.27, 0.0), "left")
    cube_gi = dr._cube_geoms[0]
    # The front camera is intentionally pinned (fixed real mount); the wrist
    # camera still jitters, so check that one for pose variation.
    cam = dr._cam["wrist"]
    samples_rgba, samples_cam = [], []
    for seed in range(5):
        scene.reset(lay)
        dr.apply(np.random.default_rng(seed))
        samples_rgba.append(scene.model.geom_rgba[cube_gi].copy())
        samples_cam.append(scene.model.cam_pos[cam].copy())
    # Colours and (wrist) camera pose actually vary across episodes.
    assert np.ptp(np.stack(samples_rgba), axis=0).max() > 1e-3
    assert np.ptp(np.stack(samples_cam), axis=0).max() > 1e-3
    # Red cube stays red-dominant.
    for rgba in samples_rgba:
        assert rgba[0] > rgba[1] and rgba[0] > rgba[2]
