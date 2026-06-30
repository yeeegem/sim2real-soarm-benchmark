"""Behavioural tests for the scripted IK expert.

These run real (headless) MuJoCo rollouts, so they are a bit slower than the
unit tests but they guard the thing that actually matters: the expert reliably
picks and places, and it does so for *both* cubes (the bimodal premise).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

from sim2real_soarm.sim import rollout  # noqa: E402
from sim2real_soarm.sim.expert import ScriptedExpert, sample_layout  # noqa: E402
from sim2real_soarm.sim.scene import Scene  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    s = Scene()
    yield s
    s.close()


def _run(scene, n, seed=0):
    exp = ScriptedExpert(scene)
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n):
        lay = sample_layout(scene.cfg, rng)
        scene.reset(lay)
        plan = exp.plan(lay)
        results.append(
            rollout.execute(
                scene, plan.setpoints, plan.target,
                attach_step=plan.attach_step, detach_step=plan.detach_step,
            )
        )
    return results


def test_expert_high_success_and_both_modes(scene):
    res = _run(scene, 16)
    success = sum(r.success for r in res)
    assert success / len(res) >= 0.8, f"expert success too low: {success}/{len(res)}"
    # The chosen cube must match the planned target on every success (weld grasp).
    for r in res:
        if r.success:
            assert r.chosen == r.target
    # Both modes must be demonstrated.
    targets = {r.target for r in res if r.success}
    assert targets == {"left", "right"}, f"only saw modes {targets}"


def test_layout_sampler_is_balanced():
    cfg = Scene.__new__(Scene)  # avoid building a model just for cfg
    import yaml
    from sim2real_soarm.sim.scene import _SCENE_CFG

    cfg = yaml.safe_load(_SCENE_CFG.read_text())
    rng = np.random.default_rng(0)
    targets = [sample_layout(cfg, rng).target for _ in range(2000)]
    frac_left = targets.count("left") / len(targets)
    assert 0.45 <= frac_left <= 0.55
