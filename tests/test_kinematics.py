"""Validate the units bridge against the real calibration and dataset stats.

The bridge is correct only if (a) it round-trips, and (b) the MuJoCo joint
limits map to a calibrated-degree range consistent with the real arm's
calibration (`valera.json`) and the observed range in the training dataset
(`stats.json`). If those don't agree, a sim-trained policy would emit
out-of-distribution numbers on the real arm.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim2real_soarm.sim import kinematics as K

ASSET = Path(__file__).resolve().parents[1] / "assets" / "so101" / "so101.xml"
CALIB = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/valera.json"
STATS = Path("/home/vasili/dev/smolvla-soarm-benchmark/recordings/redcubes_bluecup/meta/stats.json")


def test_round_trip_state():
    rng = np.random.default_rng(0)
    for _ in range(200):
        state = np.concatenate([rng.uniform(-90, 90, 5), rng.uniform(0, 100, 1)])
        back = K.qpos_to_state(K.state_to_ctrl(state))
        np.testing.assert_allclose(back, state, atol=1e-3)


def test_round_trip_qpos():
    rng = np.random.default_rng(1)
    for _ in range(200):
        qpos = np.concatenate([
            rng.uniform(-1.5, 1.5, 5),
            rng.uniform(K.GRIPPER_LO_RAD, K.GRIPPER_HI_RAD, 1),
        ])
        back = K.state_to_ctrl(K.qpos_to_state(qpos))
        np.testing.assert_allclose(back, qpos, atol=1e-6)


def test_gripper_monotonic_and_endpoints():
    assert K._gripper_rad_to_range(K.GRIPPER_LO_RAD) == pytest.approx(0.0)
    assert K._gripper_rad_to_range(K.GRIPPER_HI_RAD) == pytest.approx(100.0)
    vals = [K._gripper_rad_to_range(t) for t in np.linspace(K.GRIPPER_LO_RAD, K.GRIPPER_HI_RAD, 10)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def _mjcf_joint_ranges_deg() -> dict[str, tuple[float, float]]:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(ASSET))
    out = {}
    for j in K.ARM_JOINTS:
        ji = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = m.jnt_range[ji]
        out[j] = (math.degrees(lo), math.degrees(hi))
    return out


@pytest.mark.skipif(not CALIB.exists(), reason="real calibration not present")
def test_mjcf_ranges_match_calibration():
    """MJCF joint limits should fall within the real calibrated degree range
    (the MJCF is slightly tighter than the recorded range of motion)."""
    cal = json.loads(CALIB.read_text())
    mjcf = _mjcf_joint_ranges_deg()
    for j in K.ARM_JOINTS:
        cmin, cmax = K.lerobot_degree_range(cal[j]["range_min"], cal[j]["range_max"])
        lo, hi = mjcf[j]
        # MJCF range centred like the calibration, within ~10 deg slack.
        assert lo >= cmin - 10 and hi <= cmax + 10, f"{j}: mjcf[{lo:.1f},{hi:.1f}] vs cal[{cmin:.1f},{cmax:.1f}]"


@pytest.mark.skipif(not STATS.exists(), reason="real dataset stats not present")
def test_observed_state_within_reachable_range():
    """Every observed real state value must be reproducible by the sim, i.e.
    fall inside the calibrated reachable ranges scene.py applies to the model."""
    stats = json.loads(STATS.read_text())
    smin = stats["observation.state"]["min"]
    smax = stats["observation.state"]["max"]
    for i, j in enumerate(K.ARM_JOINTS):
        lo, hi = K.REACHABLE_DEG[j]
        assert lo <= smin[i] and smax[i] <= hi, (
            f"{j}: observed [{smin[i]:.1f},{smax[i]:.1f}] outside reachable [{lo:.1f},{hi:.1f}]"
        )
    # gripper observed values must be valid RANGE_0_100
    assert 0 <= smin[5] and smax[5] <= 100


@pytest.mark.skipif(not CALIB.exists(), reason="real calibration not present")
def test_reachable_ranges_cover_calibration():
    """The hardcoded REACHABLE_DEG must enclose the live calibration's ROM (so a
    different calibration that is tighter is still safe)."""
    cal = json.loads(CALIB.read_text())
    for j in K.ARM_JOINTS:
        cmin, cmax = K.lerobot_degree_range(cal[j]["range_min"], cal[j]["range_max"])
        lo, hi = K.REACHABLE_DEG[j]
        assert lo <= cmin + 1 and hi >= cmax - 1, f"{j}: reachable[{lo},{hi}] vs calib[{cmin:.1f},{cmax:.1f}]"


def test_apply_reachable_ranges():
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(ASSET))
    K.apply_reachable_ranges(m)
    for j, (lo_deg, hi_deg) in K.REACHABLE_DEG.items():
        ji = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        assert math.degrees(m.jnt_range[ji][0]) == pytest.approx(lo_deg, abs=1e-3)
        assert math.degrees(m.jnt_range[ji][1]) == pytest.approx(hi_deg, abs=1e-3)
