"""The generated dataset's schema must match the real dataset exactly.

This is the zero-shot contract: identical feature names, dtypes, shapes, motor
ordering, fps, and robot_type. If this drifts, a sim-trained policy will not
load/run on the real arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sim2real_soarm.data.record import MOTOR_NAMES, build_features

REAL_INFO = Path(
    "/home/vasili/dev/smolvla-soarm-benchmark/recordings/redcubes_bluecup/meta/info.json"
)
FEATURE_KEYS = [
    "observation.images.front", "observation.images.wrist",
    "observation.state", "action",
]


@pytest.mark.skipif(not REAL_INFO.exists(), reason="real dataset info.json not present")
def test_features_match_real_dataset():
    real = json.loads(REAL_INFO.read_text())["features"]
    sim = build_features()
    for key in FEATURE_KEYS:
        assert key in real, f"{key} missing from real dataset"
        r, s = real[key], sim[key]
        assert s["dtype"] == r["dtype"], f"{key} dtype {s['dtype']} != {r['dtype']}"
        assert tuple(s["shape"]) == tuple(r["shape"]), f"{key} shape mismatch"
        assert list(s["names"]) == list(r["names"]), f"{key} names mismatch"


@pytest.mark.skipif(not REAL_INFO.exists(), reason="real dataset info.json not present")
def test_motor_order_and_robot_type():
    real = json.loads(REAL_INFO.read_text())
    assert real["robot_type"] == "so_follower"
    assert real["fps"] == 30
    assert list(real["features"]["action"]["names"]) == MOTOR_NAMES


def test_build_features_self_consistent():
    f = build_features()
    assert set(f) == set(FEATURE_KEYS)
    for k in ("observation.state", "action"):
        assert f[k]["shape"] == (6,) and f[k]["names"] == MOTOR_NAMES
