"""Non-scored sim rollout helpers: execute a setpoint trajectory in the scene,
optionally record frames, and report programmatic success.

This is a *debugging / visualisation* tool only -- the benchmark's scored
evaluation is the real-arm ``soarm_eval`` harness, exactly as in the sibling
repos. Use it to sanity-check the scripted expert and, later, a trained policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim2real_soarm.sim.scene import Scene


@dataclass
class RolloutResult:
    success: bool
    target: str
    chosen: str | None        # which cube ended in the cup (if any)
    n_steps: int
    front: list[np.ndarray]   # captured frames (empty unless capture=True)
    wrist: list[np.ndarray]


def execute(
    scene: Scene,
    setpoints: np.ndarray,
    target: str,
    *,
    attach_step: int | None = None,
    detach_step: int | None = None,
    n_substeps: int = 17,
    capture: bool = False,
    capture_every: int = 1,
    settle_steps: int = 40,
) -> RolloutResult:
    """Apply ``setpoints`` (T, 6, LeRobot units) to ``scene`` and check success.

    ``scene.reset(layout)`` must have been called first. If ``attach_step`` /
    ``detach_step`` are given, the target cube is welded to / released from the
    gripper at those step indices (the reliable grasp). Returns a
    :class:`RolloutResult`; frames are captured (front+wrist) only if requested.
    """
    front: list[np.ndarray] = []
    wrist: list[np.ndarray] = []

    def maybe_capture(i):
        if capture and i % capture_every == 0:
            front.append(scene.render("front"))
            wrist.append(scene.render("wrist"))

    for i, sp in enumerate(setpoints):
        scene.step(sp, n_substeps=n_substeps)
        if i == attach_step:
            scene.attach(target)
        if i == detach_step:
            scene.detach(target)
        maybe_capture(i)

    # Let the cube settle in the cup before judging.
    last = setpoints[-1]
    for j in range(settle_steps):
        scene.step(last, n_substeps=n_substeps)
        maybe_capture(len(setpoints) + j)

    chosen = None
    for cube in ("left", "right"):
        if scene.cube_in_cup(cube):
            chosen = cube
            break
    return RolloutResult(
        success=chosen is not None,
        target=target,
        chosen=chosen,
        n_steps=len(setpoints),
        front=front,
        wrist=wrist,
    )


def save_gif(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, frames, duration=1000 / fps, loop=0)
