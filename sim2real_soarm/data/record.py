"""Generate simulated demonstrations and write a LeRobot dataset that is
schema- and unit-identical to the real ``yeeegem/redcubes_bluecup`` dataset, so
a policy trained on it deploys zero-shot on the real arm.

Each episode: sample a 50/50 left/right layout, randomize the scene, run the
scripted weld-grasp expert, render the front+wrist cameras at 30 fps, and log
``observation.state`` / ``action`` in LeRobot calibrated units (degrees +
RANGE_0_100). Only successful episodes are kept.

Usage::

    uv run python -m sim2real_soarm.data.record --num-episodes 500 \
        --out recordings/sim_redcubes_bluecup
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

MOTOR_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
DEFAULT_TASK = "Blue cup and 2 red cubes"
DEFAULT_REPO_ID = "sim/sim_redcubes_bluecup"


def build_features() -> dict:
    """Feature schema identical to the real dataset's meta/info.json."""
    img = {"dtype": "video", "shape": (480, 640, 3),
           "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": (6,), "names": MOTOR_NAMES}
    return {
        "observation.images.front": dict(img),
        "observation.images.wrist": dict(img),
        "observation.state": dict(vec),
        "action": dict(vec),
    }


def record(
    num_episodes: int,
    out: Path,
    repo_id: str = DEFAULT_REPO_ID,
    task: str = DEFAULT_TASK,
    seed: int = 0,
    use_dr: bool = True,
    max_attempts: int | None = None,
    overwrite: bool = False,
    n_substeps: int = 17,
) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from sim2real_soarm.sim.expert import ScriptedExpert, sample_layout
    from sim2real_soarm.sim.randomization import DomainRandomizer
    from sim2real_soarm.sim.scene import Scene

    out = Path(out)
    # Check before building the scene: LeRobot's create() requires a fresh dir,
    # and doing this first keeps the error path clean (no renderer to tear down).
    if out.exists():
        if overwrite:
            import shutil
            shutil.rmtree(out)
            print(f"Removed existing {out} (--overwrite).")
        else:
            raise SystemExit(
                f"Output dir already exists: {out}\n"
                f"  Pass --overwrite to replace it, or use a different --out."
            )

    scene = Scene()
    expert = ScriptedExpert(scene)
    dr = DomainRandomizer(scene) if use_dr else None
    rng = np.random.default_rng(seed)
    dr_cfg = dr.cfg if dr else {}
    obs_noise = float(dr_cfg.get("observation_noise_deg", 0.0)) if dr else 0.0

    ds = LeRobotDataset.create(
        repo_id=repo_id, fps=30, features=build_features(),
        root=out, robot_type="so_follower", use_videos=True,
    )

    kept = 0
    attempts = 0
    max_attempts = max_attempts or num_episodes * 3
    chosen = {"left": 0, "right": 0}
    while kept < num_episodes and attempts < max_attempts:
        attempts += 1
        layout = sample_layout(scene.cfg, rng)
        scene.reset(layout)
        if dr:
            dr.apply(rng)
        plan = expert.plan(layout)

        frames = []
        for i, sp in enumerate(plan.setpoints):
            state = scene.get_state().astype(np.float32)
            if obs_noise:
                state[:5] += rng.normal(0, obs_noise, 5).astype(np.float32)
            frames.append({
                "observation.images.front": scene.render("front"),
                "observation.images.wrist": scene.render("wrist"),
                "observation.state": state,
                "action": sp.astype(np.float32),
                "task": task,
            })
            scene.step(sp, n_substeps=n_substeps)
            if i == plan.attach_step:
                scene.attach(plan.target)
            if i == plan.detach_step:
                scene.detach(plan.target)

        # Settle and judge success before committing the episode.
        for _ in range(40):
            scene.step(plan.setpoints[-1], n_substeps=n_substeps)
        if not scene.cube_in_cup(plan.target):
            continue  # discard failed attempt

        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        kept += 1
        chosen[plan.target] += 1
        if kept % 25 == 0 or kept == num_episodes:
            print(f"  kept {kept}/{num_episodes} (attempts {attempts}) "
                  f"left={chosen['left']} right={chosen['right']}")

    scene.close()
    p_left = chosen["left"] / max(kept, 1)
    summary = {"kept": kept, "attempts": attempts, "p_left": p_left, "chosen": chosen}
    print(f"Done: {kept} episodes ({attempts} attempts), "
          f"mode balance left={p_left:.2f} -> {out}")
    return summary


def _generate_shard(kwargs: dict) -> dict:
    """Worker entry point: generate one shard dataset. Top-level so it is
    picklable for the 'spawn' multiprocessing start method (needed for EGL)."""
    return record(**kwargs)


def record_parallel(
    num_episodes: int,
    out: Path,
    repo_id: str = DEFAULT_REPO_ID,
    task: str = DEFAULT_TASK,
    seed: int = 0,
    use_dr: bool = True,
    workers: int = 8,
    max_attempts: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Generate episodes across ``workers`` processes (each its own MuJoCo/EGL
    context), then merge the shards into one dataset with LeRobot's
    ``aggregate_datasets``. Episodes are independent, so this scales ~linearly
    until GPU render / video-encode saturates."""
    import concurrent.futures as cf
    import multiprocessing as mp

    from lerobot.datasets.aggregate import aggregate_datasets

    out = Path(out)
    if out.exists():
        if overwrite:
            shutil.rmtree(out)
        else:
            raise SystemExit(
                f"Output dir already exists: {out}\n"
                f"  Pass --overwrite to replace it, or use a different --out."
            )

    workers = max(1, min(workers, num_episodes))
    per = [num_episodes // workers + (1 if i < num_episodes % workers else 0)
           for i in range(workers)]
    jobs, shard_roots, shard_repos = [], [], []
    for i, n in enumerate(per):
        if n == 0:
            continue
        sroot = out.parent / f".{out.name}_shard{i}"
        srepo = f"{repo_id}_shard{i}"
        shard_roots.append(sroot)
        shard_repos.append(srepo)
        jobs.append(dict(num_episodes=n, out=sroot, repo_id=srepo, task=task,
                         seed=seed + i * 10007, use_dr=use_dr,
                         max_attempts=max_attempts, overwrite=True))

    print(f"Generating {num_episodes} episodes across {len(jobs)} workers...")
    # ProcessPoolExecutor workers are non-daemonic, so each shard can still spawn
    # LeRobot's video-encoding child processes (a Pool would forbid that).
    with cf.ProcessPoolExecutor(max_workers=len(jobs),
                                mp_context=mp.get_context("spawn")) as ex:
        summaries = list(ex.map(_generate_shard, jobs))

    kept = sum(s["kept"] for s in summaries)
    chosen = {k: sum(s["chosen"][k] for s in summaries) for k in ("left", "right")}
    print(f"Merging {len(shard_roots)} shards -> {out} ...")
    aggregate_datasets(repo_ids=shard_repos, aggr_repo_id=repo_id,
                       roots=shard_roots, aggr_root=out)
    for r in shard_roots:
        shutil.rmtree(r, ignore_errors=True)

    p_left = chosen["left"] / max(kept, 1)
    print(f"Done: {kept} episodes, mode balance left={p_left:.2f} -> {out}")
    return {"kept": kept, "chosen": chosen, "p_left": p_left}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-episodes", type=int, default=500)
    p.add_argument("--out", default="recordings/sim_redcubes_bluecup")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-dr", action="store_true", help="disable domain randomization")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--overwrite", action="store_true",
                   help="replace the output dir if it already exists")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel generation processes (shards merged at the end)")
    args = p.parse_args(argv)
    common = dict(
        num_episodes=args.num_episodes, out=Path(args.out), repo_id=args.repo_id,
        task=args.task, seed=args.seed, use_dr=not args.no_dr,
        max_attempts=args.max_attempts, overwrite=args.overwrite,
    )
    if args.workers > 1:
        record_parallel(**common, workers=args.workers)
    else:
        record(**common)


if __name__ == "__main__":
    main()
