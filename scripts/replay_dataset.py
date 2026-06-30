"""Sanity-check a recorded sim dataset by replaying an episode's camera frames
to a GIF and printing its state/action stats.

Usage::

    uv run python scripts/replay_dataset.py --root recordings/sim_redcubes_bluecup \
        --episode 0 --out episode0.gif
"""

from __future__ import annotations

import argparse

import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="recordings/sim_redcubes_bluecup")
    p.add_argument("--repo-id", default="sim/sim_redcubes_bluecup")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--out", default="episode.gif")
    p.add_argument("--camera", default="front", choices=["front", "wrist"])
    args = p.parse_args(argv)

    import imageio.v3 as iio
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root)
    ep = args.episode
    fr = ds.meta.episodes["dataset_from_index"][ep]
    to = ds.meta.episodes["dataset_to_index"][ep]
    key = f"observation.images.{args.camera}"

    frames, states, actions = [], [], []
    for i in range(fr, to):
        item = ds[i]
        img = item[key]
        if hasattr(img, "numpy"):
            img = img.numpy()
        img = np.asarray(img)
        if img.shape[0] == 3:  # CHW float -> HWC uint8
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        frames.append(img)
        states.append(np.asarray(item["observation.state"]))
        actions.append(np.asarray(item["action"]))

    iio.imwrite(args.out, frames, duration=1000 / ds.fps, loop=0)
    states, actions = np.stack(states), np.stack(actions)
    print(f"episode {ep}: {len(frames)} frames -> {args.out}")
    print(f"state  min {np.round(states.min(0), 1)}")
    print(f"state  max {np.round(states.max(0), 1)}")
    print(f"action min {np.round(actions.min(0), 1)}")
    print(f"action max {np.round(actions.max(0), 1)}")


if __name__ == "__main__":
    main()
