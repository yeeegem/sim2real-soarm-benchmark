"""Rewrite the (single) task / language instruction of a local LeRobot dataset.

SmolVLA is language-conditioned and reads its prompt from the dataset's task
metadata, so this sets what SmolVLA (and the eval) are conditioned on -- without
re-recording. ACT ignores the string.

    uv run python scripts/set_task.py --root recordings/sim_redcubes_bluecup \
        --task "Pick up a red cube and put it in the blue cup"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="recordings/sim_redcubes_bluecup")
    ap.add_argument("--task", required=True, help="new task / instruction string")
    args = ap.parse_args(argv)

    path = Path(args.root) / "meta" / "tasks.parquet"
    df = pd.read_parquet(path)
    tasks = list(df.index)
    if len(tasks) != 1:
        raise SystemExit(f"expected exactly 1 task, found {len(tasks)}: {tasks}")
    df = df.rename(index={tasks[0]: args.task})
    df.index.name = "task"
    df.to_parquet(path)
    print(f"task: {tasks[0]!r} -> {args.task!r}\n  ({path})")


if __name__ == "__main__":
    main()
