"""Build a co-training dataset = sim episodes + the real episodes, mixed, with the
scarce real data oversampled, so a single ``lerobot-train`` run sees both in every
(shuffled) batch. This is the reliable sim-to-real recipe: real continuously
corrects the sim-to-real gap while the sim half keeps the balanced left/right modes.

What it does:
  1. Download the real dataset from the Hugging Face Hub into a local, editable copy.
  2. Normalize that copy so its ``features`` dict matches the sim dataset exactly
     (they differ only in extra video-encoding ``info`` keys), and rewrite its task
     string to the unified instruction -- both are required for aggregation.
  3. Aggregate the sim dataset (1x) + the real copy (``--real-oversample`` x, default
     3) into one LeRobot dataset with ``aggregate_datasets``. Passing the same real
     root K times appends it K times, which is the oversampling.

With 1000 sim eps (~174k frames) + 3x100 real eps (~98k frames), real is ~36% of
frames, so shuffled sampling makes each training batch a sim+real mix.

Usage::

    uv run python scripts/build_cotrain_dataset.py --real-oversample 3
    # -> recordings/cotrain_sim_real   (train SmolVLA on this)
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import pandas as pd

DEFAULT_TASK = "Pick up a red cube and put it in the blue cup"


def _download_real(repo_id: str, dst: Path, refresh: bool) -> None:
    if (dst / "meta" / "info.json").exists() and not refresh:
        print(f"Real dataset already present at {dst} (use --refresh-download to re-pull).")
        return
    from huggingface_hub import snapshot_download

    if refresh and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} from the Hub -> {dst} ...")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dst))
    print("Download complete.")


def _load_features(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())["features"]


def _normalize_features_to(real_root: Path, sim_features: dict) -> None:
    """Make the real copy's video-feature ``info`` identical to sim's, so
    ``aggregate_datasets``' exact-equality feature check passes. Non-video features
    are asserted already-equal (they are, by the sim dataset's design)."""
    info_path = real_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    real_features = info["features"]
    if set(real_features) != set(sim_features):
        raise SystemExit(
            "Feature keys differ between sim and real; cannot aggregate.\n"
            f"  sim : {sorted(sim_features)}\n  real: {sorted(real_features)}"
        )
    for key, feat in real_features.items():
        if feat.get("dtype") == "video":
            feat["info"] = sim_features[key]["info"]  # adopt sim's encoding-info block
        elif feat != sim_features[key]:
            raise SystemExit(
                f"Non-video feature {key!r} differs between sim and real:\n"
                f"  sim : {sim_features[key]}\n  real: {feat}"
            )
    info_path.write_text(json.dumps(info, indent=4))
    print(f"Normalized real feature info to match sim ({info_path}).")


def _unify_task(real_root: Path, task: str) -> None:
    """Rewrite the real copy's task string to ``task`` everywhere it is stored:
    the single-row ``meta/tasks.parquet`` (the frame-level conditioning source) and
    any ``tasks`` column in the per-episode metadata parquets."""
    tasks_path = real_root / "meta" / "tasks.parquet"
    df = pd.read_parquet(tasks_path)
    old = list(df.index)
    if len(old) != 1:
        raise SystemExit(f"Expected exactly 1 task in {tasks_path}, found {len(old)}: {old}")
    old_task = old[0]
    df = df.rename(index={old_task: task})
    df.index.name = "task"
    df.to_parquet(tasks_path)
    print(f"Real task: {old_task!r} -> {task!r}")

    # Cosmetic but keep consistent: replace the old string in per-episode metadata.
    for ep_path in glob.glob(str(real_root / "meta" / "episodes" / "**" / "*.parquet"),
                             recursive=True):
        try:
            edf = pd.read_parquet(ep_path)
            if "tasks" not in edf.columns:
                continue

            def _swap(v):
                if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
                    return [task if str(x) == str(old_task) else x for x in list(v)]
                return task if str(v) == str(old_task) else v

            edf["tasks"] = edf["tasks"].map(_swap)
            edf.to_parquet(ep_path)
        except Exception as e:  # noqa: BLE001 - non-fatal; tasks.parquet is authoritative
            print(f"  (skipped episodes parquet {ep_path}: {e})")


def _check_sim_task(sim_root: Path, task: str) -> None:
    df = pd.read_parquet(sim_root / "meta" / "tasks.parquet")
    tasks = list(df.index)
    if tasks != [task]:
        raise SystemExit(
            f"Sim dataset task is {tasks}, expected exactly [{task!r}].\n"
            f"  Fix with: uv run python scripts/set_task.py --root {sim_root} --task {task!r}"
        )
    print(f"Sim task OK: {task!r}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-root", default="recordings/sim_redcubes_bluecup")
    p.add_argument("--sim-repo-id", default="sim/sim_redcubes_bluecup")
    p.add_argument("--real-repo-id", default="yeeegem/redcubes_bluecup")
    p.add_argument("--real-dir", default="recordings/real_redcubes_bluecup",
                   help="local editable copy of the real dataset (downloaded here).")
    p.add_argument("--out", default="recordings/cotrain_sim_real")
    p.add_argument("--out-repo-id", default="cotrain/sim_real")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--real-oversample", type=int, default=3,
                   help="how many times to repeat the real dataset in the mix.")
    p.add_argument("--refresh-download", action="store_true",
                   help="re-download the real dataset even if a local copy exists.")
    p.add_argument("--overwrite", action="store_true",
                   help="replace the output dir if it already exists.")
    args = p.parse_args(argv)

    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    sim_root = Path(args.sim_root)
    real_dir = Path(args.real_dir)
    out = Path(args.out)

    if not (sim_root / "meta" / "info.json").exists():
        raise SystemExit(f"Sim dataset not found at {sim_root}. Generate it first "
                         f"(scripts/generate_demos.sh) then re-run.")
    if out.exists():
        if args.overwrite:
            shutil.rmtree(out)
        else:
            raise SystemExit(f"Output dir exists: {out}\n  Pass --overwrite to replace it.")

    # 1. Real copy from the Hub.
    _download_real(args.real_repo_id, real_dir, args.refresh_download)

    # 2. Make real compatible with sim (features + task), and confirm sim's task.
    sim_features = _load_features(sim_root)
    _normalize_features_to(real_dir, sim_features)
    _unify_task(real_dir, args.task)
    _check_sim_task(sim_root, args.task)

    # 3. Aggregate: sim (1x, canonical features) + real (K x for oversampling).
    k = max(1, args.real_oversample)
    repo_ids = [args.sim_repo_id] + [f"{args.real_repo_id}_dup{i}" for i in range(k)]
    roots = [sim_root] + [real_dir] * k
    print(f"\nAggregating sim (1x) + real ({k}x) -> {out} ...")
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=args.out_repo_id,
        roots=roots,
        aggr_root=out,
    )

    # 4. Verify.
    m = LeRobotDatasetMetadata(args.out_repo_id, root=out)
    sim_frames = json.loads((sim_root / "meta" / "info.json").read_text())["total_frames"]
    real_frames = m.total_frames - sim_frames
    print("\n==================== co-train dataset ====================")
    print(f"root          : {out}")
    print(f"episodes      : {m.total_episodes}")
    print(f"frames        : {m.total_frames}  (sim {sim_frames} + real {real_frames})")
    print(f"real fraction : {real_frames / max(m.total_frames, 1):.0%}")
    print(f"tasks         : {list(m.tasks.index)}")
    print("---------------------------------------------------------")
    print("Train SmolVLA on it with:")
    print(f"  DATASET_ROOT={out} DATASET_REPO_ID={args.out_repo_id} \\")
    print("  OUTPUT_DIR=runs/smolvla_cotrain scripts/train_smolvla.sh")
    print("=========================================================")


if __name__ == "__main__":
    main()
