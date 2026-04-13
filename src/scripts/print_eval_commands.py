"""
Batch evaluation script: list checkpoint folders and print eval commands for those
that don't already have test_results with an npz file.

Optionally filter folders by wandb runs of project "minerva-models" with a given tag
(wandb run name = folder name in ckpt dir).
"""

import argparse
import os
from pathlib import Path


def _load_project_env() -> None:
    """Load KEY=VALUE lines from the repository root `.env` into `os.environ`.

    Does not override variables already set in the environment (same default as python-dotenv).
    """
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_project_env()


def get_folders_from_wandb(tag: str, project: str = "minerva-models") -> set[str]:
    """Fetch wandb run names from project with the given tag."""
    import wandb

    print("Calling wandb API...")
    api = wandb.Api(timeout=60)
    entity = os.environ.get("WANDB_ENTITY") or getattr(api, "default_entity", None)
    if not entity:
        raise RuntimeError(
            "Wandb entity unknown. Set WANDB_ENTITY or log in with wandb login."
        )
    path = f"{entity}/{project}"
    print(f"  Fetching runs from {path} with tag {tag!r}...")
    runs = api.runs(path, filters={"tags": {"$in": [tag]}})
    names = {run.name for run in runs}
    print(f"  Wandb API finished: {len(names)} run(s) found with tag {tag!r}.")
    return names


def get_all_ckpt_folders(ckpt_dir: str) -> set[str]:
    """Return set of subdirectory names in ckpt_dir that look like checkpoint runs (have best_model.pt)."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_dir():
        return set()
    folders = set()
    for p in ckpt_path.iterdir():
        if p.is_dir() and (p / "best_model.pt").exists():
            folders.add(p.name)
    return folders


DATASETS = ("minerva_1A", "minerva_1B")


def has_test_results_npz_for_dataset(
    ckpt_dir: str, folder: str, dataset_name: str
) -> bool:
    """True if ckpt_dir/folder/test_results contains an .npz for this dataset (e.g. *minerva_1A*.npz)."""
    results_dir = Path(ckpt_dir) / folder / "test_results"
    if not results_dir.is_dir():
        return False
    return any(results_dir.glob(f"*{dataset_name}*.npz"))


def main():
    parser = argparse.ArgumentParser(
        description="Print eval commands for checkpoint folders missing test_results."
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="/global/cfs/cdirs/m3246/gregork/checkpoints",
        help="Checkpoint root directory (each subdir = one run)",
    )
    parser.add_argument(
        "--wandb-flag",
        "-flag",
        type=str,
        default=None,
        metavar="TAG",
        help="If set, only consider folders whose name matches a wandb run name "
        'in project "minerva-models" with this tag.',
    )
    args = parser.parse_args()

    if args.wandb_flag:
        selected = get_folders_from_wandb(args.wandb_flag)
        all_folders = get_all_ckpt_folders(args.ckpt_dir)
        folders = sorted(selected & all_folders)
        if not folders:
            print("No checkpoint folders matched the wandb tag.")
            return
    else:
        folders = sorted(get_all_ckpt_folders(args.ckpt_dir))
        if not folders:
            print("No checkpoint folders found in", args.ckpt_dir)
            return

    for folder in folders:
        for dataset_name in DATASETS:
            if has_test_results_npz_for_dataset(args.ckpt_dir, folder, dataset_name):
                continue
            cmd = (
                f"python -m src.scripts.eval --checkpoint {folder} --base_dir {args.ckpt_dir} "
                f"--dataset_name {dataset_name}"
            )
            print(cmd)


if __name__ == "__main__":
    main()
