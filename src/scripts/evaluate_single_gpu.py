"""
Batch evaluation on a single GPU: find checkpoint folders missing test_results
and either print or run eval commands sequentially.

Optionally filter folders by wandb runs of project "minerva-models" with a given tag
(wandb run name = folder name in ckpt dir).
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = ("minerva_1A", "minerva_1B")


def _load_project_env() -> None:
    """Load KEY=VALUE lines from the repository root `.env` into `os.environ`.

    Does not override variables already set in the environment (same default as python-dotenv).
    """
    env_path = REPO_ROOT / ".env"
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


def get_folders_from_wandb(tag: str, project: str = "minerva-models") -> Set[str]:
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


def get_all_ckpt_folders(ckpt_dir: str) -> Set[str]:
    """Return set of subdirectory names in ckpt_dir that look like checkpoint runs (have best_model.pt)."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.is_dir():
        return set()
    folders = set()
    for p in ckpt_path.iterdir():
        if p.is_dir() and (p / "best_model.pt").exists():
            folders.add(p.name)
    return folders


def has_test_results_npz_for_dataset(
    ckpt_dir: str, folder: str, dataset_name: str
) -> bool:
    """True if ckpt_dir/folder/test_results contains an .npz for this dataset (e.g. *minerva_1A*.npz)."""
    results_dir = Path(ckpt_dir) / folder / "test_results"
    if not results_dir.is_dir():
        return False
    return any(results_dir.glob(f"*{dataset_name}*.npz"))


def folder_matches_task(folder: str, task: str) -> bool:
    """Heuristic: training run folder names include ``_classifier_`` or ``_regression_``."""
    if task == "classifier":
        return "_classifier_" in folder
    if task == "regression":
        return "_regression_" in folder
    raise ValueError(f"Unknown task: {task!r}")


def filter_folders_by_tasks(folders: List[str], tasks: List[str]) -> List[str]:
    if set(tasks) == {"regression", "classifier"}:
        return folders
    kept = [
        folder
        for folder in folders
        if any(folder_matches_task(folder, task) for task in tasks)
    ]
    return kept


def resolve_folders(ckpt_dir: str, wandb_flag: Optional[str]) -> List[str]:
    """Return sorted checkpoint folder names to evaluate."""
    if wandb_flag:
        selected = get_folders_from_wandb(wandb_flag)
        all_folders = get_all_ckpt_folders(ckpt_dir)
        folders = sorted(selected & all_folders)
        if not folders:
            print("No checkpoint folders matched the wandb tag.")
        return folders

    folders = sorted(get_all_ckpt_folders(ckpt_dir))
    if not folders:
        print("No checkpoint folders found in", ckpt_dir)
    return folders


def build_eval_argv(ckpt_dir: str, folder: str, dataset_name: str) -> List[str]:
    """Build argv for one eval subprocess using the current interpreter."""
    return [
        sys.executable,
        "-m",
        "src.scripts.eval",
        "--checkpoint",
        folder,
        "--base_dir",
        ckpt_dir,
        "--dataset_name",
        dataset_name,
    ]


def collect_eval_jobs(ckpt_dir: str, folders: List[str]) -> List[List[str]]:
    """Return argv lists for eval jobs that do not yet have test_results npz files."""
    jobs: List[List[str]] = []
    for folder in folders:
        for dataset_name in DATASETS:
            if has_test_results_npz_for_dataset(ckpt_dir, folder, dataset_name):
                continue
            jobs.append(build_eval_argv(ckpt_dir, folder, dataset_name))
    return jobs


def format_argv(argv: List[str]) -> str:
    """Shell-join argv for dry-run printing."""
    return shlex.join(argv)


def run_eval_jobs(jobs: List[List[str]], *, dry_run: bool) -> int:
    """Print or sequentially execute eval jobs. Returns process exit code."""
    if not jobs:
        print("No eval jobs to run (all test_results already present).")
        return 0

    if dry_run:
        for argv in jobs:
            print(format_argv(argv))
        return 0

    print(f"Running {len(jobs)} eval job(s) with {sys.executable!r} …")
    for i, argv in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {format_argv(argv)}")
        try:
            subprocess.run(argv, check=True, cwd=REPO_ROOT)
        except subprocess.CalledProcessError as exc:
            print(f"Eval failed with exit code {exc.returncode}.", file=sys.stderr)
            return exc.returncode

    print(f"\nAll {len(jobs)} eval job(s) completed.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or print eval commands for checkpoint folders missing test_results. "
            "By default executes sequentially on the current node (single GPU)."
        )
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
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["regression", "classifier"],
        default=["regression", "classifier"],
        help="Which checkpoint task types to evaluate (default: both). "
        "Folders are matched by ``_regression_`` / ``_classifier_`` in the name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eval commands without executing them.",
    )
    args = parser.parse_args()

    folders = resolve_folders(args.ckpt_dir, args.wandb_flag)
    if not folders:
        return

    tasks = list(dict.fromkeys(args.tasks))
    folders = filter_folders_by_tasks(folders, tasks)
    if not folders:
        print(f"No checkpoint folders matched tasks {tasks}.")
        return

    jobs = collect_eval_jobs(args.ckpt_dir, folders)
    raise SystemExit(run_eval_jobs(jobs, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
