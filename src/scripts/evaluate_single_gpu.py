"""
Batch evaluation on a single GPU: find checkpoint folders missing test_results
and either print or run eval commands sequentially.

Optionally filter folders by wandb runs of project "minerva-models" with a given tag
(wandb run name = folder name in ckpt dir).
"""

import argparse
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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


def parse_folder_model_seed(folder: str) -> Tuple[str, str, Optional[int]]:
    """Return ``(task, model_key, seed)`` for a checkpoint folder name.

    ``task`` is ``classifier``/``regression``/``unknown``; ``model_key`` falls back
    to ``"(unparsed)"`` when the run name is not recognized; ``seed`` is ``None``
    when no ``seed`` token is present.
    """
    from src.utils.utils import (
        classification_model_cap_from_name,
        regression_model_cap_from_name,
    )

    if "_classifier_" in folder:
        task = "classifier"
        parsed = classification_model_cap_from_name(folder)
    elif "_regression_" in folder:
        task = "regression"
        parsed = regression_model_cap_from_name(folder)
    else:
        task = "unknown"
        parsed = None
    model_key = parsed[0] if parsed else "(unparsed)"

    seed: Optional[int] = None
    m = re.search(r"seed_?(-?\d+)", folder)
    if m:
        seed = int(m.group(1))
    return task, model_key, seed


def print_check_report(ckpt_dir: str, folders: List[str]) -> None:
    """Print, per (task, model), which seeds are trained and which are evaluated.

    A seed counts as *trained* if its checkpoint folder exists (best_model.pt),
    and *evaluated* if test_results npz exist for every dataset in ``DATASETS``.
    """
    # (task, model) -> seed -> set(datasets with npz)
    status: Dict[Tuple[str, str], Dict[Optional[int], Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for folder in folders:
        task, model_key, seed = parse_folder_model_seed(folder)
        done = {
            ds
            for ds in DATASETS
            if has_test_results_npz_for_dataset(ckpt_dir, folder, ds)
        }
        status[(task, model_key)][seed] |= done

    def _fmt(seeds) -> str:
        return (
            "["
            + ", ".join(
                str(s) if s is not None else "?"
                for s in sorted(seeds, key=lambda x: (x is None, x))
            )
            + "]"
        )

    print(f"\n=== Checkpoint / eval status ({len(folders)} folder(s)) ===")
    print(f"    datasets required for 'evaluated': {', '.join(DATASETS)}\n")
    for task, model_key in sorted(status.keys()):
        seed_map = status[(task, model_key)]
        trained = set(seed_map.keys())
        full = {s for s, dss in seed_map.items() if set(DATASETS) <= dss}
        partial = {
            s: seed_map[s]
            for s in seed_map
            if seed_map[s] and not set(DATASETS) <= seed_map[s]
        }
        not_eval = trained - full - set(partial.keys())
        print(f"[{task}] {model_key}")
        print(f"    trained         {len(trained):>2}: {_fmt(trained)}")
        print(f"    evaluated       {len(full):>2}: {_fmt(full)}")
        if partial:
            detail = ", ".join(
                f"{s if s is not None else '?'}({'+'.join(sorted(dss))})"
                for s, dss in sorted(
                    partial.items(), key=lambda kv: (kv[0] is None, kv[0])
                )
            )
            print(f"    partial-eval  {len(partial):>2}: {detail}")
        if not_eval:
            print(f"    trained-only  {len(not_eval):>2}: {_fmt(not_eval)}")
    print()


def format_argv(argv: List[str]) -> str:
    """Shell-join argv for dry-run printing."""
    return shlex.join(argv)


def resolve_gpu_ids(num_gpus: int) -> List[str]:
    """Return the list of GPU ids to schedule eval jobs on.

    If ``CUDA_VISIBLE_DEVICES`` is already set (e.g. by SLURM), use those ids so we
    only ever touch GPUs allocated to this job; otherwise fall back to ``0..num_gpus-1``.
    The result is capped to ``num_gpus`` entries.
    """
    existing = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if existing:
        ids = [tok.strip() for tok in existing.split(",") if tok.strip()]
    else:
        ids = [str(i) for i in range(num_gpus)]

    if not ids:
        ids = ["0"]

    if num_gpus < len(ids):
        ids = ids[:num_gpus]
    elif num_gpus > len(ids):
        print(
            f"Requested {num_gpus} GPU(s) but only {len(ids)} visible "
            f"({', '.join(ids)}); using {len(ids)}.",
            file=sys.stderr,
        )
    return ids


def run_eval_jobs(jobs: List[List[str]], *, dry_run: bool, num_gpus: int = 1) -> int:
    """Print or execute eval jobs, optionally spreading them across multiple GPUs.

    With ``num_gpus == 1`` jobs run sequentially on the current device. With
    ``num_gpus > 1`` one worker per GPU pulls jobs from a shared queue and runs them
    concurrently, pinning each subprocess to its GPU via ``CUDA_VISIBLE_DEVICES``.
    Returns a process exit code (non-zero if any job failed).
    """
    if not jobs:
        print("No eval jobs to run (all test_results already present).")
        return 0

    if dry_run:
        for argv in jobs:
            print(format_argv(argv))
        return 0

    gpu_ids = resolve_gpu_ids(num_gpus)
    if len(gpu_ids) <= 1:
        return _run_eval_jobs_sequential(jobs, gpu_id=gpu_ids[0])
    return _run_eval_jobs_parallel(jobs, gpu_ids=gpu_ids)


def _run_eval_jobs_sequential(jobs: List[List[str]], *, gpu_id: str) -> int:
    """Run eval jobs one at a time on a single GPU. Returns process exit code."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(
        f"Running {len(jobs)} eval job(s) with {sys.executable!r} " f"on GPU {gpu_id} …"
    )
    for i, argv in enumerate(jobs, start=1):
        print(f"\n[{i}/{len(jobs)}] {format_argv(argv)}")
        try:
            subprocess.run(argv, check=True, cwd=REPO_ROOT, env=env)
        except subprocess.CalledProcessError as exc:
            print(f"Eval failed with exit code {exc.returncode}.", file=sys.stderr)
            return exc.returncode

    print(f"\nAll {len(jobs)} eval job(s) completed.")
    return 0


def _run_eval_jobs_parallel(jobs: List[List[str]], *, gpu_ids: List[str]) -> int:
    """Run eval jobs concurrently with one worker per GPU. Returns process exit code."""
    total = len(jobs)
    job_queue: "queue.Queue[Tuple[int, List[str]]]" = queue.Queue()
    for i, argv in enumerate(jobs, start=1):
        job_queue.put((i, argv))

    print_lock = threading.Lock()
    failures: List[Tuple[int, int]] = []  # (job index, returncode)

    def worker(gpu_id: str) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        while True:
            try:
                index, argv = job_queue.get_nowait()
            except queue.Empty:
                return
            with print_lock:
                print(f"[gpu {gpu_id}] [{index}/{total}] start: {format_argv(argv)}")
            proc = subprocess.run(argv, cwd=REPO_ROOT, env=env)
            with print_lock:
                if proc.returncode != 0:
                    print(
                        f"[gpu {gpu_id}] [{index}/{total}] FAILED "
                        f"(exit {proc.returncode})",
                        file=sys.stderr,
                    )
                    failures.append((index, proc.returncode))
                else:
                    print(f"[gpu {gpu_id}] [{index}/{total}] done")

    print(
        f"Running {total} eval job(s) with {sys.executable!r} "
        f"across {len(gpu_ids)} GPU(s): {', '.join(gpu_ids)} …"
    )
    threads = [
        threading.Thread(target=worker, args=(gpu_id,), name=f"gpu-{gpu_id}")
        for gpu_id in gpu_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failures:
        failures.sort()
        ids = ", ".join(str(idx) for idx, _ in failures)
        print(
            f"\n{len(failures)}/{total} eval job(s) failed (indices: {ids}).",
            file=sys.stderr,
        )
        return failures[0][1] or 1

    print(f"\nAll {total} eval job(s) completed.")
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
        "--num-gpus",
        "-n",
        type=int,
        default=1,
        metavar="N",
        help="Number of GPUs to distribute eval jobs across (default: 1). With N>1, "
        "one job runs per GPU concurrently, pinned via CUDA_VISIBLE_DEVICES. Capped to "
        "the GPUs visible in CUDA_VISIBLE_DEVICES (e.g. those SLURM allocated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eval commands without executing them.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Do not run/print eval jobs; just report, per model, which seeds "
        "have been trained (checkpoint exists) and which have been evaluated "
        "(test_results npz present for every dataset).",
    )
    args = parser.parse_args()

    if args.num_gpus < 1:
        parser.error("--num-gpus must be >= 1")

    folders = resolve_folders(args.ckpt_dir, args.wandb_flag)
    if not folders:
        return

    tasks = list(dict.fromkeys(args.tasks))
    folders = filter_folders_by_tasks(folders, tasks)
    if not folders:
        print(f"No checkpoint folders matched tasks {tasks}.")
        return

    if args.check_only:
        print_check_report(args.ckpt_dir, folders)
        return

    jobs = collect_eval_jobs(args.ckpt_dir, folders)
    raise SystemExit(run_eval_jobs(jobs, dry_run=args.dry_run, num_gpus=args.num_gpus))


if __name__ == "__main__":
    main()
