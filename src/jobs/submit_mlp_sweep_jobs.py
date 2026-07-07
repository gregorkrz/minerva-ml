#!/usr/bin/env python3
"""Submit the cond-only MLP hyperparameter sweep (MLP1–MLP10).

Default: one SLURM job per training run (80 jobs for the full sweep). Each job uses
one GPU with ``bs=32000``, ``max_steps=256000``, ``warmup_steps=200``, ``num_workers=16``,
and the cond-only fast dataloader.
All jobs are submitted together so the cluster can run many in parallel (~30 at once).

Run names look like ``Run_cond_only_lowLR_MLP3_NR_full_seed50`` and map to plot keys
``MLP3`` (see ``src/utils/utils.py``).

Usage (from repo root):

    python src/jobs/submit_mlp_sweep_jobs.py
    python src/jobs/submit_mlp_sweep_jobs.py --dry-run
    python src/jobs/submit_mlp_sweep_jobs.py --first-only
    python src/jobs/submit_mlp_sweep_jobs.py --variants MLP1 MLP2 --tasks regression
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime as dt
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.constants.slurm_template import SLURM_TEMPLATE_GPU
from src.jobs.mlp_sweep_configs import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEEDS,
    DEFAULT_WARMUP_STEPS,
    MLP_SWEEP_CONFIGS,
    MLP_SWEEP_ORDER,
    build_train_cmd,
)
from src.jobs.submit_train_jobs import CONTAINER_IMAGE, get_env_vars

RUN_TAG = "run_mlp_sweep"
SLURM_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/{RUN_TAG}"
LOG_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/{RUN_TAG}"
DEFAULT_WALLTIME = "02:00:00"
DEFAULT_RUNS_PER_SLURM_JOB = 1
DEFAULT_MEM_GB = 0
DEFAULT_SHM_SIZE = "32g"


def iter_all_training_runs(
    *,
    variants: tuple[str, ...],
    tasks: tuple[str, ...],
    seeds: tuple[int, ...],
    max_steps: int,
    warmup_steps: int,
    batch_size: int,
    data_path: str,
    num_workers: int,
) -> list[tuple[str, str]]:
    """Return (log_label, train_cmd) for every (variant, task, seed) in sweep order."""
    runs: list[tuple[str, str]] = []
    for variant in variants:
        cfg = MLP_SWEEP_CONFIGS[variant]
        for task in tasks:
            for seed in seeds:
                label = f"{variant}_{task}_seed{seed}"
                cmd = build_train_cmd(
                    cfg,
                    task=task,
                    seed=seed,
                    data_path=data_path,
                    batch_size=batch_size,
                    max_steps=max_steps,
                    warmup_steps=warmup_steps,
                    num_workers=num_workers,
                )
                runs.append((label, cmd))
    return runs


def chunk_runs(
    runs: list[tuple[str, str]], batch_size: int
) -> list[list[tuple[str, str]]]:
    """Split *runs* into batches of at most *batch_size* for separate SLURM jobs."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return [runs[i : i + batch_size] for i in range(0, len(runs), batch_size)]


def build_slurm_commands(runs: list[tuple[str, str]], *, batch_label: str) -> str:
    """Build bash for one SLURM job (single run or parallel batch)."""
    lines = [
        'LOG_DIR="/global/cfs/cdirs/m3246/gregork/Minerva/logs/run_mlp_sweep/job_${SLURM_JOB_ID:-local}"',
        'mkdir -p "$LOG_DIR"',
        f'echo "=== {batch_label}: {len(runs)} run(s) ==="',
    ]
    if len(runs) == 1:
        label, cmd = runs[0]
        log_file = f'"$LOG_DIR/{label}.log"'
        lines.append(f"({cmd}) > {log_file} 2>&1")
    else:
        for label, cmd in runs:
            log_file = f'"$LOG_DIR/{label}.log"'
            lines.append(f"({cmd}) > {log_file} 2>&1 &")
        lines.append("wait")
        lines.append(
            "if [ $? -ne 0 ]; then echo 'One or more training runs failed'; exit 1; fi"
        )
    lines.append(f'echo "Finished {batch_label}."')
    return "\n".join(lines)


def submit_job(
    commands: str,
    walltime: str,
    *,
    job_name: str,
    n_runs: int,
    mem_gb: int | None,
    shm_size: str,
    dry_run: bool = False,
) -> str:
    log_path = os.path.join(LOG_DIR, f"{job_name}.log")
    error_path = os.path.join(LOG_DIR, f"{job_name}.error.log")
    slurm_path = os.path.join(SLURM_DIR, f"{job_name}.slurm")
    if mem_gb and mem_gb > 0:
        mem_directive = f"#SBATCH --mem={mem_gb}G"
    else:
        mem_directive = ""

    os.makedirs(SLURM_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(slurm_path, "w") as f:
        f.write(
            SLURM_TEMPLATE_GPU.format(
                queue_name="shared",
                time=walltime,
                cpus_per_task=32,
                gpus_per_node=1,
                job_name=job_name,
                log_dir=log_path,
                error_dir=error_path,
                mem_directive=mem_directive,
                shm_size=shm_size,
                commands=commands,
                env_vars=get_env_vars(),
                container_image=CONTAINER_IMAGE,
            )
        )

    print(f"[{job_name}] SLURM script: {slurm_path}")
    mem_note = (
        f"mem={mem_gb}G"
        if mem_gb and mem_gb > 0
        else "mem=QOS default (64G/GPU on Perlmutter shared)"
    )
    print(f"  {n_runs} run(s), {mem_note}, shm={shm_size}")
    if dry_run:
        print("  (dry run — not submitting)")
        return slurm_path

    rc = os.system(f"sbatch {slurm_path}")
    if rc != 0:
        raise RuntimeError(f"sbatch failed with exit code {rc} for {slurm_path}")
    print("  submitted")
    return slurm_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit cond-only MLP hyperparameter sweep (MLP1–MLP10). "
            "Default: one SLURM job per training run (bs=32000, max_steps=256000, "
            "warmup_steps=200, fast cond-only loader)."
        )
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(MLP_SWEEP_ORDER),
        default=list(MLP_SWEEP_ORDER),
        help="MLP variants to train (default: MLP1 … MLP10)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["regression", "classifier"],
        default=["regression", "classifier"],
        help="Which task(s) to run (default: regression classifier)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help=f"Training seeds per (variant, task) (default: {list(DEFAULT_SEEDS)})",
    )
    parser.add_argument(
        "--runs-per-slurm-job",
        type=int,
        default=DEFAULT_RUNS_PER_SLURM_JOB,
        help=(
            "Training runs per SLURM job (default: 1). Use >1 only if you accept "
            "shared-GPU contention; not recommended for this sweep."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Dataloader workers per training run (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--mem-gb",
        type=int,
        default=DEFAULT_MEM_GB,
        help=(
            "Optional SLURM --mem in GB (default: 0 = omit; Perlmutter shared gives "
            "64 GB per GPU). Values >64 may fail sbatch (core count inflation)."
        ),
    )
    parser.add_argument(
        "--shm-size",
        default=DEFAULT_SHM_SIZE,
        help=f"Container --shm-size per job (default: {DEFAULT_SHM_SIZE})",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--batch-size", "-bs", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--data-path",
        default="/global/cfs/cdirs/m3246/gregork/Minerva/20260326",
    )
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument(
        "--first-only",
        action="store_true",
        help=(
            "Submit only the first SLURM job in sweep order (smoke test; "
            "default order: MLP1 regression seed50)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write .slurm files but do not call sbatch",
    )
    args = parser.parse_args()

    variants = tuple(v for v in MLP_SWEEP_ORDER if v in args.variants)
    tasks = tuple(args.tasks)
    seeds = tuple(args.seeds)

    all_runs = iter_all_training_runs(
        variants=variants,
        tasks=tasks,
        seeds=seeds,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        data_path=args.data_path,
        num_workers=args.num_workers,
    )
    batches = chunk_runs(all_runs, args.runs_per_slurm_job)
    if args.first_only:
        batches = batches[:1]
    n_slurm_jobs = len(batches)
    ts = dt.now().strftime("%Y%m%d_%H%M%S")

    mode_note = " (first job only)" if args.first_only else ""
    print(
        f"Preparing {n_slurm_jobs} SLURM job(s) for {len(all_runs)} training run(s)"
        f"{mode_note}: "
        f"{args.runs_per_slurm_job} run(s)/job, bs={args.batch_size}, "
        f"num_workers={args.num_workers}, max_steps={args.max_steps}, "
        f"warmup_steps={args.warmup_steps}, walltime={args.walltime}"
    )

    for batch_index, batch in enumerate(batches):
        if len(batch) == 1:
            job_name = f"mlp_sweep_{batch[0][0]}_{ts}"
        else:
            job_name = f"mlp_sweep_batch{batch_index + 1}of{n_slurm_jobs}_{ts}"
        batch_label = (
            batch[0][0]
            if len(batch) == 1
            else f"batch {batch_index + 1}/{n_slurm_jobs}"
        )
        commands = build_slurm_commands(batch, batch_label=batch_label)
        submit_job(
            commands,
            args.walltime,
            job_name=job_name,
            n_runs=len(batch),
            mem_gb=args.mem_gb,
            shm_size=args.shm_size,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
