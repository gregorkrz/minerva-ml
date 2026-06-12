#!/usr/bin/env python3
"""Submit BERT-tiny (pretrained) runs with ``--bert-energy-order``.

By default submits two jobs: one regression and one classification (both seed 55),
using the same training budget as the existing BERT-tiny sweep in
``submit_train_jobs.py``.

Run names look like ``Run_1703_BERT_tiny_energy_order_regression_-1_seed55`` and map
to the plot/eval model key ``BERT-tiny-energy-order`` (see ``src/utils/utils.py``).

Usage (from repo root):

    python src/jobs/submit_bert_energy_order.py
    python src/jobs/submit_bert_energy_order.py --dry-run
    python src/jobs/submit_bert_energy_order.py --regression-seed 55 --classifier-seed 56
"""

import argparse
import os
import sys
from datetime import datetime as dt
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.constants.slurm_template import SLURM_TEMPLATE_GPU
from src.jobs.submit_train_jobs import CONTAINER_IMAGE, generate_cmd, get_env_vars

RUN_TAG = "run_bert_energy_order"
SLURM_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/{RUN_TAG}"
LOG_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/{RUN_TAG}"

DEFAULT_REGRESSION_SEED = 55
DEFAULT_CLASSIFIER_SEED = 55
DEFAULT_WALLTIME = "05:00:00"
DEFAULT_MAX_STEPS = 500_000
DEFAULT_BS = 2048


def build_jobs(
    *,
    regression_seed,
    classifier_seed,
    max_steps,
    tasks,
):
    """Return (task, seed, train_cmd) for each job."""
    jobs = []
    task_seeds = []
    if "regression" in tasks:
        task_seeds.append(("regression", regression_seed))
    if "classifier" in tasks:
        task_seeds.append(("classifier", classifier_seed))
    for task, seed in task_seeds:
        cmd = generate_cmd(
            data_cap=-1,
            seed=seed,
            task=task,
            model="BERT-tiny-energy-order",
            max_steps=max_steps,
            bs=DEFAULT_BS,
            grad_accum_steps=1,
        )
        jobs.append((task, str(seed), cmd))
    return jobs


def submit_job(
    cmd: str,
    walltime: str,
    *,
    job_index: int,
    task: str,
    seed: str,
    dry_run: bool = False,
) -> str:
    ts = dt.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"bert_energy_{task}_seed{seed}_{job_index}_{ts}"
    log_path = os.path.join(LOG_DIR, f"{job_name}.log")
    error_path = os.path.join(LOG_DIR, f"{job_name}.error.log")
    slurm_path = os.path.join(SLURM_DIR, f"{job_name}.slurm")

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
                commands=cmd,
                env_vars=get_env_vars(),
                container_image=CONTAINER_IMAGE,
            )
        )

    print(f"[{task} seed={seed}] SLURM script: {slurm_path}")
    print(f"  {cmd}")
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
            "Submit BERT-tiny (pretrained HF weights) with --bert-energy-order: "
            "one regression job and one classification job by default."
        )
    )
    parser.add_argument(
        "--regression-seed",
        type=int,
        default=DEFAULT_REGRESSION_SEED,
        help=f"Seed for the regression job (default: {DEFAULT_REGRESSION_SEED})",
    )
    parser.add_argument(
        "--classifier-seed",
        type=int,
        default=DEFAULT_CLASSIFIER_SEED,
        help=f"Seed for the classification job (default: {DEFAULT_CLASSIFIER_SEED})",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["regression", "classifier"],
        default=["regression", "classifier"],
        help="Which task(s) to submit (default: regression classifier)",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write .slurm files but do not call sbatch",
    )
    args = parser.parse_args()

    jobs = build_jobs(
        regression_seed=args.regression_seed,
        classifier_seed=args.classifier_seed,
        max_steps=args.max_steps,
        tasks=tuple(args.tasks),
    )
    print(
        f"Preparing {len(jobs)} job(s): tasks={args.tasks}, "
        f"regression_seed={args.regression_seed}, classifier_seed={args.classifier_seed}, "
        f"max_steps={args.max_steps}"
    )
    for i, (task, seed, cmd) in enumerate(jobs):
        submit_job(
            cmd,
            args.walltime,
            job_index=i,
            task=task,
            seed=seed,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
