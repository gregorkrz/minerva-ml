#!/usr/bin/env python3
"""Submit binary CCNpi± (N>=1) classifier with W-binned loss (Weigh2).

Trains ``Transformer-xsmall`` with a 2-class head (signal vs background) on
``-npi2`` Pi_labels_v2 pid classes: signal = CCN1pipm (pid 0, 1), background =
pid 2–4. Per-W-bin loss weights are unchanged from Weigh1.

Completed runs map to plot/eval key ``Transformer-xsmall-Weigh2``.

Usage (from repo root):

    python src/jobs/submit_binned_loss_ccn1pipm_binary.py
    python src/jobs/submit_binned_loss_ccn1pipm_binary.py --seed 56 --dry-run
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

RUN_TAG = "run_binned_loss_binary"
SLURM_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/{RUN_TAG}"
LOG_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/{RUN_TAG}"

DEFAULT_MODEL = "Transformer-xsmall"
DEFAULT_SEED = 55
DEFAULT_WALLTIME = "05:00:00"
DEFAULT_MAX_STEPS = 500_000

_MODEL_ALIASES = {
    "Transformer-xsmall": "Transformer1",
    "Transformer-small": "Transformer3NR",
}


def _resolve_model(model: str) -> str:
    return _MODEL_ALIASES.get(model, model)


def build_cmd(*, seed: int, model: str, max_steps: int) -> str:
    return generate_cmd(
        data_cap=-1,
        seed=seed,
        task="classifier",
        model=_resolve_model(model),
        max_steps=max_steps,
        bs=2048,
        grad_accum_steps=1,
        binned_loss_var="W",
        binned_loss_signal="CCN1pipm",
        binary_classifier=True,
    )


def submit_job(cmd: str, walltime: str, *, dry_run: bool = False) -> str:
    job_name = f"binned_CCN1pipmBin_{dt.now().strftime('%Y%m%d_%H%M%S')}"
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

    print(f"SLURM script: {slurm_path}")
    print(f"Train command:\n  {cmd}")
    if dry_run:
        print("Dry run — not submitting.")
        return slurm_path

    rc = os.system(f"sbatch {slurm_path}")
    if rc != 0:
        raise RuntimeError(f"sbatch failed with exit code {rc}")
    print("Job submitted.")
    return slurm_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Submit one binary CCNpi± (N>=1) classifier with W-binned loss "
            "(Transformer-xsmall-Weigh2)."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model tag understood by generate_cmd (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--walltime", default=DEFAULT_WALLTIME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the .slurm file but do not call sbatch",
    )
    args = parser.parse_args()

    cmd = build_cmd(seed=args.seed, model=args.model, max_steps=args.max_steps)
    submit_job(cmd, args.walltime, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
