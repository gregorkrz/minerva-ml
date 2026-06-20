#!/usr/bin/env python3
"""Submit multiclass CC1orNPi classifiers trained on cut-based baseline labels.

Completed runs map to plot/eval keys ``Transformer-small-Baseline`` and
``MLP-Baseline`` (see ``parse_predict_baseline_classifier_model_cap`` in
``src/utils/utils.py``).

Usage (from repo root):

    python src/jobs/submit_predict_baseline_ccnpi.py
    python src/jobs/submit_predict_baseline_ccnpi.py --dry-run
    python src/jobs/submit_predict_baseline_ccnpi.py --models transformer-small --seed 56
    python src/jobs/submit_predict_baseline_ccnpi.py --models mlp --mlp-variant MLP3
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
    DEFAULT_DATA_PATH,
    DEFAULT_MAX_STEPS as MLP_DEFAULT_MAX_STEPS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_WARMUP_STEPS,
    MLP_SWEEP_CONFIGS,
    build_train_cmd,
)
from src.jobs.submit_train_jobs import CONTAINER_IMAGE, generate_cmd, get_env_vars

RUN_TAG = "run_predict_baseline"
SLURM_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/{RUN_TAG}"
LOG_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/{RUN_TAG}"

DEFAULT_SEED = 55
DEFAULT_WALLTIME_MLP = "02:00:00"
DEFAULT_WALLTIME_TRANSFORMER = "05:00:00"
DEFAULT_MAX_STEPS = 500_000
DEFAULT_MLP_VARIANT = "MLP3"
DEFAULT_SHM_SIZE = "32g"

_MODEL_ALIASES = {
    "transformer-small": "Transformer3NR",
    "Transformer-small": "Transformer3NR",
}


def build_mlp_cmd(
    *,
    seed: int,
    mlp_variant: str,
    max_steps: int,
    data_path: str,
) -> str:
    cfg = MLP_SWEEP_CONFIGS[mlp_variant]
    return build_train_cmd(
        cfg,
        task="classifier",
        seed=seed,
        data_path=data_path,
        batch_size=DEFAULT_BATCH_SIZE,
        max_steps=max_steps,
        warmup_steps=DEFAULT_WARMUP_STEPS,
        num_workers=DEFAULT_NUM_WORKERS,
        predict_baseline=True,
    )


def build_transformer_cmd(*, seed: int, max_steps: int) -> str:
    model = _MODEL_ALIASES["transformer-small"]
    return generate_cmd(
        data_cap=-1,
        seed=seed,
        task="classifier",
        model=model,
        max_steps=max_steps,
        bs=2048,
        grad_accum_steps=1,
        predict_baseline=True,
    )


def submit_job(
    cmd: str,
    walltime: str,
    job_suffix: str,
    *,
    dry_run: bool = False,
) -> str:
    job_name = f"predictBaseline_{job_suffix}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
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
                mem_directive="",
                shm_size=DEFAULT_SHM_SIZE,
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
            "Submit multiclass CC1orNPi classifiers with --predict-baseline "
            "(MLP cond-only and/or Transformer-small)."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mlp", "transformer-small"],
        default=["mlp", "transformer-small"],
        help="Which model(s) to submit (default: both)",
    )
    parser.add_argument(
        "--mlp-variant",
        default=DEFAULT_MLP_VARIANT,
        choices=sorted(MLP_SWEEP_CONFIGS),
        help=f"Cond-only MLP config key (default: {DEFAULT_MLP_VARIANT})",
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--walltime-mlp",
        default=DEFAULT_WALLTIME_MLP,
        help=f"SLURM walltime for MLP jobs (default: {DEFAULT_WALLTIME_MLP})",
    )
    parser.add_argument(
        "--walltime-transformer",
        default=DEFAULT_WALLTIME_TRANSFORMER,
        help=(
            "SLURM walltime for Transformer-small jobs "
            f"(default: {DEFAULT_WALLTIME_TRANSFORMER})"
        ),
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Dataset root passed to train.py --data_path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write .slurm files but do not call sbatch",
    )
    args = parser.parse_args()

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    submitted: list[str] = []

    if "mlp" in args.models:
        cmd = build_mlp_cmd(
            seed=args.seed,
            mlp_variant=args.mlp_variant,
            max_steps=args.max_steps,
            data_path=args.data_path,
        )
        submitted.append(
            submit_job(
                cmd,
                args.walltime_mlp,
                f"{args.mlp_variant}_seed{args.seed}",
                dry_run=args.dry_run,
            )
        )

    if "transformer-small" in args.models:
        cmd = build_transformer_cmd(seed=args.seed, max_steps=args.max_steps)
        submitted.append(
            submit_job(
                cmd,
                args.walltime_transformer,
                f"Transformer3NR_seed{args.seed}",
                dry_run=args.dry_run,
            )
        )

    print(f"\nDone: {len(submitted)} job(s) {'prepared' if args.dry_run else 'submitted'}.")


if __name__ == "__main__":
    main()
