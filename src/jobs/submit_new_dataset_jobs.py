#!/usr/bin/env python3
"""Submit training jobs on the fixed NEW dataset (20260326_NEW).

Trains the comparison model set on
``/global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW`` for both regression
and classification. One SLURM job per (model, task, seed).

Models (plot keys):
  HyperScale-small, HyperScale-small-rw,
  OmniLearned-small, OmniLearned-small-rw,
  OmniLearned-medium (frozen backbone via OLM_FB),
  HyperScale-medium, HyperScale-medium-rw,
  BERT-tiny (BERT-small pretrained), BERT-tiny-rw (BERT-small random init)

MLP (MLP3) commands are printed when ``--models`` includes ``MLP`` (local only, no SLURM).

Usage (from repo root):

    python src/jobs/submit_new_dataset_jobs.py --seeds 55 56 57 58
    python src/jobs/submit_new_dataset_jobs.py --seeds 55 --models HyperScale-small
    python src/jobs/submit_new_dataset_jobs.py --seeds 55 60 61 62 --models BERT-tiny BERT-tiny-rw
    python src/jobs/submit_new_dataset_jobs.py --seeds 55 --models HyperScale-small --tasks regression
    python src/jobs/submit_new_dataset_jobs.py --seeds 55 --models MLP
    python src/jobs/submit_new_dataset_jobs.py --seeds 55 --dry-run
    python src/jobs/submit_new_dataset_jobs.py --list-models
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from src.constants.slurm_template import SLURM_TEMPLATE_GPU
from src.jobs.mlp_sweep_configs import (
    DEFAULT_BATCH_SIZE as MLP_BATCH_SIZE,
    DEFAULT_MAX_STEPS as MLP_MAX_STEPS,
    DEFAULT_NUM_WORKERS as MLP_NUM_WORKERS,
    DEFAULT_WARMUP_STEPS as MLP_WARMUP_STEPS,
    MLP_SWEEP_CONFIGS,
    build_train_cmd,
)
from src.jobs.submit_train_jobs import CONTAINER_IMAGE, generate_cmd, get_env_vars

DEFAULT_DATA_PATH = "/global/cfs/cdirs/m3246/gregork/Minerva/20260326_NEW"
RUN_TAG = "run_new_dataset"
SLURM_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/{RUN_TAG}"
LOG_DIR = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/{RUN_TAG}"
DEFAULT_SHM_SIZE = "32g"

# HyperScale jobs use the embedding variant (matches existing pretrained ckpts).
_HS_VARIANT = "embedding"


@dataclass(frozen=True)
class ModelSpec:
    plot_key: str
    generate_model: str
    bs: int
    grad_accum_steps: int
    max_steps_regression: int
    max_steps_classifier: int
    walltime_regression: str
    walltime_classifier: str
    fp16: bool = False


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        plot_key="HyperScale-small",
        generate_model=f"HyperScale-small-{_HS_VARIANT}",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=200_000,
        max_steps_classifier=70_000,
        walltime_regression="04:00:00",
        walltime_classifier="02:00:00",
        fp16=True,
    ),
    ModelSpec(
        plot_key="HyperScale-small-rw",
        generate_model=f"HyperScale-small-rw-{_HS_VARIANT}",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=200_000,
        max_steps_classifier=70_000,
        walltime_regression="04:00:00",
        walltime_classifier="03:00:00",
        fp16=True,
    ),
    ModelSpec(
        plot_key="OmniLearned-small",
        generate_model="OLS",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=500_000,
        max_steps_classifier=500_000,
        walltime_regression="12:00:00",
        walltime_classifier="12:00:00",
    ),
    ModelSpec(
        plot_key="OmniLearned-small-rw",
        generate_model="OLS_RW",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=500_000,
        max_steps_classifier=500_000,
        walltime_regression="12:00:00",
        walltime_classifier="12:00:00",
    ),
    ModelSpec(
        plot_key="OmniLearned-medium",
        generate_model="OLM_FB",
        bs=512,
        grad_accum_steps=4,
        max_steps_regression=500_000,
        max_steps_classifier=500_000,
        walltime_regression="15:00:00",
        walltime_classifier="15:00:00",
    ),
    ModelSpec(
        plot_key="HyperScale-medium",
        generate_model=f"HyperScale-medium-{_HS_VARIANT}",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=200_000,
        max_steps_classifier=70_000,
        walltime_regression="15:00:00",
        walltime_classifier="05:00:00",
        fp16=True,
    ),
    ModelSpec(
        plot_key="HyperScale-medium-rw",
        generate_model=f"HyperScale-medium-rw-{_HS_VARIANT}",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=200_000,
        max_steps_classifier=70_000,
        walltime_regression="04:00:00",
        walltime_classifier="03:00:00",
        fp16=True,
    ),
    ModelSpec(
        plot_key="BERT-tiny",
        generate_model="BERT-tiny",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=500_000,
        max_steps_classifier=500_000,
        walltime_regression="04:00:00",
        walltime_classifier="04:00:00",
    ),
    ModelSpec(
        plot_key="BERT-tiny-rw",
        generate_model="BERT-tiny-rw",
        bs=2048,
        grad_accum_steps=1,
        max_steps_regression=500_000,
        max_steps_classifier=500_000,
        walltime_regression="04:00:00",
        walltime_classifier="04:00:00",
    ),
)

SLURM_MODEL_KEYS: tuple[str, ...] = tuple(spec.plot_key for spec in MODEL_SPECS)
ALL_MODEL_KEYS: tuple[str, ...] = SLURM_MODEL_KEYS + ("MLP",)


def resolve_model_selection(models: tuple[str, ...]) -> tuple[tuple[ModelSpec, ...], bool]:
    """Return SLURM model specs and whether to print local MLP commands."""
    unknown = sorted(set(models) - set(ALL_MODEL_KEYS))
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. "
            f"Choose from: {list(ALL_MODEL_KEYS)}"
        )
    include_mlp = "MLP" in models
    slurm_keys = [m for m in models if m != "MLP"]
    specs = tuple(spec for spec in MODEL_SPECS if spec.plot_key in slurm_keys)
    if not specs and not include_mlp:
        raise ValueError("No jobs selected. Pass at least one model via --models.")
    return specs, include_mlp


def list_models() -> None:
    print("SLURM models:")
    for key in SLURM_MODEL_KEYS:
        print(f"  {key}")
    print("Local only:")
    print("  MLP")


def build_mlp_training_cmd(*, task: str, seed: int, data_path: str) -> str:
    max_steps = MLP_MAX_STEPS
    return build_train_cmd(
        MLP_SWEEP_CONFIGS["MLP3"],
        task=task,
        seed=seed,
        data_path=data_path,
        batch_size=MLP_BATCH_SIZE,
        max_steps=max_steps,
        warmup_steps=MLP_WARMUP_STEPS,
        grad_accum_steps=1,
        event_sampler_seed=seed,
        num_workers=MLP_NUM_WORKERS,
    )


def iter_mlp_local_cmds(
    *,
    seeds: tuple[int, ...],
    tasks: tuple[str, ...],
    data_path: str,
) -> list[tuple[str, str]]:
    """Return (label, train_cmd) for MLP runs to execute locally."""
    runs: list[tuple[str, str]] = []
    for task in tasks:
        for seed in seeds:
            label = f"MLP_{task}_seed{seed}"
            cmd = build_mlp_training_cmd(task=task, seed=seed, data_path=data_path)
            runs.append((label, cmd))
    return runs


def print_local_cmds(runs: list[tuple[str, str]]) -> None:
    print(f"\n{'=' * 72}")
    print(f"MLP — run locally ({len(runs)} command(s), not submitted via SLURM)")
    print(f"{'=' * 72}")
    for label, cmd in runs:
        print(f"\n# {label}")
        print(cmd)


def build_training_cmd(
    spec: ModelSpec,
    *,
    task: str,
    seed: int,
    data_path: str,
) -> str:
    if task == "regression":
        max_steps = spec.max_steps_regression
    elif task == "classifier":
        max_steps = spec.max_steps_classifier
    else:
        raise ValueError(f"Unknown task: {task!r}")

    return generate_cmd(
        data_cap=-1,
        seed=seed,
        task=task,
        model=spec.generate_model,
        max_steps=max_steps,
        bs=spec.bs,
        grad_accum_steps=spec.grad_accum_steps,
        data_path=data_path,
        fp16=spec.fp16,
    )


def iter_training_runs(
    *,
    model_specs: tuple[ModelSpec, ...],
    seeds: tuple[int, ...],
    tasks: tuple[str, ...],
    data_path: str,
) -> list[tuple[str, str, str]]:
    """Return (label, walltime, train_cmd) in sweep order."""
    runs: list[tuple[str, str, str]] = []
    for spec in model_specs:
        for task in tasks:
            walltime = (
                spec.walltime_regression
                if task == "regression"
                else spec.walltime_classifier
            )
            for seed in seeds:
                label = f"{spec.plot_key}_{task}_seed{seed}"
                cmd = build_training_cmd(
                    spec, task=task, seed=seed, data_path=data_path
                )
                runs.append((label, walltime, cmd))
    return runs


def submit_job(
    cmd: str,
    walltime: str,
    *,
    job_name: str,
    dry_run: bool = False,
) -> str:
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

    print(f"[{job_name}] SLURM script: {slurm_path}")
    print(f"  walltime={walltime}")
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
            "Submit training jobs on the NEW dataset (20260326_NEW) for the "
            "standard comparison model set."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model keys and exit",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Training seed(s) to submit (required unless --list-models)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(ALL_MODEL_KEYS),
        default=list(SLURM_MODEL_KEYS),
        metavar="MODEL",
        help=(
            "Model(s) to run (default: all SLURM models). "
            "Include MLP to print local train commands only."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["regression", "classifier"],
        default=["regression", "classifier"],
        help="Which task(s) to run (default: regression classifier)",
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"Preprocessed dataset root (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write .slurm files but do not call sbatch",
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if not args.seeds:
        parser.error("--seeds is required unless --list-models is set")

    seeds = tuple(args.seeds)
    tasks = tuple(args.tasks)
    models = tuple(dict.fromkeys(args.models))  # preserve order, drop dupes
    slurm_specs, include_mlp = resolve_model_selection(models)
    runs = iter_training_runs(
        model_specs=slurm_specs,
        seeds=seeds,
        tasks=tasks,
        data_path=args.data_path,
    )
    ts = dt.now().strftime("%Y%m%d_%H%M%S")

    if runs:
        slurm_model_list = ", ".join(spec.plot_key for spec in slurm_specs)
        print(
            f"Preparing {len(runs)} SLURM job(s): "
            f"[{slurm_model_list}] × {len(tasks)} task(s) × {len(seeds)} seed(s)"
        )
        print(f"  data_path={args.data_path}")

        for index, (label, walltime, cmd) in enumerate(runs):
            job_name = f"newds_{index:03d}_{label}_{ts}"
            submit_job(cmd, walltime, job_name=job_name, dry_run=args.dry_run)
    else:
        print("No SLURM jobs selected.")

    if include_mlp:
        mlp_runs = iter_mlp_local_cmds(
            seeds=seeds, tasks=tasks, data_path=args.data_path
        )
        print_local_cmds(mlp_runs)


if __name__ == "__main__":
    main()
