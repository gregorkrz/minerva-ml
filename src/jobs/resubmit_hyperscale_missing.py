#!/usr/bin/env python3
"""Write SLURM scripts for HyperScale sweep jobs that still need to run.

Usage (from repo root):
  python src/jobs/resubmit_hyperscale_missing.py

Prints one ``sbatch <path>`` per line (does not submit).
"""
import glob
import os
import re
import subprocess
from datetime import datetime as dt

from src.constants.slurm_template import SLURM_TEMPLATE_GPU

CONTAINER_IMAGE = "docker.io/gkrz/minerva_ml:v1"
SLURM_DIR = "/global/cfs/cdirs/m3246/gregork/Minerva/slurm/run_100326"
LOG_DIR = "/global/cfs/cdirs/m3246/gregork/Minerva/logs/run_100326"
BATCH_GLOB = "run_{i}_20260530_1909*.slurm"
# Original sweep used 2h; timed-out rw jobs reached ~19k/200k steps — use 8h on resubmit.
WALLTIME_RESUBMIT = "08:00:00"
MAX_STEPS = 200_000
HS_PRETRAINED = (
    "/global/cfs/cdirs/m3246/gregork/Minerva/HyperscaleV5/"
    "emb_6e17_d4_e448_bs512_lr5e-4_run3_normalized"
)
# Older slurm scripts may still embed the pre-move path.
_HS_PRETRAINED_OLD_PREFIXES = (
    "/global/cfs/cdirs/m3246/jaluus/Hyperscale_V5/Pretrain_Scaling/"
    "emb_6e17_d4_e448_bs512_lr5e-4_run3_normalized",
)


def get_env_vars():
    env_commands = ""
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("WANDB_"):
                continue
            env_commands += f"export {key.strip()}={value.strip()}\n"
    return env_commands


def _batch_slurm_states():
    """Map sweep index -> latest sacct State for the 20260530_1909* batch."""
    states = {}
    try:
        out = subprocess.check_output(
            [
                "sacct",
                "-u",
                os.environ.get("USER", ""),
                "--starttime",
                "2026-05-30",
                "--format=JobName,State",
                "-n",
                "-P",
            ],
            universal_newlines=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return states
    for line in out.splitlines():
        if "20260530_1909" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_name, state = parts[0], parts[1]
        m = re.match(r"run_(\d+)_20260530_1909", job_name)
        if m:
            states[int(m.group(1))] = state
    return states


def _active_batch_jobs():
    """Job base names (run_{i}_20260530_1909*) still in the queue."""
    active = set()
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"],
            universal_newlines=True,
        )
        for name in out.splitlines():
            name = name.strip()
            if "20260530_1909" in name:
                active.add(name)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return active


def slurm_state(index, batch_states):
    if index in batch_states:
        return batch_states[index]
    pattern = os.path.join(SLURM_DIR, BATCH_GLOB.format(i=index))
    old = sorted(glob.glob(pattern))
    if not old:
        return "UNKNOWN"
    base = os.path.basename(old[-1]).replace(".slurm", "")
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "")], universal_newlines=True
        )
        if base in out:
            return "RUNNING"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return "UNKNOWN"


def extract_train_cmd(slurm_path):
    text = open(slurm_path).read()
    m = re.search(r"(python -m src\.scripts\.train[^\n]+)", text)
    if not m:
        raise ValueError(f"No train command in {slurm_path}")
    cmd = m.group(1).strip()
    for old in _HS_PRETRAINED_OLD_PREFIXES:
        cmd = cmd.replace(old, HS_PRETRAINED)
    return cmd


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    env_vars = get_env_vars()
    ts = dt.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(SLURM_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("# index | init | task | seed | SLURM state | action")
    to_submit = []
    batch_states = _batch_slurm_states()
    active_jobs = _active_batch_jobs()

    for i in range(16):
        pattern = os.path.join(SLURM_DIR, BATCH_GLOB.format(i=i))
        old_slurms = sorted(glob.glob(pattern))
        if not old_slurms:
            print(f"# {i:2d}  (no prior slurm — skipped)")
            continue
        cmd = extract_train_cmd(old_slurms[-1])
        state = slurm_state(i, batch_states)
        init = "rw" if "_rw_" in cmd else "pretrained"
        task = "regression" if "--mode regression" in cmd else "classifier"
        seed_m = re.search(r"seed(\d+)", cmd)
        seed = seed_m.group(1) if seed_m else "?"

        old_base = os.path.basename(old_slurms[-1]).replace(".slurm", "")
        if old_base in active_jobs:
            print(f"# {i:2d}  {init:10s}  {task:12s}  seed{seed}  {state:10s}  SKIP (still in queue)")
            continue
        if state == "COMPLETED":
            print(f"# {i:2d}  {init:10s}  {task:12s}  seed{seed}  {state:10s}  SKIP (completed)")
            continue

        job_name = f"run_{i}_resubmit_{ts}"
        slurm_file = os.path.join(SLURM_DIR, f"{job_name}.slurm")
        log_file = os.path.join(LOG_DIR, f"{job_name}.log")
        err_file = os.path.join(LOG_DIR, f"{job_name}.error.log")
        with open(slurm_file, "w") as f:
            f.write(
                SLURM_TEMPLATE_GPU.format(
                    queue_name="shared",
                    time=WALLTIME_RESUBMIT,
                    cpus_per_task=32,
                    gpus_per_node=1,
                    job_name=job_name,
                    log_dir=log_file,
                    error_dir=err_file,
                    commands=cmd,
                    env_vars=env_vars,
                    container_image=CONTAINER_IMAGE,
                )
            )
        reason = state if state != "UNKNOWN" else "not finished"
        print(
            f"# {i:2d}  {init:10s}  {task:12s}  seed{seed}  {state:10s}  RESUBMIT ({reason})"
        )
        to_submit.append(slurm_file)

    print()
    for path in to_submit:
        print(f"sbatch {path}")


if __name__ == "__main__":
    main()
