# Train the transformer locally
from src.constants.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt


def get_env_vars():
    # Read .env
    env_commands = ""
    with open(".env", "r") as f:
        for line in f:
            if line.startswith("="):
                continue
            key, value = line.strip().split("=")
            env_commands += f"export {key}={value}\n"
    return env_commands


def generate_cmd(data_cap=-1, seed=42, task="regression", model="Transformer1", bs=2048, max_steps=250000, grad_accum_steps=1, continue_from="", resume_run_id="", resume_run_name=""):
    if continue_from:
        base = f"python -m src.scripts.train --resume {continue_from} -name {resume_run_name} --resume-run-id {resume_run_id}"
        return base.format(continue_from=continue_from)
    base = "python -m src.scripts.train -bs {bs} --mode {task} {detailed_task} -name {name} --d_model 128 --depth 4 --n_heads 8 --dropout 0.0 --attn_dropout 0.0 {cap} --seed {seed} -seed-event-sampler {seed}  --max_steps {max_steps} --grad_accum_steps {grad_accum_steps} {extra} --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260313"
    # Model options: "Transformer1", "OLS" (OmniLearned Small), "OLS_RW" (OLS Random Weights)
    # Seed both the event sampler and the whole training with seed
    detailed_task = "-E-available-no-muon" if task == "regression" else "-npi2"
    if model == "OLS":
        name = f"Run_1203_OLS_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small --use-pretrained pretrain_s"
    elif model == "OLS_RW":
        name = f"Run_1203_OLS_RW_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small"
    elif model == "OLM":
        name = f"Run_1203_OLM_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned medium --use-pretrained pretrain_m"
    elif model == "Transformer1":  # Transformer1
        name = f"Run_1203_{task}_{model}_data_cap_{data_cap}_seed_{seed}"
        extra = ""
    else:
        raise ValueError(f"Invalid model: {model}")
    cap = f" -cap {data_cap} " if data_cap > 0 else ""
    return base.format(bs=bs, task=task, detailed_task=detailed_task, name=name, cap=cap, seed=seed, max_steps=max_steps, extra=extra, grad_accum_steps=grad_accum_steps)


def get_cmds_and_slurm_times():
    times_data_cap = { # for 20k: for OLS_RW 100 minutes, for the rest OLS 50 minutes
        "20000": {
            "OLS_RW": "01:30:00",
            "OLS": "00:50:00",
            "Transformer1": "00:20:00",
        },
        "50000": {
            "OLS_RW": "03:00:00",
            "OLS": "01:00:00",
            "Transformer1": "00:20:00",
        },
        "100000":{
            "OLS_RW": "04:00:00",
            "Transformer1": "00:20:00",
            "OLS": "01:00:00"
        },
        "200000": {
            "OLS_RW": "05:00:00",
            "OLS": "01:30:00",
            "Transformer1": "00:20:00",
        }
    }
    cmds = []
    slurm_times = []
    for seed in [46, 47, 48, 49, 50]:
        for data_cap in [20000, 50000, 100000, 200000]:
            for task in ["regression", "classifier"]:
                for model in ["OLS", "OLS_RW", "Transformer1"]: #["Transformer1", "OLS", "OLS_RW", "OLM"]:
                    if "OL" in model:
                        bs = 2048
                        grad_accum_steps = 1
                        if "OLM" in model:
                            bs = 512
                            grad_accum_steps = 4
                            if data_cap == -1:
                                slurm_times.append("15:00:00")
                            else:
                                slurm_times.append(times_data_cap[str(data_cap)][model])
                        else:
                            if data_cap == -1:
                                slurm_times.append("12:00:00")
                            else:
                                slurm_times.append(times_data_cap[str(data_cap)][model])
                    else:
                        bs = 2048
                        grad_accum_steps = 1
                        if task == "regression":
                            if data_cap == -1:
                                slurm_times.append("05:00:00")
                            else:
                                slurm_times.append(times_data_cap[str(data_cap)][model])
                        else:
                            if data_cap == -1:
                                slurm_times.append("05:00:00")
                            else:
                                slurm_times.append(times_data_cap[str(data_cap)][model])
                    max_steps = {
                        "regression": 500000,
                        "classifier": 500000,
                    }
                    cmd = generate_cmd(data_cap=data_cap, seed=seed, task=task, model=model, max_steps=max_steps[task], bs=bs, grad_accum_steps=grad_accum_steps)
                    cmds.append(cmd)
    return cmds, slurm_times

def get_cmds_and_slurm_times_continue():
    to_resume = ["Run_1203_OLS_RW_regression_-1_seed42_20260312_211002", "Run_1203_OLS_RW_classifier_-1_seed42_20260313_011017"]
    names = ["Run_1203_OLS_RW_regression_20000_seed42", "Run_1203_OLS_RW_classifier_20000_seed42"]
    run_ids = ["uw7qe1ix", "7ovfe9wn"]
    CKPT_DIR = "/global/cfs/cdirs/m3246/gregork/checkpoints"
    cmds = []
    slurm_times = []
    for i, ckpt in enumerate(to_resume):
        ckpt_path = os.path.join(CKPT_DIR, ckpt)
        cmd = generate_cmd(continue_from=ckpt_path, resume_run_name=names[i], resume_run_id=run_ids[i])
        cmds.append(cmd)
        slurm_times.append("12:00:00")
    return cmds, slurm_times


if __name__ == "__main__":
    cmds, slurm_times = get_cmds_and_slurm_times()
    for i, cmd in enumerate(cmds):
        job_name = f"run_{i}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/run_100326/{job_name}.log"
        error_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/run_100326/{job_name}.error.log"
        slurm_file = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/run_100326/{job_name}.slurm"
        os.makedirs(os.path.dirname(slurm_file), exist_ok=True)
        os.makedirs(os.path.dirname(log_dir), exist_ok=True)
        with open(slurm_file, "w") as f:
            f.write(SLURM_TEMPLATE_GPU.format(
                queue_name="shared",
                time=slurm_times[i],
                cpus_per_task=32,
                gpus_per_node=1,
                job_name=job_name,
                log_dir=log_dir,
                error_dir=error_dir,
                commands=cmd,
                env_vars=get_env_vars(),
                ))
        print(f"Saved slurm file to {slurm_file}")
        os.system(f"sbatch {slurm_file}")
        print("Job submitted")
