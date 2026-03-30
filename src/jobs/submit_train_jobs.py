# Train the transformer locally
from src.constants.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt


def get_env_vars():
    """Build `export` lines from `.env` for SLURM, excluding WANDB_* (no API key in job files).

    Training loads repo-root `.env` at runtime via `src.scripts.train`.
    """
    env_commands = ""
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.startswith("WANDB_"):
                continue
            value = value.strip()
            env_commands += f"export {key}={value}\n"
    return env_commands


def generate_cmd(data_cap=-1, seed=42, task="regression", model="Transformer1", bs=2048, max_steps=250000, grad_accum_steps=1, continue_from="", resume_run_id="", resume_run_name="", model_n_layers=4):
    if continue_from:
        base = f"python -m src.scripts.train --resume {continue_from} -name {resume_run_name} --resume-run-id {resume_run_id} --max_steps 1000000"
        return base.format(continue_from=continue_from)
    base = "python -m src.scripts.train -bs {bs} --mode {task} {detailed_task} -name {name} --d_model {model_dim} --depth {model_depth} --n_heads {model_n_heads} --dropout {model_dropout} --attn_dropout {model_attn_dropout} {cap} --seed {seed} -seed-event-sampler {seed}  --max_steps {max_steps} --grad_accum_steps {grad_accum_steps} {extra} --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260326 "
    # Model options: "Transformer1", "Transformer1NR" (no E_recoil: --zero-cond-feature 2), "Transformer2", "Transformer3", "OLS", "OLS_RW", "OLM", "BERT-tiny"
    # Seed both the event sampler and the whole training with seed
    detailed_task = "-E-available-no-muon" if task == "regression" else "-npi2"
    model_dim = 128
    model_depth = 4
    model_n_heads = 8
    model_dropout = 0.0
    model_attn_dropout = 0.0
    if model == "OLS":
        name = f"Run_1703_OLS_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small --use-pretrained pretrain_s --zero-cond-feature 2 "
    elif model == "OLS_RW":
        name = f"Run_1703_OLS_RW_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small --zero-cond-feature 2 "
    elif model == "OLM":
        name = f"Run_1703_OLM_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned medium --use-pretrained pretrain_m --zero-cond-feature 2 "
    elif model == "BERT-tiny":
        name = f"Run_1703_BERT_tiny_{task}_{data_cap}_seed{seed}"
        extra = " --use-bert tiny --zero-cond-feature 2 "
    elif model in ("Transformer1", "Transformer1NR", "Transformer2", "Transformer3", "Transformer3NR"):
        name = f"Run_1703_{task}_{model}_data_cap_{data_cap}_seed_{seed}"
        #extra = " --zero-cond-feature 2 "
        extra = ""
        if model == "Transformer1NR":
            extra = " --zero-cond-feature 2 "
        elif model == "Transformer2":
            model_n_heads = 12
            model_dim = 384
            model_depth = 8
        elif model == "Transformer3":
            model_dim = 184
            model_depth = 7
            model_n_heads = 8
        elif model == "Transformer3NR":
            model_dim = 184
            model_depth = 7
            model_n_heads = 8
            extra = " --zero-cond-feature 2 "
    else:
        raise ValueError(f"Invalid model: {model}")
    cap = f" -cap {data_cap} " if data_cap > 0 else ""
    return base.format(bs=bs, task=task, detailed_task=detailed_task, name=name, cap=cap, seed=seed, max_steps=max_steps, extra=extra, grad_accum_steps=grad_accum_steps,
    model_dim=model_dim, model_depth=model_depth, model_n_heads=model_n_heads, model_dropout=model_dropout, model_attn_dropout=model_attn_dropout, model_n_layers=model_n_layers)


def get_cmds_and_slurm_times():
    times_data_cap = { # for 20k: for OLS_RW 100 minutes, for the rest OLS 50 minutes
        "20000": {
            "OLS_RW": "01:30:00",
            "OLS": "00:50:00",
            "BERT-tiny": "00:50:00",
            "Transformer1": "00:20:00",
            "Transformer1NR": "00:20:00"
        },
        "50000": {
            "OLS_RW": "03:00:00",
            "OLS": "01:00:00",
            "BERT-tiny": "01:00:00",
            "Transformer1": "00:20:00",
            "Transformer1NR": "00:20:00",
        },
        "100000":{
            "OLS_RW": "04:00:00",
            "Transformer1": "00:20:00",
            "Transformer1NR": "00:20:00",
            "OLS": "01:00:00",
            "BERT-tiny": "01:00:00",
        },
        "200000": {
            "OLS_RW": "05:00:00",
            "OLS": "01:30:00",
            "BERT-tiny": "01:30:00",
            "Transformer1": "00:20:00",
            "Transformer1NR": "00:20:00",
        }
    }
    cmds = []
    slurm_times = []
    for seed in [50, 51, 52, 53]:
        for data_cap in [-1]:
            for task in ["regression", "classifier"]:
                for model in ["OLS", "OLS_RW", "OLM", "BERT-tiny","Transformer1NR", "Transformer3NR"]:
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
                    elif model == "BERT-tiny":
                        bs = 2048
                        grad_accum_steps = 1
                        if data_cap == -1:
                            slurm_times.append("12:00:00")
                        else:
                            slurm_times.append(times_data_cap[str(data_cap)][model])
                    else:
                        bs = 2048
                        grad_accum_steps = 1
                        if task == "regression":
                            if data_cap == -1:
                                slurm_times.append("08:00:00")
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
    # Use this to continue runs that were cut short. TODO: change run names and wandb IDs
    to_resume = [
        "Run_1703_OLM_regression_-1_seed50_20260328_115708",
        "Run_1703_OLS_RW_regression_-1_seed50_20260327_174607"
    ]
    names = [
        "Run_1703_OLM_regression_-1_seed50",
        "Run_1703_OLS_RW_regression_-1_seed50"
    ]
    run_ids = [
        "ofpk105k",
        "rd9azt5b"
    ]
    CKPT_DIR = "/global/cfs/cdirs/m3246/gregork/checkpoints"
    cmds = []
    slurm_times = []
    for i, ckpt in enumerate(to_resume):
        ckpt_path = os.path.join(CKPT_DIR, ckpt, "best_model.pt")
        cmd = generate_cmd(continue_from=ckpt_path, resume_run_name=names[i], resume_run_id=run_ids[i])
        cmds.append(cmd)
        slurm_times.append("12:00:00")
    return cmds, slurm_times


if __name__ == "__main__":
    cmds, slurm_times = get_cmds_and_slurm_times_continue()
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

