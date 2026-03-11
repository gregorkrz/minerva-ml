# Train the transformer locally
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt


def generate_cmd(data_cap=-1, seed=42, task="regression", model="Transformer1", bs=2048, max_steps=250000):
    base = "python -m src.scripts.train -bs {bs} --mode {task} {detailed_task} -name {name} --d_model 64 --depth 4 --n_heads 4 {cap} --seed {seed} -seed-event-sampler {seed}  --max_steps {max_steps} {extra}"
    # Model options: "Transformer1", "OLS" (OmniLearned Small), "OLS_RW" (OLS Random Weights)
    # Seed both the event sampler and the whole training with seed
    detailed_task = "-E-available-no-muon" if task == "regression" else "-npi2"
    if model == "OLS":
        name = f"OLS_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small --use-pretrained pretrain_s"
    elif model == "OLS_RW":
        name = f"OLS_RW_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned small"
    elif model == "OLM":
        name = f"OLM_{task}_{data_cap}_seed{seed}"
        extra = " --use-omnilearned medium --use-pretrained pretrain_m"
    elif model == "Transformer1":  # Transformer1
        name = f"Run_1003_{task}_{model}_data_cap_{data_cap}_seed_{seed}"
        extra = ""
    else:
        raise ValueError(f"Invalid model: {model}")
    cap = f" -cap {data_cap} " if data_cap > 0 else ""
    return base.format(bs=bs, task=task, detailed_task=detailed_task, name=name, cap=cap, seed=seed, max_steps=max_steps, extra=extra)

cmds = []
slurm_times = []

for seed in [42]:
    for data_cap in [-1]:
        for task in ["regression", "classifier"]:
            for model in ["Transformer1", "OLS", "OLS_RW", "OLM"]:
                if "OL" in model:
                    slurm_times.append("10:00:00")
                else:
                    if task == "regression":
                        slurm_times.append("05:00:00")
                    else:
                        slurm_times.append("03:00:00")
                max_steps = {
                    "regression": 250000,
                    "classifier": 100000,
                }
                cmd = generate_cmd(data_cap=data_cap, seed=seed, task=task, model=model, max_steps=max_steps[task])
                cmds.append(cmd)

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
            commands="srun " + cmd,
            env_commands="",
            ))
    print(f"Saved slurm file to {slurm_file}")
    os.system(f"sbatch {slurm_file}")
    print("Job submitted")

