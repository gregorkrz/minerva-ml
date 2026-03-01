# Train the transformer locally
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt

cmds = ["python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon -name E_avail_HuberEWeighted -wl --d_model 128 --depth 4 --n_heads 8 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260216_additional_info1_split --num_workers 10 --eval_interval 1000",
        "python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon  -log-mse -name E_avail_LogMSE --d_model 128 --depth 4 --n_heads 8 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260216_additional_info1_split --num_workers 10 --eval_interval 1000"]


for i, cmd in enumerate(cmds):
    job_name = f"Tr_{i}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/transformer/{job_name}.log"
    error_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/transformer/{job_name}.error.log"
    slurm_file = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/transformer/{job_name}.slurm"
    os.makedirs(os.path.dirname(slurm_file), exist_ok=True)
    with open(slurm_file, "w") as f:
        f.write(SLURM_TEMPLATE_GPU.format(
            queue_name="shared",
            time="20:00:00",
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
    print("SUBMITTED")

