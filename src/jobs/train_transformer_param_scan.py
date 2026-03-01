from src.jobs.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt

base_cmd = "python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon -name ParamScan_d{d_model}_L{depth}_H{n_heads} --d_model {d_model} --depth {depth} --n_heads {n_heads} --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260216_additional_info1_split --num_workers 10 --log1p_loss --eval_interval 1000 --epochs 1000000"

configs = [
    {"d_model": 64,  "depth": 4, "n_heads": 4},
    {"d_model": 128, "depth": 4, "n_heads": 8},
    {"d_model": 256, "depth": 4, "n_heads": 8},
    {"d_model": 128, "depth": 2, "n_heads": 8},
    {"d_model": 128, "depth": 6, "n_heads": 8},
    {"d_model": 128, "depth": 8, "n_heads": 8},
    {"d_model": 128, "depth": 4, "n_heads": 4},
    {"d_model": 128, "depth": 4, "n_heads": 16},
    {"d_model": 256, "depth": 6, "n_heads": 8},
]

cmds = []
for cfg in configs:
    cmd = base_cmd.format(**cfg)
    cmds.append(cmd)

for i, cmd in enumerate(cmds):
    cfg = configs[i]
    job_name = f"PS_d{cfg['d_model']}_L{cfg['depth']}_H{cfg['n_heads']}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/transformer_param_scan/{job_name}.log"
    error_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/transformer_param_scan/{job_name}.error.log"
    slurm_file = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/transformer_param_scan/{job_name}.slurm"
    os.makedirs(os.path.dirname(slurm_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    with open(slurm_file, "w") as f:
        f.write(SLURM_TEMPLATE_GPU.format(
            queue_name="shared",
            time="03:00:00",
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
