# Train the transformer locally
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU
import os
from datetime import datetime as dt

base_cmd = "python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon -name SmallDataset_E_avail_Log1PLoss_{suffix} --d_model 128 --depth 4 --n_heads 8 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260216_additional_info1_split --num_workers 10  --log1p_loss --eval_interval 1000 -cap {data_cap} -seed-event-sampler {seed} --epochs 1000000"

cmds = []

for data_cap in [10000, 50000]:
    for seed in [42, 43, 44]:
        cmd = base_cmd.format(suffix=f"{data_cap}_Evts_seed_{seed}", data_cap=data_cap, seed=seed)
        cmds.append(cmd)

cmds = ["python -m src.scripts.train -bs 2048 --mode regression -E-available-no-muon -name E_avail_Log1PLoss_NoGlobalFeatures --d_model 128 --depth 4 --n_heads 8 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260216_additional_info1_split --num_workers 10  --log1p_loss --eval_interval 1000  --epochs 1000000 --use_cond False"]

for i, cmd in enumerate(cmds):
    job_name = f"Tr_{i}_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/transformer_small_dataset/{job_name}.log"
    error_dir = f"/global/cfs/cdirs/m3246/gregork/Minerva/logs/transformer_small_dataset/{job_name}.error.log"
    slurm_file = f"/global/cfs/cdirs/m3246/gregork/Minerva/slurm/transformer_small_dataset/{job_name}.slurm"
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

