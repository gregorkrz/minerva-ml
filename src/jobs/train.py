
import argparse
import os
import datetime
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="default_Minerva", help="Dataset to use")
parser.add_argument("--data-dir", type=str, default="/global/cfs/cdirs/m3246/gregork/data")
parser.add_argument("--output-dir", type=str, default="/global/cfs/cdirs/m3246/gregork/checkpoints")
parser.add_argument("--logs-dir", type=str, default="/pscratch/sd/g/gregork/logs")
parser.add_argument("--training-name", "-name", type=str, required=True) #Training name; will be suffixed with the current timestamp
parser.add_argument("--playlist", type=str, default="1A", help="Playlist to use")
parser.add_argument("--loss", type=str, default="l1")
parser.add_argument("--use-pretrained", type=str, default=None) # If turned on, use a pretrained small model.
parser.add_argument("--run", action="store_true", default=False) # If turned on, run the job immediately.


args = parser.parse_args()

DATA_DIR = args.data_dir

DATASETS = {
    "default_Minerva": f"{DATA_DIR}/Minerva/20260129_split_all"
}

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
job_name = f"{args.training_name}_{args.playlist}_{timestamp}"
output_dir = os.path.join(args.output_dir, job_name)
assert not os.path.exists(output_dir), f"Output directory {output_dir} already exists"
os.makedirs(output_dir)
log_output = os.path.join(args.logs_dir, job_name + ".out")
os.makedirs(os.path.dirname(log_output), exist_ok=True)
log_error = os.path.join(args.logs_dir, job_name + ".err")
sbatch_file = os.path.join(args.logs_dir, job_name + ".sh")
env_commands = f"""export DATA_DIR={args.data_dir}
export DATASET_PATH=$DATA_DIR/{DATASETS[args.dataset]}
export CHECKPOINT_DIR={output_dir}
"""

train_cmd = f"""srun omnilearned train \
  --output_dir $CHECKPOINT_DIR \
  --save-tag {args.training_name} \
  --dataset minerva_{args.playlist} \
  --path $DATASET_PATH \
  --mode regression \
  --regression-loss {args.loss} \
  --batch 128 \
  --epoch 100 \
  --lr 5e-5 \
  --size small \
  --wd 0.1 \
  --num-workers 32 \
  --use-pid\
  --wandb """


if args.use_pretrained is not None:
    train_cmd += f"--pretrain-tag {args.use_pretrained} --fine-tune"


slurm_file_content = SLURM_TEMPLATE_GPU.format(
    time="00:20:00",
    cpus_per_task=15,
    gpus_per_node=1,
    job_name=job_name,
    log_dir=log_output,
    error_dir=log_error,
    env_commands=env_commands,
    commands=train_cmd,
    queue_name="debug"
)
with open(sbatch_file, "w") as f:
    f.write(slurm_file_content)
    print("Written to sbatch file: ", sbatch_file)

if args.run:
    os.system(f"sbatch {sbatch_file}")
