
import argparse
import os
import shutil
import datetime
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="default_Minerva", help="Dataset to use")
parser.add_argument("--data-dir", type=str, default="/global/cfs/cdirs/m3246/gregork")
parser.add_argument("--output-dir", type=str, default="/global/cfs/cdirs/m3246/gregork/checkpoints")
parser.add_argument("--logs-dir", type=str, default="/pscratch/sd/g/gregork/logs")
parser.add_argument("--training-name", "-name", type=str, required=True) #Training name; will be suffixed with the current timestamp
parser.add_argument("--playlist", type=str, default="1A", help="Playlist to use")
parser.add_argument("--loss", type=str, default="l1")
parser.add_argument("--use-pretrained", type=str, default=None) # If turned on, use a pretrained small model for fine-tuning.
parser.add_argument("--resume-from", type=str, default=None) # If turned on, resume training from this checkpoint.
parser.add_argument("--run", action="store_true", default=False) # If turned on, run the job immediately.
parser.add_argument("--max-particles", type=int, default=33)
parser.add_argument("--print-cmd-only", action="store_true", default=False)
parser.add_argument("--regress-log", action="store_true", default=False)
parser.add_argument("--batch-size", "-bs", type=int, default=128)
parser.add_argument("--num-workers", "-nw", type=int, default=32)
# add --class-event-type  and --class-current-type
parser.add_argument("--class-event-type", action="store_true", default=False)
parser.add_argument("--class-current-type", action="store_true", default=False)
args = parser.parse_args()

DATA_DIR = args.data_dir

DATASETS = {
    "default_Minerva": f"{DATA_DIR}/Minerva/20260129_split_all",
    "Minerva_v2": f"{DATA_DIR}/Minerva/20260201_all_max_blobs_and_prongs_split_fix_anomalies"
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
export DATASET_PATH={DATASETS[args.dataset]}
export CHECKPOINT_DIR={output_dir}
"""

regress_log_flag = ""
regress_loss_flag = ""
class_event_type_flag = ""
class_current_type_flag = ""

if args.regress_log:
    regress_log_flag = " --regress-log "
    regress_loss_flag = f" --regression-loss {args.loss} "
if args.class_event_type and args.class_current_type:
    assert False, "Cannot use both class-event-type and class-current-type"
elif args.class_event_type:
    class_event_type_flag = " --class-event-type --num-classes 5 "
    mode = "classifier"
elif args.class_current_type:
    class_current_type_flag = " --class-current-type --num-classes 2 "
    mode = "classifier"
else:
    mode = "regression"

train_cmd = f"""  -m omnilearned.cli train \
  --output_dir $CHECKPOINT_DIR \
  --save-tag {job_name} \
  --dataset minerva_{args.playlist} \
  --path $DATASET_PATH \
  --mode {mode} \
  {regress_loss_flag} \
  --batch {args.batch_size} \
  --epoch 100 \
  --lr 5e-5 \
  --size small \
  --wd 0.1 \
  --num-workers {args.num_workers} \
  --use-pid \
  --wandb \
  --max-particles {args.max_particles} {regress_log_flag} {class_event_type_flag} {class_current_type_flag}"""

# Handle pretrained checkpoint for fine-tuning
if args.use_pretrained is not None and args.resume_from is not None:
    raise ValueError("Cannot use both --use-pretrained and --resume-from. Choose one.")

if args.use_pretrained is not None:
    # Check if it's a file path
    if os.path.isfile(args.use_pretrained):
        print(f"Detected pretrained checkpoint as file path: {args.use_pretrained}")
        
        # Copy the checkpoint to the output directory
        checkpoint_basename = os.path.basename(args.use_pretrained)
        dest_path = os.path.join(output_dir, checkpoint_basename)
        print(f"Copying checkpoint: {args.use_pretrained} -> {dest_path}")
        shutil.copy2(args.use_pretrained, dest_path)
        
        # Extract the tag from the filename (remove "best_model_" prefix and ".pt" suffix)
        if checkpoint_basename.startswith("best_model_") and checkpoint_basename.endswith(".pt"):
            pretrain_tag = checkpoint_basename[11:-3]  # Remove "best_model_" and ".pt"
        else:
            # If it doesn't follow the expected naming convention, use the basename without extension
            pretrain_tag = os.path.splitext(checkpoint_basename)[0]
        
        print(f"Using pretrain_tag: {pretrain_tag}")
        train_cmd += f" --pretrain-tag {pretrain_tag} --fine-tune"
    else:
        # It's just a tag name, use it directly
        train_cmd += f" --pretrain-tag {args.use_pretrained} --fine-tune"

# Handle resuming from checkpoint
elif args.resume_from is not None:
    if os.path.isfile(args.resume_from):
        print(f"Resuming training from checkpoint: {args.resume_from}")
        
        # Copy the checkpoint to the output directory with the current job_name
        checkpoint_basename = f"best_model_{job_name}.pt"
        dest_path = os.path.join(output_dir, checkpoint_basename)
        print(f"Copying checkpoint: {args.resume_from} -> {dest_path}")
        shutil.copy2(args.resume_from, dest_path)
        
        train_cmd += " --resuming"
    else:
        raise ValueError(f"Resume checkpoint not found: {args.resume_from}")

slurm_file_content = SLURM_TEMPLATE_GPU.format(
    time="16:00:00",
    cpus_per_task=15,
    gpus_per_node=1,
    job_name=job_name,
    log_dir=log_output,
    error_dir=log_error,
    env_commands=env_commands,
    commands="srun "  + train_cmd,
    queue_name="regular"
)
if args.print_cmd_only:
    print(env_commands)
    print("torchrun --nproc_per_node=4 " +train_cmd)
    exit()
with open(sbatch_file, "w") as f:
    f.write(slurm_file_content)
    print("Written to sbatch file: ", sbatch_file)

if args.run:
    os.system(f"sbatch {sbatch_file}")

