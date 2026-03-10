
import argparse
import os
import shutil
import datetime
from src.jobs.slurm_template import SLURM_TEMPLATE_GPU

"""
--------------------------------
Current tasks:

E available regression:
python -m src.jobs.gen_train_cmds_OmniLearned -name E_avail_LogMSE --regress-E-available-no-muon -nw 10  --loss-type mse --log --run
python -m src.jobs.gen_train_cmds_OmniLearned -name E_avail_LogMSE_PT --regress-E-available-no-muon -nw 10  --loss-type mse --log --use-pretrained pretrain_s --run

python -m src.jobs.gen_train_cmds_OmniLearned -name E_avail_HuberWeighted_PT_NoLog --regress-E-available-no-muon -nw 10 --use-pretrained pretrain_s --loss-type huber --weighted-loss -p
python -m src.jobs.gen_train_cmds_OmniLearned -name E_avail_HuberWeighted_NoLog --regress-E-available-no-muon -nw 10  --loss-type huber --weighted-loss -p

python -m src.jobs.gen_train_cmds_OmniLearned -name E_avail_Log1PLoss --regress-E-available-no-muon -nw 10  --loss-type log1p  -p
python -m src.jobs.gen_train_cmds_OmniLearned -name E_avail_Log1PLoss_PT_M --regress-E-available-no-muon -nw 10  --loss-type log1p --use-pretrained pretrain_m --run

--------------------------------
Ch. pion classification:
python -m src.jobs.gen_train_cmds_OmniLearned -name Train_CC1orNPi_v3_PT_S -nw 10 --class-pions -p --resume-from /global/cfs/cdirs/m3246/gregork/checkpoints/Train_CC1orNPi_v3_PT_S_1A_20260303_082607/best_model_Train_CC1orNPi_v3_PT_S_1A_20260303_082607.pt  --dataset Minerva_100Blobs_v1
python -m src.jobs.gen_train_cmds_OmniLearned -name Train_CC1orNPi_v3 -nw 10 --class-pions -p --dataset Minerva_100Blobs_v1 --resume-from /global/cfs/cdirs/m3246/gregork/checkpoints/Train_CC1orNPi_v3_1A_20260303_082351/best_model_Train_CC1orNPi_v3_1A_20260303_082351.pt


--------------------------------
python -m src.jobs.gen_train_cmds_OmniLearned -name Train_CC1orNPi_v3_PT_S -nw 10 --class-pions  --dataset Minerva_100Blobs_v1 --use-pretrained pretrain_s --run
python -m src.jobs.gen_train_cmds_OmniLearned -name Train_CC1orNPi_v3_PT_M -nw 10 --class-pions  --dataset Minerva_100Blobs_v1 --use-pretrained pretrain_m --run
python -m src.jobs.gen_train_cmds_OmniLearned -name Train_CC1orNPi_v3 -nw 10 --class-pions --dataset Minerva_100Blobs_v1  --run
--------------------------------

python -m src.scripts.train -bs 2048 --mode classifier -npi2 -name Pi_Class_v3 --d_model 128 --depth 4 --n_heads 8 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260227_100Blobs_v1_split --num_workers 10 --eval_interval 1000 

"""


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
parser.add_argument("--print-cmd-only", "-print-only", action="store_true", default=False)
parser.add_argument("--batch-size", "-bs", type=int, default=2048)
parser.add_argument("--num-workers", "-nw", type=int, default=2)
parser.add_argument("--loss-type", "-loss_type", type=str, default="l1")
parser.add_argument("--log", "-log", action="store_true", default=False)
# add --class-event-type  and --class-current-type
parser.add_argument("--class-current-type", "-cc",  action="store_true", default=False)
parser.add_argument("--class-pions", "-cp",  action="store_true", default=False)
parser.add_argument("--regress-E-available", "-E-available",  action="store_true", default=False)
parser.add_argument("--regress-E-available-no-muon", "-E-available-no-muon",  action="store_true", default=False)
parser.add_argument("--weighted-loss", "-wl", action="store_true", default=False)
parser.add_argument("--dataset-cap", "-cap", type=int, default=-1, help="If set, cap the dataset to this number of training samples.")
parser.add_argument("--dataset-cap-seed", "-seed", type=int, default=42, help="If set, use this seed for the dataset cap.")

args = parser.parse_args()

DATA_DIR = args.data_dir

DATASETS = {
    "default_Minerva": f"{DATA_DIR}/Minerva/20260216_additional_info1_split",
    "Minerva_v2": f"{DATA_DIR}/Minerva/20260201_all_max_blobs_and_prongs_split_fix_anomalies",
    "Minerva_150obj": f"{DATA_DIR}/Minerva/20260223_150obj_split",
    "Minerva_100Blobs": f"{DATA_DIR}/Minerva/20260227_100Blobs_split",
    "Minerva_100Blobs_v1": f"{DATA_DIR}/Minerva/20260227_100Blobs_v1_split"
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

class_current_type_flag = ""
class_pions_flag = ""
regress_E_available_flag = ""
regress_E_available_no_muon_flag = ""
log_flag = ""
weighted_loss_flag = ""
dataset_cap_flag = ""
if args.log:
    log_flag = " --regress-log "
if args.weighted_loss:
    weighted_loss_flag = " --weight-loss "
if args.dataset_cap is not None and args.dataset_cap > 0:
    dataset_cap_flag = f" --nevts {args.dataset_cap} --event-sampler-random-state {args.dataset_cap_seed} "
if args.class_current_type:
    class_current_type_flag = " --class-current-type "
    mode = "classifier"
elif args.class_pions:
    class_pions_flag = " --class-pions "
    mode = "classifier"
elif args.regress_E_available:
    regress_E_available_flag = " --regress-e-available "
    mode = "regression"
elif args.regress_E_available_no_muon:
    regress_E_available_no_muon_flag = " --regress-e-available-no-muon "
    mode = "regression"
else:
    raise ValueError(f"Invalid task")

train_cmd = f""" -m omnilearned.cli train \
  --output_dir $CHECKPOINT_DIR  {log_flag} \
  --save-tag {job_name} \
  --dataset minerva_{args.playlist} \
  --path $DATASET_PATH \
  --mode {mode} \
  --batch {args.batch_size} \
  --epoch 100000 \
  --size small \
   {dataset_cap_flag} \
  --num-workers {args.num_workers} \
  --use-pid \
  --wandb --regression-loss {args.loss_type} \
  --max-particles {args.max_particles} {weighted_loss_flag} {class_current_type_flag} {class_pions_flag} {regress_E_available_flag} {regress_E_available_no_muon_flag}"""


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
    time="08:00:00",
    cpus_per_task=32,
    gpus_per_node=1,
    job_name=job_name,
    log_dir=log_output,
    error_dir=log_error,
    env_commands=env_commands,
    commands="srun python "  + train_cmd,
    queue_name="shared"
)

if args.print_cmd_only:
    print(env_commands)
    #print("torchrun --nproc_per_node=4 " + train_cmd)
    print("python " + train_cmd)
    exit()

with open(sbatch_file, "w") as f:
    f.write(slurm_file_content)
    print("Written to sbatch file: ", sbatch_file)

if args.run:
    os.system(f"sbatch {sbatch_file}")
