# Script to generate eval commands for specific datasets.

import argparse
import os
import json

parser = argparse.ArgumentParser()
parser.add_argument("--training-path", "-p", type=str, required=True)
parser.add_argument("--playlist", "-pl", type=str, default="1A")

args = parser.parse_args()

settings_path = os.path.join(args.training_path, "settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)

dataset_path = settings["path"]

# get the dataset name from the settings /global/cfs/cdirs/m3246/gregork/checkpoints/max_blob_and_prong_1A_20260202_102139/
dataset_name = os.path.basename(args.training_path)
# 
checkpoint_name = os.path.join(args.training_path, f"best_model_{os.path.basename(args.training_path)}.pt")

eval_cmd = f"""omnilearned evaluate \
  --output_dir {os.path.join(args.training_path, "test_results")} \
  --dataset minerva_{args.playlist} \
  --path {dataset_path} \
  --mode regression \
  --max-particles {settings["max_particles"]} \
  --size {settings["model_size"]} \
  --use-pid \
  --run --checkpoint {checkpoint_name} --batch 1024 --num-workers 16 \
  -i {args.training_path} --save-tag {os.path.basename(args.training_path)}"""

print(eval_cmd)
