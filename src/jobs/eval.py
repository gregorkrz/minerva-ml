# Script to generate eval commands for specific datasets.

import argparse
import os
import json
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--training-path", "-p", type=str, required=True)
parser.add_argument("--playlists", "-pl", nargs="+", type=str, required=False, default=None)
parser.add_argument("--event-classifier", "-ec", action="store_true", required=False, default=False)
parser.add_argument("--current-classifier", "-cc", action="store_true", required=False, default=False)
parser.add_argument("--print-only", "-po", action="store_true", required=False, default=False) # If toggled on, print only 

args = parser.parse_args()

settings_path = os.path.join(args.training_path, "settings.json")
with open(settings_path, "r") as f:
    settings = json.load(f)

dataset_path = settings["path"]

# get the dataset name from the settings /global/cfs/cdirs/m3246/gregork/checkpoints/max_blob_and_prong_1A_20260202_102139/
dataset_name = os.path.basename(args.training_path)
# 
checkpoint_name = os.path.join(args.training_path, f"best_model_{os.path.basename(args.training_path)}.pt")
playlists = args.playlists if args.playlists is not None else os.listdir(dataset_path)

mode_tag = " --mode regression "
if args.event_classifier:
    mode_tag = " --mode classifier --class-event-type  --num-classes 5 "
elif args.current_classifier:
    mode_tag = " --mode classifier --class-current-type --num-classes 2 "

succeeded = []
failed = []

for playlist in playlists: # Loop over all playlists
    eval_cmd = f"""omnilearned evaluate \
    --output_dir {os.path.join(args.training_path, "test_results")} \
    --dataset minerva_{playlist} \
  --path {dataset_path} \
  {mode_tag} \
  --max-particles {settings["max_particles"]} \
  --size {settings["model_size"]} \
  --use-pid \
  --batch 1024 --num-workers 16   \
  -i {args.training_path} --save-tag {os.path.basename(args.training_path)}"""
    
    if args.print_only:
        print(eval_cmd)
    else:
        print(f"\nEvaluating playlist: {playlist}")
        try:
            result = subprocess.run(eval_cmd, shell=True, check=True, capture_output=False)
            succeeded.append(playlist)
            print(f"✓ Successfully evaluated {playlist}")
        except subprocess.CalledProcessError as e:
            failed.append(playlist)
            print(f"✗ Failed to evaluate {playlist}")

if not args.print_only:
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"\nSucceeded ({len(succeeded)}/{len(playlists)}):")
    for playlist in succeeded:
        print(f"  ✓ {playlist}")
    if failed:
        print(f"\nFailed ({len(failed)}/{len(playlists)}):")
        for playlist in failed:
            print(f"  ✗ {playlist}")
    else:
        print("\nAll playlists evaluated successfully!")

