import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import h5py
import os
import pickle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--val-ratio", type=float, required=False, default=0.1)
parser.add_argument("--test-ratio", type=float, required=False, default=0.1)
parser.add_argument("--seed", type=int, required=False, default=42)

args = parser.parse_args()


def filter_weird_events(truth_labels):
    # Only keep events where the interaction type is [1, 2, 3, 4, 7, 8]
    passing_idx = np.where(np.isin(truth_labels[:, 1], [1, 2, 3, 4, 7, 8]))[0]
    return passing_idx
    
result = {} # Dictionary of {file: {train_idx, val_idx, test_idx}} - used to debug. In the end, store it in output_dir too.

for file in os.listdir(args.input_dir):
    if file.endswith(".h5"):
        dataset_name = file.split(".")[0]
        with h5py.File(os.path.join(args.input_dir, file), "r") as f:
            print(f"Processing file: {file}")
            truth_labels = f["truth_labels"]
            passing_idx = filter_weird_events(truth_labels)
            # Now split the remaining idx using train_test_split
            truth_event_type = truth_labels[passing_idx, 1]
          
            # Split into train and temp (val+test)
            train_idx, temp_idx = train_test_split(
                passing_idx, 
                test_size=args.test_ratio + args.val_ratio, 
                random_state=args.seed, 
                stratify=truth_event_type
            )
            temp_idx = sorted(temp_idx)
            
            # Split temp into val and test
            temp_event_types = truth_labels[temp_idx, 1]
            val_ratio_adjusted = args.val_ratio / (args.test_ratio + args.val_ratio)
            val_idx, test_idx = train_test_split(
                temp_idx, 
                test_size=(1 - val_ratio_adjusted), 
                random_state=args.seed, 
                stratify=temp_event_types
            )

            print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")
            result[file] = [train_idx, val_idx, test_idx]
            
            # Load all data from input file
            input_data = f["data"][:]
            input_global = f["global"][:]
            input_truth_labels = f["truth_labels"][:]
            input_num_particles = f["number_of_particles"][:]
            
            # Save the h5 files split in this way to <DATASET>/<train/val/test>.h5
            Path(os.path.join(args.output_dir, dataset_name)).mkdir(parents=True, exist_ok=True)
            
            with h5py.File(os.path.join(args.output_dir, dataset_name, "train.h5"), "w") as f_out:
                f_out["data"] = input_data[train_idx]
                f_out["global"] = input_global[train_idx]
                f_out["truth_labels"] = input_truth_labels[train_idx]
                f_out["number_of_particles"] = input_num_particles[train_idx]
                
            with h5py.File(os.path.join(args.output_dir, dataset_name, "val.h5"), "w") as f_out:
                f_out["data"] = input_data[val_idx]
                f_out["global"] = input_global[val_idx]
                f_out["truth_labels"] = input_truth_labels[val_idx]
                f_out["number_of_particles"] = input_num_particles[val_idx]
                
            with h5py.File(os.path.join(args.output_dir, dataset_name, "test.h5"), "w") as f_out:
                f_out["data"] = input_data[test_idx]
                f_out["global"] = input_global[test_idx]
                f_out["truth_labels"] = input_truth_labels[test_idx]
                f_out["number_of_particles"] = input_num_particles[test_idx]
                
            print(f"✓ {file} split and written to {args.output_dir}")

# Save the result to the output directory
with open(os.path.join(args.output_dir, "result.pkl"), "wb") as f:
    pickle.dump(result, f)
