import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import h5py
import os
import torch
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
    allowed_event_types = np.array([1, 2, 3, 4, 7, 8])
    passing_idx = np.where(np.isin(truth_labels.astype(np.int32), allowed_event_types, kind="table"))[0]
    return passing_idx

result = {} # Dictionary of {dataset: {train_idx, val_idx, test_idx}} - used to debug. In the end, store it in output_dir too.

for dataset in os.listdir(args.input_dir):
    # This assumes the data for each dataset can fit in memory, which is reasonable for now. We could optimize this later if needed.
    # Dataset: 1A, 1B...
    truth_labels = []
    global_features = []
    data = []
    filenames = sorted(os.listdir(os.path.join(args.input_dir, dataset)))
    for file in filenames:
        if file.endswith(".pb"): # There's 'data', 'truth_labels', 'global_features' keys in the file; add it to the list
            with torch.load(os.path.join(args.input_dir, file)) as f:
                print(f"Processing file: {file}")
                data.append(f["data"])
                truth_labels.append(f["truth_labels"])
                global_features.append(f["global_features"])
    truth_labels = torch.concat(truth_labels) # concat the different chunks
    global_features = torch.concat(global_features) # concat the different chunks
    data = torch.concat(data) # concat nested tensors?
    truth_event_types = truth_labels[:, 1]
    passing_idx = filter_weird_events(truth_event_types)
    # Now split the remaining idx using train_test_split
    train_idx, temp_idx = train_test_split(
        passing_idx, 
        test_size=args.test_ratio + args.val_ratio, 
        random_state=args.seed, 
        stratify=truth_event_types[passing_idx]
    )
    temp_idx = sorted(temp_idx)
    
    # Split temp into val and test
    temp_event_types = truth_labels[temp_idx]
    val_ratio_adjusted = args.val_ratio / (args.test_ratio + args.val_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx, 
        test_size=(1 - val_ratio_adjusted), 
        random_state=args.seed, 
        stratify=temp_event_types
    )

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")
    result[file] = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx
    }
    
    # Save the files split in this way to <DATASET>/<train/val/test>/0.pb
    Path(os.path.join(args.output_dir, "train")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "val")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "test")).mkdir(parents=True, exist_ok=True)
    print("Splitting data...")
    train = {
        "data": data[train_idx],
        "truth_labels": truth_labels[train_idx],
        "global_features": global_features[train_idx]
    }
    val = {
        "data": data[val_idx],
        "truth_labels": truth_labels[val_idx],
        "global_features": global_features[val_idx]
    }
    test = {
        "data": data[test_idx],
        "truth_labels": truth_labels[test_idx],
        "global_features": global_features[test_idx]
    }
    
    torch.save(train, os.path.join(args.output_dir, "train", "0.pb"))
    torch.save(val, os.path.join(args.output_dir, "val", "0.pb"))
    torch.save(test, os.path.join(args.output_dir, "test", "0.pb"))
    
    print(f"✓ {file} split and written to {args.output_dir}")

# Save the result to the output directory
with open(os.path.join(args.output_dir, "result.pkl"), "wb") as f:
    pickle.dump(result, f)
