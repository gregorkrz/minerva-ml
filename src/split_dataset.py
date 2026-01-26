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
    allowed_event_types = np.array([1, 2, 3, 4, 7, 8])
    passing_idx = np.where(np.isin(truth_labels.astype(np.int32), allowed_event_types, kind="table"))[0]
    return passing_idx


def write_h5_in_chunks(input_file, output_path, indices, chunk_size=10000):
    """
    Write HDF5 file in chunks to avoid loading all data into memory.
    
    Parameters:
    -----------
    input_file : h5py.File
        Open input HDF5 file
    output_path : str
        Path to output HDF5 file
    indices : np.ndarray
        Indices to extract from input file (must be sorted!)
    chunk_size : int
        Number of events to process at a time
    """
    num_events = len(indices)
    
    # Get shapes and dtypes from input file
    datasets_info = {}
    for key in ['data', 'global', 'truth_labels', 'number_of_particles']:
        if key in input_file:
            shape = list(input_file[key].shape)
            shape[0] = num_events  # Update first dimension
            datasets_info[key] = {
                'shape': tuple(shape),
                'dtype': input_file[key].dtype,
                'chunks': True  # Enable chunking for efficient I/O
            }
    
    # Create output file and initialize datasets
    with h5py.File(output_path, 'w') as f_out:
        # Create empty datasets with correct shapes
        datasets = {}
        for key, info in datasets_info.items():
            datasets[key] = f_out.create_dataset(
                key, 
                shape=info['shape'], 
                dtype=info['dtype'],
                chunks=info['chunks']
            )
        # Write data in chunks
        for chunk_start in range(0, num_events, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_events)
            chunk_indices = indices[chunk_start:chunk_end]
            print(f"  Writing chunk {chunk_start//chunk_size + 1}/{(num_events-1)//chunk_size + 1} "
                  f"(events {chunk_start}-{chunk_end}/{num_events})")
            # Read and write each dataset
            for key in datasets_info.keys():
                if key in input_file:
                    chunk_data = input_file[key][chunk_indices]
                    datasets[key][chunk_start:chunk_end] = chunk_data
    print(f"  ✓ Wrote {num_events} events to {output_path}")
    
result = {} # Dictionary of {file: {train_idx, val_idx, test_idx}} - used to debug. In the end, store it in output_dir too.

for file in os.listdir(args.input_dir):
    if file.endswith(".h5"):
        dataset_name = file.split(".")[0]
        with h5py.File(os.path.join(args.input_dir, file), "r") as f:
            print(f"Processing file: {file}")
            truth_labels = f["truth_labels"][:, 1]
            passing_idx = filter_weird_events(truth_labels)
            # Now split the remaining idx using train_test_split
            truth_event_type = truth_labels[passing_idx]
            # Split into train and temp (val+test)
            train_idx, temp_idx = train_test_split(
                passing_idx, 
                test_size=args.test_ratio + args.val_ratio, 
                random_state=args.seed, 
                stratify=truth_event_type
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
            
            # Save the h5 files split in this way to <DATASET>/<train/val/test>.h5
            Path(os.path.join(args.output_dir, dataset_name)).mkdir(parents=True, exist_ok=True)
            
            # Write train data in chunks
            print("Writing train data...")
            write_h5_in_chunks(
                f, 
                os.path.join(args.output_dir, dataset_name, "train.h5"),
                train_idx,
                chunk_size=10000
            )
            
            # Write val data in chunks
            print("Writing val data...")
            write_h5_in_chunks(
                f,
                os.path.join(args.output_dir, dataset_name, "val.h5"),
                val_idx,
                chunk_size=10000
            )
            
            # Write test data in chunks
            print("Writing test data...")
            write_h5_in_chunks(
                f,
                os.path.join(args.output_dir, dataset_name, "test.h5"),
                test_idx,
                chunk_size=10000
            )
            
            print(f"✓ {file} split and written to {args.output_dir}")

# Save the result to the output directory
with open(os.path.join(args.output_dir, "result.pkl"), "wb") as f:
    pickle.dump(result, f)
