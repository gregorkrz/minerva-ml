# Filters and splits the dataset into train, val, test sets. It's not super optimized, but the datasets are not large, so it kind of works in the ened.


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
parser.add_argument(
    "--playlists", nargs="+", required=False, default=None
)  # restrict to a specific playlist (if None, all playlists will be processed)
parser.add_argument("--val-ratio", type=float, required=False, default=0.1)
parser.add_argument("--test-ratio", type=float, required=False, default=0.1)
parser.add_argument("--seed", type=int, required=False, default=42)
parser.add_argument(
    "--only-save-result", action="store_true", required=False, default=False
)
# If set to True, only save the result.pkl file and don't save the train, val, test files.

args = parser.parse_args()


def filter_weird_events(truth_labels):
    # Only keep events where the interaction type is [1, 2, 3, 4, 8] (the other events are weird).
    allowed_event_types = torch.tensor([1, 2, 3, 4, 8])
    passing_idx = torch.where(torch.isin(truth_labels.int(), allowed_event_types))[0]
    return passing_idx


result = (
    {}
)  # Dictionary of {dataset: {train_idx, val_idx, test_idx}} - used to debug. In the end, store it in output_dir too.


def select_from_list(lst, idx):
    return [lst[i] for i in idx]


for dataset in os.listdir(args.input_dir):
    if dataset == "baselines":
        continue
    if args.playlists is not None and dataset not in args.playlists:
        continue
    # This assumes the data for each dataset can fit in memory, which is reasonable for now. We could optimize this later if needed.
    # Dataset: 1A, 1B...
    truth_labels = []
    global_features = []
    data = []
    additional_info = []
    filenames = sorted(os.listdir(os.path.join(args.input_dir, dataset)))
    for file in filenames:
        if file.endswith(
            ".pb"
        ):  # Keys: 'data', 'truth_labels', 'global_features'; optionally 'data_additional_info' (legacy)
            print(f"Processing file: {file}")
            f = torch.load(
                os.path.join(args.input_dir, dataset, file), weights_only=False
            )
            print(f.keys())
            data.append(f["data"])
            truth_labels.append(f["truth_labels"])
            global_features.append(f["global_features"])
            additional_info.append(
                f.get("data_additional_info")
            )  # None in new format (data is 10-col concat)
    # New format: single "data" (n_particles, 10); legacy: "data" (5-col) + "data_additional_info" (5-col)
    is_legacy_format = all(ai is not None for ai in additional_info)
    if is_legacy_format:
        print(f"  Format: legacy (data + data_additional_info)")
    else:
        print(f"  Format: new (data only, 10-col)")
    # Concatenate the different chunks
    # For nested tensors, we need to flatten the list first
    print("Concatenating truth_labels")
    truth_labels = torch.cat(
        [
            torch.tensor(x) if not isinstance(x, torch.Tensor) else x
            for x in truth_labels
        ],
        dim=0,
    )
    global_features = torch.cat(
        [
            torch.tensor(x) if not isinstance(x, torch.Tensor) else x
            for x in global_features
        ],
        dim=0,
    )
    truth_event_types = truth_labels[:, 1]
    passing_idx = filter_weird_events(truth_event_types)

    # Convert to numpy for sklearn
    passing_idx_np = passing_idx.cpu().numpy()
    truth_event_types_np = truth_event_types.cpu().numpy()

    # Now split the remaining idx using train_test_split
    train_idx, temp_idx = train_test_split(
        passing_idx_np,
        test_size=args.test_ratio + args.val_ratio,
        random_state=args.seed,
        stratify=truth_event_types_np[passing_idx_np],
    )

    # Split temp into val and test
    temp_event_types = truth_event_types_np[temp_idx]
    val_ratio_adjusted = args.val_ratio / (args.test_ratio + args.val_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio_adjusted),
        random_state=args.seed,
        stratify=temp_event_types,
    )

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    print(
        f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}"
    )
    result[dataset] = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
    if not args.only_save_result:
        # Don't toggle this if --only-save-result is set to True!
        # Save the files split in this way to <DATASET>/<train/val/test>/0.pb
        Path(os.path.join(args.output_dir, dataset, "train")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(args.output_dir, dataset, "val")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(args.output_dir, dataset, "test")).mkdir(
            parents=True, exist_ok=True
        )
        # if len(data) > 0 and isinstance(data[0], torch.Tensor) and data[0].is_nested:
        # Nested tensors: flatten all nested components into a single list
        # This may not be the best way to do it...
        print("Concatenating data")
        flattened_data = []
        for nested_tensor in data:
            print(f"Nested tensor: {nested_tensor.shape}")
            flattened_data += list(nested_tensor.unbind())
        if is_legacy_format:
            flattened_additional_info = []
            for nested_tensor in additional_info:
                flattened_additional_info += list(nested_tensor.unbind())
        print("Splitting data...")
        # assert that the indices dont overlap
        assert len(
            set(train_idx.tolist() + val_idx.tolist() + test_idx.tolist())
        ) == len(
            train_idx.tolist() + val_idx.tolist() + test_idx.tolist()
        ), "Indices overlap"

        def make_split(idx_list):
            out = {
                "data": torch.nested.nested_tensor(
                    select_from_list(flattened_data, idx_list), layout=torch.jagged
                ),
                "truth_labels": truth_labels[idx_list],
                "global_features": global_features[idx_list],
            }
            if is_legacy_format:
                out["data_additional_info"] = torch.nested.nested_tensor(
                    select_from_list(flattened_additional_info, idx_list),
                    layout=torch.jagged,
                )
            return out

        train = make_split(train_idx.tolist())
        val = make_split(val_idx.tolist())
        test = make_split(test_idx.tolist())

        torch.save(train, os.path.join(args.output_dir, dataset, "train", "0.pb"))
        torch.save(val, os.path.join(args.output_dir, dataset, "val", "0.pb"))
        torch.save(test, os.path.join(args.output_dir, dataset, "test", "0.pb"))

        print(f"✓ {dataset} split and written to {args.output_dir}")

# Save the result to the output directory (merge with existing result.pkl if present, so 1A then 1B don't overwrite)
result_path = os.path.join(args.output_dir, "result.pkl")
if os.path.exists(result_path):
    with open(result_path, "rb") as f:
        existing_result = pickle.load(f)
    for k, v in result.items():
        existing_result[k] = v
    result = existing_result

with open(result_path, "wb") as f:
    pickle.dump(result, f)
