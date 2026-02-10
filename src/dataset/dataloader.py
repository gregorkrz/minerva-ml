import torch
import h5py
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import requests
import re
import os
from urllib.parse import urljoin
import numpy as np
from pathlib import Path
import torch._dynamo


def collate_point_cloud(batch):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]
    batch_attention_mask = [item["attention_mask"] for item in batch]

    # Stack once to avoid repeated slicing
    point_clouds = torch.stack(batch_X)  # (B, N, F)
    labels = torch.stack(batch_y)  # (B, num_classes)
    attention_masks = torch.stack(batch_attention_mask)  # (B, N)
    max_particles = point_clouds.shape[1]
    truncated_X = point_clouds[:, :max_particles, :].contiguous()  # (B, M, F)
    truncated_attention_mask = attention_masks[:, :max_particles].contiguous()  # (B, M)
    result = {"X": truncated_X, "y": labels, "attention_mask": truncated_attention_mask}

    # Handle optional fields in a loop to reduce code duplication
    optional_fields = ["cond", "pid", "add_info", "data_pid", "vertex_pid"]
    for field in optional_fields:
        if all(field in item for item in batch):
            stacked = torch.stack([item[field] for item in batch])
            # Truncate if it's sequence-like (i.e., has 2 or more dims)
            if stacked.dim() >= 2 and stacked.shape[1] >= max_particles:
                stacked = stacked[:, :max_particles].contiguous()
            result[field] = stacked
        else:
            result[field] = None
    return result


def get_class_counts(class_idx, label_idx_to_class_idx, files_truth_labels, truth_labels_idx):
    n_class = len(class_idx)
    class_counts = np.zeros(n_class)
    for file_idx in range(len(files_truth_labels)):
        labels = files_truth_labels[file_idx][:, truth_labels_idx] # labels contain 1's and 2's, rewrite them into 0's and 1's based on class_idx indices of 1 and 2 in there
        # 'rewrite' labels in a way 
        labels = np.array([label_idx_to_class_idx[int(label.item())] for label in labels])
        class_counts += np.bincount(labels, minlength=n_class)
    return class_counts


class HEPTorchDataset(Dataset):
    def __init__(
        self,
        folder,
        use_cond=False,
        pid_idx=4,
        use_pid=True,
        use_add=False,
        num_add=4,
        mode="",
        nevts=-1,
        max_particles=150,
        classes=None,
        regress_log=False,
        classification_event_type=False, #  if True, it will classify the event type (1, 2, 3, 4, 7, 8)
        classification_current=False # if True, it will classify the event current (1, 2)
    ):
        """
        Args:
            file_paths (list): List of file paths.
            use_pid (bool): Flag to select if PID information is used during training
            use_add (bool): Flags to select if additional information besides kinematics are used.
            regress_log (bool): Apply log transformation to regression targets
        """
        self.use_cond = use_cond
        self.use_add = use_add
        self.num_add = num_add
        self.pid_idx = pid_idx
        self.use_pid = use_pid
        self.folder = folder
        self.regress_log = regress_log
        self.file_paths = sorted(list([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.pb')]))
        self.files = [torch.load(file, weights_only=True, mmap=True) for file in self.file_paths]
        self.files_n_events = np.array([len(file["data"].offsets())-1 for file in self.files]) # -1 because the last offset is the total number of events
        self.files_n_events_sum = np.cumsum(self.files_n_events)
        self.files_values = [file["data"].values() for file in self.files]
        self.files_offsets = [file["data"].offsets() for file in self.files]
        # truth_labels and global_features are regular tensors, not nested
        self.files_truth_labels = [file["truth_labels"] for file in self.files]
        self.files_global_features = [file["global_features"] for file in self.files]
        self.mode = mode
        self.nevts = int(nevts)
        self.max_particles = max_particles
        self.classification_event_type = classification_event_type
        self.classification_current = classification_current
        if classification_event_type:
            self.class_idx = np.array([1, 2, 3, 4, 8]) # 5 classes for the classification task; TODO: make more flexible
            self.class_idx_map = {1: 0, 2: 1, 3: 2, 4: 3, 8: 4}
            # Estimate the class weights
            self.class_counts = get_class_counts(self.class_idx, self.class_idx_map, self.files_truth_labels, 1)
            self.class_weights = 1 / (self.class_counts / np.sum(self.class_counts))
        elif classification_current:
            self.class_idx = np.array([1, 2])
            self.class_idx_map = {1: 0, 2: 1}
            self.class_counts = get_class_counts(self.class_idx, self.class_idx_map, self.files_truth_labels, -1)
            self.class_weights = 1 / (self.class_counts / np.sum(self.class_counts))
        elif mode == "classification":
            raise ValueError("Invalid classification task")

    def __len__(self):
        if self.nevts > 0:
            return min(self.nevts, np.sum(self.files_n_events))
        print("Number of events per file", self.files_n_events)
        return np.sum(self.files_n_events)

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.files_n_events_sum, idx, side='right')
        if file_idx > 0:
            sample_idx = idx - self.files_n_events_sum[file_idx - 1]
        else:
            sample_idx = idx
        
        data = self.files_values[file_idx][self.files_offsets[file_idx][sample_idx]:self.files_offsets[file_idx][sample_idx+1]]
        
        # pad up to max_particles
        if data.shape[0] <= self.max_particles:
            valid_attention_mask = torch.ones(data.shape[0])
            n_padding = self.max_particles - data.shape[0]
            data = torch.cat([data, torch.zeros(n_padding, data.shape[1])], dim=0)
            valid_attention_mask = torch.cat([valid_attention_mask, torch.zeros(n_padding)], dim=0)
        else:
            raise ValueError("Data has more particles than max_particles")
        
        sample = {}

        # Handle labels
        if self.mode == "classifier":
            if self.classification_event_type:
                i = 1
            elif self.classification_current:
                i = -1
            else:
                raise ValueError("Invalid classification task")
            label = self.files_truth_labels[file_idx][sample_idx, i]
            label_int = int(label.item()) if torch.is_tensor(label) else int(label)
            sample["y"] = torch.tensor(self.class_idx_map[label_int], dtype=torch.long)
        elif self.mode == "regression":
            label = torch.log(self.files_truth_labels[file_idx][sample_idx, 0] / 1000.0 + 1e-6) # regression target: Enu in GeV (TODO: change to a better quantity to regress)
            label_val = label.item() if torch.is_tensor(label) else label
            sample["y"] = torch.tensor(label_val, dtype=torch.float32)
        else:
            # Default: return first truth label
            label = self.files_truth_labels[file_idx][sample_idx, 0]
            sample["y"] = label.clone().detach().float() if torch.is_tensor(label) else torch.tensor(label, dtype=torch.float32)
        
        if self.use_cond: # Use global features
            cond = self.files_global_features[file_idx][sample_idx]
            sample["cond"] = cond.clone().detach().float() if torch.is_tensor(cond) else torch.tensor(cond, dtype=torch.float32)
        
        if self.use_pid:
            sample["pid"] = data[:, self.pid_idx].int()
           
        sample["X"] = data
        sample["attention_mask"] = valid_attention_mask
        return sample


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    distributed=True,
    use_cond=False,
    use_pid=False,
    pid_idx=4,
    use_add=False,
    num_add=4,
    num_workers=16,
    rank=0,
    size=1,
    clip_inputs=False,
    mode="",
    shuffle=True,
    nevts=-1,
    regress_log=False,
    max_particles=33,
    classification_event_type=False,
    classification_current=False,
):
    supported_datasets = ["minerva_1A", "minerva_1B", "minerva_1C", "minerva_1D", "minerva_1E", "minerva_1F",
    "minerva_1G", "minerva_1L", "minerva_1M", "minerva_1N", "minerva_1O", "minerva_1P"]
    if dataset_name not in supported_datasets:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}."
        )

    if dataset_name in supported_datasets:
        dataset_playlist = dataset_name.split("_")[1]
        dataset_path = Path(path) / dataset_playlist / dataset_type
    
        data = HEPTorchDataset(
            folder=str(dataset_path),
            use_cond=use_cond,
            use_pid=use_pid,
            pid_idx=pid_idx,
            use_add=use_add,
            num_add=num_add,
            mode=mode,
            nevts=nevts,
            regress_log=regress_log,
            max_particles=max_particles,
            classification_event_type=classification_event_type,
            classification_current=classification_current,
        )
        loader = DataLoader(
            data,
            batch_size=batch,
            pin_memory=torch.cuda.is_available(),
            shuffle=shuffle,
            sampler=None,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=collate_point_cloud,
        )
        return loader, data.class_weights if hasattr(data, "class_weights") else None
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}.")
