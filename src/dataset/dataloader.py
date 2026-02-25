import torch
import h5py
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader, Subset
import requests
import re
import os
from urllib.parse import urljoin
import numpy as np
from pathlib import Path
import torch._dynamo
from dataclasses import dataclass, field

@dataclass
class Task:
    type: str = field(default="regression") # "regression" or "classifier"
    regress_log: bool = field(default=True) # If True, it will apply log transformation to the regression target
    classification_event_type: bool = field(default=False) # If True, it will classify the event type (1, 2, 3, 4, 7, 8)
    classification_current: bool = field(default=False) # If True, it will classify the event current (1, 2)
    classification_cc_1pi: bool = field(default=False) # If True, it will classify the event type [0, 1, 2]: 0=other, 1=CC pi+, 2=CC pi-
    classification_n_pions: bool = field(default=False) # If True, it will classify whether the event is multi-pion or not (>1 pion produced)
    classification_CC1orNPi: bool = field(default=False) # If True, it will classify CC 1pi or n pions, according to signal definition in Eberly et al. 2015
    class_idx: list[int] = field(default=None) # List of class indices for the classification task # e.g. [1, 2, 3] means 3 classes: 1, 2, 3
    class_idx_map: dict[int, int] = field(default=None) # Map of class indices to class labels
    class_label_idx: int = field(default=None) # Index of the label in the truth_labels tensor for the classification task
    class_weights: torch.Tensor = field(default=None) # Weights for the classification task
    regress_E_available: bool = field(default=False) # If True, it will regress the available energy of the event
    regress_E_available_no_muon: bool = field(default=False) # If True, it will regress the available energy of the event, without the muon energy
    
def collate_point_cloud(batch, max_particles=33):
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
    def _pad_or_truncate(tensor, target_len):
        if tensor.shape[0] == target_len:
            return tensor
        if tensor.shape[0] > target_len:
            return tensor[:target_len]
        pad_shape = (target_len - tensor.shape[0],) + tuple(tensor.shape[1:])
        padding = tensor.new_zeros(pad_shape)
        return torch.cat([tensor, padding], dim=0)
    batch_X = [_pad_or_truncate(item["X"], max_particles) for item in batch]
    batch_y = [item["y"] for item in batch]
    batch_attention_mask = [_pad_or_truncate(item["attention_mask"], max_particles) for item in batch]
    #batch_additional_info = [_pad_or_truncate(item["data_additional_info"], max_particles) for item in batch]
    point_clouds = torch.stack(batch_X)  # (B, M, F)
    labels = torch.stack(batch_y)  # (B, num_classes)
    attention_masks = torch.stack(batch_attention_mask)  # (B, M)
    #additional_info = torch.stack(batch_additional_info) # (B, M, 5)
    result = {"X": point_clouds, "y": labels, "attention_mask": attention_masks}#, "data_additional_info": additional_info}

    # Handle optional fields in a loop to reduce code duplication
    optional_fields = ["cond", "pid", "add_info", "data_pid", "vertex_pid"]
    for field in optional_fields:
        if all(field in item and item[field] is not None for item in batch):
            values = [item[field] for item in batch]
            # Pad per-particle optional tensors to max_particles before stacking.
            is_particle_aligned = all(
                torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == batch[i]["X"].shape[0]
                for i, v in enumerate(values)
            )
            if is_particle_aligned:
                values = [_pad_or_truncate(v, max_particles) for v in values]
            stacked = torch.stack(values)
            result[field] = stacked
        else:
            result[field] = None
    return result

def get_class_counts(class_idx, label_idx_to_class_idx, files_truth_labels, truth_labels_idx):
    n_class = len(class_idx)
    class_counts = np.zeros(n_class)
    for file_idx in range(len(files_truth_labels)):
        labels = files_truth_labels[file_idx][:, truth_labels_idx] # labels contain 1's and 2's, rewrite them into 0's and 1's based on class_idx indices of 1 and 2 in there
        labels = np.array([label_idx_to_class_idx[int(label.item())] for label in labels])
        class_counts += np.bincount(labels, minlength=n_class)
    return class_counts

def get_CC1orNPi_labels(file_truth_labels):
    # Generate labels for CC 1pi or n pions, according to signal definition in Eberly et al. 2015, on the fly
    is_cc = file_truth_labels[:, 3] == 1
    n_pi_plus = file_truth_labels[:, 5]
    n_pi_minus = file_truth_labels[:, 6]
    labels = torch.zeros(len(file_truth_labels), dtype=torch.long)
    one_pi_plus = n_pi_plus == 1
    one_pi_minus = n_pi_minus == 1
    multi_pions = (n_pi_plus + n_pi_minus) > 1
    labels[is_cc & one_pi_plus & ~one_pi_minus] = 0 # CC 1pi+
    labels[is_cc & one_pi_minus & ~one_pi_plus] = 1 # CC 1pi-
    labels[is_cc & multi_pions] = 2 # CC N Pi +-
    labels[~is_cc] = 3 # OTHER
    return labels

class HEPTorchDataset(Dataset):
    def __init__(
        self,
        folder,
        use_cond=False,
        pid_idx=4,
        use_pid=True,
        use_add=False,
        num_add=4,
        nevts=-1,
        max_particles=150,
        task: Task = Task(),
        concat_additional_info=True
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
        self.concat_additional_info = concat_additional_info
        self.folder = folder
        self.file_paths = sorted(list([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.pb')]))
        print("Loading files into memory")
        self.files = [torch.load(file, weights_only=True, mmap=False) for file in self.file_paths]
        print("Files loaded into memory")
        self.files_n_events = np.array([len(file["data"].offsets())-1 for file in self.files]) # -1 because the last offset is the total number of events
        self.files_n_events_sum = np.cumsum(self.files_n_events)
        self.files_values = [file["data"].values() for file in self.files]
        self.files_offsets = [file["data"].offsets() for file in self.files]
        self.files_values_additional_info = [file["data_additional_info"].values() for file in self.files]
        self.files_offsets_additional_info = [file["data_additional_info"].offsets() for file in self.files]
        # truth_labels and global_features are regular tensors, not nested
        self.files_truth_labels = [file["truth_labels"] for file in self.files]
        # add a column with CC1orNPi labels
        if task.classification_CC1orNPi:
            self.files_truth_labels = [np.concatenate([file_truth_labels, get_CC1orNPi_labels(file_truth_labels).reshape(-1, 1)], axis=1) for file_truth_labels in self.files_truth_labels]
        self.files_global_features = [file["global_features"] for file in self.files]
        self.nevts = int(nevts)
        self.max_particles = max_particles
        self.task = task
        if self.task.type == "classifier":
            self.class_counts = get_class_counts(self.task.class_idx, self.task.class_idx_map, self.files_truth_labels, self.task.class_label_idx)
            self.class_weights = 1 / (self.class_counts / np.sum(self.class_counts))
            print("Class weights", self.class_weights)
        elif self.task.type == "regression":
            self.regress_log = self.task.regress_log
            if self.regress_log:
                print("Regressing log!")
            else:
                print("Not regressing log")
        else:
            raise ValueError("Invalid task type")

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
        data_additional_info = self.files_values_additional_info[file_idx][self.files_offsets_additional_info[file_idx][sample_idx]:self.files_offsets_additional_info[file_idx][sample_idx+1]]
        valid_attention_mask = torch.ones(data.shape[0], dtype=data.dtype)
        
        # pad up to max_particles
        #if data.shape[0] <= self.max_particles:
        #    valid_attention_mask = torch.ones(data.shape[0])
        #    n_padding = self.max_particles - data.shape[0]
        #    data = torch.cat([data, torch.zeros(n_padding, data.shape[1])], dim=0)
        #    data_additional_info = torch.cat([data_additional_info, torch.zeros(n_padding, data_additional_info.shape[1])], dim=0)
        #    valid_attention_mask = torch.cat([valid_attention_mask, torch.zeros(n_padding)], dim=0)
        #else:
        #    raise ValueError("Data has more particles than max_particles")
        sample = {}
        # Handle labels
        if self.task.type == "classifier":
            i = self.task.class_label_idx
            label = self.files_truth_labels[file_idx][sample_idx, i]
            label_int = int(label.item()) if torch.is_tensor(label) else int(label)
            sample["y"] = torch.tensor(self.task.class_idx_map[label_int], dtype=torch.long)
        elif self.task.type == "regression":
            regression_label_idx = 0
            if self.task.regress_E_available or self.task.regress_E_available_no_muon:
                regression_label_idx = self.task.class_label_idx
            label = torch.log(self.files_truth_labels[file_idx][sample_idx, regression_label_idx] / 1000.0 + 1e-6) if self.task.regress_log else self.files_truth_labels[file_idx][sample_idx, regression_label_idx] / 1000.0
            label_val = label.item() if torch.is_tensor(label) else label
            sample["y"] = torch.tensor(label_val, dtype=torch.float32)
        else:
            raise ValueError("Invalid task type")
        
        if self.use_cond: # Use global features
            cond = self.files_global_features[file_idx][sample_idx]
            sample["cond"] = cond.clone().detach().float() if torch.is_tensor(cond) else torch.tensor(cond, dtype=torch.float32)
        
        if self.use_pid:
            sample["pid"] = data[:, self.pid_idx].int()
        #sample["data_additional_info"] = data_additional_info # shape (N, 5)
        if self.concat_additional_info: # concate data and data_additional_info into a single tensor
            sample["X"] = torch.cat([data, data_additional_info], dim=1) # shape (N, F+5)
        else:
            sample["X"] = data
        #else:
        #    sample["data_additional_info"] = data_additional_info # shape (N, 5)
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
    shuffle=True,
    nevts=-1,
    max_particles=33,
    task: Task = Task(),
    concat_additional_info=True,
    event_sampler_random_state=42
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
            max_particles=max_particles,
            task=task,
            concat_additional_info=concat_additional_info,
            nevts=-1 # Here, keep the whole dataset, only later we will sample a subset of the data
        )
        if nevts > 0 and nevts < len(data):
            print(f"Using a subset of the data: {nevts} events out of {len(data)}")
            rng = np.random.RandomState(event_sampler_random_state)
            indices = rng.choice(len(data), size=nevts, replace=False)
            data = Subset(data, indices)
        loader = DataLoader(
            data,
            batch_size=batch,
            pin_memory=torch.cuda.is_available(),
            shuffle=shuffle,
            sampler=None,
            num_workers=num_workers,
            drop_last=False,
            collate_fn=lambda x: collate_point_cloud(x, max_particles=max_particles),
            prefetch_factor=2 if distributed else None,
            persistent_workers=distributed
        )
        if task.type == "classifier":
            return loader, data.class_weights
        else:
            return loader, None
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}.")
