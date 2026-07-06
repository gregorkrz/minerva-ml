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
from typing import Iterable

from src.dataset.baseline_labels import load_baseline_pi_labels_v2
from src.dataset.binned_loss_weights import (
    assign_bin_indices,
    bin_edges_for_var,
    compute_binned_class_weights,
    load_kinematic_values,
    log_binned_weight_summary,
    per_event_loss_weights,
)
from src.eval.classification_plots._signal_definitions import resolve_signal_classes


# GENIE interaction-type codes stored in truth_labels[:, 1] (see DATASET.md).
INT_TYPE_NAME_TO_CODE: dict[str, int] = {
    "QE": 1,
    "RES": 2,
    "DIS": 3,
    "COH": 4,
    "MEC": 8,
}
INT_TYPE_CODE_TO_NAME: dict[int, str] = {v: k for k, v in INT_TYPE_NAME_TO_CODE.items()}
# Column index of mc_intType in the truth_labels tensor.
TRUTH_INT_TYPE_COL: int = 1


def parse_event_types(values: Iterable) -> list[int]:
    """Parse user-supplied event-type identifiers into GENIE interaction codes.

    Accepts mixed names (``"DIS"``, case-insensitive) and integer codes (``3``).
    Raises ``ValueError`` on unknown entries.
    """
    if values is None:
        return []
    codes: list[int] = []
    for v in values:
        if isinstance(v, str) and not v.strip().lstrip("-").isdigit():
            key = v.strip().upper()
            if key not in INT_TYPE_NAME_TO_CODE:
                raise ValueError(
                    f"Unknown event type '{v}'. Allowed names: "
                    f"{sorted(INT_TYPE_NAME_TO_CODE)}; allowed codes: "
                    f"{sorted(INT_TYPE_NAME_TO_CODE.values())}"
                )
            codes.append(INT_TYPE_NAME_TO_CODE[key])
        else:
            code = int(v)
            if code not in INT_TYPE_CODE_TO_NAME:
                raise ValueError(
                    f"Unknown event-type code {code}. Allowed codes: "
                    f"{sorted(INT_TYPE_CODE_TO_NAME)}"
                )
            codes.append(code)
    # Preserve order, drop duplicates.
    seen: set[int] = set()
    unique: list[int] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


@dataclass
class Task:
    type: str = field(default="regression")  # "regression" or "classifier"
    regress_log: bool = field(
        default=True
    )  # If True, it will apply log transformation to the regression target
    classification_event_type: bool = field(
        default=False
    )  # If True, it will classify the event type (1, 2, 3, 4, 7, 8)
    classification_current: bool = field(
        default=False
    )  # If True, it will classify the event current (1, 2)
    classification_cc_1pi: bool = field(
        default=False
    )  # If True, it will classify the event type [0, 1, 2]: 0=other, 1=CC pi+, 2=CC pi-
    classification_n_pions: bool = field(
        default=False
    )  # If True, it will classify whether the event is multi-pion or not (>1 pion produced)
    classification_CC1orNPi: bool = field(
        default=False
    )  # If True, it will classify CC 1pi or n pions, according to signal definition in Eberly et al. 2015
    classification_binary: bool = field(
        default=False
    )  # If True, map Pi_labels_v2 pid classes to binary signal/background for training
    binary_signal_pid_classes: list[int] = field(
        default_factory=list
    )  # Pi_labels_v2 indices treated as signal (e.g. [0, 1] for CCN1pipm)
    predict_baseline: bool = field(
        default=False
    )  # If True, train on cut-based baseline Pi_labels_v2 instead of MC truth
    class_idx: list[int] = field(
        default=None
    )  # List of class indices for the classification task # e.g. [1, 2, 3] means 3 classes: 1, 2, 3
    class_idx_map: dict[int, int] = field(
        default=None
    )  # Map of class indices to class labels
    class_label_idx: int = field(
        default=None
    )  # Index of the label in the truth_labels tensor for the classification task
    class_weights: torch.Tensor = field(
        default=None
    )  # Weights for the classification task
    regress_E_available: bool = field(
        default=False
    )  # If True, it will regress the available energy of the event
    regress_E_available_no_muon: bool = field(
        default=False
    )  # If True, it will regress the available energy of the event, without the muon energy


def _pad_or_truncate(tensor, target_len):
    if tensor.shape[0] == target_len:
        return tensor
    if tensor.shape[0] > target_len:
        return tensor[:target_len]
    pad_shape = (target_len - tensor.shape[0],) + tuple(tensor.shape[1:])
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=0)


def collate_tabular_cond(batch):
    """Collate function for tabular cond-only datasets (no point clouds)."""
    result = {
        "cond": torch.stack([item["cond"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
    }
    if all("loss_weight" in item for item in batch):
        result["loss_weight"] = torch.stack([item["loss_weight"] for item in batch])
    if all("pid_label" in item for item in batch):
        result["pid_label"] = torch.stack([item["pid_label"] for item in batch])
    return result


def collate_point_cloud(batch, max_particles=33):
    """
    Collate function for point clouds and labels with truncation performed per batch.
    """
    B = len(batch)
    # Fast path: all samples already have exactly max_particles (no pad/truncate)
    first_X = batch[0]["X"]
    if all(item["X"].shape[0] == max_particles for item in batch):
        point_clouds = torch.stack([item["X"] for item in batch])
        attention_masks = torch.stack([item["attention_mask"] for item in batch])
    else:
        point_clouds = torch.stack(
            [_pad_or_truncate(item["X"], max_particles) for item in batch]
        )
        attention_masks = torch.stack(
            [_pad_or_truncate(item["attention_mask"], max_particles) for item in batch]
        )
    labels = torch.stack([item["y"] for item in batch])
    result = {"X": point_clouds, "y": labels, "attention_mask": attention_masks}

    # Optional fields: only stack if present in all items
    optional_fields = [
        "cond",
        "pid",
        "add_info",
        "data_pid",
        "vertex_pid",
        "energy_sums",
        "loss_weight",
        "pid_label",
    ]
    for field in optional_fields:
        if not all(field in item and item[field] is not None for item in batch):
            result[field] = None
            continue
        values = [item[field] for item in batch]
        v0 = values[0]
        need_pad = (
            torch.is_tensor(v0)
            and v0.dim() >= 1
            and all(
                torch.is_tensor(v) and v.shape[0] == batch[i]["X"].shape[0]
                for i, v in enumerate(values)
            )
        )
        if need_pad:
            values = [_pad_or_truncate(v, max_particles) for v in values]
        result[field] = torch.stack(values)
    return result


def get_class_counts(
    class_idx, label_idx_to_class_idx, files_truth_labels, truth_labels_idx
):
    n_class = len(class_idx)
    class_counts = np.zeros(n_class)
    for file_idx in range(len(files_truth_labels)):
        labels = files_truth_labels[file_idx][
            :, truth_labels_idx
        ]  # labels contain 1's and 2's, rewrite them into 0's and 1's based on class_idx indices of 1 and 2 in there
        labels = np.array(
            [label_idx_to_class_idx[int(label.item())] for label in labels]
        )
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
    labels[is_cc & one_pi_plus & ~one_pi_minus] = 0  # CC 1pi+
    labels[is_cc & one_pi_minus & ~one_pi_plus] = 1  # CC 1pi-
    labels[is_cc & multi_pions] = 2  # CC N Pi +-
    labels[~is_cc] = 3  # Not CC
    labels[is_cc & ((n_pi_plus + n_pi_minus) == 0)] = 4  # CC other
    return labels  # Labels: 0=CC 1pi+, 1=CC 1pi-, 2=CC N charged pions-, 3=OTHER


def get_Pi_labels_v2(file_truth_labels):
    # label0 = CC, exactly 1 charged pion (pi+ or pi-)
    # label1 = CC, N >1 charged pion, the rest whatever
    # label2 = CC, 1 pi0, no charged pions
    # label3 = CC, all others
    # label4 = NC
    is_cc = file_truth_labels[:, 3] == 1
    n_pi_plus = file_truth_labels[:, 5]
    n_pi_minus = file_truth_labels[:, 6]
    n_pi_zero = file_truth_labels[:, 10]
    n_charged = n_pi_plus + n_pi_minus
    labels = (
        torch.ones(len(file_truth_labels), dtype=torch.long) * 4
    )  # 4 is the label for other
    labels[is_cc & (n_charged == 1)] = 0
    labels[is_cc & (n_charged > 1)] = 1
    labels[is_cc & (n_charged == 0)] = 3
    labels[is_cc & (n_pi_zero == 1) & (n_pi_plus == 0) & (n_pi_minus == 0)] = 2
    return labels


def pid_to_binary_label(pid_class: int, signal_pid_classes: list[int]) -> int:
    """Map a Pi_labels_v2 class (0–4) to binary training label (1=signal, 0=background)."""
    return 1 if int(pid_class) in signal_pid_classes else 0


def binary_class_counts_from_pid_column(
    pid_labels: np.ndarray,
    event_idx_map: np.ndarray,
    signal_pid_classes: list[int],
) -> np.ndarray:
    """Class counts for binary training labels on the active event subset."""
    signal_set = set(signal_pid_classes)
    binary = np.array(
        [1 if int(p) in signal_set else 0 for p in pid_labels[event_idx_map]],
        dtype=np.int64,
    )
    return np.bincount(binary, minlength=2).astype(np.float64)


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
        concat_additional_info=True,
        use_energy_sums=False,
        event_types: list[int] | None = None,
        binned_loss_var: str | None = None,
        binned_loss_signal: str | None = None,
        data_path: str | Path | None = None,
        dataset_playlist: str | None = None,
        dataset_split: str | None = None,
        tabular_only: bool = False,
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
        self.use_energy_sums = use_energy_sums
        self.tabular_only = tabular_only
        self.folder = folder
        self.file_paths = sorted(
            list(
                [
                    os.path.join(folder, file)
                    for file in os.listdir(folder)
                    if file.endswith(".pb")
                ]
            )
        )
        if tabular_only and not use_cond:
            raise ValueError("tabular_only requires use_cond=True")
        if tabular_only:
            print("Loading tabular features only (no point clouds)")
            self.files_truth_labels = []
            self.files_global_features = []
            files_n_events = []
            for file in self.file_paths:
                obj = torch.load(file, weights_only=True, mmap=False)
                self.files_truth_labels.append(obj["truth_labels"])
                self.files_global_features.append(obj["global_features"])
                gf = obj["global_features"]
                files_n_events.append(
                    gf.shape[0] if torch.is_tensor(gf) else len(gf)
                )
                del obj
            self.files_n_events = np.array(files_n_events, dtype=np.int64)
            self.files = None
            self.files_values = None
            self.files_offsets = None
            self.files_values_additional_info = None
            self.files_offsets_additional_info = None
            self.data_feature_dim = None
        else:
            print("Loading files into memory")
            self.files = [
                torch.load(file, weights_only=True, mmap=False)
                for file in self.file_paths
            ]
            print("Files loaded into memory")
            self.files_n_events = np.array(
                [len(file["data"].offsets()) - 1 for file in self.files]
            )  # -1 because the last offset is the total number of events
            self.files_values = [file["data"].values() for file in self.files]
            self.files_offsets = [file["data"].offsets() for file in self.files]
            # data_feature_dim: 10 = new format (data already has features+additional_info concat), 5 = legacy (data + data_additional_info)
            self.data_feature_dim = self.files_values[0].shape[1]
            if "data_additional_info" in self.files[0]:
                self.files_values_additional_info = [
                    file["data_additional_info"].values() for file in self.files
                ]
                self.files_offsets_additional_info = [
                    file["data_additional_info"].offsets() for file in self.files
                ]
            else:
                self.files_values_additional_info = None
                self.files_offsets_additional_info = None
            # truth_labels and global_features are regular tensors, not nested
            self.files_truth_labels = [file["truth_labels"] for file in self.files]
            self.files_global_features = [file["global_features"] for file in self.files]
        self.files_n_events_sum = np.cumsum(self.files_n_events)
        # add a column with CC1orNPi labels
        if task.classification_CC1orNPi:
            self.files_truth_labels = [
                np.concatenate(
                    [
                        file_truth_labels,
                        get_Pi_labels_v2(file_truth_labels).reshape(-1, 1),
                    ],
                    axis=1,
                )
                for file_truth_labels in self.files_truth_labels
            ]
        self.global_feature_dim = self.files_global_features[0].shape[1]
        # Flatten truth and global features for single-index access (faster __getitem__)
        self._truth_flat = np.vstack(
            [f.numpy() if torch.is_tensor(f) else f for f in self.files_truth_labels]
        )
        self._global_flat = torch.cat(self.files_global_features, dim=0)
        self.nevts = int(nevts)
        self.max_particles = max_particles
        self.task = task
        self._baseline_labels_flat: np.ndarray | None = None
        if task.predict_baseline:
            if not task.classification_CC1orNPi:
                raise ValueError(
                    "predict_baseline requires classification_CC1orNPi (-npi2)."
                )
            if data_path is None or dataset_playlist is None or dataset_split is None:
                raise ValueError(
                    "predict_baseline requires data_path, dataset_playlist, and "
                    "dataset_split."
                )
            self._baseline_labels_flat = load_baseline_pi_labels_v2(
                data_path, dataset_playlist, dataset_split
            )
            if self._baseline_labels_flat.shape[0] != self._truth_flat.shape[0]:
                raise ValueError(
                    f"Baseline labels length {self._baseline_labels_flat.shape[0]} "
                    f"does not match split size {self._truth_flat.shape[0]} in "
                    f"{self.folder}."
                )
            print("Training targets: cut-based baseline Pi_labels_v2 (not MC truth)")
        self._len_full = int(np.sum(self.files_n_events))
        # Optional filter on GENIE interaction type (truth_labels col 1).
        # self._event_idx_map[logical_idx] -> physical_idx into the flattened arrays.
        # When no filter is active it is still the identity map so downstream logic stays uniform.
        self.event_types: list[int] = list(event_types) if event_types else []
        if self.event_types:
            int_type_flat = self._truth_flat[:, TRUTH_INT_TYPE_COL].astype(np.int64)
            allowed = np.asarray(self.event_types, dtype=np.int64)
            mask = np.isin(int_type_flat, allowed)
            self._event_idx_map = np.nonzero(mask)[0].astype(np.int64)
            kept = int(self._event_idx_map.size)
            total = int(self._truth_flat.shape[0])
            names = [INT_TYPE_CODE_TO_NAME.get(c, str(c)) for c in self.event_types]
            print(
                f"Filtering to event types {names} (codes {self.event_types}): "
                f"kept {kept:,} / {total:,} events "
                f"({100.0 * kept / max(total, 1):.2f}%)"
            )
            if kept == 0:
                raise ValueError(
                    f"No events remain after filtering to event types {names}. "
                    "Check that --event-types matches data in this split."
                )
        else:
            self._event_idx_map = np.arange(self._truth_flat.shape[0], dtype=np.int64)
        if self.nevts <= 0:
            print("Number of events per file", self.files_n_events)
        if self.task.type == "classifier":
            if self.task.predict_baseline:
                label_source = self._baseline_labels_flat.astype(np.int64)
            else:
                label_source = self._truth_flat[:, self.task.class_label_idx].astype(
                    np.int64
                )
            if self.task.classification_binary:
                self.class_counts = binary_class_counts_from_pid_column(
                    label_source,
                    self._event_idx_map,
                    self.task.binary_signal_pid_classes,
                )
            elif self.event_types:
                # Compute class counts on the filtered subset so class weights reflect training data.
                mapped = np.asarray(
                    [self.task.class_idx_map[int(l)] for l in label_source],
                    dtype=np.int64,
                )
                mapped = mapped[self._event_idx_map]
                self.class_counts = np.bincount(
                    mapped, minlength=len(self.task.class_idx)
                ).astype(np.float64)
            else:
                self.class_counts = get_class_counts(
                    self.task.class_idx,
                    self.task.class_idx_map,
                    [label_source.reshape(-1, 1)],
                    0,
                )
            self.class_weights = 1 / (self.class_counts / np.sum(self.class_counts))
            print("Class weights", self.class_weights)
            self.use_binned_loss = (
                binned_loss_var is not None and binned_loss_signal is not None
            )
            self._loss_weight_per_phys: np.ndarray | None = None
            if self.use_binned_loss:
                if not task.classification_CC1orNPi:
                    raise ValueError(
                        "Binned loss weighting requires --classification_CC1orNPi "
                        "(signal tags map to pid classes 0-4)."
                    )
                if data_path is None or dataset_playlist is None or dataset_split is None:
                    raise ValueError(
                        "Binned loss weighting requires data_path, playlist, and split."
                    )
                n_phys = self._truth_flat.shape[0]
                if self.task.predict_baseline:
                    raw_labels = self._baseline_labels_flat.astype(np.int64)
                else:
                    raw_labels = self._truth_flat[:, self.task.class_label_idx].astype(
                        np.int64
                    )
                if self.task.classification_binary:
                    signal_pid = set(self.task.binary_signal_pid_classes)
                    mapped_labels = np.array(
                        [1 if int(l) in signal_pid else 0 for l in raw_labels],
                        dtype=np.int64,
                    )
                    binned_signal_classes = [1]
                else:
                    mapped_labels = np.asarray(
                        [self.task.class_idx_map[int(l)] for l in raw_labels],
                        dtype=np.int64,
                    )
                    binned_signal_classes = resolve_signal_classes(binned_loss_signal)
                kin = load_kinematic_values(
                    data_path, dataset_playlist, dataset_split, binned_loss_var
                )
                if kin.shape[0] != n_phys:
                    raise ValueError(
                        f"Kinematic values length {kin.shape[0]} does not match "
                        f"split size {n_phys} in {self.folder}."
                    )
                edges = bin_edges_for_var(binned_loss_var)
                bin_indices = assign_bin_indices(kin, edges)
                labels_sub = mapped_labels[self._event_idx_map]
                bin_sub = bin_indices[self._event_idx_map]
                n_classes = len(self.task.class_idx)
                weight_table = compute_binned_class_weights(
                    labels_sub,
                    bin_sub,
                    n_classes,
                    self.class_weights,
                    binned_signal_classes,
                )
                weights_sub = per_event_loss_weights(
                    labels_sub,
                    bin_sub,
                    weight_table,
                    self.class_weights,
                )
                self._loss_weight_per_phys = np.ones(n_phys, dtype=np.float32)
                self._loss_weight_per_phys[self._event_idx_map] = weights_sub.astype(
                    np.float32
                )
                log_binned_weight_summary(
                    labels_sub,
                    bin_sub,
                    weight_table,
                    self.class_weights,
                    binned_signal_classes,
                    edges,
                    binned_loss_var,
                    binned_loss_signal,
                )
        elif self.task.type == "regression":
            self.regress_log = self.task.regress_log
            if self.regress_log:
                print("Regressing log!")
            else:
                print("Not regressing log")
        else:
            raise ValueError("Invalid task type")

    def __len__(self):
        n_available = int(self._event_idx_map.size)
        if self.nevts > 0:
            return min(self.nevts, n_available)
        print("Number of events per file", self.files_n_events)
        return n_available

    def __getitem__(self, idx):
        # Remap logical idx -> physical idx in the flattened/concatenated ordering.
        phys_idx = int(self._event_idx_map[idx])
        idx = phys_idx
        sample = {}

        # Labels from pre-flattened arrays (single index)
        if self.task.type == "classifier":
            mc_pid_class = int(self._truth_flat[idx, self.task.class_label_idx])
            train_pid_class = (
                int(self._baseline_labels_flat[idx])
                if self.task.predict_baseline
                else mc_pid_class
            )
            if self.task.classification_binary:
                sample["y"] = torch.tensor(
                    pid_to_binary_label(
                        train_pid_class, self.task.binary_signal_pid_classes
                    ),
                    dtype=torch.long,
                )
                sample["pid_label"] = torch.tensor(mc_pid_class, dtype=torch.long)
            else:
                sample["y"] = torch.tensor(
                    self.task.class_idx_map[train_pid_class], dtype=torch.long
                )
            if getattr(self, "use_binned_loss", False):
                sample["loss_weight"] = torch.tensor(
                    self._loss_weight_per_phys[idx], dtype=torch.float32
                )
        elif self.task.type == "regression":
            regression_label_idx = 0
            if self.task.regress_E_available or self.task.regress_E_available_no_muon:
                regression_label_idx = self.task.class_label_idx
            label_val = float(self._truth_flat[idx, regression_label_idx]) / 1000.0
            sample["y"] = torch.tensor(label_val, dtype=torch.float32)
        else:
            raise ValueError("Invalid task type")

        if self.use_cond:
            sample["cond"] = self._global_flat[idx].float()

        if self.tabular_only:
            return sample

        file_idx = np.searchsorted(self.files_n_events_sum, phys_idx, side="right")
        if file_idx > 0:
            sample_idx = phys_idx - self.files_n_events_sum[file_idx - 1]
        else:
            sample_idx = phys_idx
        off = self.files_offsets[file_idx]
        start, end = off[sample_idx].item(), off[sample_idx + 1].item()
        data = self.files_values[file_idx][start:end]
        n_pt = data.shape[0]

        if self.use_pid:
            sample["pid"] = data[:, self.pid_idx].int()
        if self.data_feature_dim == 10:
            # New format: data is (n_particles, 10), cols 0-4 = features, 5-9 = additional_info
            if self.concat_additional_info:
                sample["X"] = data.float()
            else:
                sample["X"] = data[:, :5].float()
                sample["add_info"] = data[:, 5:10].float()
        else:
            # Legacy: data (5 cols) + data_additional_info (5 cols)
            off_add = self.files_offsets_additional_info[file_idx]
            s, e = off_add[sample_idx].item(), off_add[sample_idx + 1].item()
            data_additional_info = self.files_values_additional_info[file_idx][s:e]
            if self.concat_additional_info:
                sample["X"] = torch.cat([data, data_additional_info], dim=1)
            else:
                sample["X"] = data.float()
                sample["add_info"] = data_additional_info.float()
        sample["attention_mask"] = torch.ones(n_pt, dtype=torch.float32)

        if self.use_energy_sums and self.global_feature_dim == 4:
            # Legacy: 4-col global_features; compute energy sums on-the-fly (cols 0-4 are features in both formats)
            pid = data[:, self.pid_idx]
            E = torch.exp(data[:, 3])  # index 3 is log(E + 1e-6)
            # PID 2=blob, 3=prong(3), 4=prong(8), 5=prong(13), 6=agg_blob, 7=agg_prong
            energy_sums = torch.zeros(6, dtype=torch.float32)
            for i, pid_val in enumerate([2, 3, 4, 5, 6, 7]):
                mask = pid == pid_val
                energy_sums[i] = E[mask].sum()
            sample["energy_sums"] = energy_sums
        # When global_feature_dim is 10 (legacy), 13 (7 base + 6), or 16 (10 base + 6), cond is complete

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
    event_sampler_random_state=42,
    use_energy_sums=False,
    event_types: list[int] | None = None,
    binned_loss_var: str | None = None,
    binned_loss_signal: str | None = None,
    tabular_only: bool = False,
):
    supported_datasets = [
        "minerva_1A",
        "minerva_1B",
        "minerva_1C",
        "minerva_1D",
        "minerva_1E",
        "minerva_1F",
        "minerva_1G",
        "minerva_1L",
        "minerva_1M",
        "minerva_1N",
        "minerva_1O",
        "minerva_1P",
    ]
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
            nevts=-1,  # Here, keep the whole dataset, only later we will sample a subset of the data
            use_energy_sums=use_energy_sums,
            event_types=event_types,
            binned_loss_var=binned_loss_var,
            binned_loss_signal=binned_loss_signal,
            data_path=path,
            dataset_playlist=dataset_playlist,
            dataset_split=dataset_type,
            tabular_only=tabular_only,
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
            collate_fn=(
                collate_tabular_cond
                if tabular_only
                else lambda x: collate_point_cloud(x, max_particles=max_particles)
            ),
            prefetch_factor=4 if distributed and num_workers > 0 else None,
            persistent_workers=distributed and num_workers > 0,
        )
        if task.type == "classifier":
            base_ds = data.dataset if isinstance(data, Subset) else data
            return loader, base_ds.class_weights, getattr(base_ds, "use_binned_loss", False)
        else:
            return loader, None, False
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not supported. Choose from {supported_datasets}."
        )
