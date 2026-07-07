"""Validation loss for cut-based reco baselines (classification and regression).

Used as horizontal references on training-curve plots (no wandb run / no training).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.dataset.baseline_labels import load_baseline_pi_labels_v2
from src.dataset.dataloader import HEPTorchDataset, Task
from src.eval.e_available_plots._constants import EAVAILABLE_SCALE

RECO_BASELINE_MODEL_KEY = "Reco-baseline"


def _npi2_task() -> Task:
    return Task(
        type="classifier",
        classification_CC1orNPi=True,
        class_idx=[0, 1, 2, 3, 4],
        class_idx_map={0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
        class_label_idx=-1,
    )


def _class_weights_from_train_counts(
    data_path: Path,
    playlists: tuple[str, ...],
) -> torch.Tensor:
    """Inverse-frequency class weights from the train split (same recipe as ``HEPTorchDataset``)."""
    task = _npi2_task()
    counts = np.zeros(len(task.class_idx), dtype=np.float64)
    for playlist in playlists:
        train_dir = data_path / playlist / "train"
        ds = HEPTorchDataset(folder=str(train_dir), task=task, data_path=data_path)
        counts += ds.class_counts
    if counts.sum() <= 0:
        raise ValueError("Empty train class counts for reco-baseline val loss.")
    weights = 1.0 / (counts / counts.sum())
    return torch.tensor(weights, dtype=torch.float32)


def _mc_pi_labels_v2_val(data_path: Path, playlist: str) -> np.ndarray:
    """MC-truth Pi_labels_v2 on the val split (last column appended by ``HEPTorchDataset``)."""
    task = _npi2_task()
    val_dir = data_path / playlist / "val"
    ds = HEPTorchDataset(
        folder=str(val_dir),
        task=task,
        data_path=data_path,
        dataset_playlist=playlist,
    )
    return ds._truth_flat[:, task.class_label_idx].astype(np.int64)


def compute_reco_baseline_classification_val_loss(
    data_path: str | Path,
    playlists: tuple[str, ...] = ("1A",),
) -> float:
    """Weighted CE of cut-based baseline vs MC truth on val (matches ``evaluate`` for ``-npi2``).

    Baseline predictions are treated as a deterministic classifier (logit 0 on the
    predicted class, -1 elsewhere) so ``F.cross_entropy`` matches the training
    ``evaluate()`` recipe without numerical overflow from one-hot logits.
    """
    data_path = Path(data_path)
    class_weights = _class_weights_from_train_counts(data_path, playlists)

    mc_parts: list[np.ndarray] = []
    bl_parts: list[np.ndarray] = []
    for playlist in playlists:
        mc_parts.append(_mc_pi_labels_v2_val(data_path, playlist))
        bl_parts.append(
            load_baseline_pi_labels_v2(data_path, playlist, "val").astype(np.int64)
        )

    mc_true = torch.from_numpy(np.concatenate(mc_parts)).long()
    baseline_pred = torch.from_numpy(np.concatenate(bl_parts)).long()
    num_classes = len(_npi2_task().class_idx)

    # Deterministic classifier: logit 0 on predicted class, -1 elsewhere (stable softmax CE).
    logits = torch.full((mc_true.shape[0], num_classes), -1.0, dtype=torch.float32)
    logits[torch.arange(mc_true.shape[0]), baseline_pred] = 0.0
    return F.cross_entropy(logits, mc_true, weight=class_weights).item()


def reco_baseline_loss_series(val_loss: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Single-point history compatible with ``loss_histories`` / wandb BDT entries."""
    return [(np.array([0.0], dtype=float), np.array([float(val_loss)], dtype=float))]


def inject_reco_baseline_loss_history(
    loss_histories: dict,
    *,
    data_path: str | Path,
    playlists: tuple[str, ...] = ("1A",),
) -> None:
    """Add ``Reco-baseline`` to *loss_histories* for classification training-curve refs."""
    try:
        val_loss = compute_reco_baseline_classification_val_loss(data_path, playlists)
    except Exception as exc:
        print(f"  Warning: skipped {RECO_BASELINE_MODEL_KEY} val loss: {exc}")
        return
    loss_histories[RECO_BASELINE_MODEL_KEY] = reco_baseline_loss_series(val_loss)
    print(f"  {RECO_BASELINE_MODEL_KEY} val CE (cut-based vs MC truth): {val_loss:.4f}")


def _regression_e_available_no_muon_task() -> Task:
    return Task(
        type="regression",
        regress_E_available_no_muon=True,
        class_label_idx=9,
    )


def _blob_recoil_baseline_val_gev(
    data_path: Path,
    playlist: str,
    val_global: np.ndarray,
) -> np.ndarray:
    bl_path = data_path / "baselines" / f"{playlist}_enu_baselines.npz"
    if not bl_path.exists():
        raise FileNotFoundError(f"Missing baseline file: {bl_path}")
    baselines = np.load(bl_path)
    if "blob_recoil_E" not in baselines:
        raise KeyError(f"'blob_recoil_E' not found in {bl_path}")
    raw = baselines["blob_recoil_E"][val_global].astype(np.float64) / 1000.0
    return raw * EAVAILABLE_SCALE


def compute_reco_baseline_regression_val_loss(
    data_path: str | Path,
    playlists: tuple[str, ...] = ("1A",),
) -> float:
    """Huber loss of MINERvA blob-recoil E_available baseline vs MC truth on val.

    Matches ``evaluate()`` for ``-E-available-no-muon`` regressors (default Huber, GeV).
    """
    data_path = Path(data_path)
    result_path = data_path / "result.pkl"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing split index file: {result_path}")
    with open(result_path, "rb") as f:
        split_idx = pickle.load(f)

    task = _regression_e_available_no_muon_task()
    target_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    for playlist in playlists:
        if playlist not in split_idx:
            raise KeyError(f"Playlist {playlist!r} not in {result_path}")
        val_dir = data_path / playlist / "val"
        ds = HEPTorchDataset(
            folder=str(val_dir),
            task=task,
            data_path=data_path,
            dataset_playlist=playlist,
        )
        targets = ds._truth_flat[:, task.class_label_idx].astype(np.float64) / 1000.0
        val_global = np.asarray(split_idx[playlist]["val_idx"], dtype=np.int64)
        preds = _blob_recoil_baseline_val_gev(data_path, playlist, val_global)
        if targets.shape[0] != preds.shape[0]:
            raise ValueError(
                f"Val size mismatch for {playlist}: targets={targets.shape[0]}, "
                f"baseline={preds.shape[0]}"
            )
        target_parts.append(targets)
        pred_parts.append(preds)

    target_t = torch.from_numpy(np.concatenate(target_parts).astype(np.float32))
    pred_t = torch.from_numpy(np.concatenate(pred_parts).astype(np.float32))
    return F.huber_loss(pred_t, target_t).item()


def inject_reco_baseline_regression_loss_history(
    loss_histories: dict,
    *,
    data_path: str | Path,
    playlists: tuple[str, ...] = ("1A",),
) -> None:
    """Add ``Reco-baseline`` to regression *loss_histories* (horizontal val-loss ref)."""
    try:
        val_loss = compute_reco_baseline_regression_val_loss(data_path, playlists)
    except Exception as exc:
        print(
            f"  Warning: skipped {RECO_BASELINE_MODEL_KEY} regression val loss: {exc}"
        )
        return
    loss_histories[RECO_BASELINE_MODEL_KEY] = reco_baseline_loss_series(val_loss)
    print(
        f"  {RECO_BASELINE_MODEL_KEY} val Huber (blob-recoil E_available vs MC): "
        f"{val_loss:.4f}"
    )
