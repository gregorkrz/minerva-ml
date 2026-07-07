"""Per-kinematic-bin class weights for classifier training."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.eval.classification_plots._constants import (
    DEFAULT_Q3_BIN_EDGES,
    DEFAULT_W_BIN_EDGES_GEV,
)
from src.eval.classification_plots._metrics_binned import mc_value_in_bin
from src.eval.classification_plots._signal_definitions import resolve_signal_classes

INVALID_BIN = -1


def bin_edges_for_var(var: str) -> np.ndarray:
    """Return default histogram edges for kinematic variable *var*."""
    if var == "W":
        return DEFAULT_W_BIN_EDGES_GEV.copy()
    if var == "q3":
        return DEFAULT_Q3_BIN_EDGES.copy()
    raise ValueError(f"Unknown kinematic variable '{var}'. Allowed: W, q3")


def load_kinematic_values(
    data_path: str | Path,
    playlist: str,
    split: str,
    var: str,
) -> np.ndarray:
    """Load per-split-event kinematic values aligned with split-local indices.

    Parameters
    ----------
    data_path
        Dataset root containing ``result.pkl`` and ``baselines/``.
    playlist
        Playlist name (e.g. ``"1A"``).
    split
        One of ``"train"``, ``"val"``, ``"test"``.
    var
        ``"W"`` (true MC hadronic invariant mass, GeV) or ``"q3"`` (GeV).
    """
    data_path = Path(data_path)
    result_path = data_path / "result.pkl"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Missing split index file: {result_path}. "
            "Run split_dataset.py on this dataset first."
        )
    with open(result_path, "rb") as f:
        split_idx = pickle.load(f)
    if playlist not in split_idx:
        raise KeyError(
            f"Playlist '{playlist}' not found in {result_path}. "
            f"Available: {sorted(split_idx)}"
        )
    split_key = f"{split}_idx"
    if split_key not in split_idx[playlist]:
        raise KeyError(
            f"Split '{split}' not found for playlist '{playlist}' in {result_path}."
        )
    local_to_global = np.asarray(split_idx[playlist][split_key], dtype=np.int64)

    baseline_file = data_path / "baselines" / f"{playlist}_enu_baselines.npz"
    if not baseline_file.exists():
        raise FileNotFoundError(
            f"Missing baseline file: {baseline_file}. "
            "Run src/scripts/extract_baselines.py to generate baselines."
        )
    baselines = dict(np.load(baseline_file))

    if var == "W":
        if "mc_true_hadronic_W_GeV" not in baselines:
            raise KeyError(
                "Baselines must contain 'mc_true_hadronic_W_GeV'. "
                "Re-run src/scripts/extract_baselines.py."
            )
        w = np.asarray(
            baselines["mc_true_hadronic_W_GeV"][local_to_global], dtype=np.float64
        )
        return np.where((w < 0.0) | ~np.isfinite(w), np.nan, w)
    if var == "q3":
        return np.asarray(baselines["q3"][local_to_global], dtype=np.float64) / 1000.0
    raise ValueError(f"Unknown kinematic variable '{var}'. Allowed: W, q3")


def assign_bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign each value to a histogram bin index, or ``INVALID_BIN`` (-1).

    Uses the same convention as :func:`mc_value_in_bin` (last bin closed on
    upper edge). Non-finite or out-of-range values map to ``INVALID_BIN``.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    if n_bins < 1:
        raise ValueError("edges must contain at least two points")

    out = np.full(values.shape, INVALID_BIN, dtype=np.int64)
    ok = np.isfinite(values)
    for i in range(n_bins):
        in_bin = mc_value_in_bin(values, edges, i, require_finite=True)
        out[in_bin & ok] = i
    return out


def _inverse_freq_weights(class_counts: np.ndarray) -> np.ndarray:
    total = float(np.sum(class_counts))
    if total <= 0:
        return np.ones_like(class_counts, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = total / np.maximum(class_counts, 1.0)
    weights[class_counts == 0] = 0.0
    return weights


def compute_binned_class_weights(
    labels: np.ndarray,
    bin_indices: np.ndarray,
    n_classes: int,
    global_weights: np.ndarray,
    signal_classes: list[int],
) -> np.ndarray:
    """Per-bin class weight table with global fallback.

    Returns
    -------
    weight_table
        Shape ``(n_bins, n_classes)``. Row ``INVALID_BIN`` is unused; callers
        should index global weights directly for events with ``bin_index == -1``.
    """
    labels = np.asarray(labels, dtype=np.int64)
    bin_indices = np.asarray(bin_indices, dtype=np.int64)
    global_weights = np.asarray(global_weights, dtype=np.float64)
    signal_set = set(signal_classes)

    n_bins = int(bin_indices.max()) + 1 if bin_indices.size else 0
    if n_bins < 1:
        n_bins = len(np.unique(bin_indices[bin_indices >= 0])) or 0
    # Derive n_bins from edges caller will use; if only INVALID_BIN present, 0 rows.
    valid_bins = bin_indices[bin_indices >= 0]
    if valid_bins.size:
        n_bins = int(valid_bins.max()) + 1
    else:
        n_bins = 0

    weight_table = np.tile(global_weights, (max(n_bins, 0), 1))
    if n_bins == 0:
        return weight_table

    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        bin_labels = labels[mask]
        n_signal = int(np.sum(np.isin(bin_labels, list(signal_set))))
        n_background = int(bin_labels.size - n_signal)
        if n_signal == 0 or n_background == 0:
            weight_table[b] = global_weights
            continue
        counts = np.bincount(bin_labels, minlength=n_classes).astype(np.float64)
        weight_table[b] = _inverse_freq_weights(counts)

    return weight_table


def per_event_loss_weights(
    labels: np.ndarray,
    bin_indices: np.ndarray,
    weight_table: np.ndarray,
    global_weights: np.ndarray,
) -> np.ndarray:
    """Per-event scalar loss weights from class labels and bin assignments."""
    labels = np.asarray(labels, dtype=np.int64)
    bin_indices = np.asarray(bin_indices, dtype=np.int64)
    global_weights = np.asarray(global_weights, dtype=np.float64)
    out = np.empty(labels.shape, dtype=np.float64)
    fallback = bin_indices == INVALID_BIN
    valid = ~fallback
    if np.any(valid):
        out[valid] = weight_table[bin_indices[valid], labels[valid]]
    if np.any(fallback):
        out[fallback] = global_weights[labels[fallback]]
    return out


def log_binned_weight_summary(
    labels: np.ndarray,
    bin_indices: np.ndarray,
    weight_table: np.ndarray,
    global_weights: np.ndarray,
    signal_classes: list[int],
    edges: np.ndarray,
    var: str,
    signal_tag: str,
) -> None:
    """Print per-bin signal/background counts and whether global fallback was used."""
    signal_set = set(signal_classes)
    n_bins = len(edges) - 1
    print(
        f"Binned loss weighting: var={var}, signal={signal_tag}, "
        f"classes={signal_classes}, global_weights={global_weights.round(4).tolist()}"
    )
    for b in range(n_bins):
        mask = bin_indices == b
        n_events = int(mask.sum())
        if n_events == 0:
            lo, hi = edges[b], edges[b + 1]
            print(f"  bin {b} [{lo:.2f}, {hi:.2f}]: empty")
            continue
        bin_labels = labels[mask]
        n_signal = int(np.sum(np.isin(bin_labels, list(signal_set))))
        n_background = n_events - n_signal
        used_global = n_signal == 0 or n_background == 0
        lo, hi = edges[b], edges[b + 1]
        br = "]" if b == n_bins - 1 else ")"
        weights = global_weights if used_global else weight_table[b]
        print(
            f"  bin {b} [{lo:.2f}, {hi:.2f}{br}: "
            f"n={n_events}, signal={n_signal}, background={n_background}, "
            f"fallback={'yes' if used_global else 'no'}, "
            f"weights={weights.round(4).tolist()}"
        )
    n_invalid = int((bin_indices == INVALID_BIN).sum())
    if n_invalid:
        print(f"  invalid/out-of-range kinematics: n={n_invalid} (global weights)")


def build_binned_loss_weights(
    labels: np.ndarray,
    data_path: str | Path,
    playlist: str,
    split: str,
    var: str,
    signal_tag: str,
    global_weights: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """End-to-end: kinematic values -> per-event loss weights.

    Returns
    -------
    per_event_weights, weight_table
    """
    signal_classes = resolve_signal_classes(signal_tag)
    edges = bin_edges_for_var(var)
    kin = load_kinematic_values(data_path, playlist, split, var)
    if kin.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Kinematic array length {kin.shape[0]} does not match "
            f"labels length {labels.shape[0]} for split '{split}'."
        )
    bin_indices = assign_bin_indices(kin, edges)
    weight_table = compute_binned_class_weights(
        labels,
        bin_indices,
        n_classes,
        global_weights,
        signal_classes,
    )
    per_event = per_event_loss_weights(
        labels, bin_indices, weight_table, global_weights
    )
    log_binned_weight_summary(
        labels,
        bin_indices,
        weight_table,
        global_weights,
        signal_classes,
        edges,
        var,
        signal_tag,
    )
    return per_event, weight_table
