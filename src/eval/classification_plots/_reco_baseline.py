"""Reconstruction-baseline recall/FPR per kinematic bin."""

from __future__ import annotations

import numpy as np

from ._metrics_binned import mc_value_in_bin


def compute_reco_baseline_recall_per_bin(
    reco_pred: np.ndarray,
    is_signal: np.ndarray,
    bin_var: np.ndarray,
    bin_edges: np.ndarray,
    has_pion: np.ndarray | None = None,
    finite_bin_var: bool = False,
) -> np.ndarray:
    """Per-bin recall of a binary reconstruction-level baseline."""
    recalls = []
    for i in range(len(bin_edges) - 1):
        bm = mc_value_in_bin(bin_var, bin_edges, i, require_finite=finite_bin_var)
        if has_pion is not None:
            bm = bm & has_pion
        sig_in_bin = is_signal & bm
        n_sig = sig_in_bin.sum()
        if n_sig == 0:
            recalls.append(np.nan)
        else:
            recalls.append(((reco_pred == 1) & sig_in_bin).sum() / n_sig)
    return np.array(recalls)


def compute_reco_baseline_fpr_per_bin(
    reco_pred: np.ndarray,
    y_true_binary: np.ndarray,
    bin_var: np.ndarray,
    bin_edges: np.ndarray,
    event_mask: np.ndarray | None = None,
    *,
    has_pion: np.ndarray | None = None,
    finite_bin_var: bool = False,
) -> np.ndarray:
    """Per-bin FPR of a binary reconstruction baseline on **true background**.

    In each kinematic bin (same edges as the stacked count histogram), among
    events passing ``event_mask`` (and optional ``has_pion``) with
    ``y_true_binary == 0``, returns the fraction with ``reco_pred == 1``.
    Bins with no true background yield ``nan``.
    """
    if event_mask is None:
        event_mask = np.ones(len(bin_var), dtype=bool)
    y_true_binary = np.asarray(y_true_binary)
    reco_pred = np.asarray(reco_pred)
    out: list[float] = []
    for i in range(len(bin_edges) - 1):
        bm = mc_value_in_bin(bin_var, bin_edges, i, require_finite=finite_bin_var)
        if has_pion is not None:
            bm = bm & has_pion
        bm = bm & event_mask
        bg = bm & (y_true_binary == 0)
        n_bg = int(bg.sum())
        if n_bg == 0:
            out.append(float("nan"))
        else:
            fp = int(((reco_pred == 1) & bg).sum())
            out.append(fp / n_bg)
    return np.asarray(out, dtype=np.float64)
