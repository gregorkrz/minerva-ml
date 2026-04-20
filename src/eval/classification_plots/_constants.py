"""Constants and small helpers for classification eval plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_curve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MC_INT_TYPE: dict[int, str] = {
    1: "QE",
    2: "RES",
    3: "DIS",
    4: "COH",
    8: "MEC/2p2h",
}

# Merged interaction-type grouping used by the polished classification plots:
# everything that is not DIS (3) or RES (2) is collapsed into "Other" (sentinel
# code 0, which never appears in raw ``int_type_arr``).
MC_INT_TYPE_MERGED: dict[int, str] = {
    3: "DIS",
    2: "RES",
    0: "Other",
}

# Fixed colors for the three merged interaction-type categories; used in all
# composition plots (signal / background / S/B) and shared with any future
# int-type-colored overlays for style consistency.
INT_TYPE_COLORS: dict[str, str] = {
    "DIS": "#1f77b4",
    "RES": "#ff7f0e",
    "Other": "#7f7f7f",
}


def merge_int_type_arr(int_type_arr: np.ndarray) -> np.ndarray:
    """Collapse ``int_type_arr`` into {3=DIS, 2=RES, 0=Other}.

    Any code that is not 2 or 3 (including unknown values) becomes 0.
    Shape and dtype semantics match :func:`numpy.asarray` on the input.
    """
    arr = np.asarray(int_type_arr)
    merged = np.zeros_like(arr, dtype=int)
    merged[arr == 3] = 3
    merged[arr == 2] = 2
    return merged


def _default_signal_label(signal_classes: list[int]) -> str:
    """Short label for the binary signal definition used in classification plots."""
    key = tuple(sorted(signal_classes))
    if key == (0,):
        return r"$CC1\pi^\pm$"
    if key == (0, 1):
        return r"CCN$\pi$"
    if key == (2,):
        return r"$CC\pi^0$"
    return "signal"


DEFAULT_FIXED_FPR = [0.2]

# Shared font sizes for all classification performance figures.
_LABEL_FS = 12
_TICK_FS = 11
_LEGEND_FS = 10


def _tpr_column_title_vs_kinematics(use_global_fpr: bool) -> str:
    """Third-column title for kinematic-bin performance plots."""
    return "TPR @ fixed FPR" if use_global_fpr else "TPR @ per-bin FPR"


def _tpr_line_legend_label(
    model_name: str, _fpr_val: float, _use_global_fpr: bool
) -> str:
    """Legend entry for a model TPR line (no FPR suffix; baseline uses :func:`_baseline_legend_with_global_fpr`)."""
    return model_name


def _global_reco_baseline_fpr(
    reco_pred: np.ndarray, y_true_binary: np.ndarray
) -> float:
    """Full-sample FPR of a binary baseline: FP / (FP + TN) on true background."""
    y_true_binary = np.asarray(y_true_binary)
    reco_pred = np.asarray(reco_pred)
    bg = y_true_binary == 0
    n_bg = int(bg.sum())
    if n_bg == 0:
        return float("nan")
    fp = int(((reco_pred == 1) & bg).sum())
    return fp / n_bg


def _reco_baseline_fpr_on_mask(
    reco_pred: np.ndarray,
    y_true_binary: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Baseline FPR restricted to rows where *mask* is True: FP / (FP+TN) on true background in mask."""
    reco_pred = np.asarray(reco_pred)
    y_true_binary = np.asarray(y_true_binary)
    mask = np.asarray(mask, dtype=bool)
    bg = mask & (y_true_binary == 0)
    n_bg = int(bg.sum())
    if n_bg == 0:
        return float("nan")
    fp = int(((reco_pred == 1) & bg).sum())
    return fp / n_bg


def _baseline_legend_with_global_fpr(base_label: str, global_fpr: float | None) -> str:
    """Append ``(FPR x.x%)`` when *global_fpr* is finite; else return *base_label*."""
    if global_fpr is None or not np.isfinite(global_fpr):
        return base_label
    return f"{base_label} (FPR {100.0 * float(global_fpr):.1f}%)"


def _global_score_thresholds_at_target_fprs(
    y_true: np.ndarray,
    scores: np.ndarray,
    fixed_fpr: list[float],
) -> dict[float, float]:
    """Return one score threshold per target FPR from a **global** ROC curve.

    Thresholds follow :func:`sklearn.metrics.roc_curve` (same ``searchsorted``
    indexing as the legacy per-bin extraction).  Events are classified as
    signal when ``scores >= threshold`` (sklearn convention for ``y_score``).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    valid = ~np.isnan(scores)
    y_true, scores = y_true[valid], scores[valid]
    out: dict[float, float] = {}
    n_neg = int((y_true == 0).sum())
    n_pos = int((y_true == 1).sum())
    if n_neg == 0 or n_pos == 0:
        for target_fpr in fixed_fpr:
            out[target_fpr] = float("nan")
        return out
    fpr_arr, _tpr_arr, thr = roc_curve(y_true, scores)
    n_thr = len(thr)
    if n_thr == 0:
        for target_fpr in fixed_fpr:
            out[target_fpr] = float("nan")
        return out
    for target_fpr in fixed_fpr:
        idx = int(np.searchsorted(fpr_arr, target_fpr, side="right") - 1)
        idx = max(0, min(idx, n_thr - 1))
        out[target_fpr] = float(thr[idx])
    return out


DEFAULT_N_BINS = 5
DEFAULT_Q3_BIN_EDGES = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25])

# Hadronic invariant mass W (GeV bin edges for classification plots).
DEFAULT_W_BIN_EDGES_GEV = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
# Fixed x-axis [GeV] for every figure that plots metrics or counts vs *W* (not data-driven).
W_METRICS_XLIM_GEV: tuple[float, float] = (
    float(DEFAULT_W_BIN_EDGES_GEV[0]),
    float(DEFAULT_W_BIN_EDGES_GEV[-1]),
)

# PDG-like masses for W² (MeV), consistent with ``extract_baselines.py``.
PROTON_MASS_MEV = 938.2720813
MUON_MASS_MEV = 105.6583755

# Default legend title on performance plots (pass ``legend_title=None`` to omit).
CLASSIFICATION_PERFORMANCE_LEGEND_TITLE = None


def _classification_legend_kw(
    fontsize: int, legend_title: str | None
) -> dict[str, Any]:
    kw: dict[str, Any] = {"fontsize": fontsize}
    if legend_title:
        kw["title"] = legend_title
        kw["title_fontsize"] = 10 if fontsize >= 9 else 8
    return kw
