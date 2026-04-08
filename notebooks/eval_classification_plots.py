"""
Evaluation plotting utilities for charged-pion classification models.

Loads evaluation results from checkpoint directories and produces
AUPRC / AUROC / TPR-at-fixed-FPR summary plots binned by pion energy,
pion angle, or true q3.  Supports multi-run uncertainty bands.

Usage example::

    from eval_classification_plots import (
        load_results, load_truth_and_baselines,
        compute_all_metrics, plot_cc1pi_vs_pion_kinematics,
    )

    training_names = {
        "Transformer": ["run_A", "run_B"],
        "OmniLearned":  ["run_C"],
    }

    results = load_results(CKPT_DIR, training_names)
    data    = load_truth_and_baselines(CKPT_DIR, training_names)
    metrics = compute_all_metrics(results, data, signal_classes=[0])
    fig     = plot_cc1pi_vs_pion_kinematics(metrics, data, uncertainties=True)
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve, roc_curve

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
DEFAULT_N_BINS = 5
DEFAULT_Q3_BIN_EDGES = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 20, 25])


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_results(
    ckpt_dir: str | Path,
    training_names: dict[str, list[str]],
    playlists: list[str] | None = None,
    verbose: bool = True,
) -> dict[str, list[dict[str, dict]]]:
    """Load ``.npz`` evaluation results for every model and run.

    Parameters
    ----------
    ckpt_dir : root checkpoint directory.
    training_names : ``{model_name: [run_name_1, run_name_2, ...]}``.
    playlists : evaluation playlists (default ``["1A"]``).

    Returns
    -------
    ``{model_name: [run0_results, run1_results, ...]}`` where each
    ``runN_results`` is ``{playlist: {prediction, pid, ...}}``.
    """
    ckpt_dir = Path(ckpt_dir)
    if playlists is None:
        playlists = ["1A"]

    all_results: dict[str, list[dict]] = {}
    for model_name in sorted(training_names.keys()):
        run_names = training_names[model_name]
        all_results[model_name] = []
        for run_name in run_names:
            run_results: dict[str, dict] = {}
            for playlist in playlists:
                p1 = ckpt_dir / run_name / "test_results" / f"outputs_{run_name}_minerva_{playlist}_0.npz"
                p2 = ckpt_dir / run_name / "test_results" / f"outputs_best_model_minerva_{playlist}_0.npz"
                if p1.exists():
                    run_results[playlist] = dict(np.load(p1))
                elif p2.exists():
                    run_results[playlist] = dict(np.load(p2))
                else:
                    raise FileNotFoundError(
                        f"No results for {run_name} playlist {playlist}: "
                        f"tried {p1} and {p2}"
                    )
                run_results[playlist]["prediction"] = _softmax(
                    run_results[playlist]["prediction"]
                )
            all_results[model_name].append(run_results)
        if verbose:
            print(f"[{model_name}] loaded {len(run_names)} run(s)")
    return all_results


def load_truth_and_baselines(
    ckpt_dir: str | Path,
    training_names: dict[str, list[str]],
    playlists: list[str] | None = None,
    n_pion_bins: int = DEFAULT_N_BINS,
    q3_bin_edges: np.ndarray | None = None,
    *,
    pion_E_bin_edges: np.ndarray | Sequence[float] | None = None,
    pion_theta_bin_edges: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Load truth labels, baselines and derived kinematic arrays.

    Uses the first run of the first model to locate the data path.

    **Pion kinematic bins** default to ``np.quantile`` on events with
    ``has_pion`` (and :math:`E>0` / finite :math:`\\theta`), with *n_pion_bins*
    bins. Alternatively pass both *pion_E_bin_edges* and *pion_theta_bin_edges*
    (strictly increasing, length :math:`N+1` each) to use fixed edges; *n_pion_bins*
    is then ignored for pion binning. Same validation as
    :func:`data_with_signal_pion_bins` ``custom`` mode.

    Returns
    -------
    dict with keys: ``truth_labels``, ``test_idx``, ``baselines``,
    ``pion_E_MC``, ``pion_theta_MC``, ``has_pion``, ``q3_GeV``,
    ``int_type_arr``, ``pion_E_MC_bins``, ``pion_E_MC_bins_mid``,
    ``pion_theta_MC_bins``, ``pion_theta_MC_bins_mid``,
    ``q3_bin_edges``, ``q3_bin_mids``.
    """
    ckpt_dir = Path(ckpt_dir)
    if playlists is None:
        playlists = ["1A"]
    if q3_bin_edges is None:
        q3_bin_edges = DEFAULT_Q3_BIN_EDGES.copy()

    first_model = next(iter(training_names))
    first_run = training_names[first_model][0]

    # Find data_path from best_model.pt or settings.json
    data_path = None
    model_pt = ckpt_dir / first_run / "best_model.pt"
    if model_pt.exists():
        ckpt = torch.load(model_pt, weights_only=False, map_location="cpu")
        data_path = Path(ckpt["args"]["data_path"])
    else:
        settings_path = ckpt_dir / first_run / "settings.json"
        if settings_path.exists():
            with open(settings_path) as f:
                cfg = json.load(f)
            data_path = Path(cfg.get("data_path", cfg.get("dataset_path", "")))
    if data_path is None:
        raise FileNotFoundError(
            f"Cannot determine data_path from {first_run}: "
            "no best_model.pt or settings.json found"
        )

    split_idx = pickle.load(open(data_path / "result.pkl", "rb"))

    truth_labels: dict[str, torch.Tensor] = {}
    baselines_dict: dict[str, dict] = {}
    test_idx_dict: dict[str, np.ndarray] = {}

    for playlist in playlists:
        test_idx = split_idx[playlist]["test_idx"]
        test_idx_dict[playlist] = test_idx

        truth_labels[playlist] = torch.load(
            open(data_path / playlist / "test" / "0.pb", "rb"),
            weights_only=False,
            map_location="cpu",
        )["truth_labels"]

        baseline_file = f"{playlist}_enu_baselines.npz"
        loaded = False
        for subdir in ["baselines2", "baselines1", "baselines"]:
            candidate = data_path / subdir / baseline_file
            if candidate.exists():
                baselines_dict[playlist] = dict(np.load(candidate))
                loaded = True
                break
        if not loaded:
            print(f"[{playlist}] WARNING: no baselines found in {data_path}")

    playlist = playlists[0]
    test_idx = test_idx_dict[playlist]
    tl = truth_labels[playlist]

    # Pion kinematics
    pion_fv = baselines_dict[playlist]["pion_four_vectors"][test_idx] / 1000.0
    pion_E_MC = pion_fv[:, -1]
    has_pion = pion_fv[:, -1] > 0
    pion_p_MC = np.linalg.norm(pion_fv[:, 1:4], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pion_theta_MC = np.arccos(pion_fv[:, 2] / pion_p_MC)

    if (pion_E_bin_edges is None) ^ (pion_theta_bin_edges is None):
        raise ValueError(
            "pion_E_bin_edges and pion_theta_bin_edges must both be set or both be None"
        )
    if pion_E_bin_edges is not None:
        pion_E_bins = _as_strictly_increasing_bin_edges(pion_E_bin_edges, "pion_E_bin_edges")
        pion_theta_bins = _as_strictly_increasing_bin_edges(
            pion_theta_bin_edges, "pion_theta_bin_edges"
        )
    else:
        # Pion bins (quantile-based on events with a pion)
        pion_E_signal = pion_E_MC[has_pion & (pion_E_MC > 0)]
        pion_theta_signal = pion_theta_MC[has_pion & np.isfinite(pion_theta_MC)]
        pion_E_bins = np.quantile(pion_E_signal, np.linspace(0, 1, n_pion_bins + 1))
        pion_theta_bins = np.quantile(pion_theta_signal, np.linspace(0, 1, n_pion_bins + 1))

    # q3
    q3_GeV = baselines_dict[playlist]["q3"][test_idx] / 1000.0

    # Interaction type
    int_type_arr = tl[:, 1].numpy()

    return {
        "truth_labels": truth_labels,
        "test_idx": test_idx_dict,
        "baselines": baselines_dict,
        "pion_E_MC": pion_E_MC,
        "pion_theta_MC": pion_theta_MC,
        "has_pion": has_pion,
        "q3_GeV": q3_GeV,
        "int_type_arr": int_type_arr,
        "pion_E_MC_bins": pion_E_bins,
        "pion_E_MC_bins_mid": (pion_E_bins[:-1] + pion_E_bins[1:]) / 2,
        "pion_theta_MC_bins": pion_theta_bins,
        "pion_theta_MC_bins_mid": (pion_theta_bins[:-1] + pion_theta_bins[1:]) / 2,
        "q3_bin_edges": q3_bin_edges,
        "q3_bin_mids": (q3_bin_edges[:-1] + q3_bin_edges[1:]) / 2,
    }


def equal_frequency_bin_edges(
    x: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Histogram edges so **true-signal** rows in *mask* are split ~evenly across bins.

    Sorts finite ``x[mask]`` and places cuts between consecutive blocks of
    ``~n // n_bins`` sorted signal values. Interior cut between block
    boundaries is the midpoint of the two adjacent values (ties broken with
    ``np.nextafter``). ``edges[0]`` and ``edges[-1]`` are the signal min and
    max so all signal land in some bin under :func:`mc_value_in_bin`.

    All events (signal + background) are then assigned with :func:`mc_value_in_bin`.
    Background **histogram** counts per bin still vary with kinematics; only
    **signal** counts are balanced by construction.
    """
    vals = np.asarray(x[mask], dtype=float)
    vals = vals[np.isfinite(vals)]
    vals.sort()
    n = len(vals)
    if n < n_bins:
        raise ValueError(
            f"equal_frequency_bin_edges: need at least n_bins={n_bins} "
            f"finite values under mask, got {n}"
        )
    splits = (np.arange(n_bins + 1, dtype=np.int64) * n / n_bins).astype(np.int64)
    splits[-1] = n
    for j in range(1, n_bins + 1):
        splits[j] = max(int(splits[j]), int(splits[j - 1]))
    splits[0] = 0

    edges = np.empty(n_bins + 1, dtype=np.float64)
    edges[0] = float(vals[0])
    edges[-1] = float(vals[-1])
    for j in range(1, n_bins):
        sj = int(splits[j])
        sj = min(max(sj, 1), n - 1)
        left_max = float(vals[sj - 1])
        right_min = float(vals[sj])
        if right_min > left_max:
            edges[j] = 0.5 * (left_max + right_min)
        else:
            edges[j] = float(np.nextafter(left_max, np.inf))

    for j in range(1, n_bins):
        if edges[j] <= edges[j - 1]:
            edges[j] = float(np.nextafter(edges[j - 1], np.inf))
    edges[-1] = float(vals[-1])
    return edges


def _as_strictly_increasing_bin_edges(edges: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Validate user-supplied histogram edges (1-D, finite, strictly increasing)."""
    arr = np.asarray(edges, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1-D sequence with at least 2 edges, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def data_with_signal_pion_bins(
    data: dict[str, Any],
    pid: np.ndarray,
    signal_classes: list[int],
    n_bins: int | None = None,
    pion_quantile_require_has_pion: bool = True,
    pion_bin_edge_method: str = "equal_frequency",
    *,
    pion_E_bin_edges: np.ndarray | Sequence[float] | None = None,
    pion_theta_bin_edges: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Shallow copy of *data* with pion E/θ bin edges from **true signal** only.

    **Edge methods** (*pion_bin_edge_method*):

    * ``"equal_frequency"`` (default): edges split sorted signal :math:`E` / θ
      into nearly equal counts per bin (best-effort with ties).
    * ``"quantile"``: ``np.quantile`` on the same signal subset (can differ from
      exact equal counts when there are duplicate values).
    * ``"custom"``: use keyword arguments *pion_E_bin_edges* and *pion_theta_bin_edges*
      (each strictly increasing, length :math:`N+1` for :math:`N` bins). *n_bins* is
      ignored in this mode; :math:`E` and θ may use different bin counts.

    **Masks for which signal rows define edges** (*pion_quantile_require_has_pion*):

    * True (legacy): ``has_pion``, ``E > 0``, finite θ, restricted to signal.
    * False: ``signal & finite E`` for :math:`E`, ``signal & finite θ`` for θ,
      matching ``compute_all_metrics(..., pion_bins_require_has_pion=False)``.

    **ROC / AUPRC per bin** (:func:`bin_separation_metrics`): positives are
    signal in that kinematic bin; **negatives are the full background set**
    (every non-signal event). So **the same background events — and the same
    count of negatives — are used for every bin’s curve**; only the in-bin
    signal positives change.

    **Count histograms** (``N_\\mathrm{total}``, ``N_\\mathrm{signal}``): background
    per bin is “non-signal whose :math:`E` or θ falls in that bin”; those counts
    **cannot** all match each other while also using signal-balanced edges,
    unless signal and background share the same kinematic distribution.

    Use separate calls for CC1π± (e.g. ``[0]``) and CCπ⁰ (e.g. ``[2]``).
    """
    if n_bins is None:
        n_bins = len(data["pion_E_MC_bins"]) - 1

    pid_i = np.asarray(pid).astype(int, copy=False)
    sig = np.isin(pid_i, np.asarray(signal_classes, dtype=int))

    has_pion = data["has_pion"]
    pion_E = data["pion_E_MC"]
    pion_theta = data["pion_theta_MC"]

    if pion_quantile_require_has_pion:
        m_e = sig & has_pion & (pion_E > 0)
        m_th = sig & has_pion & np.isfinite(pion_theta)
    else:
        m_e = sig & np.isfinite(pion_E)
        m_th = sig & np.isfinite(pion_theta)

    if m_e.sum() < 2:
        raise ValueError(
            f"Too few events for E binning (n={int(m_e.sum())}); "
            f"signal_classes={signal_classes}"
        )
    if m_th.sum() < 2:
        raise ValueError(
            f"Too few events for θ binning (n={int(m_th.sum())}); "
            f"signal_classes={signal_classes}"
        )

    if pion_bin_edge_method != "custom" and (
        pion_E_bin_edges is not None or pion_theta_bin_edges is not None
    ):
        raise ValueError(
            "pion_E_bin_edges / pion_theta_bin_edges are only used when "
            "pion_bin_edge_method='custom'"
        )

    if pion_bin_edge_method == "quantile":
        pion_E_bins = np.quantile(pion_E[m_e], np.linspace(0, 1, n_bins + 1))
        pion_theta_bins = np.quantile(pion_theta[m_th], np.linspace(0, 1, n_bins + 1))
    elif pion_bin_edge_method == "equal_frequency":
        pion_E_bins = equal_frequency_bin_edges(pion_E, m_e, n_bins)
        pion_theta_bins = equal_frequency_bin_edges(pion_theta, m_th, n_bins)
    elif pion_bin_edge_method == "custom":
        if pion_E_bin_edges is None or pion_theta_bin_edges is None:
            raise ValueError(
                "pion_bin_edge_method='custom' requires keyword arguments "
                "pion_E_bin_edges and pion_theta_bin_edges"
            )
        pion_E_bins = _as_strictly_increasing_bin_edges(pion_E_bin_edges, "pion_E_bin_edges")
        pion_theta_bins = _as_strictly_increasing_bin_edges(
            pion_theta_bin_edges, "pion_theta_bin_edges"
        )
    else:
        raise ValueError(
            f"pion_bin_edge_method must be 'quantile', 'equal_frequency', or 'custom', "
            f"got {pion_bin_edge_method!r}"
        )

    out = dict(data)
    out["pion_E_MC_bins"] = pion_E_bins
    out["pion_E_MC_bins_mid"] = (pion_E_bins[:-1] + pion_E_bins[1:]) / 2
    out["pion_theta_MC_bins"] = pion_theta_bins
    out["pion_theta_MC_bins_mid"] = (pion_theta_bins[:-1] + pion_theta_bins[1:]) / 2
    return out


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def mc_value_in_bin(
    x: np.ndarray,
    edges: np.ndarray,
    bin_index: int,
    *,
    require_finite: bool = False,
) -> np.ndarray:
    """Boolean mask: *x* lies in histogram bin *bin_index* for *edges*.

    Uses the same cell convention as :func:`numpy.histogram` with fixed
    *edges*: interior bins are half-open ``[lo, hi)``, the **last** bin is
    ``[lo, hi]`` closed on both ends. This avoids dropping events exactly at
    the minimum quantile (the old ``(lo, hi]`` convention excluded ``x == lo``).

    When *require_finite* is True, non-finite *x* never match any bin.
    """
    edges = np.asarray(edges, dtype=float)
    n_bins = len(edges) - 1
    if n_bins < 1:
        raise ValueError("edges must contain at least two points")
    if bin_index < 0 or bin_index >= n_bins:
        raise IndexError(f"bin_index {bin_index} out of range for {n_bins} bins")

    if require_finite:
        ok = np.isfinite(x)
    else:
        ok = np.ones(x.shape, dtype=bool)

    lo, hi = float(edges[bin_index]), float(edges[bin_index + 1])
    if n_bins == 1:
        in_bin = (x >= lo) & (x <= hi)
    elif bin_index == n_bins - 1:
        in_bin = (x >= lo) & (x <= hi)
    else:
        in_bin = (x >= lo) & (x < hi)
    return ok & in_bin


def _pion_kinematic_bin_mask(
    data: dict,
    *,
    kind: str,
    bin_index: int,
    edges: np.ndarray,
    require_has_pion: bool,
) -> np.ndarray:
    """MC pion E or θ in histogram bin *bin_index*, optionally ``has_pion``."""
    if kind == "E":
        x = data["pion_E_MC"]
        req_fin = False
    elif kind == "theta":
        x = data["pion_theta_MC"]
        req_fin = True
    else:
        raise ValueError(f"kind must be 'E' or 'theta', got {kind!r}")
    bm = mc_value_in_bin(x, edges, bin_index, require_finite=req_fin)
    if require_has_pion:
        bm = bm & data["has_pion"]
    return bm


def get_signal_probabilities(
    result: dict,
    signal_classes: list[int],
    playlist: str = "1A",
) -> dict[str, np.ndarray]:
    """Compute binary y_true / y_pred from a single run result dict."""
    pid = result[playlist]["pid"]
    signal_set = set(signal_classes)
    y_true = np.array([1 if x in signal_set else 0 for x in pid])
    probs = result[playlist]["prediction"]
    y_pred = sum(probs[:, c] for c in signal_classes)
    # convert nans in y_pred to 0
    y_pred = np.nan_to_num(y_pred, 0.0)
    return {"ytrue": y_true, "ypred": y_pred}


def bin_separation_metrics(
    bin_mask: np.ndarray,
    is_signal: np.ndarray,
    is_background: np.ndarray,
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
) -> dict | None:
    """Signal in *bin_mask* vs **the full background sample**: AUPRC, AUROC, TPR@FPR.

    Positives are true signal events that pass *bin_mask* (e.g. in a kinematic
    bin). Negatives are **every** background event, not only those in the bin.
    So the **same** background set — and the same number of negative examples —
    is used for each bin’s ROC/AUPRC; only the in-bin signal positives change.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    sig_in_bin = is_signal & bin_mask
    n_sig = sig_in_bin.sum()
    n_bg = is_background.sum()
    if n_sig == 0 or n_bg == 0:
        return None

    eval_mask = sig_in_bin | is_background
    y_eval = y_true[eval_mask]
    p_eval = probs[eval_mask]

    # Skip NaNs (sklearn does not accept them)
    valid = ~(np.isnan(y_eval) | np.isnan(p_eval))
    y_eval = y_eval[valid]
    p_eval = p_eval[valid]
    n_sig = (y_eval == 1).sum()
    n_bg = (y_eval == 0).sum()
    if n_sig == 0 or n_bg == 0:
        return None

    prec, rec, _ = precision_recall_curve(y_eval, p_eval)
    auprc_val = auc(rec, prec)
    fpr_arr, tpr_arr, _ = roc_curve(y_eval, p_eval)
    auroc_val = auc(fpr_arr, tpr_arr)

    selected = p_eval > threshold
    tp = (selected & (y_eval == 1)).sum()

    tpr_at_fpr = {}
    for target_fpr in fixed_fpr:
        idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
        idx = max(0, min(idx, len(tpr_arr) - 1))
        tpr_at_fpr[target_fpr] = float(tpr_arr[idx])

    return {
        "auprc": auprc_val,
        "auroc": auroc_val,
        "efficiency": tp / n_sig if n_sig > 0 else 0.0,
        "purity": tp / selected.sum() if selected.sum() > 0 else 0.0,
        "n_signal": int(n_sig),
        "tpr_at_fpr": tpr_at_fpr,
    }


def compute_binned_metrics_single(
    result: dict,
    data: dict,
    signal_classes: list[int],
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
    pion_bins_require_has_pion: bool = True,
) -> dict[str, list[dict | None]]:
    """Per-bin metrics for pion E and pion theta (single run).

    If ``pion_bins_require_has_pion`` is False, every event can land in an
    E bin (by ``pion_E_MC``); θ bins use finite ``pion_theta_MC`` only.
    Binary signal/background is unchanged (all non-signal remain background).
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR

    sig = get_signal_probabilities(result, signal_classes, playlist)
    y_true, probs = sig["ytrue"], sig["ypred"]
    is_signal = y_true == 1
    is_background = y_true == 0

    if event_mask is not None:
        is_signal = is_signal & event_mask
        is_background = is_background & event_mask

    E_bins = data["pion_E_MC_bins"]
    theta_bins = data["pion_theta_MC_bins"]

    metrics_E = []

    for i in range(len(E_bins) - 1):
        bm = _pion_kinematic_bin_mask(
            data,
            kind="E",
            bin_index=i,
            edges=E_bins,
            require_has_pion=pion_bins_require_has_pion,
        )
        metrics_E.append(
            bin_separation_metrics(bm, is_signal, is_background, y_true, probs, threshold, fixed_fpr)
        )

    metrics_theta = []
    for i in range(len(theta_bins) - 1):
        bm = _pion_kinematic_bin_mask(
            data,
            kind="theta",
            bin_index=i,
            edges=theta_bins,
            require_has_pion=pion_bins_require_has_pion,
        )
        metrics_theta.append(
            bin_separation_metrics(bm, is_signal, is_background, y_true, probs, threshold, fixed_fpr)
        )

    return {"E": metrics_E, "theta": metrics_theta}


def compute_binned_metrics_q3(
    result: dict,
    data: dict,
    signal_classes: list[int],
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
) -> list[dict | None]:
    """Per-q3-bin metrics (single run)."""
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR

    sig = get_signal_probabilities(result, signal_classes, playlist)
    y_true, probs = sig["ytrue"], sig["ypred"]

    q3 = data["q3_GeV"]
    edges = data["q3_bin_edges"]

    metrics = []
    for i in range(len(edges) - 1):
        bm = mc_value_in_bin(q3, edges, i, require_finite=False)
        if event_mask is not None:
            bm = bm & event_mask
        y_bin = y_true[bm]
        p_bin = probs[bm]
        valid = ~(np.isnan(y_bin) | np.isnan(p_bin))
        y_bin = y_bin[valid]
        p_bin = p_bin[valid]
        n_sig = (y_bin == 1).sum()
        n_bg = (y_bin == 0).sum()
        if n_sig == 0 or n_bg == 0:
            metrics.append(None)
        else:
            prec, rec, _ = precision_recall_curve(y_bin, p_bin)
            auprc_val = auc(rec, prec)
            fpr_arr, tpr_arr, _ = roc_curve(y_bin, p_bin)
            auroc_val = auc(fpr_arr, tpr_arr)
            tpr_at_fpr = {}
            for target_fpr in fixed_fpr:
                idx = np.searchsorted(fpr_arr, target_fpr, side="right") - 1
                idx = max(0, min(idx, len(tpr_arr) - 1))
                tpr_at_fpr[target_fpr] = float(tpr_arr[idx])
            metrics.append({
                "auprc": auprc_val,
                "auroc": auroc_val,
                "n_signal": int(n_sig),
                "tpr_at_fpr": tpr_at_fpr,
            })
    return metrics


# ---------------------------------------------------------------------------
# Multi-run aggregation
# ---------------------------------------------------------------------------

def _extract_metric_array(
    runs: list[list[dict | None]],
    key: str,
    fpr_value: float | None = None,
) -> np.ndarray:
    """Build (n_runs, n_bins) array from a list of per-run bin-metric lists."""
    n_runs = len(runs)
    n_bins = len(runs[0])
    arr = np.full((n_runs, n_bins), np.nan)
    for r in range(n_runs):
        for b in range(n_bins):
            m = runs[r][b]
            if m is not None:
                if key == "tpr_at_fpr" and fpr_value is not None:
                    arr[r, b] = m["tpr_at_fpr"].get(fpr_value, np.nan)
                else:
                    arr[r, b] = m[key]
    return arr


def aggregate_metrics(
    runs: list[list[dict | None]],
    fixed_fpr: list[float] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate per-bin metrics across multiple runs.

    Parameters
    ----------
    runs : list (one per run) of lists (one per bin) of metric dicts.

    Returns
    -------
    ``{metric_key: {"mean": array, "std": array}}`` for auprc, auroc,
    and each tpr_at_fpr entry.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR

    out: dict[str, dict[str, np.ndarray]] = {}
    for key in ("auprc", "auroc"):
        arr = _extract_metric_array(runs, key)
        out[key] = {"mean": np.nanmean(arr, axis=0), "std": np.nanstd(arr, axis=0)}

    for fpr_val in fixed_fpr:
        arr = _extract_metric_array(runs, "tpr_at_fpr", fpr_val)
        out[f"tpr@{fpr_val}"] = {"mean": np.nanmean(arr, axis=0), "std": np.nanstd(arr, axis=0)}

    return out


# ---------------------------------------------------------------------------
# High-level metric computation helpers
# ---------------------------------------------------------------------------

def compute_all_metrics(
    results: dict[str, list[dict]],
    data: dict,
    signal_classes: list[int],
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
    pion_bins_require_has_pion: bool = True,
) -> dict[str, dict]:
    """Compute aggregated pion-E/theta metrics for all models.

    Returns ``{model_name: {"E": agg, "theta": agg}}`` where each ``agg``
    is the output of :func:`aggregate_metrics`.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    out = {}
    for model_name, run_list in sorted(results.items(), key=lambda kv: kv[0]):
        runs_E, runs_theta = [], []
        for run_result in run_list:
            m = compute_binned_metrics_single(
                run_result,
                data,
                signal_classes,
                threshold,
                fixed_fpr,
                event_mask,
                playlist,
                pion_bins_require_has_pion=pion_bins_require_has_pion,
            )
            runs_E.append(m["E"])
            runs_theta.append(m["theta"])
        out[model_name] = {
            "E": aggregate_metrics(runs_E, fixed_fpr),
            "theta": aggregate_metrics(runs_theta, fixed_fpr),
        }
    return out


def compute_all_metrics_q3(
    results: dict[str, list[dict]],
    data: dict,
    signal_classes: list[int],
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
) -> dict[str, dict]:
    """Compute aggregated q3-binned metrics for all models.

    Returns ``{model_name: agg}`` where ``agg`` is the output of
    :func:`aggregate_metrics`.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    out = {}
    for model_name, run_list in sorted(results.items(), key=lambda kv: kv[0]):
        runs = []
        for run_result in run_list:
            runs.append(
                compute_binned_metrics_q3(
                    run_result, data, signal_classes, threshold, fixed_fpr, event_mask, playlist,
                )
            )
        out[model_name] = aggregate_metrics(runs, fixed_fpr)
    return out


def compute_signal_baseline(
    results: dict[str, list[dict]],
    data: dict,
    signal_classes: list[int],
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
    pion_bins_require_has_pion: bool = True,
) -> dict[str, np.ndarray]:
    """Compute random-classifier baseline (signal fraction) per bin.

    Returns dict with keys ``"E"``, ``"theta"``, ``"q3"``.
    """
    first_model = next(iter(results))
    first_run = results[first_model][0]
    sig = get_signal_probabilities(first_run, signal_classes, playlist)
    y_true = sig["ytrue"]
    is_signal = y_true == 1
    is_background = y_true == 0

    if event_mask is not None:
        is_signal = is_signal & event_mask
        is_background = is_background & event_mask

    n_bg = is_background.sum()

    # Pion E bins
    E_bins = data["pion_E_MC_bins"]
    baseline_E = []
    for i in range(len(E_bins) - 1):
        bm = _pion_kinematic_bin_mask(
            data,
            kind="E",
            bin_index=i,
            edges=E_bins,
            require_has_pion=pion_bins_require_has_pion,
        )
        n_sig = (is_signal & bm).sum()
        baseline_E.append(n_sig / (n_sig + n_bg) if n_sig > 0 else np.nan)

    # Pion theta bins
    theta_bins = data["pion_theta_MC_bins"]
    baseline_theta = []
    for i in range(len(theta_bins) - 1):
        bm = _pion_kinematic_bin_mask(
            data,
            kind="theta",
            bin_index=i,
            edges=theta_bins,
            require_has_pion=pion_bins_require_has_pion,
        )
        n_sig = (is_signal & bm).sum()
        baseline_theta.append(n_sig / (n_sig + n_bg) if n_sig > 0 else np.nan)

    # q3 bins
    q3 = data["q3_GeV"]
    q3_edges = data["q3_bin_edges"]
    baseline_q3 = []
    for i in range(len(q3_edges) - 1):
        bm = mc_value_in_bin(q3, q3_edges, i, require_finite=False)
        if event_mask is not None:
            bm = bm & event_mask
        n_sig = (y_true[bm] == 1).sum() if bm.sum() > 0 else 0
        baseline_q3.append(y_true[bm].mean() if n_sig > 0 else np.nan)

    return {
        "E": np.array(baseline_E),
        "theta": np.array(baseline_theta),
        "q3": np.array(baseline_q3),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

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


def _plot_metric_line(
    ax: plt.Axes,
    x: np.ndarray,
    agg: dict[str, np.ndarray],
    label: str,
    uncertainties: bool,
    **kwargs,
) -> None:
    """Plot mean line with optional +/-1 std band."""
    mean = agg["mean"]
    line, = ax.plot(x, mean, "o-", label=label, **kwargs)
    if uncertainties and len(agg["std"]) > 0:
        std = agg["std"]
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=line.get_color())


def _format_axes_grid(
    axes: np.ndarray,
    n_rows: int,
    col_labels: list[str],
    xlabel: str,
    row_titles: list[str] | None = None,
    log_x: bool = False,
    fixed_fpr: list[float] | None = None,
) -> None:
    """Apply labels, titles, legend and grid to an (n_rows, 3) axes array."""
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    for row in range(n_rows):
        for col in range(3):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(col_labels[col])
            title_prefix = f"{row_titles[row]} — " if row_titles else ""
            if col < 2:
                ax.set_title(f"{title_prefix}{col_labels[col]} vs. {xlabel.split('[')[0].strip()}")
            else:
                ax.set_title(f"{title_prefix}TPR @ fixed FPR vs. {xlabel.split('[')[0].strip()}")
            ax.legend(fontsize=7)
            ax.grid(True)
            if log_x:
                ax.set_xscale("log")


# ---------------------------------------------------------------------------
# Top-level plotting functions
# ---------------------------------------------------------------------------

def plot_cc1pi_vs_pion_kinematics(
    all_metrics: dict[str, dict],
    data: dict,
    baseline: dict[str, np.ndarray],
    fixed_fpr: list[float] | None = None,
    uncertainties: bool = False,
    reco_baseline_tpr: dict[str, np.ndarray] | None = None,
    reco_baseline_label: str = "Reco baseline",
    colors: dict[str, str] | None = None,
) -> plt.Figure:
    """2x3 figure: pion E (top row) and pion theta (bottom row).

    Columns: AUPRC, AUROC, TPR@FPR.

    Parameters
    ----------
    reco_baseline_tpr : optional dict with keys ``"E"`` and ``"theta"``,
        each a per-bin recall array for a reconstruction-level baseline.
        Plotted on the rightmost (TPR@FPR) panels.
    reco_baseline_label : label for the reco baseline in the legend.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), tight_layout=True)

    E_mid = data["pion_E_MC_bins_mid"]
    theta_mid = data["pion_theta_MC_bins_mid"]

    # Random baseline
    axes[0, 0].plot(E_mid, baseline["E"], ".--", color="black", label="Random baseline")
    axes[1, 0].plot(theta_mid, baseline["theta"], ".--", color="black", label="Random baseline")

    for model_name, metrics in sorted(all_metrics.items(), key=lambda kv: kv[0]):
        agg_E = metrics["E"]
        agg_theta = metrics["theta"]
        clr = {} if colors is None else {"color": colors.get(model_name)}

        _plot_metric_line(axes[0, 0], E_mid, agg_E["auprc"], model_name, uncertainties, **clr)
        _plot_metric_line(axes[1, 0], theta_mid, agg_theta["auprc"], model_name, uncertainties, **clr)
        _plot_metric_line(axes[0, 1], E_mid, agg_E["auroc"], model_name, uncertainties, **clr)
        _plot_metric_line(axes[1, 1], theta_mid, agg_theta["auroc"], model_name, uncertainties, **clr)

        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            _plot_metric_line(
                axes[0, 2], E_mid, agg_E[key],
                f"{model_name} (FPR={fpr_val:.0%})", uncertainties, **clr,
            )
            _plot_metric_line(
                axes[1, 2], theta_mid, agg_theta[key],
                f"{model_name} (FPR={fpr_val:.0%})", uncertainties, **clr,
            )

    if reco_baseline_tpr is not None:
        if "E" in reco_baseline_tpr:
            axes[0, 2].plot(E_mid, reco_baseline_tpr["E"], "s--", color="black",
                            label=reco_baseline_label)
        if "theta" in reco_baseline_tpr:
            axes[1, 2].plot(theta_mid, reco_baseline_tpr["theta"], "s--", color="black",
                            label=reco_baseline_label)

    col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
    for row, kinematic in enumerate([r"True $E_\pi$ [GeV]", r"$\theta_\pi$ [rad]"]):
        for col, metric in enumerate(col_labels):
            ax = axes[row, col]
            ax.set_xlabel(kinematic)
            ax.set_ylabel(metric)
            if col < 2:
                ax.set_title(f"{metric} vs. {kinematic.split('[')[0].strip()}")
            else:
                ax.set_title(f"TPR @ fixed FPR vs. {kinematic.split('[')[0].strip()}")
            ax.legend(fontsize=7)
            ax.grid(True)
            if row == 0:
                ax.set_xlim(E_mid[0] * 0.8, E_mid[-1] * 1.2)
                ax.set_xscale("log")

    fig.suptitle(r"$CC1\pi^\pm$ event tagging", fontsize=14)
    return fig


def plot_multi_pion_vs_q3(
    all_metrics_q3: dict[str, dict],
    data: dict,
    baseline_q3: np.ndarray,
    fixed_fpr: list[float] | None = None,
    uncertainties: bool = False,
    reco_baseline_tpr_q3: np.ndarray | None = None,
    reco_baseline_label: str = "Reco baseline",
    colors: dict[str, str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """1x3 figure: AUPRC / AUROC / TPR@FPR vs q3.

    Parameters
    ----------
    reco_baseline_tpr_q3 : optional per-bin recall array for a
        reconstruction-level baseline.  Plotted on the rightmost
        (TPR@FPR) panel.
    reco_baseline_label : label for the reco baseline in the legend.
    title : optional figure super-title.  Defaults to a multi-pion
        description when *None*.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), tight_layout=True)

    q3_mid = data["q3_bin_mids"]

    axes[0].plot(q3_mid, baseline_q3, ".--", color="black", label="Random baseline")

    for model_name, agg in sorted(all_metrics_q3.items(), key=lambda kv: kv[0]):
        clr = {} if colors is None else {"color": colors.get(model_name)}
        _plot_metric_line(axes[0], q3_mid, agg["auprc"], model_name, uncertainties, **clr)
        _plot_metric_line(axes[1], q3_mid, agg["auroc"], model_name, uncertainties, **clr)
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            _plot_metric_line(
                axes[2], q3_mid, agg[key],
                f"{model_name} (FPR={fpr_val:.0%})", uncertainties, **clr,
            )

    if reco_baseline_tpr_q3 is not None:
        axes[2].plot(q3_mid, reco_baseline_tpr_q3, "s--", color="black",
                     label=reco_baseline_label)

    col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
    for col, metric in enumerate(col_labels):
        ax = axes[col]
        ax.set_xlabel(r"$q_{3}^{\mathrm{true}}$ [GeV]")
        ax.set_ylabel(metric)
        if col < 2:
            ax.set_title(f"{metric} vs. $q_3$")
        else:
            ax.set_title(r"TPR @ fixed FPR vs. $q_3$")
        ax.legend(fontsize=7)
        ax.grid(True)

    if title is None:
        title = r"Multi-pion tagging (one or more charged pions) vs. $q_{3}^{\mathrm{true}}$"
    fig.suptitle(title, fontsize=14)
    return fig


def plot_binned_by_inttype(
    results: dict[str, list[dict]],
    data: dict,
    signal_classes: list[int],
    x_var: str,
    xlabel: str,
    title: str,
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    log_x: bool = False,
    uncertainties: bool = False,
    int_types: dict[int, str] | None = None,
    playlist: str = "1A",
    reco_baseline_pred: np.ndarray | None = None,
    reco_baseline_label: str = "Reco baseline",
    colors: dict[str, str] | None = None,
    signal_label: str | None = None,
    pion_bins_require_has_pion: bool = True,
) -> plt.Figure:
    """One row per interaction type, 4 columns: AUPRC, AUROC, TPR@FPR,
    and an event-count histogram.

    Parameters
    ----------
    x_var : ``"pion_E"``, ``"pion_theta"``, or ``"q3"``.
    reco_baseline_pred : optional binary prediction array (same length as
        test set). When provided, the per-bin recall is overlaid on the
        TPR@FPR panel for each interaction type.
    reco_baseline_label : legend label for the reco baseline.
    signal_label : optional name for the signal class definition (e.g.
        ``r"$CC\\pi^0$"``). Used when there are events in an interaction
        type but no signal positives; defaults from *signal_classes*.
    pion_bins_require_has_pion : if False, pion E/θ histograms
        and binned metrics include all events (θ requires finite MC angle).
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    if int_types is None:
        int_types = MC_INT_TYPE

    int_type_arr = data["int_type_arr"]
    n_int = len(int_types)
    fig, axes = plt.subplots(n_int, 4, figsize=(22, 4.5 * n_int), tight_layout=True)
    if n_int == 1:
        axes = axes[np.newaxis, :]

    first_model = next(iter(results))
    y_true_binary = get_signal_probabilities(
        results[first_model][0], signal_classes, playlist
    )["ytrue"]
    label_for_signal = signal_label if signal_label is not None else _default_signal_label(
        signal_classes
    )

    has_pion = data["has_pion"]

    # Resolve bin edges for the histogram column
    if x_var == "q3":
        hist_var = data["q3_GeV"]
        bin_edges = data["q3_bin_edges"]
        hist_pion_mask = np.ones(len(hist_var), dtype=bool)
    elif x_var == "pion_E":
        hist_var = data["pion_E_MC"]
        bin_edges = data["pion_E_MC_bins"]
        hist_pion_mask = has_pion if pion_bins_require_has_pion else np.ones(len(hist_var), dtype=bool)
    elif x_var == "pion_theta":
        hist_var = data["pion_theta_MC"]
        bin_edges = data["pion_theta_MC_bins"]
        if pion_bins_require_has_pion:
            hist_pion_mask = has_pion
        else:
            hist_pion_mask = np.isfinite(hist_var)
    else:
        raise ValueError(f"Unknown x_var: {x_var}")

    for row_idx, (int_code, int_name) in enumerate(int_types.items()):
        int_mask = int_type_arr == int_code

        # Count events actually entering the plots
        plot_mask = int_mask & hist_pion_mask
        n_events = int(plot_mask.sum())

        # Choose x-axis midpoints
        if x_var == "q3":
            x_mid = data["q3_bin_mids"]
        elif x_var == "pion_E":
            x_mid = data["pion_E_MC_bins_mid"]
        else:
            x_mid = data["pion_theta_MC_bins_mid"]

        if n_events == 0:
            for col in range(4):
                axes[row_idx, col].text(
                    0.5, 0.5, "No data", transform=axes[row_idx, col].transAxes,
                    ha="center", va="center", fontsize=14, color="gray",
                )
                axes[row_idx, col].set_title(f"{int_name} (N=0)")
            continue

        n_signal = int(((y_true_binary == 1) & int_mask).sum())
        no_signal_msg = (
            f"No {label_for_signal} signal in this interaction type"
        )

        if n_signal == 0:
            for col in (0, 1, 2):
                ax = axes[row_idx, col]
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    no_signal_msg,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="gray",
                )
                ax.set_title(f"{int_name} (N={n_events:,})")
            # Histogram still shows kinematic counts for this interaction type
            ax_h = axes[row_idx, 3]
            counts, _ = np.histogram(hist_var[plot_mask], bins=bin_edges)
            widths = np.diff(bin_edges)
            ax_h.bar(
                bin_edges[:-1],
                counts,
                width=widths,
                align="edge",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            for i, c in enumerate(counts):
                if c > 0:
                    ax_h.text(
                        bin_edges[i] + widths[i] / 2,
                        c,
                        str(c),
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )
            ax_h.set_xlabel(xlabel)
            ax_h.set_ylabel("Events")
            ax_h.set_title(f"{int_name} (N={n_events:,}) — event counts")
            ax_h.grid(True, axis="y", alpha=0.3)
            if log_x:
                ax_h.set_xscale("log")
            continue

        # Compute baseline
        bl = compute_signal_baseline(
            results,
            data,
            signal_classes,
            int_mask,
            playlist,
            pion_bins_require_has_pion=pion_bins_require_has_pion,
        )

        if x_var == "q3":
            bl_values = bl["q3"]
            all_agg = compute_all_metrics_q3(
                results, data, signal_classes, threshold, fixed_fpr, int_mask, playlist,
            )
        else:
            bl_values = bl[{"pion_E": "E", "pion_theta": "theta"}[x_var]]
            all_agg_full = compute_all_metrics(
                results,
                data,
                signal_classes,
                threshold,
                fixed_fpr,
                int_mask,
                playlist,
                pion_bins_require_has_pion=pion_bins_require_has_pion,
            )
            sub_key = {"pion_E": "E", "pion_theta": "theta"}[x_var]
            all_agg = {mn: m[sub_key] for mn, m in all_agg_full.items()}

        # Random baseline
        axes[row_idx, 0].plot(x_mid, bl_values, ".--", color="black", label="Random baseline")

        for model_name, agg in sorted(all_agg.items(), key=lambda kv: kv[0]):
            clr = {} if colors is None else {"color": colors.get(model_name)}
            _plot_metric_line(axes[row_idx, 0], x_mid, agg["auprc"], model_name, uncertainties, **clr)
            _plot_metric_line(axes[row_idx, 1], x_mid, agg["auroc"], model_name, uncertainties, **clr)
            for fpr_val in fixed_fpr:
                key = f"tpr@{fpr_val}"
                _plot_metric_line(
                    axes[row_idx, 2], x_mid, agg[key],
                    f"{model_name} (FPR={fpr_val:.0%})", uncertainties, **clr,
                )

        # Reco baseline on TPR panel
        if reco_baseline_pred is not None:
            is_signal_masked = (y_true_binary == 1) & int_mask
            if x_var == "q3":
                reco_bl = compute_reco_baseline_recall_per_bin(
                    reco_baseline_pred, is_signal_masked,
                    data["q3_GeV"], data["q3_bin_edges"],
                )
            else:
                var_key = {"pion_E": "pion_E_MC", "pion_theta": "pion_theta_MC"}[x_var]
                edges_key = {"pion_E": "pion_E_MC_bins", "pion_theta": "pion_theta_MC_bins"}[x_var]
                reco_bl = compute_reco_baseline_recall_per_bin(
                    reco_baseline_pred,
                    is_signal_masked,
                    data[var_key],
                    data[edges_key],
                    has_pion=data["has_pion"] if pion_bins_require_has_pion else None,
                )
            axes[row_idx, 2].plot(x_mid, reco_bl, "s--", color="black",
                                  label=reco_baseline_label)

        # --- Metric columns ---
        col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
        for col, metric in enumerate(col_labels):
            ax = axes[row_idx, col]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            if col < 2:
                ax.set_title(f"{int_name} (N={n_events:,}) — {metric} vs. {xlabel.split('[')[0].strip()}")
            else:
                ax.set_title(f"{int_name} (N={n_events:,}) — TPR @ fixed FPR vs. {xlabel.split('[')[0].strip()}")
            ax.legend(fontsize=7)
            ax.grid(True)
            if log_x:
                ax.set_xlim(x_mid[0] * 0.8, x_mid[-1] * 1.2)
                ax.set_xscale("log")

        # --- Histogram column (col 3) ---
        ax_h = axes[row_idx, 3]
        counts, _ = np.histogram(hist_var[plot_mask], bins=bin_edges)
        widths = np.diff(bin_edges)
        ax_h.bar(bin_edges[:-1], counts, width=widths, align="edge",
                 edgecolor="black", linewidth=0.5, alpha=0.7)
        for i, c in enumerate(counts):
            if c > 0:
                ax_h.text(bin_edges[i] + widths[i] / 2, c, str(c),
                          ha="center", va="bottom", fontsize=7)
        ax_h.set_xlabel(xlabel)
        ax_h.set_ylabel("Events")
        ax_h.set_title(f"{int_name} (N={n_events:,}) — event counts")
        ax_h.grid(True, axis="y", alpha=0.3)
        if log_x:
            ax_h.set_xscale("log")

    fig.suptitle(title, fontsize=14, y=1.005)
    return fig


def plot_event_counts_by_inttype(
    data: dict,
    x_var: str,
    xlabel: str,
    title: str,
    log_x: bool = False,
    int_types: dict[int, str] | None = None,
    pion_bins_require_has_pion: bool = True,
) -> plt.Figure:
    """Histogram of event counts per bin, one row per interaction type.

    The bar heights in each row sum to the total N for that interaction
    type (shown in the row title).

    Parameters
    ----------
    x_var : ``"pion_E"``, ``"pion_theta"``, or ``"q3"``.
    pion_bins_require_has_pion : same meaning as in :func:`plot_binned_by_inttype`.
    """
    if int_types is None:
        int_types = MC_INT_TYPE

    int_type_arr = data["int_type_arr"]

    if x_var == "q3":
        var = data["q3_GeV"]
        bin_edges = data["q3_bin_edges"]
    elif x_var == "pion_E":
        var = data["pion_E_MC"]
        bin_edges = data["pion_E_MC_bins"]
    elif x_var == "pion_theta":
        var = data["pion_theta_MC"]
        bin_edges = data["pion_theta_MC_bins"]
    else:
        raise ValueError(f"Unknown x_var: {x_var}")

    if x_var == "q3":
        kin_mask = np.ones(len(var), dtype=bool)
    elif x_var == "pion_E":
        kin_mask = data["has_pion"] if pion_bins_require_has_pion else np.ones(len(var), dtype=bool)
    else:
        kin_mask = data["has_pion"] if pion_bins_require_has_pion else np.isfinite(var)

    n_int = len(int_types)
    fig, axes = plt.subplots(n_int, 1, figsize=(8, 3.0 * n_int), tight_layout=True)
    if n_int == 1:
        axes = np.array([axes])

    for row_idx, (int_code, int_name) in enumerate(int_types.items()):
        ax = axes[row_idx]
        int_mask = int_type_arr == int_code

        mask = int_mask & kin_mask
        counts, _ = np.histogram(var[mask], bins=bin_edges)
        n_plotted = int(counts.sum())

        widths = np.diff(bin_edges)
        ax.bar(bin_edges[:-1], counts, width=widths, align="edge",
               edgecolor="black", linewidth=0.5, alpha=0.7)

        for i, c in enumerate(counts):
            if c > 0:
                ax.text(
                    bin_edges[i] + widths[i] / 2, c, str(c),
                    ha="center", va="bottom", fontsize=7,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Events")
        ax.set_title(f"{int_name} (N={n_plotted:,})")
        ax.grid(True, axis="y", alpha=0.3)
        if log_x:
            ax.set_xscale("log")

    fig.suptitle(title, fontsize=14, y=1.005)
    return fig


# ---------------------------------------------------------------------------
# PRC curves
# ---------------------------------------------------------------------------

def _compute_prc_curves(
    results: dict[str, list[dict]],
    signal_classes: list[int],
    playlist: str = "1A",
    recall_grid: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute interpolated PRC curves for all models/runs.

    Returns ``{model: {"precision_mean", "precision_std", "recall",
    "auprc_mean", "auprc_std"}}``.
    """
    if recall_grid is None:
        recall_grid = np.linspace(0, 1, 200)

    out = {}
    for model_name, run_list in sorted(results.items(), key=lambda kv: kv[0]):
        precisions_interp = []
        thresholds_interp = []
        auprcs = []
        for run_result in run_list:
            sig = get_signal_probabilities(run_result, signal_classes, playlist)
            prec, rec, thresholds = precision_recall_curve(sig["ytrue"], sig["ypred"])
            auprc_val = auc(rec, prec)
            auprcs.append(auprc_val)
            prec_interp = np.interp(recall_grid, rec[::-1], prec[::-1])
            precisions_interp.append(prec_interp)
            thresh_ext = np.append(thresholds, thresholds[-1])
            thresholds_interp.append(
                np.interp(recall_grid, rec[::-1], thresh_ext[::-1])
            )

        prec_arr = np.array(precisions_interp)
        thresh_arr = np.array(thresholds_interp)
        out[model_name] = {
            "precision_mean": np.mean(prec_arr, axis=0),
            "precision_std": np.std(prec_arr, axis=0),
            "recall": recall_grid,
            "threshold_mean": np.mean(thresh_arr, axis=0),
            "auprc_mean": np.mean(auprcs),
            "auprc_std": np.std(auprcs),
        }
    return out


def plot_prc_curves(
    results: dict[str, list[dict]],
    signal_classes: list[int],
    title: str = "Precision-Recall Curve",
    playlist: str = "1A",
    uncertainties: bool = False,
    max_threshold: float | None = None,
    colors: dict[str, str] | None = None,
) -> plt.Figure:
    """Plot PRC curves for all models with optional uncertainty bands.

    Parameters
    ----------
    max_threshold : if set, the right (log-scale) plot only shows the
        portion of each curve where the mean classification threshold is
        below this value.
    """
    curves = _compute_prc_curves(results, signal_classes, playlist)

    first_model = next(iter(results))
    sig = get_signal_probabilities(results[first_model][0], signal_classes, playlist)
    signal_frac = sig["ytrue"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), tight_layout=True)

    for ax in axes:
        ax.axhline(signal_frac, color="black", linestyle="--", linewidth=1,
                    label=f"Random baseline ({signal_frac:.1%} signal)")

    for model_name, c in sorted(curves.items(), key=lambda kv: kv[0]):
        rec = c["recall"]
        prec_mean = c["precision_mean"]
        prec_std = c["precision_std"]
        auprc_m = c["auprc_mean"]
        auprc_s = c["auprc_std"]
        if uncertainties and auprc_s > 0:
            label = f"{model_name} (AUPRC={auprc_m:.3f}±{auprc_s:.3f})"
        else:
            label = f"{model_name} (AUPRC={auprc_m:.3f})"
        clr = {} if colors is None else {"color": colors.get(model_name)}

        for ax_idx, ax in enumerate(axes):
            if ax_idx == 1 and max_threshold is not None:
                mask = c["threshold_mean"] < max_threshold
                r, p, s = rec[mask], prec_mean[mask], prec_std[mask]
            else:
                r, p, s = rec, prec_mean, prec_std

            line, = ax.plot(r, p, "-", label=label, **clr)
            if uncertainties:
                ax.fill_between(r, p - s, p + s, alpha=0.2, color=line.get_color())

    for ax, scale in zip(axes, ["linear", "log"]):
        ax.set_xlabel(r"Recall (TPR)")
        ax.set_ylabel(r"Precision (purity)")
        ax.set_title(f"{title} ({scale} scale)")
        ax.legend(fontsize=9)
        ax.grid(True)
        ax.set_xlim(0, 1)
        if scale == "log":
            ax.set_yscale("log")
            #ax.set_xscale("log")
            ax.set_ylim(bottom=signal_frac * 0.5, top=1.05)
        else:
            ax.set_ylim(0, 1)
    return fig


# ---------------------------------------------------------------------------
# Multi-figure PDF export
# ---------------------------------------------------------------------------

def save_figures_to_pdf(figures: list[plt.Figure], path: str | Path) -> None:
    """Save a list of matplotlib figures into a single multi-page PDF."""
    from matplotlib.backends.backend_pdf import PdfPages
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    print(f"Saved {len(figures)} page(s) to {path}")

