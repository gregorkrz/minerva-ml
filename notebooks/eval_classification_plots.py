"""
Evaluation plotting utilities for charged-pion classification models.

Loads evaluation results from checkpoint directories and produces
AUPRC / AUROC / TPR vs kinematic bins for pion energy, pion angle, true *q₃*,
or true MC hadronic invariant mass *W* (GeV) from baselines.  TPR at a target FPR can use either a **single
score threshold** fit on the full masked sample (``use_global_fpr=True``) or
**per-bin** ROC cuts (``use_global_fpr=False``).  Supports multi-run uncertainty bands.

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


def _tpr_column_title_vs_kinematics(use_global_fpr: bool) -> str:
    """Third-column title for kinematic-bin performance plots."""
    return "TPR @ fixed FPR" if use_global_fpr else "TPR @ per-bin FPR"


def _tpr_line_legend_label(model_name: str, _fpr_val: float, _use_global_fpr: bool) -> str:
    """Legend entry for a model TPR line (no FPR suffix; baseline uses :func:`_baseline_legend_with_global_fpr`)."""
    return model_name


def _global_reco_baseline_fpr(reco_pred: np.ndarray, y_true_binary: np.ndarray) -> float:
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


def _classification_legend_kw(fontsize: int, legend_title: str | None) -> dict[str, Any]:
    kw: dict[str, Any] = {"fontsize": fontsize}
    if legend_title:
        kw["title"] = legend_title
        kw["title_fontsize"] = 10 if fontsize >= 9 else 8
    return kw


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
        subdir = "baselines"
        candidate = data_path / subdir / baseline_file
        print("Baseline file: ", candidate)
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


# ---------------------------------------------------------------------------
# Hadronic invariant mass W (MC truth from baselines; optional reco-derived W)
# ---------------------------------------------------------------------------


def mc_true_hadronic_W_gev_from_baselines(
    baselines: dict[str, np.ndarray],
    test_idx: np.ndarray,
) -> np.ndarray:
    """True MC hadronic *W* (GeV) per test event from baseline ``mc_true_hadronic_W_GeV``.

    This array is written by ``extract_baselines.py`` using
    :func:`extract_baselines.true_hadronic_invariant_W_gev_from_mc_part`
    (sum of non-lepton final-state four-momenta; see that script).  Sentinel
    invalid values are ``-1``; they become ``nan`` here for binning and metrics.

    Parameters
    ----------
    baselines
        Dictionary loaded from ``*_enu_baselines.npz`` (must include the key
        ``mc_true_hadronic_W_GeV``).
    test_idx
        Indices of the test split (same convention as :func:`load_truth_and_baselines`).

    Raises
    ------
    KeyError
        If ``mc_true_hadronic_W_GeV`` is missing — regenerate baselines with
        ``src/scripts/extract_baselines.py``.
    """
    if "mc_true_hadronic_W_GeV" not in baselines:
        raise KeyError(
            "Baselines must contain 'mc_true_hadronic_W_GeV' (true MC hadronic W in GeV). "
            "Re-run src/scripts/extract_baselines.py to regenerate *_enu_baselines.npz files."
        )
    w = np.asarray(baselines["mc_true_hadronic_W_GeV"][test_idx], dtype=np.float64)
    return np.where((w < 0.0) | ~np.isfinite(w), np.nan, w)


def hadronic_invariant_W_gev_from_baselines(
    baselines: dict[str, np.ndarray],
    test_idx: np.ndarray,
) -> np.ndarray:
    """Reco-derived hadronic *W* (GeV) per test event from baseline kinematics (**not** MC truth).

    For **classification vs. true MC** *W*, use :func:`mc_true_hadronic_W_gev_from_baselines`
    via :func:`add_hadronic_W_to_classification_data` (default).  This function remains
    available for comparisons that use the lab-frame expression below.

    Uses the lab-frame expression

        ``W² = M_p² + 2 M_p E_recoil - 2 (E_μ + E_recoil) (E_μ - |p_μ| cos θ_μ) + m_μ²``,

    with all energies in MeV.  The combination ``(E_μ - |p_μ| cos θ_μ)`` is
    reconstructed from the same *q0*, *q3*, and MC incoming neutrino energy
    *E_true* stored in the baseline file as

        ``E_μ - |p_μ| cos θ_μ = (Q² + m_μ²) / (2 E_ν)``,

    where ``Q² = q₃² - q₀²`` (MeV²) with *q₃* the magnitude returned by
    ``extract_baselines.get_q3`` and ``q₀ = E_ν - E_μ``.

    ``E_recoil`` is ``MasterAnaDev_hadron_recoil`` (``E_recoil_only`` in the
    npz); invalid recoil rows (``< 0``) yield NaN for *W*.

    Parameters
    ----------
    baselines
        Dictionary loaded from ``*_enu_baselines.npz``.
    test_idx
        Indices of the test split (same convention as :func:`load_truth_and_baselines`).
    """
    E_mu = np.asarray(baselines["E_muon"][test_idx], dtype=np.float64)
    E_rec = np.asarray(baselines["E_recoil_only"][test_idx], dtype=np.float64)
    q0 = np.asarray(baselines["q0"][test_idx], dtype=np.float64)
    q3 = np.asarray(baselines["q3"][test_idx], dtype=np.float64)
    E_nu = np.asarray(baselines["E_true"][test_idx], dtype=np.float64)

    Mp, mm = PROTON_MASS_MEV, MUON_MASS_MEV
    Q2 = q3 * q3 - q0 * q0
    with np.errstate(divide="ignore", invalid="ignore"):
        emu_minus_pl = (Q2 + mm * mm) / (2.0 * E_nu)

    valid = (E_mu > 0) & (E_nu > 0) & np.isfinite(emu_minus_pl) & (E_rec >= 0)
    W2 = Mp * Mp + 2.0 * Mp * E_rec - 2.0 * (E_mu + E_rec) * emu_minus_pl + mm * mm
    W_gev = np.sqrt(np.maximum(W2, 0.0)) / 1000.0
    W_gev[~valid] = np.nan
    return W_gev


def add_hadronic_W_to_classification_data(
    data: dict[str, Any],
    playlist: str,
    w_bin_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Shallow copy of *data* with ``W_GeV``, ``W_bin_edges``, and ``W_bin_mids``.

    ``W_GeV`` is **true MC hadronic invariant mass** (GeV) from the baselines
    field ``mc_true_hadronic_W_GeV`` produced by ``extract_baselines.py`` — not
    the reco-derived lab-frame *W* from :func:`hadronic_invariant_W_gev_from_baselines`.

    Required keys: ``baselines``, ``test_idx`` (as returned by
    :func:`load_truth_and_baselines`).
    """
    if w_bin_edges is None:
        w_bin_edges = DEFAULT_W_BIN_EDGES_GEV.copy()
    else:
        w_bin_edges = _as_strictly_increasing_bin_edges(w_bin_edges, "w_bin_edges")

    out = dict(data)
    test_idx = data["test_idx"][playlist]
    bl = data["baselines"][playlist]
    out["W_GeV"] = mc_true_hadronic_W_gev_from_baselines(bl, test_idx)
    out["W_bin_edges"] = w_bin_edges
    out["W_bin_mids"] = (w_bin_edges[:-1] + w_bin_edges[1:]) / 2
    return out


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
    global_score_thresholds: dict[float, float] | None = None,
) -> dict | None:
    """Signal in *bin_mask* vs **the full background sample**: AUPRC, AUROC, TPR.

    Positives are true signal events that pass *bin_mask* (e.g. in a kinematic
    bin). Negatives are **every** background event, not only those in the bin.
    So the **same** background set — and the same number of negative examples —
    is used for each bin’s ROC/AUPRC; only the in-bin signal positives change.

    If ``global_score_thresholds`` is set (mapping target FPR → score cut fit
    on the **global** masked sample), ``tpr_at_fpr`` is the fraction of in-bin
    true signal with ``probs >= threshold`` (one cut for all bins). Otherwise
    ``tpr_at_fpr`` is read from the bin’s local ROC (legacy behaviour).
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
    if global_score_thresholds is not None:
        n_sig_bin = int(sig_in_bin.sum())
        if n_sig_bin == 0:
            for target_fpr in fixed_fpr:
                tpr_at_fpr[target_fpr] = float("nan")
        else:
            for target_fpr in fixed_fpr:
                t_cut = global_score_thresholds.get(target_fpr, float("nan"))
                if np.isnan(t_cut):
                    tpr_at_fpr[target_fpr] = float("nan")
                else:
                    passed = (probs >= t_cut) & sig_in_bin
                    tpr_at_fpr[target_fpr] = float(passed.sum() / n_sig_bin)
    else:
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
    use_global_fpr: bool = True,
) -> dict[str, list[dict | None]]:
    """Per-bin metrics for pion E and pion theta (single run).

    If ``pion_bins_require_has_pion`` is False, every event can land in an
    E bin (by ``pion_E_MC``); θ bins use finite ``pion_theta_MC`` only.
    Binary signal/background is unchanged (all non-signal remain background).

    If ``use_global_fpr`` is True, ``tpr_at_fpr`` uses one score cut per target
    FPR from the global (masked) ROC; if False, TPR is taken from each bin's
    local ROC at the target FPR (see :func:`bin_separation_metrics`).
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

    if event_mask is not None:
        m = event_mask
        y_glob, p_glob = y_true[m], probs[m]
    else:
        y_glob, p_glob = y_true, probs
    vg = ~np.isnan(p_glob)
    y_glob, p_glob = y_glob[vg], p_glob[vg]
    global_thr = (
        _global_score_thresholds_at_target_fprs(y_glob, p_glob, fixed_fpr)
        if use_global_fpr
        else None
    )

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
            bin_separation_metrics(
                bm,
                is_signal,
                is_background,
                y_true,
                probs,
                threshold,
                fixed_fpr,
                global_score_thresholds=global_thr,
            )
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
            bin_separation_metrics(
                bm,
                is_signal,
                is_background,
                y_true,
                probs,
                threshold,
                fixed_fpr,
                global_score_thresholds=global_thr,
            )
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
    use_global_fpr: bool = True,
) -> list[dict | None]:
    """Per-q3-bin metrics (single run).

    AUPRC/AUROC use the usual in-bin ROC.  If ``use_global_fpr`` is True,
    ``tpr_at_fpr`` uses one score threshold per target FPR fit on the **global**
    (optionally ``event_mask``'ed) sample, then reports in-bin signal efficiency
    at that cut.  If False, ``tpr_at_fpr`` is read from each bin's local ROC.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR

    sig = get_signal_probabilities(result, signal_classes, playlist)
    y_true, probs = sig["ytrue"], sig["ypred"]

    if event_mask is not None:
        y_glob, p_glob = y_true[event_mask], probs[event_mask]
    else:
        y_glob, p_glob = y_true, probs
    vg = ~np.isnan(p_glob)
    y_glob, p_glob = y_glob[vg], p_glob[vg]
    global_thr_map: dict[float, float] | None = (
        _global_score_thresholds_at_target_fprs(y_glob, p_glob, fixed_fpr)
        if use_global_fpr
        else None
    )

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
                if use_global_fpr:
                    t_cut = global_thr_map.get(target_fpr, float("nan")) if global_thr_map else float("nan")
                    if np.isnan(t_cut):
                        tpr_at_fpr[target_fpr] = float("nan")
                    else:
                        sig_mask = y_bin == 1
                        tpr_at_fpr[target_fpr] = float(
                            ((p_bin >= t_cut) & sig_mask).sum() / max(int(sig_mask.sum()), 1)
                        )
                else:
                    idx = int(np.searchsorted(fpr_arr, target_fpr, side="right") - 1)
                    idx = max(0, min(idx, len(tpr_arr) - 1))
                    tpr_at_fpr[target_fpr] = float(tpr_arr[idx])
            metrics.append({
                "auprc": auprc_val,
                "auroc": auroc_val,
                "n_signal": int(n_sig),
                "tpr_at_fpr": tpr_at_fpr,
            })
    return metrics


def compute_binned_metrics_W(
    result: dict,
    data: dict,
    signal_classes: list[int],
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
    use_global_fpr: bool = True,
) -> list[dict | None]:
    """Per-*W*-bin metrics (single run); same structure as :func:`compute_binned_metrics_q3`.

    AUPRC/AUROC use in-bin ROC.  ``tpr_at_fpr`` follows ``use_global_fpr`` like
    :func:`compute_binned_metrics_q3`.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR

    sig = get_signal_probabilities(result, signal_classes, playlist)
    y_true, probs = sig["ytrue"], sig["ypred"]

    if event_mask is not None:
        y_glob, p_glob = y_true[event_mask], probs[event_mask]
    else:
        y_glob, p_glob = y_true, probs
    vg = ~np.isnan(p_glob)
    y_glob, p_glob = y_glob[vg], p_glob[vg]
    global_thr_map: dict[float, float] | None = (
        _global_score_thresholds_at_target_fprs(y_glob, p_glob, fixed_fpr)
        if use_global_fpr
        else None
    )

    w = data["W_GeV"]
    edges = data["W_bin_edges"]

    metrics = []
    for i in range(len(edges) - 1):
        bm = mc_value_in_bin(w, edges, i, require_finite=False)
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
                if use_global_fpr:
                    t_cut = global_thr_map.get(target_fpr, float("nan")) if global_thr_map else float("nan")
                    if np.isnan(t_cut):
                        tpr_at_fpr[target_fpr] = float("nan")
                    else:
                        sig_mask = y_bin == 1
                        tpr_at_fpr[target_fpr] = float(
                            ((p_bin >= t_cut) & sig_mask).sum() / max(int(sig_mask.sum()), 1)
                        )
                else:
                    idx = int(np.searchsorted(fpr_arr, target_fpr, side="right") - 1)
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
    use_global_fpr: bool = True,
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
                use_global_fpr=use_global_fpr,
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
    use_global_fpr: bool = True,
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
                    run_result,
                    data,
                    signal_classes,
                    threshold,
                    fixed_fpr,
                    event_mask,
                    playlist,
                    use_global_fpr=use_global_fpr,
                )
            )
        out[model_name] = aggregate_metrics(runs, fixed_fpr)
    return out


def compute_all_metrics_W(
    results: dict[str, list[dict]],
    data: dict,
    signal_classes: list[int],
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
    use_global_fpr: bool = True,
) -> dict[str, dict]:
    """Aggregated *W*-binned metrics for all models (requires ``W_GeV`` / ``W_bin_edges`` on *data*)."""
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    out = {}
    for model_name, run_list in sorted(results.items(), key=lambda kv: kv[0]):
        runs = []
        for run_result in run_list:
            runs.append(
                compute_binned_metrics_W(
                    run_result,
                    data,
                    signal_classes,
                    threshold,
                    fixed_fpr,
                    event_mask,
                    playlist,
                    use_global_fpr=use_global_fpr,
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


def compute_signal_baseline_W(
    results: dict[str, list[dict]],
    data: dict,
    signal_classes: list[int],
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
) -> np.ndarray:
    """Random-classifier baseline (signal fraction) per *W* bin."""
    first_model = next(iter(results))
    first_run = results[first_model][0]
    sig = get_signal_probabilities(first_run, signal_classes, playlist)
    y_true = sig["ytrue"]

    w = data["W_GeV"]
    edges = data["W_bin_edges"]
    baseline_w = []
    for i in range(len(edges) - 1):
        bm = mc_value_in_bin(w, edges, i, require_finite=False)
        if event_mask is not None:
            bm = bm & event_mask
        n_sig = (y_true[bm] == 1).sum() if bm.sum() > 0 else 0
        baseline_w.append(y_true[bm].mean() if n_sig > 0 else np.nan)
    return np.array(baseline_w)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _set_xlim_w_metrics(ax: plt.Axes) -> None:
    """Set a consistent *W* [GeV] axis span on metric / histogram panels."""
    lo, hi = W_METRICS_XLIM_GEV
    ax.set_xlim(lo, hi)


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


def _histogram_inttype_counts_with_positives(
    ax: plt.Axes,
    hist_var: np.ndarray,
    plot_mask: np.ndarray,
    y_true_binary: np.ndarray,
    bin_edges: np.ndarray,
    *,
    log_x: bool = False,
    reco_baseline_pred: np.ndarray | None = None,
    finite_bin_var: bool = False,
    has_pion_for_binning: np.ndarray | None = None,
) -> None:
    """Stacked histogram: orange = MC signal (positives) on the **bottom**,
    blue = not signal stacked **above**.

    Total bar height per bin matches ``np.histogram(hist_var[plot_mask], ...)``.
    Signal is drawn first (``bottom=0``); ``Other`` uses ``bottom=counts_sig``
    so the smaller category is not hidden under a large lower block.

    If ``reco_baseline_pred`` is set, a **right** *y* axis shows per-bin baseline
    FPR on true background within ``plot_mask`` (``Baseline FPR`` line + legend).
    """
    counts_all, _ = np.histogram(hist_var[plot_mask], bins=bin_edges)
    pos_mask = plot_mask & (y_true_binary == 1)
    counts_sig, _ = np.histogram(hist_var[pos_mask], bins=bin_edges)
    counts_all = np.asarray(counts_all, dtype=np.int64)
    counts_sig = np.asarray(counts_sig, dtype=np.int64)
    counts_sig = np.minimum(counts_sig, counts_all)
    counts_bg = counts_all - counts_sig
    widths = np.diff(bin_edges)
    x0 = bin_edges[:-1]

    ax.bar(
        x0,
        counts_sig.astype(float),
        width=widths,
        align="edge",
        color="tab:orange",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.9,
        label="Signal (positives)",
    )
    ax.bar(
        x0,
        counts_bg.astype(float),
        width=widths,
        align="edge",
        bottom=counts_sig.astype(float),
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.65,
        label="Other (not signal)",
    )
    for i, ctot in enumerate(counts_all):
        if ctot > 0:
            ax.text(
                x0[i] + widths[i] / 2,
                float(ctot),
                str(int(ctot)),
                ha="center",
                va="bottom",
                fontsize=7,
            )
        cs = int(counts_sig[i])
        if cs > 0:
            y_mid = float(counts_sig[i]) / 2.0
            ax.text(
                x0[i] + widths[i] / 2,
                y_mid,
                str(cs),
                ha="center",
                va="center",
                fontsize=6,
                color="white" if cs >= 3 else "black",
            )
    ax.grid(True, axis="y", alpha=0.3)
    if log_x:
        ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)

    if reco_baseline_pred is not None:
        if len(reco_baseline_pred) != len(hist_var):
            raise ValueError(
                f"len(reco_baseline_pred)={len(reco_baseline_pred)} != len(hist_var)={len(hist_var)}"
            )
        fpr_bin = compute_reco_baseline_fpr_per_bin(
            reco_baseline_pred,
            y_true_binary,
            hist_var,
            bin_edges,
            event_mask=plot_mask,
            has_pion=has_pion_for_binning,
            finite_bin_var=finite_bin_var,
        )
        x_mid = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        ax2 = ax.twinx()
        ax2.plot(
            x_mid,
            fpr_bin,
            color="black",
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.2,
            label="Baseline FPR",
            clip_on=False,
            zorder=5,
        )
        ax2.set_ylabel("Baseline FPR", color="black")
        ax2.tick_params(axis="y", labelcolor="black")
        ymax = float(np.nanmax(fpr_bin)) if np.any(np.isfinite(fpr_bin)) else 1.0
        ax2.set_ylim(0.0, min(1.0, max(0.05, ymax * 1.15)))
        ax2.legend(loc="upper left", fontsize=6, framealpha=0.9)


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
    use_global_fpr: bool = True,
) -> None:
    """Apply labels, titles, legend and grid to an (n_rows, 3) axes array."""
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    for row in range(n_rows):
        for col in range(3):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(col_labels[col])
            title_prefix = f"{row_titles[row]} — " if row_titles else ""
            if col < 2:
                ax.set_title(f"{title_prefix}{col_labels[col]} vs. {xlabel.split('[')[0].strip()}")
            else:
                ax.set_title(f"{title_prefix}{tpr_title} vs. {xlabel.split('[')[0].strip()}")
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
    reco_baseline_label: str = "Baseline",
    reco_baseline_global_fpr: float | None = None,
    reco_baseline_pred: np.ndarray | None = None,
    results: dict[str, list[dict]] | None = None,
    signal_classes: list[int] | None = None,
    pion_bins_require_has_pion: bool = True,
    colors: dict[str, str] | None = None,
    legend_title: str | None = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    suptitle: str | None = None,
    use_global_fpr: bool = True,
    playlist: str = "1A",
) -> plt.Figure:
    """2×3 or 2×4 figure: pion *E* (top) and pion *θ* (bottom).

    Columns: AUPRC, AUROC, TPR vs kinematics (global or per-bin FPR; see ``use_global_fpr``).
    If both *results* and *signal_classes* are passed, a fourth column shows stacked
    event counts vs *E* or *θ* (same convention as :func:`plot_multi_classification_vs_W`).
    Optional *reco_baseline_pred* draws per-bin **Baseline FPR** on a twin *y* axis on
    those histograms.

    Parameters
    ----------
    reco_baseline_tpr : optional dict with keys ``"E"`` and ``"theta"``,
        each a per-bin recall array for a reconstruction-level baseline.
        Plotted on the TPR panels (column index 2).
    reco_baseline_global_fpr : optional scalar FPR for the baseline legend on the TPR panels.
        If omitted but *reco_baseline_pred* and *results* / *signal_classes* are set,
        FPR is computed on the full masked sample like :func:`plot_multi_classification_vs_W`.
    reco_baseline_pred : optional binary predictions (same length as the test set).
    results, signal_classes : together enable the fourth-column histograms; must both
        be set or both omitted.
    pion_bins_require_has_pion : same meaning as in :func:`compute_binned_metrics_single`
        and :func:`plot_binned_by_inttype` for the count histograms.
    reco_baseline_label : label for the reconstruction baseline in the legend.
    legend_title : optional legend title (e.g. dataset line); ``None`` to omit.
    suptitle : figure super-title; default
        ``$CC1\\pi^\\pm$ tagging - MINERvA Open Data Playlist {playlist}``.
    playlist : playlist id for the default *suptitle* and for *y_true* from *results*.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    if (results is None) ^ (signal_classes is None):
        raise ValueError("results and signal_classes must both be set or both be omitted")

    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    n_cols = 4 if results is not None else 3
    fig_w = 22.0 if n_cols == 4 else 17.0
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, 9), tight_layout=True)

    E_mid = data["pion_E_MC_bins_mid"]
    theta_mid = data["pion_theta_MC_bins_mid"]

    y_true_binary: np.ndarray | None = None
    if n_cols == 4:
        first_model = next(iter(results))
        y_true_binary = get_signal_probabilities(
            results[first_model][0], signal_classes, playlist
        )["ytrue"]
        pE = data["pion_E_MC"]
        pTh = data["pion_theta_MC"]
        if len(pE) != len(y_true_binary):
            raise ValueError(
                f"len(pion_E_MC)={len(pE)} != len(y_true)={len(y_true_binary)}; "
                "check playlist alignment for the event-count panels."
            )
        if reco_baseline_pred is not None and len(reco_baseline_pred) != len(pE):
            raise ValueError(
                f"len(reco_baseline_pred)={len(reco_baseline_pred)} != len(pion_E_MC)={len(pE)}"
            )

    bl_tpr_label = reco_baseline_label
    if reco_baseline_tpr is not None:
        fpr_for_legend = reco_baseline_global_fpr
        if (
            fpr_for_legend is None
            and reco_baseline_pred is not None
            and results is not None
            and signal_classes is not None
        ):
            fm = next(iter(results))
            y_tb = get_signal_probabilities(results[fm][0], signal_classes, playlist)["ytrue"]
            if len(reco_baseline_pred) == len(y_tb):
                fpr_for_legend = _global_reco_baseline_fpr(reco_baseline_pred, y_tb)
        bl_tpr_label = _baseline_legend_with_global_fpr(reco_baseline_label, fpr_for_legend)

    # Random baseline (circle markers like models; dashed + gray to distinguish)
    axes[0, 0].plot(E_mid, baseline["E"], "o--", color="gray", label="Random baseline")
    axes[1, 0].plot(theta_mid, baseline["theta"], "o--", color="gray", label="Random baseline")

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
            lbl = _tpr_line_legend_label(model_name, fpr_val, use_global_fpr)
            _plot_metric_line(
                axes[0, 2], E_mid, agg_E[key],
                lbl, uncertainties, **clr,
            )
            _plot_metric_line(
                axes[1, 2], theta_mid, agg_theta[key],
                lbl, uncertainties, **clr,
            )

    if reco_baseline_tpr is not None:
        if "E" in reco_baseline_tpr:
            axes[0, 2].plot(
                E_mid, reco_baseline_tpr["E"], "s--", color="black",
                label=bl_tpr_label,
            )
        if "theta" in reco_baseline_tpr:
            axes[1, 2].plot(
                theta_mid, reco_baseline_tpr["theta"], "s--", color="black",
                label=bl_tpr_label,
            )

    if n_cols == 4 and y_true_binary is not None:
        has_pion = data["has_pion"]
        hp_bin = has_pion if pion_bins_require_has_pion else None
        plot_mask_e = has_pion if pion_bins_require_has_pion else np.ones(len(data["pion_E_MC"]), dtype=bool)
        plot_mask_th = (
            has_pion if pion_bins_require_has_pion else np.isfinite(data["pion_theta_MC"])
        )
        _histogram_inttype_counts_with_positives(
            axes[0, 3],
            data["pion_E_MC"],
            plot_mask_e,
            y_true_binary,
            data["pion_E_MC_bins"],
            log_x=True,
            reco_baseline_pred=reco_baseline_pred,
            finite_bin_var=False,
            has_pion_for_binning=hp_bin,
        )
        _histogram_inttype_counts_with_positives(
            axes[1, 3],
            data["pion_theta_MC"],
            plot_mask_th,
            y_true_binary,
            data["pion_theta_MC_bins"],
            log_x=False,
            reco_baseline_pred=reco_baseline_pred,
            finite_bin_var=True,
            has_pion_for_binning=hp_bin,
        )

    col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
    xlabels = [r"True $E_\pi$ [GeV]", r"True $\theta_\pi$ [rad]"]
    for row, kinematic in enumerate(xlabels):
        for col, metric in enumerate(col_labels):
            ax = axes[row, col]
            ax.set_xlabel(kinematic)
            ax.set_ylabel(metric)
            x_short = kinematic.split("[")[0].strip()
            if col < 2:
                ax.set_title(f"{metric} vs. {x_short}")
            else:
                ax.set_title(f"{tpr_title} vs. {x_short}")
            ax.legend(**_classification_legend_kw(7, legend_title))
            ax.grid(True)
            if row == 0:
                ax.set_xlim(E_mid[0] * 0.8, E_mid[-1] * 1.2)
                ax.set_xscale("log")
        if n_cols == 4:
            axh = axes[row, 3]
            axh.set_xlabel(kinematic)
            axh.set_ylabel("Events")
            x_short = kinematic.split("[")[0].strip()
            axh.set_title(
                rf"Event counts vs. {x_short} (top = $N_{{\mathrm{{tot}}}}$; orange = $N_{{\mathrm{{sig}}}}$)"
            )
            axh.grid(True, axis="y", alpha=0.3)
            if row == 0:
                axh.set_xlim(E_mid[0] * 0.8, E_mid[-1] * 1.2)
                axh.set_xscale("log")

    if suptitle is None:
        suptitle = fr"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}"
    fig.suptitle(suptitle, fontsize=14)
    return fig


def plot_multi_pion_vs_q3(
    all_metrics_q3: dict[str, dict],
    data: dict,
    baseline_q3: np.ndarray,
    fixed_fpr: list[float] | None = None,
    uncertainties: bool = False,
    reco_baseline_tpr_q3: np.ndarray | None = None,
    reco_baseline_label: str = "Baseline",
    colors: dict[str, str] | None = None,
    title: str | None = None,
    legend_title: str | None = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    use_global_fpr: bool = True,
    playlist: str = "1A",
) -> plt.Figure:
    """1x3 figure: AUPRC / AUROC / TPR vs *q₃* (global or per-bin FPR).

    Parameters
    ----------
    reco_baseline_tpr_q3 : optional per-bin recall array for a
        reconstruction-level baseline.  Plotted on the rightmost
        (TPR @ fixed FPR) panel.
    reco_baseline_label : label for the reconstruction baseline in the legend.
    title : optional figure super-title.  Defaults to a multi-pion
        description when *None*.
    legend_title : optional legend title; ``None`` to omit.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), tight_layout=True)

    q3_mid = data["q3_bin_mids"]

    axes[0].plot(q3_mid, baseline_q3, "o--", color="gray", label="Random baseline")

    for model_name, agg in sorted(all_metrics_q3.items(), key=lambda kv: kv[0]):
        clr = {} if colors is None else {"color": colors.get(model_name)}
        _plot_metric_line(axes[0], q3_mid, agg["auprc"], model_name, uncertainties, **clr)
        _plot_metric_line(axes[1], q3_mid, agg["auroc"], model_name, uncertainties, **clr)
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            _plot_metric_line(
                axes[2], q3_mid, agg[key],
                _tpr_line_legend_label(model_name, fpr_val, use_global_fpr),
                uncertainties, **clr,
            )

    if reco_baseline_tpr_q3 is not None:
        axes[2].plot(q3_mid, reco_baseline_tpr_q3, "s--", color="black",
                     label=reco_baseline_label)

    col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
    for col, metric in enumerate(col_labels):
        ax = axes[col]
        ax.set_xlabel(r"True $q_3$ [GeV]")
        ax.set_ylabel(metric)
        x_short = r"True $q_3$"
        if col < 2:
            ax.set_title(f"{metric} vs. {x_short}")
        else:
            ax.set_title(f"{tpr_title} vs. {x_short}")
        ax.legend(**_classification_legend_kw(7, legend_title))
        ax.grid(True)

    if title is None:
        title = (
            fr"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}"
        )
    fig.suptitle(title, fontsize=14)
    return fig


def plot_multi_classification_vs_W(
    all_metrics_W: dict[str, dict],
    data: dict,
    baseline_W: np.ndarray,
    fixed_fpr: list[float] | None = None,
    uncertainties: bool = False,
    reco_baseline_tpr_W: np.ndarray | None = None,
    reco_baseline_label: str = "Baseline",
    reco_baseline_global_fpr: float | None = None,
    reco_baseline_pred: np.ndarray | None = None,
    colors: dict[str, str] | None = None,
    title: str | None = None,
    legend_title: str | None = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    use_global_fpr: bool = True,
    playlist: str = "1A",
    results: dict[str, list[dict]] | None = None,
    signal_classes: list[int] | None = None,
) -> plt.Figure:
    """1×3 or 1×4 figure: AUPRC / AUROC / TPR vs *W* (global or per-bin FPR).

    Same layout as :func:`plot_multi_pion_vs_q3` but with *W* on the *x* axis.
    If both *results* and *signal_classes* are passed, a fourth panel shows a
    stacked histogram of **all** test events vs *W* (blue = not signal,
    orange = signal), with total *N* on top of each bar and *N* signal inside
    the orange segment — same convention as :func:`plot_binned_by_inttype`.

    Pass *reco_baseline_global_fpr* (scalar FP/(FP+TN) on the test set, same
    convention as the notebook) to show it in the column-3 baseline legend,
    e.g. ``Baseline (FPR 3.5%)``.  If omitted but *reco_baseline_pred* is set
    and the fourth panel is used, FPR is computed from *reco_baseline_pred* and
    the task ``y_true`` from *results* / *signal_classes*.  With the fourth
    panel, *reco_baseline_pred* also draws **Baseline FPR** per *W* bin on a
    right-hand axis over the stacked counts.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    if (results is None) ^ (signal_classes is None):
        raise ValueError("results and signal_classes must both be set or both be omitted")

    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    n_cols = 4 if results is not None else 3
    fig_w = 22.0 if n_cols == 4 else 17.0
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 5), tight_layout=True)

    w_mid = data["W_bin_mids"]

    axes[0].plot(w_mid, baseline_W, "o--", color="gray", label="Random baseline")

    for model_name, agg in sorted(all_metrics_W.items(), key=lambda kv: kv[0]):
        clr = {} if colors is None else {"color": colors.get(model_name)}
        _plot_metric_line(axes[0], w_mid, agg["auprc"], model_name, uncertainties, **clr)
        _plot_metric_line(axes[1], w_mid, agg["auroc"], model_name, uncertainties, **clr)
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            _plot_metric_line(
                axes[2], w_mid, agg[key],
                _tpr_line_legend_label(model_name, fpr_val, use_global_fpr),
                uncertainties, **clr,
            )

    if reco_baseline_tpr_W is not None:
        fpr_for_legend = reco_baseline_global_fpr
        if fpr_for_legend is None and reco_baseline_pred is not None and results is not None and signal_classes is not None:
            first_model = next(iter(results))
            y_tb = get_signal_probabilities(
                results[first_model][0], signal_classes, playlist
            )["ytrue"]
            if len(reco_baseline_pred) == len(y_tb):
                fpr_for_legend = _global_reco_baseline_fpr(reco_baseline_pred, y_tb)
        bl_lbl = _baseline_legend_with_global_fpr(reco_baseline_label, fpr_for_legend)
        axes[2].plot(w_mid, reco_baseline_tpr_W, "s--", color="black", label=bl_lbl)

    col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
    for col, metric in enumerate(col_labels):
        ax = axes[col]
        ax.set_xlabel(r"$W$ [GeV]")
        ax.set_ylabel(metric)
        x_short = r"$W$"
        if col < 2:
            ax.set_title(f"{metric} vs. {x_short}")
        else:
            ax.set_title(f"{tpr_title} vs. {x_short}")
        ax.legend(**_classification_legend_kw(7, legend_title))
        ax.grid(True)
        _set_xlim_w_metrics(ax)

    if n_cols == 4:
        first_model = next(iter(results))
        y_true_binary = get_signal_probabilities(
            results[first_model][0], signal_classes, playlist
        )["ytrue"]
        w_gev = data["W_GeV"]
        if len(w_gev) != len(y_true_binary):
            raise ValueError(
                f"len(W_GeV)={len(w_gev)} != len(y_true)={len(y_true_binary)}; "
                "check playlist alignment for the event-count panel."
            )
        if reco_baseline_pred is not None and len(reco_baseline_pred) != len(w_gev):
            raise ValueError(
                f"len(reco_baseline_pred)={len(reco_baseline_pred)} != len(W_GeV)={len(w_gev)}"
            )
        ax_h = axes[3]
        all_mask = np.ones(len(w_gev), dtype=bool)
        _histogram_inttype_counts_with_positives(
            ax_h,
            w_gev,
            all_mask,
            y_true_binary,
            data["W_bin_edges"],
            reco_baseline_pred=reco_baseline_pred,
        )
        ax_h.set_xlabel(r"$W$ [GeV]")
        ax_h.set_ylabel("Events")
        ax_h.set_title(
            r"Event counts vs. $W$ (top = $N_{\mathrm{tot}}$; orange = $N_{\mathrm{sig}}$)"
        )
        _set_xlim_w_metrics(ax_h)

    if title is None:
        title = (
            fr"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}"
        )
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
    reco_baseline_label: str = "Baseline",
    colors: dict[str, str] | None = None,
    signal_label: str | None = None,
    pion_bins_require_has_pion: bool = True,
    legend_title: str | None = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    use_global_fpr: bool = True,
) -> plt.Figure:
    """One row per interaction type, 4 columns: AUPRC, AUROC, TPR (global or per-bin FPR),
    and a stacked event-count histogram (blue: not signal, orange: MC positives).

    Parameters
    ----------
    x_var : ``"pion_E"``, ``"pion_theta"``, ``"q3"``, or ``"W"`` (hadronic
        invariant mass; requires ``W_GeV`` / ``W_bin_edges`` on *data*).
    reco_baseline_pred : optional binary prediction array (same length as
        test set). When provided, the per-bin recall is overlaid on the
        TPR panel for each interaction type, the baseline legend **FPR** is
        computed on the **same row mask** as the histogram (interaction type
        ∩ pion / finiteness rules), and per-bin **Baseline FPR** on that slice
        is drawn on the right axis of the stacked histogram.
    use_global_fpr : if True, one global score cut per target FPR; if False,
        TPR is taken from each bin's local ROC (and plot titles/legends match).
    reco_baseline_label : legend label for the reconstruction baseline.
    signal_label : optional name for the signal class definition (e.g.
        ``r"$CC\\pi^0$"``). Used when there are events in an interaction
        type but no signal positives; defaults from *signal_classes*.
    pion_bins_require_has_pion : if False, pion E/θ histograms
        and binned metrics include all events (θ requires finite MC angle).
    legend_title : optional legend title on metric panels; ``None`` to omit.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    if int_types is None:
        int_types = MC_INT_TYPE
    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)

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
    elif x_var == "W":
        hist_var = data["W_GeV"]
        bin_edges = data["W_bin_edges"]
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
        raise ValueError(f"Unknown x_var: {x_var!r}")

    finite_hist = x_var == "pion_theta"
    hp_for_bins = (
        data["has_pion"]
        if (x_var in ("pion_E", "pion_theta") and pion_bins_require_has_pion)
        else None
    )

    for row_idx, (int_code, int_name) in enumerate(int_types.items()):
        int_mask = int_type_arr == int_code

        # Count events actually entering the plots
        plot_mask = int_mask & hist_pion_mask
        n_events = int(plot_mask.sum())

        # Choose x-axis midpoints
        if x_var == "q3":
            x_mid = data["q3_bin_mids"]
        elif x_var == "W":
            x_mid = data["W_bin_mids"]
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
            if x_var == "W":
                for col in range(4):
                    _set_xlim_w_metrics(axes[row_idx, col])
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
            # Histogram: stacked all events vs MC signal positives in this interaction type
            ax_h = axes[row_idx, 3]
            _histogram_inttype_counts_with_positives(
                ax_h,
                hist_var,
                plot_mask,
                y_true_binary,
                bin_edges,
                log_x=log_x and x_var != "W",
                reco_baseline_pred=reco_baseline_pred,
                finite_bin_var=finite_hist,
                has_pion_for_binning=hp_for_bins,
            )
            ax_h.set_xlabel(xlabel)
            ax_h.set_ylabel("Events")
            ax_h.set_title(f"{int_name} (N={n_events:,}) — events (orange = signal, bottom)")
            if x_var == "W":
                _set_xlim_w_metrics(ax_h)
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
                results,
                data,
                signal_classes,
                threshold,
                fixed_fpr,
                int_mask,
                playlist,
                use_global_fpr=use_global_fpr,
            )
        elif x_var == "W":
            bl_values = compute_signal_baseline_W(
                results, data, signal_classes, int_mask, playlist,
            )
            all_agg = compute_all_metrics_W(
                results,
                data,
                signal_classes,
                threshold,
                fixed_fpr,
                int_mask,
                playlist,
                use_global_fpr=use_global_fpr,
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
                use_global_fpr=use_global_fpr,
            )
            sub_key = {"pion_E": "E", "pion_theta": "theta"}[x_var]
            all_agg = {mn: m[sub_key] for mn, m in all_agg_full.items()}

        # Random baseline (circle markers like models; dashed + gray to distinguish)
        axes[row_idx, 0].plot(x_mid, bl_values, "o--", color="gray", label="Random baseline")

        for model_name, agg in sorted(all_agg.items(), key=lambda kv: kv[0]):
            clr = {} if colors is None else {"color": colors.get(model_name)}
            _plot_metric_line(axes[row_idx, 0], x_mid, agg["auprc"], model_name, uncertainties, **clr)
            _plot_metric_line(axes[row_idx, 1], x_mid, agg["auroc"], model_name, uncertainties, **clr)
            for fpr_val in fixed_fpr:
                key = f"tpr@{fpr_val}"
                _plot_metric_line(
                    axes[row_idx, 2], x_mid, agg[key],
                    _tpr_line_legend_label(model_name, fpr_val, use_global_fpr),
                    uncertainties, **clr,
                )

        # Reconstruction baseline on TPR panel
        if reco_baseline_pred is not None:
            is_signal_masked = (y_true_binary == 1) & int_mask
            if x_var == "q3":
                reco_bl = compute_reco_baseline_recall_per_bin(
                    reco_baseline_pred, is_signal_masked,
                    data["q3_GeV"], data["q3_bin_edges"],
                )
            elif x_var == "W":
                reco_bl = compute_reco_baseline_recall_per_bin(
                    reco_baseline_pred, is_signal_masked,
                    data["W_GeV"], data["W_bin_edges"],
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
            # FPR in legend: same interaction-type (and pion/hist) slice as this row's histogram
            fpr_row = _reco_baseline_fpr_on_mask(reco_baseline_pred, y_true_binary, plot_mask)
            bl_lbl = _baseline_legend_with_global_fpr(reco_baseline_label, fpr_row)
            axes[row_idx, 2].plot(x_mid, reco_bl, "s--", color="black", label=bl_lbl)

        # --- Metric columns ---
        col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
        for col, metric in enumerate(col_labels):
            ax = axes[row_idx, col]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            if col < 2:
                ax.set_title(f"{int_name} (N={n_events:,}) — {metric} vs. {xlabel.split('[')[0].strip()}")
            else:
                ax.set_title(
                    f"{int_name} (N={n_events:,}) — {tpr_title} vs. {xlabel.split('[')[0].strip()}"
                )
            ax.legend(**_classification_legend_kw(7, legend_title))
            ax.grid(True)
            if x_var == "W":
                _set_xlim_w_metrics(ax)
            elif log_x:
                ax.set_xlim(x_mid[0] * 0.8, x_mid[-1] * 1.2)
                ax.set_xscale("log")

        # --- Histogram column (col 3): stacked not-signal (blue) + signal positives (orange)
        ax_h = axes[row_idx, 3]
        _histogram_inttype_counts_with_positives(
            ax_h,
            hist_var,
            plot_mask,
            y_true_binary,
            bin_edges,
            log_x=log_x and x_var != "W",
            reco_baseline_pred=reco_baseline_pred,
            finite_bin_var=finite_hist,
            has_pion_for_binning=hp_for_bins,
        )
        ax_h.set_xlabel(xlabel)
        ax_h.set_ylabel("Events")
        ax_h.set_title(f"{int_name} (N={n_events:,}) — events (orange = signal, bottom)")
        if x_var == "W":
            _set_xlim_w_metrics(ax_h)

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
    x_var : ``"pion_E"``, ``"pion_theta"``, ``"q3"``, or ``"W"``.
    pion_bins_require_has_pion : same meaning as in :func:`plot_binned_by_inttype`.
    """
    if int_types is None:
        int_types = MC_INT_TYPE

    int_type_arr = data["int_type_arr"]

    if x_var == "q3":
        var = data["q3_GeV"]
        bin_edges = data["q3_bin_edges"]
    elif x_var == "W":
        var = data["W_GeV"]
        bin_edges = data["W_bin_edges"]
    elif x_var == "pion_E":
        var = data["pion_E_MC"]
        bin_edges = data["pion_E_MC_bins"]
    elif x_var == "pion_theta":
        var = data["pion_theta_MC"]
        bin_edges = data["pion_theta_MC_bins"]
    else:
        raise ValueError(f"Unknown x_var: {x_var!r}")

    if x_var == "q3":
        kin_mask = np.ones(len(var), dtype=bool)
    elif x_var == "W":
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
        if x_var == "W":
            _set_xlim_w_metrics(ax)
        elif log_x:
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
    legend_title: str | None = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
) -> plt.Figure:
    """Plot PRC curves for all models with optional uncertainty bands.

    Parameters
    ----------
    max_threshold : if set, the right (log-scale) plot only shows the
        portion of each curve where the mean classification threshold is
        below this value.
    legend_title : optional legend title; ``None`` to omit.
    """
    curves = _compute_prc_curves(results, signal_classes, playlist)

    first_model = next(iter(results))
    sig = get_signal_probabilities(results[first_model][0], signal_classes, playlist)
    signal_frac = sig["ytrue"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), tight_layout=True)

    for ax in axes:
        ax.axhline(signal_frac, color="gray", linestyle="--", linewidth=1,
                    label="Random baseline")

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
        ax.legend(**_classification_legend_kw(9, legend_title))
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

