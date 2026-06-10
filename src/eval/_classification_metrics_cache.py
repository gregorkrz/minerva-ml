"""Pre-computed metrics cache for the three classification plot scripts.

Build once with :func:`build_metrics_cache` (or ``plot_classification_q3`` /
``plot_classification_Pions`` / ``plot_classification_W`` on a normal non
``--plots-only`` run), then load with :func:`load_metrics_cache`` to recover
everything needed by ``--plots-only`` runs without touching the 16 GB source
pickle.

Cache file: ``plots/tmp_results/classification_metrics.pkl`` by default.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.eval.classification_plots import (
    MC_INT_TYPE_MERGED,
    add_hadronic_W_to_classification_data,
    compute_all_metrics,
    compute_all_metrics_q3,
    compute_all_metrics_W,
    compute_reco_baseline_recall_per_bin,
    compute_signal_baseline,
    compute_signal_baseline_W,
    data_with_signal_pion_bins,
    get_signal_probabilities,
    merge_int_type_arr,
)
from src.eval.classification_plots._metrics_binned import (
    _align_per_event_array,
    _resolve_test_idx,
)
from src.eval.classification_plots._plotting import _compute_prc_curves

CLF_METRICS_CACHE_NAME = "classification_metrics.pkl"

_CC1PI_CLASSES = [0]
_CC1PI0_CLASSES = [2]
_CCNPI_GE1_CLASSES = [0, 1]
_CCNPI_GT1_CLASSES = [1]
_N_CLASSES = 5

# Keys used in the cache for each signal definition.
_SIG_TAGS = {
    "cc1pi": _CC1PI_CLASSES,
    "cc1pi0": _CC1PI0_CLASSES,
    "ccnpi_ge1": _CCNPI_GE1_CLASSES,
    "ccnpi_gt1": _CCNPI_GT1_CLASSES,
}


def _load_test_idx_from_clf(clf: dict, playlist: str) -> np.ndarray:
    """Load test-split indices from the dataset ``result.pkl`` referenced by *clf*."""
    ckpt_dir = Path(clf["ckpt_dir"])
    training_names = clf["training_names"]
    first_run = training_names[next(iter(training_names))][0]

    data_path = None
    model_pt = ckpt_dir / first_run / "best_model.pt"
    if model_pt.exists():
        ckpt = torch.load(model_pt, weights_only=False, map_location="cpu")
        data_path = Path(ckpt["args"]["data_path"])
    else:
        settings_path = ckpt_dir / first_run / "settings.json"
        with open(settings_path) as f:
            cfg = json.load(f)
        data_path = Path(cfg.get("data_path", cfg.get("dataset_path", "")))
    if data_path is None:
        raise FileNotFoundError(
            f"Cannot determine data_path from {first_run} for test_idx lookup"
        )

    split_idx = pickle.load(open(data_path / "result.pkl", "rb"))
    return np.asarray(split_idx[playlist]["test_idx"])


def _ensure_data_test_idx(data: dict, playlist: str, clf: dict) -> dict:
    """Ensure ``data["test_idx"][playlist]`` exists (older pickles may omit it)."""
    out = dict(data)
    ti = out.get("test_idx")
    if isinstance(ti, dict) and playlist in ti:
        return out
    test_idx = _load_test_idx_from_clf(clf, playlist)
    if isinstance(ti, dict):
        merged = dict(ti)
        merged[playlist] = test_idx
        out["test_idx"] = merged
    else:
        out["test_idx"] = {playlist: test_idx}
    return out


def _normalize_data_arrays(data: dict, playlist: str, n_events: int) -> dict:
    """Shallow copy with per-event arrays indexed onto the test split."""
    out = dict(data)
    for key in (
        "q3_GeV",
        "W_GeV",
        "pion_E_MC",
        "pion_theta_MC",
        "has_pion",
        "int_type_arr",
    ):
        if key in out:
            out[key] = _align_per_event_array(
                out[key], out, playlist, n_events, key=key
            )
    return out


def _int_masks(
    data: dict,
    playlist: str = "1A",
    n_events: int | None = None,
) -> dict[int, np.ndarray]:
    int_arr = merge_int_type_arr(data["int_type_arr"])
    if n_events is not None and len(int_arr) != n_events:
        int_arr = merge_int_type_arr(
            _align_per_event_array(
                int_arr, data, playlist, n_events, key="int_type_arr"
            )
        )
    return {code: int_arr == code for code in MC_INT_TYPE_MERGED}


def _compute_baseline_fpr(
    results: dict,
    data: dict,
    pid: np.ndarray,
    signal_classes: list[int],
    playlist: str,
    baselines_key: str,
    *,
    n_muons_cond: str = "cc1pi",
) -> float:
    """Return the reco-baseline FPR for a given signal definition."""
    test_idx = _resolve_test_idx(data, playlist)
    baselines_pl = data["baselines"][playlist]
    n_muons = baselines_pl["n_muons"][test_idx]

    y_true = np.isin(pid, signal_classes).astype(int)

    if n_muons_cond == "cc1pi":
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        y_pred = ((n_muons == 1) & (n_charged_prongs == 1) & (improved_nmichel == 1)).astype(int)
    elif n_muons_cond == "ccnpi_ge1":
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        y_pred = ((n_muons == 1) & (n_charged_prongs >= 1) & (improved_nmichel >= 1)).astype(int)
    elif n_muons_cond == "ccnpi_gt1":
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        y_pred = ((n_muons == 1) & (n_charged_prongs >= 2) & (improved_nmichel >= 1)).astype(int)
    elif n_muons_cond == "cc1pi0":
        PI0_MASS = 134.977
        is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
        two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
        n_michel = baselines_pl["improved_nmichel"][test_idx]
        y_pred = (
            (n_muons == 1)
            & (is_pizero_signal == 2)
            & (np.abs(two_gamma_inv_mass - PI0_MASS) < PI0_MASS)
            & (n_michel == 0)
        ).astype(int)
    else:
        raise ValueError(f"Unknown n_muons_cond: {n_muons_cond!r}")

    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    return float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan")


def _compute_reco_pred(
    data: dict,
    pid: np.ndarray,
    playlist: str,
    n_muons_cond: str,
) -> np.ndarray:
    """Binary reco-baseline prediction array for a given signal definition."""
    test_idx = _resolve_test_idx(data, playlist)
    baselines_pl = data["baselines"][playlist]
    n_muons = baselines_pl["n_muons"][test_idx]
    if n_muons_cond == "cc1pi":
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        return ((n_muons == 1) & (n_charged_prongs == 1) & (improved_nmichel == 1)).astype(int)
    elif n_muons_cond == "ccnpi_ge1":
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        return ((n_muons == 1) & (n_charged_prongs >= 1) & (improved_nmichel >= 1)).astype(int)
    elif n_muons_cond == "ccnpi_gt1":
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        return ((n_muons == 1) & (n_charged_prongs >= 2) & (improved_nmichel >= 1)).astype(int)
    elif n_muons_cond == "cc1pi0":
        PI0_MASS = 134.977
        is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
        two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
        n_michel = baselines_pl["improved_nmichel"][test_idx]
        return (
            (n_muons == 1)
            & (is_pizero_signal == 2)
            & (np.abs(two_gamma_inv_mass - PI0_MASS) < PI0_MASS)
            & (n_michel == 0)
        ).astype(int)
    raise ValueError(n_muons_cond)


def build_metrics_cache(clf: dict) -> dict:
    """Compute all expensive classification metrics and return the cache dict.

    The returned dict is ready to be pickled.  Pass *clf* = the full
    classification pickle dict loaded from ``classification_<flag>.pkl``.
    """
    results = clf["results"]
    data_by_playlist = clf["data_by_playlist"]
    data_w_by_playlist = clf.get("data_w_by_playlist", {})
    playlists = clf["playlists"]
    use_global_fpr_W = clf.get("use_global_fpr_W", True)

    first_model = next(iter(results))

    # pid is model-independent: MC truth class labels.
    pid_by_pl = {pl: results[first_model][0][pl]["pid"] for pl in playlists}

    # Older pickles may store full-dataset kinematic arrays; align to test predictions.
    for pl in playlists:
        n_events = len(pid_by_pl[pl])
        data_by_playlist[pl] = _ensure_data_test_idx(data_by_playlist[pl], pl, clf)
        data_by_playlist[pl] = _normalize_data_arrays(data_by_playlist[pl], pl, n_events)
        if pl in data_w_by_playlist:
            data_w_by_playlist[pl] = _ensure_data_test_idx(
                data_w_by_playlist[pl], pl, clf
            )
            data_w_by_playlist[pl] = _normalize_data_arrays(
                data_w_by_playlist[pl], pl, n_events
            )

    # -----------------------------------------------------------------------
    # Confusion matrices (fast — argmax only)
    # -----------------------------------------------------------------------
    confusion_matrices: dict = {}
    for model_name, run_list in results.items():
        confusion_matrices[model_name] = {}
        for pl in playlists:
            y_true_pl = run_list[0][pl]["pid"]
            y_pred_pl = run_list[0][pl]["prediction"].argmax(axis=1)
            confusion_matrices[model_name][pl] = confusion_matrix(
                y_true_pl, y_pred_pl, labels=list(range(_N_CLASSES))
            )

    # -----------------------------------------------------------------------
    # Baseline FPRs and reco prediction arrays (cheap numpy)
    # -----------------------------------------------------------------------
    _BASELINE_CONDS = {
        "cc1pi": "cc1pi",
        "cc1pi0": "cc1pi0",
        "ccnpi_ge1": "ccnpi_ge1",
        "ccnpi_gt1": "ccnpi_gt1",
    }
    baseline_fpr: dict = {tag: {} for tag in _BASELINE_CONDS}
    reco_pred: dict = {tag: {} for tag in _BASELINE_CONDS}

    for tag, cond in _BASELINE_CONDS.items():
        sc = _SIG_TAGS[tag]
        # For plot_classification_W only playlist 1A is used; for others both.
        for pl in playlists:
            data = data_by_playlist[pl]
            pid = pid_by_pl[pl]
            baseline_fpr[tag][pl] = _compute_baseline_fpr(results, data, pid, sc, pl, cond, n_muons_cond=cond)
            reco_pred[tag][pl] = _compute_reco_pred(data, pid, pl, cond)

    # -----------------------------------------------------------------------
    # Helper: compute metrics for all int_codes including "all" (no mask)
    # Returns {mask_tag: {model: agg}}  where mask_tag is "all" or int_code
    # -----------------------------------------------------------------------
    def _metrics_q3(sig_classes, fpr, data, pl, use_gfpr=True):
        n_events = len(pid_by_pl[pl])
        masks = _int_masks(data, pl, n_events)
        result = {
            "all": compute_all_metrics_q3(
                results, data, sig_classes, fixed_fpr=[fpr], playlist=pl, use_global_fpr=use_gfpr
            ),
        }
        for code, mask in masks.items():
            result[code] = compute_all_metrics_q3(
                results,
                data,
                sig_classes,
                fixed_fpr=[fpr],
                playlist=pl,
                event_mask=mask,
                use_global_fpr=use_gfpr,
            )
        return result

    def _metrics_W(sig_classes, fpr, data_w, pl, use_gfpr=True):
        n_events = len(pid_by_pl[pl])
        masks = _int_masks(data_w, pl, n_events)
        result = {
            "all": compute_all_metrics_W(
                results, data_w, sig_classes, fixed_fpr=[fpr], playlist=pl, use_global_fpr=use_gfpr
            ),
        }
        for code, mask in masks.items():
            result[code] = compute_all_metrics_W(
                results,
                data_w,
                sig_classes,
                fixed_fpr=[fpr],
                playlist=pl,
                event_mask=mask,
                use_global_fpr=use_gfpr,
            )
        return result

    def _metrics_pion(sig_classes, fpr, data_pion, pl, pion_bins_require_has_pion=False):
        n_events = len(pid_by_pl[pl])
        masks = _int_masks(data_pion, pl, n_events)
        result_full = {
            "all": compute_all_metrics(
                results,
                data_pion,
                sig_classes,
                fixed_fpr=[fpr],
                playlist=pl,
                pion_bins_require_has_pion=pion_bins_require_has_pion,
            ),
        }
        for code, mask in masks.items():
            result_full[code] = compute_all_metrics(
                results,
                data_pion,
                sig_classes,
                fixed_fpr=[fpr],
                playlist=pl,
                event_mask=mask,
                pion_bins_require_has_pion=pion_bins_require_has_pion,
            )
        return result_full

    # -----------------------------------------------------------------------
    # q3 metrics (plot_classification_q3)
    # -----------------------------------------------------------------------
    print("  Computing q3 metrics …")
    metrics_q3: dict = {"cc1pi": {}, "ccnpi_ge1": {}}
    for pl in playlists:
        data = data_by_playlist[pl]
        metrics_q3["cc1pi"][pl] = _metrics_q3(_CC1PI_CLASSES, baseline_fpr["cc1pi"][pl], data, pl)
        metrics_q3["ccnpi_ge1"][pl] = _metrics_q3(_CCNPI_GE1_CLASSES, baseline_fpr["ccnpi_ge1"][pl], data, pl)

    # -----------------------------------------------------------------------
    # W metrics from data_w_by_playlist (plot_classification_W)
    # -----------------------------------------------------------------------
    print("  Computing W-clf metrics …")
    metrics_W_clf: dict = {tag: {} for tag in ("cc1pi", "cc1pi0", "ccnpi_ge1", "ccnpi_gt1")}
    if data_w_by_playlist:
        for pl in ["1A"]:  # plot_classification_W only uses 1A
            if pl not in data_w_by_playlist:
                continue
            data_w = data_w_by_playlist[pl]
            metrics_W_clf["cc1pi"][pl] = _metrics_W(_CC1PI_CLASSES, baseline_fpr["cc1pi"][pl], data_w, pl, use_gfpr=use_global_fpr_W)
            metrics_W_clf["cc1pi0"][pl] = _metrics_W(_CC1PI0_CLASSES, baseline_fpr["cc1pi0"][pl], data_w, pl, use_gfpr=use_global_fpr_W)
            metrics_W_clf["ccnpi_ge1"][pl] = _metrics_W(_CCNPI_GE1_CLASSES, baseline_fpr["ccnpi_ge1"][pl], data_w, pl, use_gfpr=use_global_fpr_W)
            metrics_W_clf["ccnpi_gt1"][pl] = _metrics_W(_CCNPI_GT1_CLASSES, baseline_fpr["ccnpi_gt1"][pl], data_w, pl, use_gfpr=use_global_fpr_W)

    # -----------------------------------------------------------------------
    # Pion E/theta + W metrics from pion-augmented data (plot_classification_Pions)
    # -----------------------------------------------------------------------
    print("  Computing pion-kinematic metrics …")
    metrics_pion: dict = {"cc1pi": {}, "cc1pi0": {}}
    metrics_W_pion: dict = {"cc1pi": {}, "cc1pi0": {}}
    data_cc1pi_by_pl: dict = {}
    data_cc1pi_w_by_pl: dict = {}
    data_cc1pi0_by_pl: dict = {}
    data_cc1pi0_w_by_pl: dict = {}

    for pl in playlists:
        data = data_by_playlist[pl]
        pid = pid_by_pl[pl]
        first_run0 = results[first_model][0]

        data_cc1pi = data_with_signal_pion_bins(
            data, pid, _CC1PI_CLASSES, pion_quantile_require_has_pion=False,
            pion_bin_edge_method="equal_frequency",
        )
        data_cc1pi_w = add_hadronic_W_to_classification_data(data_cc1pi, pl)
        data_cc1pi_by_pl[pl] = data_cc1pi
        data_cc1pi_w_by_pl[pl] = data_cc1pi_w

        data_cc1pi0 = data_with_signal_pion_bins(
            data, pid, _CC1PI0_CLASSES, pion_quantile_require_has_pion=False,
            pion_bin_edge_method="equal_frequency",
        )
        data_cc1pi0_w = add_hadronic_W_to_classification_data(data_cc1pi0, pl)
        data_cc1pi0_by_pl[pl] = data_cc1pi0
        data_cc1pi0_w_by_pl[pl] = data_cc1pi0_w

        metrics_pion["cc1pi"][pl] = _metrics_pion(_CC1PI_CLASSES, baseline_fpr["cc1pi"][pl], data_cc1pi, pl)
        metrics_pion["cc1pi0"][pl] = _metrics_pion(_CC1PI0_CLASSES, baseline_fpr["cc1pi0"][pl], data_cc1pi0, pl)
        metrics_W_pion["cc1pi"][pl] = _metrics_W(_CC1PI_CLASSES, baseline_fpr["cc1pi"][pl], data_cc1pi_w, pl, use_gfpr=True)
        metrics_W_pion["cc1pi0"][pl] = _metrics_W(_CC1PI0_CLASSES, baseline_fpr["cc1pi0"][pl], data_cc1pi0_w, pl, use_gfpr=True)

    # -----------------------------------------------------------------------
    # PRC curves
    # -----------------------------------------------------------------------
    print("  Computing PRC curves …")
    prc: dict = {}
    signal_frac: dict = {}
    for sig_tag, sig_classes in [("cc1pi", _CC1PI_CLASSES), ("ccnpi_ge1", _CCNPI_GE1_CLASSES), ("cc1pi0", _CC1PI0_CLASSES)]:
        prc[sig_tag] = {}
        signal_frac[sig_tag] = {}
        for pl in playlists:
            prc[sig_tag][pl] = _compute_prc_curves(results, sig_classes, pl)
            signal_frac[sig_tag][pl] = float(np.isin(pid_by_pl[pl], sig_classes).mean())

    # -----------------------------------------------------------------------
    # Signal baselines (cheap, but precompute to avoid needing results at draw time)
    # -----------------------------------------------------------------------
    print("  Computing signal baselines …")
    signal_baseline: dict = {"cc1pi": {}, "ccnpi_ge1": {}, "cc1pi0": {}}
    W_pion_baseline: dict = {"cc1pi": {}, "cc1pi0": {}}
    reco_tpr: dict = {}

    for pl in playlists:
        data = data_by_playlist[pl]
        pid = pid_by_pl[pl]
        is_signal_cc1pi = np.isin(pid, _CC1PI_CLASSES)
        is_signal_ccnpi = np.isin(pid, _CCNPI_GE1_CLASSES)
        is_signal_cc1pi0 = np.isin(pid, _CC1PI0_CLASSES)

        signal_baseline["cc1pi"][pl] = compute_signal_baseline(results, data, _CC1PI_CLASSES, playlist=pl, pion_bins_require_has_pion=False)
        signal_baseline["ccnpi_ge1"][pl] = compute_signal_baseline(results, data, _CCNPI_GE1_CLASSES, playlist=pl)
        signal_baseline["cc1pi0"][pl] = compute_signal_baseline(results, data, _CC1PI0_CLASSES, playlist=pl, pion_bins_require_has_pion=False)

        reco_tpr[f"q3_cc1pi_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi"][pl], is_signal_cc1pi, data["q3_GeV"], data["q3_bin_edges"])
        reco_tpr[f"q3_ccnpi_ge1_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["ccnpi_ge1"][pl], is_signal_ccnpi, data["q3_GeV"], data["q3_bin_edges"])

        data_cc1pi = data_cc1pi_by_pl[pl]
        data_cc1pi0 = data_cc1pi0_by_pl[pl]
        data_cc1pi_w = data_cc1pi_w_by_pl[pl]
        data_cc1pi0_w = data_cc1pi0_w_by_pl[pl]

        reco_tpr[f"pion_E_cc1pi_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi"][pl], is_signal_cc1pi, data_cc1pi["pion_E_MC"], data_cc1pi["pion_E_MC_bins"], has_pion=None)
        reco_tpr[f"pion_theta_cc1pi_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi"][pl], is_signal_cc1pi, data_cc1pi["pion_theta_MC"], data_cc1pi["pion_theta_MC_bins"], has_pion=None, finite_bin_var=True)
        reco_tpr[f"pion_E_cc1pi0_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi0"][pl], is_signal_cc1pi0, data_cc1pi0["pion_E_MC"], data_cc1pi0["pion_E_MC_bins"], has_pion=None)
        reco_tpr[f"pion_theta_cc1pi0_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi0"][pl], is_signal_cc1pi0, data_cc1pi0["pion_theta_MC"], data_cc1pi0["pion_theta_MC_bins"], has_pion=None, finite_bin_var=True)
        reco_tpr[f"W_pion_cc1pi_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi"][pl], is_signal_cc1pi, data_cc1pi_w["W_GeV"], data_cc1pi_w["W_bin_edges"])
        reco_tpr[f"W_pion_cc1pi0_{pl}"] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi0"][pl], is_signal_cc1pi0, data_cc1pi0_w["W_GeV"], data_cc1pi0_w["W_bin_edges"])
        W_pion_baseline["cc1pi"][pl] = compute_signal_baseline_W(results, data_cc1pi_w, _CC1PI_CLASSES, playlist=pl)
        W_pion_baseline["cc1pi0"][pl] = compute_signal_baseline_W(results, data_cc1pi0_w, _CC1PI0_CLASSES, playlist=pl)

    W_clf_baseline: dict = {tag: {} for tag in ("cc1pi", "cc1pi0", "ccnpi_ge1", "ccnpi_gt1")}
    W_clf_reco_tpr: dict = {tag: {} for tag in ("cc1pi", "cc1pi0", "ccnpi_ge1", "ccnpi_gt1")}
    if data_w_by_playlist:
        for pl in ["1A"]:
            if pl not in data_w_by_playlist:
                continue
            data_w = data_w_by_playlist[pl]
            pid = pid_by_pl[pl]
            is_signal_cc1pi = np.isin(pid, _CC1PI_CLASSES)
            is_signal_cc1pi0 = np.isin(pid, _CC1PI0_CLASSES)
            is_signal_ccnpi_ge1 = np.isin(pid, _CCNPI_GE1_CLASSES)
            is_signal_ccnpi_gt1 = np.isin(pid, _CCNPI_GT1_CLASSES)

            W_clf_baseline["cc1pi"][pl] = compute_signal_baseline_W(results, data_w, _CC1PI_CLASSES, playlist=pl)
            W_clf_baseline["cc1pi0"][pl] = compute_signal_baseline_W(results, data_w, _CC1PI0_CLASSES, playlist=pl)
            W_clf_baseline["ccnpi_ge1"][pl] = compute_signal_baseline_W(results, data_w, _CCNPI_GE1_CLASSES, playlist=pl)
            W_clf_baseline["ccnpi_gt1"][pl] = compute_signal_baseline_W(results, data_w, _CCNPI_GT1_CLASSES, playlist=pl)

            W_clf_reco_tpr["cc1pi"][pl] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi"][pl], is_signal_cc1pi, data_w["W_GeV"], data_w["W_bin_edges"])
            W_clf_reco_tpr["cc1pi0"][pl] = compute_reco_baseline_recall_per_bin(reco_pred["cc1pi0"][pl], is_signal_cc1pi0, data_w["W_GeV"], data_w["W_bin_edges"])
            W_clf_reco_tpr["ccnpi_ge1"][pl] = compute_reco_baseline_recall_per_bin(reco_pred["ccnpi_ge1"][pl], is_signal_ccnpi_ge1, data_w["W_GeV"], data_w["W_bin_edges"])
            W_clf_reco_tpr["ccnpi_gt1"][pl] = compute_reco_baseline_recall_per_bin(reco_pred["ccnpi_gt1"][pl], is_signal_ccnpi_gt1, data_w["W_GeV"], data_w["W_bin_edges"])

    # -----------------------------------------------------------------------
    # Per-inttype signal baselines (for plot_binned_by_inttype's bl_values)
    # -----------------------------------------------------------------------
    print("  Computing per-inttype baselines …")
    signal_baseline_inttype: dict = {}
    W_clf_baseline_inttype: dict = {}
    W_pion_baseline_inttype: dict = {}

    for pl in playlists:
        data = data_by_playlist[pl]
        n_events = len(pid_by_pl[pl])
        int_masks = _int_masks(data, pl, n_events)
        data_cc1pi = data_cc1pi_by_pl[pl]
        data_cc1pi0 = data_cc1pi0_by_pl[pl]
        data_cc1pi_w = data_cc1pi_w_by_pl[pl]
        data_cc1pi0_w = data_cc1pi0_w_by_pl[pl]

        for tag, sig_classes, pion_has_pion in [
            ("cc1pi", _CC1PI_CLASSES, False),
            ("ccnpi_ge1", _CCNPI_GE1_CLASSES, True),
            ("cc1pi0", _CC1PI0_CLASSES, False),
        ]:
            if tag not in signal_baseline_inttype:
                signal_baseline_inttype[tag] = {}
            if pl not in signal_baseline_inttype[tag]:
                signal_baseline_inttype[tag][pl] = {}

            for code, mask in int_masks.items():
                bl = compute_signal_baseline(results, data, sig_classes, event_mask=mask, playlist=pl, pion_bins_require_has_pion=pion_has_pion)
                # bl has keys "E", "theta", "q3" — the x_var-specific value is extracted at draw time
                signal_baseline_inttype[tag][pl][code] = bl

        # W clf per-inttype baselines
        if data_w_by_playlist and pl in data_w_by_playlist:
            data_w = data_w_by_playlist[pl]
            int_masks_w = _int_masks(data_w, pl, n_events)
            for tag, sig_classes in [("cc1pi", _CC1PI_CLASSES), ("cc1pi0", _CC1PI0_CLASSES), ("ccnpi_ge1", _CCNPI_GE1_CLASSES), ("ccnpi_gt1", _CCNPI_GT1_CLASSES)]:
                if tag not in W_clf_baseline_inttype:
                    W_clf_baseline_inttype[tag] = {}
                if pl not in W_clf_baseline_inttype[tag]:
                    W_clf_baseline_inttype[tag][pl] = {}
                for code, mask in int_masks_w.items():
                    W_clf_baseline_inttype[tag][pl][code] = compute_signal_baseline_W(results, data_w, sig_classes, event_mask=mask, playlist=pl)

        # W pion per-inttype baselines
        for tag, data_pion_w, sig_classes in [("cc1pi", data_cc1pi_w, _CC1PI_CLASSES), ("cc1pi0", data_cc1pi0_w, _CC1PI0_CLASSES)]:
            if tag not in W_pion_baseline_inttype:
                W_pion_baseline_inttype[tag] = {}
            if pl not in W_pion_baseline_inttype[tag]:
                W_pion_baseline_inttype[tag][pl] = {}
            int_masks_pw = _int_masks(data_pion_w, pl, n_events)
            for code, mask in int_masks_pw.items():
                W_pion_baseline_inttype[tag][pl][code] = compute_signal_baseline_W(results, data_pion_w, sig_classes, event_mask=mask, playlist=pl)

    return {
        "schema_version": 1,
        # Metadata
        "clrs_dict_full": clf["clrs_dict_full"],
        "playlists": playlists,
        "use_global_fpr_W": use_global_fpr_W,
        # MC truth (model-independent)
        "pid": pid_by_pl,
        # Reco-baseline binary prediction arrays (small int8-equivalent)
        "reco_pred": reco_pred,
        # Data arrays (no per-model predictions)
        "data_by_playlist": data_by_playlist,
        "data_w_by_playlist": data_w_by_playlist,
        # Pion-augmented data (for plot_classification_Pions draw calls)
        "data_cc1pi": data_cc1pi_by_pl,
        "data_cc1pi_w": data_cc1pi_w_by_pl,
        "data_cc1pi0": data_cc1pi0_by_pl,
        "data_cc1pi0_w": data_cc1pi0_w_by_pl,
        # Baseline FPRs and reco prediction arrays
        "baseline_fpr": baseline_fpr,
        # Signal baseline arrays (for "all" mask)
        "signal_baseline": signal_baseline,
        "W_clf_baseline": W_clf_baseline,
        "W_pion_baseline": W_pion_baseline,
        "W_clf_reco_tpr": W_clf_reco_tpr,
        # Per-inttype baselines
        "signal_baseline_inttype": signal_baseline_inttype,
        "W_clf_baseline_inttype": W_clf_baseline_inttype,
        "W_pion_baseline_inttype": W_pion_baseline_inttype,
        # Reco TPR arrays
        "reco_tpr": reco_tpr,
        # Precomputed metrics
        "metrics_q3": metrics_q3,
        "metrics_W_clf": metrics_W_clf,
        "metrics_W_pion": metrics_W_pion,
        "metrics_pion": metrics_pion,
        # PRC
        "prc": prc,
        "signal_frac": signal_frac,
        # Confusion matrices
        "confusion_matrices": confusion_matrices,
    }


def save_metrics_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved classification metrics cache → {path}")


def load_metrics_cache(path: Path) -> dict:
    print(f"Loading classification metrics cache from {path} …")
    with open(path, "rb") as f:
        return pickle.load(f)


def precomputed_inttype_agg(metrics_by_mask: dict) -> dict:
    """Extract per-inttype agg from a metrics_by_mask dict.

    *metrics_by_mask* has structure ``{"all": {model: agg}, int_code: {model: agg}, ...}``.
    Returns ``{int_code: {model: agg}}`` for the non-"all" keys.
    """
    return {k: v for k, v in metrics_by_mask.items() if k != "all"}


def precomputed_bl_from_signal_baseline(
    signal_baseline_inttype: dict, sig_tag: str, pl: str, x_var: str
) -> dict:
    """Build precomputed_bl_values for ``plot_binned_by_inttype``.

    Returns ``{int_code: array}`` where the array is the per-bin baseline
    for the given *x_var* (``"q3"``, ``"pion_E"``, or ``"pion_theta"``).
    """
    sub_key = {"q3": "q3", "pion_E": "E", "pion_theta": "theta"}[x_var]
    return {
        code: bl[sub_key]
        for code, bl in signal_baseline_inttype[sig_tag][pl].items()
    }
