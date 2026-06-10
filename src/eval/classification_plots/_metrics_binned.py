"""Per-bin classification metrics (AUPRC, AUROC, TPR@FPR)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from ._constants import DEFAULT_FIXED_FPR, _global_score_thresholds_at_target_fprs


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


def _resolve_test_idx(data: dict, playlist: str) -> np.ndarray:
    """Return test-split indices for *playlist* from a classification data dict.

    ``load_truth_and_baselines(..., playlists=[pl])`` stores only that playlist's
    indices under ``data["test_idx"]``.  Fall back to the sole entry when the
    requested key is missing but the dict has exactly one playlist.
    """
    test_idx = data.get("test_idx")
    if test_idx is None:
        raise KeyError("classification data dict has no 'test_idx'")
    if isinstance(test_idx, dict):
        if playlist in test_idx:
            return np.asarray(test_idx[playlist])
        if len(test_idx) == 1:
            return np.asarray(next(iter(test_idx.values())))
        raise KeyError(
            f"test_idx keys {list(test_idx.keys())!r} do not include playlist {playlist!r}"
        )
    return np.asarray(test_idx)


def _rebuild_per_event_array(
    data: dict,
    playlist: str,
    n_pred: int,
    key: str,
) -> np.ndarray | None:
    """Rebuild a test-sized per-event array from baselines (same as ``_io.load``)."""
    baselines = data.get("baselines", {})
    bl_pl = baselines.get(playlist)
    if bl_pl is None:
        return None
    try:
        test_idx = _resolve_test_idx(data, playlist)
    except KeyError:
        return None

    if key == "q3_GeV" and "q3" in bl_pl:
        return bl_pl["q3"][test_idx] / 1000.0

    if key == "W_GeV" and "mc_true_hadronic_W_GeV" in bl_pl:
        from ._hadronic_w import mc_true_hadronic_W_gev_from_baselines

        return mc_true_hadronic_W_gev_from_baselines(bl_pl, test_idx)

    if key in ("pion_E_MC", "pion_theta_MC", "has_pion") and "pion_four_vectors" in bl_pl:
        pion_fv = bl_pl["pion_four_vectors"][test_idx] / 1000.0
        if key == "pion_E_MC":
            return pion_fv[:, -1]
        if key == "has_pion":
            return pion_fv[:, -1] > 0
        pion_p_MC = np.linalg.norm(pion_fv[:, 1:4], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.arccos(pion_fv[:, 2] / pion_p_MC)

    if key == "int_type_arr":
        truth_labels = data.get("truth_labels", {})
        tl = truth_labels.get(playlist)
        if tl is None and len(truth_labels) == 1:
            tl = next(iter(truth_labels.values()))
        if tl is not None:
            col = tl[:, 1]
            return col.numpy() if hasattr(col, "numpy") else np.asarray(col)

    return None


def _align_per_event_array(
    arr: np.ndarray,
    data: dict,
    playlist: str,
    n_pred: int,
    *,
    key: str | None = None,
) -> np.ndarray:
    """Index *arr* onto the test split so its length matches model predictions."""
    arr = np.asarray(arr)
    if len(arr) == n_pred:
        return arr

    if key is not None:
        rebuilt = _rebuild_per_event_array(data, playlist, n_pred, key)
        if rebuilt is not None:
            return rebuilt

    test_idx = _resolve_test_idx(data, playlist)
    if len(arr) > n_pred and len(test_idx) == n_pred:
        return arr[test_idx]
    raise ValueError(
        f"Per-event array length {len(arr)} does not match prediction count "
        f"{n_pred} (test_idx length {len(test_idx)})."
    )


def _pion_kinematic_bin_mask(
    data: dict,
    *,
    kind: str,
    bin_index: int,
    edges: np.ndarray,
    require_has_pion: bool,
    playlist: str = "1A",
    n_pred: int | None = None,
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
    if n_pred is not None:
        x = _align_per_event_array(
            x, data, playlist, n_pred, key="pion_E_MC" if kind == "E" else "pion_theta_MC"
        )
    bm = mc_value_in_bin(x, edges, bin_index, require_finite=req_fin)
    if require_has_pion:
        has_pion = data["has_pion"]
        if n_pred is not None:
            has_pion = _align_per_event_array(has_pion, data, playlist, n_pred, key="has_pion")
        bm = bm & has_pion
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
    n_pred = len(y_true)
    is_signal = y_true == 1
    is_background = y_true == 0

    if event_mask is not None:
        event_mask = _align_per_event_array(event_mask, data, playlist, n_pred)
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
            playlist=playlist,
            n_pred=n_pred,
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
            playlist=playlist,
            n_pred=n_pred,
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
    n_pred = len(y_true)

    if event_mask is not None:
        event_mask = _align_per_event_array(event_mask, data, playlist, n_pred)
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

    q3 = _align_per_event_array(data["q3_GeV"], data, playlist, n_pred, key="q3_GeV")
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
                    t_cut = (
                        global_thr_map.get(target_fpr, float("nan"))
                        if global_thr_map
                        else float("nan")
                    )
                    if np.isnan(t_cut):
                        tpr_at_fpr[target_fpr] = float("nan")
                    else:
                        sig_mask = y_bin == 1
                        tpr_at_fpr[target_fpr] = float(
                            ((p_bin >= t_cut) & sig_mask).sum()
                            / max(int(sig_mask.sum()), 1)
                        )
                else:
                    idx = int(np.searchsorted(fpr_arr, target_fpr, side="right") - 1)
                    idx = max(0, min(idx, len(tpr_arr) - 1))
                    tpr_at_fpr[target_fpr] = float(tpr_arr[idx])
            metrics.append(
                {
                    "auprc": auprc_val,
                    "auroc": auroc_val,
                    "n_signal": int(n_sig),
                    "tpr_at_fpr": tpr_at_fpr,
                }
            )
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
    n_pred = len(y_true)

    if event_mask is not None:
        event_mask = _align_per_event_array(event_mask, data, playlist, n_pred)
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

    w = _align_per_event_array(data["W_GeV"], data, playlist, n_pred, key="W_GeV")
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
                    t_cut = (
                        global_thr_map.get(target_fpr, float("nan"))
                        if global_thr_map
                        else float("nan")
                    )
                    if np.isnan(t_cut):
                        tpr_at_fpr[target_fpr] = float("nan")
                    else:
                        sig_mask = y_bin == 1
                        tpr_at_fpr[target_fpr] = float(
                            ((p_bin >= t_cut) & sig_mask).sum()
                            / max(int(sig_mask.sum()), 1)
                        )
                else:
                    idx = int(np.searchsorted(fpr_arr, target_fpr, side="right") - 1)
                    idx = max(0, min(idx, len(tpr_arr) - 1))
                    tpr_at_fpr[target_fpr] = float(tpr_arr[idx])
            metrics.append(
                {
                    "auprc": auprc_val,
                    "auroc": auroc_val,
                    "n_signal": int(n_sig),
                    "tpr_at_fpr": tpr_at_fpr,
                }
            )
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
        out[f"tpr@{fpr_val}"] = {
            "mean": np.nanmean(arr, axis=0),
            "std": np.nanstd(arr, axis=0),
        }

    return out
