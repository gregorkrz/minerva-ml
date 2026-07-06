"""Aggregate metrics across models (pion, q3, W)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._constants import DEFAULT_FIXED_FPR
from ._metrics_binned import (
    _align_per_event_array,
    _pion_kinematic_bin_mask,
    aggregate_metrics,
    compute_binned_metrics_q3,
    compute_binned_metrics_single,
    compute_binned_metrics_W,
    get_signal_probabilities,
    mc_value_in_bin,
)


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


def metrics_W_by_model_data(
    results: dict[str, list[dict]],
    default_data: dict,
    data_by_model: dict[str, dict] | None,
    signal_classes: list[int],
    threshold: float = 0.5,
    fixed_fpr: list[float] | None = None,
    event_mask: np.ndarray | None = None,
    playlist: str = "1A",
    use_global_fpr: bool = True,
) -> dict[str, dict]:
    """W-binned metrics with optional per-model aligned ``data`` dicts."""
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    out: dict[str, dict] = {}
    for model_name in sorted(results.keys()):
        data = (
            data_by_model.get(model_name, default_data)
            if data_by_model
            else default_data
        )
        out.update(
            compute_all_metrics_W(
                {model_name: results[model_name]},
                data,
                signal_classes,
                threshold=threshold,
                fixed_fpr=fixed_fpr,
                event_mask=event_mask,
                playlist=playlist,
                use_global_fpr=use_global_fpr,
            )
        )
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
    n_pred = len(y_true)
    is_signal = y_true == 1
    is_background = y_true == 0

    if event_mask is not None:
        event_mask = _align_per_event_array(event_mask, data, playlist, n_pred)
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
            playlist=playlist,
            n_pred=n_pred,
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
            playlist=playlist,
            n_pred=n_pred,
        )
        n_sig = (is_signal & bm).sum()
        baseline_theta.append(n_sig / (n_sig + n_bg) if n_sig > 0 else np.nan)

    # q3 bins
    q3 = _align_per_event_array(data["q3_GeV"], data, playlist, n_pred, key="q3_GeV")
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
    n_pred = len(y_true)

    if event_mask is not None:
        event_mask = _align_per_event_array(event_mask, data, playlist, n_pred)

    w = _align_per_event_array(data["W_GeV"], data, playlist, n_pred, key="W_GeV")
    edges = data["W_bin_edges"]
    baseline_w = []
    for i in range(len(edges) - 1):
        bm = mc_value_in_bin(w, edges, i, require_finite=False)
        if event_mask is not None:
            bm = bm & event_mask
        n_sig = (y_true[bm] == 1).sum() if bm.sum() > 0 else 0
        baseline_w.append(y_true[bm].mean() if n_sig > 0 else np.nan)
    return np.array(baseline_w)
