"""Compact "light" classification PDFs for paper-style summaries.

For each playlist, **single-pion** tasks (CC1π±, CCπ⁰) emit three separate PDFs:
AUPRC / AUROC / TPR@fixed FPR vs *q₃*, the same vs *W*, and a 2×3 figure with
pion *E* (top row) and *θ* (bottom row).

CCNπ (multi-pion) emits vs *q₃* and vs *W* only (two PDFs when *W* data exist).

``plot_classification_q3`` passes ``components=("q3",)`` (CCNπ only);
``plot_classification_Pions`` passes ``components=("pion",)`` (CC1π± and CCπ⁰).

Two-stage API (preferred for caching)
--------------------------------------
1. :func:`compute_light_classification_data` — expensive metric computation, returns
   a list of *draw specs* (picklable dicts).
2. :func:`draw_light_classification_from_cache` — fast figure drawing from specs.

:func:`save_light_classification_pdfs` is the legacy single-call wrapper (compute+draw).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

from src.eval._constants import plot_model_label
from src.eval.classification_plots import (
    compute_all_metrics,
    compute_all_metrics_q3,
    compute_all_metrics_W,
    compute_reco_baseline_recall_per_bin,
    compute_signal_baseline,
    compute_signal_baseline_W,
    data_with_signal_pion_bins,
    _plot_metric_line,
)
from src.eval.classification_plots._constants import (
    _baseline_legend_with_global_fpr,
    _tpr_line_legend_label,
)

if TYPE_CHECKING:
    from src.eval._plot_config import PlotConfig

LightComponent = Literal["pion", "q3"]

# Axis labels match ``plot_steps`` (12); ticks 11; shared legend slightly smaller for density.
_LABEL_FS = 12
_TICK_FS = 11
_LEGEND_FS = 10

_COL_LABELS = ("AUPRC", "AUROC", "Efficiency (TPR)")


def _shared_light_legend(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    """One legend below the figure; first-seen label order, one handle per label."""
    by_label: dict[str, plt.Artist] = {}
    labels_order: list[str] = []
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if li in by_label:
                continue
            by_label[li] = hi
            labels_order.append(li)
    if not labels_order:
        return
    handles = [by_label[k] for k in labels_order]
    n = len(labels_order)
    ncol = max(3, min(6, (n + 2) // 3)) if n > 2 else n
    legend_kw: dict = dict(
        ncol=ncol,
        fontsize=_LEGEND_FS,
        frameon=True,
        fancybox=True,
        facecolor="white",
        edgecolor="0.4",
        columnspacing=1.0,
        handletextpad=0.5,
    )
    try:
        fig.legend(handles, labels_order, loc="outside lower center", **legend_kw)
    except (TypeError, ValueError):
        fig.legend(
            handles,
            labels_order,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            **legend_kw,
        )


def _save_single_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def _figure_metrics_1x3(
    all_metrics: dict[str, dict],
    x: np.ndarray,
    xlabel: str,
    baseline_auprc: np.ndarray,
    fixed_fpr: list[float],
    reco_baseline_tpr: np.ndarray,
    reco_label: str,
    colors: dict[str, str],
    *,
    log_x: bool = False,
    reco_baseline_global_fpr: float | None = None,
) -> plt.Figure:
    """One row: AUPRC | AUROC | TPR vs a common *x* (global FPR only)."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    axes[0].plot(
        x, baseline_auprc, "o--", color="gray", label="Random baseline", zorder=1
    )

    for model_name, agg in sorted(all_metrics.items(), key=lambda kv: kv[0]):
        clr = {"color": colors.get(model_name, "tab:gray")}
        _plot_metric_line(axes[0], x, agg["auprc"], plot_model_label(model_name), True, **clr)
        _plot_metric_line(axes[1], x, agg["auroc"], plot_model_label(model_name), True, **clr)
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            lbl = _tpr_line_legend_label(model_name, fpr_val, True)
            _plot_metric_line(axes[2], x, agg[key], lbl, True, **clr)

    bl_lbl = (
        _baseline_legend_with_global_fpr(reco_label, reco_baseline_global_fpr)
        if reco_baseline_global_fpr is not None
        and np.isfinite(reco_baseline_global_fpr)
        else reco_label
    )
    axes[2].plot(x, reco_baseline_tpr, "s--", color="black", label=bl_lbl, zorder=2)

    for col, metric in enumerate(_COL_LABELS):
        ax = axes[col]
        ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
        ax.set_ylabel(metric, fontsize=_LABEL_FS)
        ax.tick_params(axis="both", labelsize=_TICK_FS)
        ax.grid(True, alpha=0.35)
        if log_x:
            ax.set_xscale("log")

    _shared_light_legend(fig, axes.ravel())
    return fig


def _figure_metrics_2x3_pion(
    all_metrics: dict[str, dict],
    x_E: np.ndarray,
    x_theta: np.ndarray,
    baseline_E: np.ndarray,
    baseline_theta: np.ndarray,
    fixed_fpr: list[float],
    reco_tpr_E: np.ndarray,
    reco_tpr_theta: np.ndarray,
    reco_label: str,
    reco_baseline_global_fpr: float,
    colors: dict[str, str],
) -> plt.Figure:
    """Two rows: pion *E* (top) and *θ* (bottom); matches ``plot_cc1pi_vs_pion_kinematics`` (3-col part)."""
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(
        x_E, baseline_E, "o--", color="gray", label="Random baseline", zorder=1
    )
    axes[1, 0].plot(
        x_theta, baseline_theta, "o--", color="gray", label="Random baseline", zorder=1
    )

    for model_name, m in sorted(all_metrics.items(), key=lambda kv: kv[0]):
        clr = {"color": colors.get(model_name, "tab:gray")}
        agg_E, agg_th = m["E"], m["theta"]
        _plot_metric_line(
            axes[0, 0], x_E, agg_E["auprc"], plot_model_label(model_name), True, **clr
        )
        _plot_metric_line(
            axes[0, 1], x_E, agg_E["auroc"], plot_model_label(model_name), True, **clr
        )
        _plot_metric_line(
            axes[1, 0], x_theta, agg_th["auprc"], plot_model_label(model_name), True, **clr
        )
        _plot_metric_line(
            axes[1, 1], x_theta, agg_th["auroc"], plot_model_label(model_name), True, **clr
        )
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            lab = _tpr_line_legend_label(model_name, fpr_val, True)
            _plot_metric_line(axes[0, 2], x_E, agg_E[key], lab, True, **clr)
            _plot_metric_line(axes[1, 2], x_theta, agg_th[key], lab, True, **clr)

    bl_lbl = _baseline_legend_with_global_fpr(reco_label, reco_baseline_global_fpr)
    axes[0, 2].plot(x_E, reco_tpr_E, "s--", color="black", label=bl_lbl, zorder=2)
    axes[1, 2].plot(
        x_theta, reco_tpr_theta, "s--", color="black", label=bl_lbl, zorder=2
    )

    for col, metric in enumerate(_COL_LABELS):
        ax0 = axes[0, col]
        ax1 = axes[1, col]
        ax0.set_xlabel(r"True $E_\pi$ [GeV]", fontsize=_LABEL_FS)
        ax1.set_xlabel(r"True $\theta_\pi$ [rad]", fontsize=_LABEL_FS)
        ax0.set_ylabel(metric, fontsize=_LABEL_FS)
        ax1.set_ylabel(metric, fontsize=_LABEL_FS)
        ax0.tick_params(axis="both", labelsize=_TICK_FS)
        ax1.tick_params(axis="both", labelsize=_TICK_FS)
        ax0.grid(True, alpha=0.35)
        ax1.grid(True, alpha=0.35)
        ax0.set_xscale("log")
        if len(x_E) > 0 and np.all(np.isfinite(x_E[[0, -1]])):
            ax0.set_xlim(float(x_E[0]) * 0.8, float(x_E[-1]) * 1.2)

    _shared_light_legend(fig, axes.ravel())
    return fig


# ---------------------------------------------------------------------------
# Compute helpers (private)
# ---------------------------------------------------------------------------


def _compute_pion_bundle_specs(
    *,
    results: dict,
    data: dict,
    data_w: dict | None,
    pid: np.ndarray,
    signal_classes: list[int],
    y_pred: np.ndarray,
    baseline_fpr: float,
    tag: str,
    playlist: str,
) -> list[dict[str, Any]]:
    """Compute metrics for one pion task; return draw specs (no matplotlib calls)."""
    if not np.isfinite(baseline_fpr):
        return []
    fpr = [baseline_fpr]
    data_sp = data_with_signal_pion_bins(
        data,
        pid,
        signal_classes,
        pion_quantile_require_has_pion=False,
        pion_bin_edge_method="equal_frequency",
    )
    metrics_q3 = compute_all_metrics_q3(
        results, data, signal_classes=signal_classes, fixed_fpr=fpr, playlist=playlist,
    )
    bl_q3 = compute_signal_baseline(
        results, data, signal_classes=signal_classes, playlist=playlist,
        pion_bins_require_has_pion=False,
    )["q3"]
    y_true = np.isin(pid, signal_classes).astype(int)
    is_signal = y_true == 1
    reco_q3 = compute_reco_baseline_recall_per_bin(
        y_pred, is_signal, data["q3_GeV"], data["q3_bin_edges"],
    )
    specs: list[dict[str, Any]] = [
        {
            "type": "1x3",
            "all_metrics": metrics_q3,
            "x": data["q3_bin_mids"],
            "xlabel": r"True $q_3$ [GeV]",
            "baseline_auprc": bl_q3,
            "fixed_fpr": fpr,
            "reco_baseline_tpr": reco_q3,
            "reco_label": "Baseline",
            "log_x": False,
            "reco_baseline_global_fpr": baseline_fpr,
            "filename": f"eval_classification_light_{tag}_q3_{playlist}.pdf",
        }
    ]
    if data_w is not None:
        metrics_W = compute_all_metrics_W(
            results, data_w, signal_classes=signal_classes, fixed_fpr=fpr,
            playlist=playlist, use_global_fpr=True,
        )
        bl_W = compute_signal_baseline_W(
            results, data_w, signal_classes=signal_classes, playlist=playlist,
        )
        reco_W = compute_reco_baseline_recall_per_bin(
            y_pred, is_signal, data_w["W_GeV"], data_w["W_bin_edges"],
        )
        specs.append({
            "type": "1x3",
            "all_metrics": metrics_W,
            "x": data_w["W_bin_mids"],
            "xlabel": r"True hadronic $W$ [GeV]",
            "baseline_auprc": bl_W,
            "fixed_fpr": fpr,
            "reco_baseline_tpr": reco_W,
            "reco_label": "Baseline",
            "log_x": False,
            "reco_baseline_global_fpr": baseline_fpr,
            "filename": f"eval_classification_light_{tag}_W_{playlist}.pdf",
        })
    metrics_pion = compute_all_metrics(
        results, data_sp, signal_classes=signal_classes, fixed_fpr=fpr,
        playlist=playlist, pion_bins_require_has_pion=False,
    )
    bl = compute_signal_baseline(
        results, data_sp, signal_classes=signal_classes, playlist=playlist,
        pion_bins_require_has_pion=False,
    )
    reco_E = compute_reco_baseline_recall_per_bin(
        y_pred, is_signal, data_sp["pion_E_MC"], data_sp["pion_E_MC_bins"], has_pion=None,
    )
    reco_th = compute_reco_baseline_recall_per_bin(
        y_pred, is_signal, data_sp["pion_theta_MC"], data_sp["pion_theta_MC_bins"],
        has_pion=None, finite_bin_var=True,
    )
    specs.append({
        "type": "2x3_pion",
        "all_metrics": metrics_pion,
        "x_E": data_sp["pion_E_MC_bins_mid"],
        "x_theta": data_sp["pion_theta_MC_bins_mid"],
        "baseline_E": bl["E"],
        "baseline_theta": bl["theta"],
        "fixed_fpr": fpr,
        "reco_tpr_E": reco_E,
        "reco_tpr_theta": reco_th,
        "reco_label": "Baseline",
        "reco_baseline_global_fpr": baseline_fpr,
        "filename": f"eval_classification_light_{tag}_pion_kinematics_{playlist}.pdf",
    })
    return specs


def _compute_q3_bundle_specs(
    *,
    results: dict,
    data: dict,
    data_w: dict | None,
    pid: np.ndarray,
    multi_pi_classes: list[int],
    n_muons: np.ndarray,
    n_charged_prongs: np.ndarray,
    improved_nmichel: np.ndarray,
    playlist: str,
) -> list[dict[str, Any]]:
    """Compute CCNπ vs q₃ (and W) metrics; return draw specs."""
    y_true_ccnpi = np.isin(pid, multi_pi_classes).astype(int)
    y_pred_ccnpi = (
        (n_muons == 1) & (n_charged_prongs >= 1) & (improved_nmichel >= 1)
    ).astype(int)
    fpn = int(np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 0)))
    tnn = int(np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 0)))
    baseline_fpr_ccnpi = fpn / (fpn + tnn) if (fpn + tnn) > 0 else float("nan")
    if not np.isfinite(baseline_fpr_ccnpi):
        return []
    fpr_n = [baseline_fpr_ccnpi]
    metrics_q3 = compute_all_metrics_q3(
        results, data, signal_classes=multi_pi_classes, fixed_fpr=fpr_n, playlist=playlist,
    )
    bl_q3 = compute_signal_baseline(
        results, data, signal_classes=multi_pi_classes, playlist=playlist,
        pion_bins_require_has_pion=False,
    )["q3"]
    is_signal_ccnpi = y_true_ccnpi == 1
    reco_q3 = compute_reco_baseline_recall_per_bin(
        y_pred_ccnpi, is_signal_ccnpi, data["q3_GeV"], data["q3_bin_edges"],
    )
    specs: list[dict[str, Any]] = [
        {
            "type": "1x3",
            "all_metrics": metrics_q3,
            "x": data["q3_bin_mids"],
            "xlabel": r"True $q_3$ [GeV]",
            "baseline_auprc": bl_q3,
            "fixed_fpr": fpr_n,
            "reco_baseline_tpr": reco_q3,
            "reco_label": "Baseline",
            "log_x": False,
            "reco_baseline_global_fpr": baseline_fpr_ccnpi,
            "filename": f"eval_classification_light_ccnpi_q3_{playlist}.pdf",
        }
    ]
    if data_w is not None:
        metrics_W = compute_all_metrics_W(
            results, data_w, signal_classes=multi_pi_classes, fixed_fpr=fpr_n,
            playlist=playlist, use_global_fpr=True,
        )
        bl_W = compute_signal_baseline_W(
            results, data_w, signal_classes=multi_pi_classes, playlist=playlist,
        )
        reco_W = compute_reco_baseline_recall_per_bin(
            y_pred_ccnpi, is_signal_ccnpi, data_w["W_GeV"], data_w["W_bin_edges"],
        )
        specs.append({
            "type": "1x3",
            "all_metrics": metrics_W,
            "x": data_w["W_bin_mids"],
            "xlabel": r"True hadronic $W$ [GeV]",
            "baseline_auprc": bl_W,
            "fixed_fpr": fpr_n,
            "reco_baseline_tpr": reco_W,
            "reco_label": "Baseline",
            "log_x": False,
            "reco_baseline_global_fpr": baseline_fpr_ccnpi,
            "filename": f"eval_classification_light_ccnpi_W_{playlist}.pdf",
        })
    return specs


# ---------------------------------------------------------------------------
# Public two-stage API
# ---------------------------------------------------------------------------


def compute_light_classification_data(
    results: dict,
    data_by_playlist: dict,
    playlists: list[str],
    components: tuple[LightComponent, ...] = ("pion", "q3"),
    *,
    data_w_by_playlist: dict | None = None,
) -> list[dict[str, Any]]:
    """Compute all metrics for light classification figures (expensive).

    Returns a list of *draw specs* — picklable dicts that can be passed directly
    to :func:`draw_light_classification_from_cache` without re-running any
    sklearn metric calls.
    """
    specs: list[dict[str, Any]] = []
    do_pion = "pion" in components
    do_q3 = "q3" in components

    cc1pi_classes = [0]
    cc1pi0_classes = [2]
    multi_pi_classes = [0, 1]
    PI0_MASS = 134.977
    DELTA_M = PI0_MASS

    for playlist in playlists:
        data = data_by_playlist[playlist]
        data_w = data_w_by_playlist[playlist] if data_w_by_playlist else None

        test_idx = data["test_idx"][playlist]
        baselines_pl = data["baselines"][playlist]
        first_model = next(iter(results))
        run0 = results[first_model][0][playlist]
        pid = run0["pid"]
        n_muons = baselines_pl["n_muons"][test_idx]
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]

        if do_pion:
            # --- CC1π± ---
            y_true_cc1pi = np.isin(pid, cc1pi_classes).astype(int)
            y_pred_cc1pi = (
                (n_muons == 1) & (n_charged_prongs == 1) & (improved_nmichel == 1)
            ).astype(int)
            fp = int(np.sum((y_pred_cc1pi == 1) & (y_true_cc1pi == 0)))
            tn = int(np.sum((y_pred_cc1pi == 0) & (y_true_cc1pi == 0)))
            baseline_fpr_cc1pi = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
            specs.extend(
                _compute_pion_bundle_specs(
                    results=results,
                    data=data,
                    data_w=data_w,
                    pid=pid,
                    signal_classes=cc1pi_classes,
                    y_pred=y_pred_cc1pi,
                    baseline_fpr=baseline_fpr_cc1pi,
                    tag="cc1pi",
                    playlist=playlist,
                )
            )

            # --- CCπ⁰ ---
            is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
            two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
            n_michel = baselines_pl["improved_nmichel"][test_idx]
            y_true_pi0 = np.isin(pid, cc1pi0_classes).astype(int)
            y_pred_pi0 = (
                (n_muons == 1)
                & (is_pizero_signal == 2)
                & (np.abs(two_gamma_inv_mass - PI0_MASS) < DELTA_M)
                & (n_michel == 0)
            ).astype(int)
            fp0 = int(np.sum((y_pred_pi0 == 1) & (y_true_pi0 == 0)))
            tn0 = int(np.sum((y_pred_pi0 == 0) & (y_true_pi0 == 0)))
            baseline_fpr_pi0 = fp0 / (fp0 + tn0) if (fp0 + tn0) > 0 else float("nan")
            specs.extend(
                _compute_pion_bundle_specs(
                    results=results,
                    data=data,
                    data_w=data_w,
                    pid=pid,
                    signal_classes=cc1pi0_classes,
                    y_pred=y_pred_pi0,
                    baseline_fpr=baseline_fpr_pi0,
                    tag="cc1pi0",
                    playlist=playlist,
                )
            )

        if do_q3:
            specs.extend(
                _compute_q3_bundle_specs(
                    results=results,
                    data=data,
                    data_w=data_w,
                    pid=pid,
                    multi_pi_classes=multi_pi_classes,
                    n_muons=n_muons,
                    n_charged_prongs=n_charged_prongs,
                    improved_nmichel=improved_nmichel,
                    playlist=playlist,
                )
            )

    return specs


def draw_light_classification_from_cache(
    specs: list[dict[str, Any]],
    clrs_dict_full: dict[str, str],
    out_dir: Path,
    *,
    cfg: PlotConfig | None = None,
) -> None:
    """Draw light classification PDFs from pre-computed draw specs (fast).

    *cfg* optionally filters metrics to the config model subset and applies
    config colors before drawing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    clrs = clrs_dict_full
    if cfg is not None:
        clrs = {**clrs, **cfg.colors()}

    for spec in specs:
        all_metrics: dict = spec["all_metrics"]
        if cfg is not None:
            all_metrics = cfg.filter_dict(all_metrics)

        out_path = out_dir / spec["filename"]

        if spec["type"] == "1x3":
            fig = _figure_metrics_1x3(
                all_metrics,
                spec["x"],
                spec["xlabel"],
                spec["baseline_auprc"],
                spec["fixed_fpr"],
                spec["reco_baseline_tpr"],
                spec["reco_label"],
                clrs,
                log_x=spec.get("log_x", False),
                reco_baseline_global_fpr=spec.get("reco_baseline_global_fpr"),
            )
        elif spec["type"] == "2x3_pion":
            fig = _figure_metrics_2x3_pion(
                all_metrics,
                spec["x_E"],
                spec["x_theta"],
                spec["baseline_E"],
                spec["baseline_theta"],
                spec["fixed_fpr"],
                spec["reco_tpr_E"],
                spec["reco_tpr_theta"],
                spec["reco_label"],
                spec["reco_baseline_global_fpr"],
                clrs,
            )
        else:
            raise ValueError(f"Unknown draw spec type: {spec['type']!r}")

        _save_single_fig(fig, out_path)


def save_light_classification_pdfs(
    out_dir: Path,
    results: dict,
    data_by_playlist: dict,
    clrs_dict_full: dict[str, str],
    playlists: list[str],
    components: tuple[LightComponent, ...] = ("pion", "q3"),
    *,
    data_w_by_playlist: dict | None = None,
) -> None:
    """Write light PDFs under *out_dir* (compute + draw in one shot).

    For caching, prefer calling :func:`compute_light_classification_data` once,
    saving the result, then calling :func:`draw_light_classification_from_cache`.
    """
    specs = compute_light_classification_data(
        results,
        data_by_playlist,
        playlists,
        components,
        data_w_by_playlist=data_w_by_playlist,
    )
    draw_light_classification_from_cache(specs, clrs_dict_full, out_dir)


# --- Legacy notebook helpers (multi-panel counts / TPR); not used by ``save_light_classification_pdfs``. ---


def per_bin_total_and_signal(
    y_true_signal: np.ndarray,
    bin_var: np.ndarray,
    bin_edges: np.ndarray,
    has_pion: np.ndarray | None = None,
    require_finite_bin_var: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    from src.eval.classification_plots import mc_value_in_bin

    n_tot, n_sig = [], []
    for i in range(len(bin_edges) - 1):
        bm = mc_value_in_bin(
            bin_var, bin_edges, i, require_finite=require_finite_bin_var
        )
        if has_pion is not None:
            bm = bm & has_pion
        n_tot.append(int(bm.sum()))
        n_sig.append(int((bm & y_true_signal).sum()))
    return np.array(n_tot), np.array(n_sig)


def sb_ratio(n_tot: np.ndarray, n_sig: np.ndarray) -> np.ndarray:
    n_bg = n_tot - n_sig
    return np.divide(
        n_sig, n_bg, out=np.full_like(n_sig, np.nan, dtype=float), where=n_bg > 0
    )


def sb_ratio_vs_global_bkg(n_sig: np.ndarray, n_bg_global: int) -> np.ndarray:
    if n_bg_global <= 0:
        return np.full_like(n_sig, np.nan, dtype=float)
    return np.asarray(n_sig, dtype=float) / float(n_bg_global)


def plot_histogram_counts(
    ax: plt.Axes,
    bin_edges: np.ndarray,
    n_tot: np.ndarray,
    n_sig: np.ndarray,
    xlabel: str,
    log_x: bool = False,
    leg_title: str | None = None,
    n_bg_global: int | None = None,
) -> None:
    COLOR_N_TOTAL = "#9e9e9e"
    COLOR_N_SIGNAL = "#1f77b4"
    HIST_EDGE = "0.25"
    HIST_EDGELINE = 0.35
    HIST_ALPHA_TOTAL = 0.55
    HIST_ALPHA_SIGNAL = 0.9

    edges = np.asarray(bin_edges, dtype=float)
    w = np.diff(edges)
    x0 = edges[:-1]
    ax.bar(
        x0,
        n_tot,
        width=w,
        align="edge",
        color=COLOR_N_TOTAL,
        alpha=HIST_ALPHA_TOTAL,
        edgecolor=HIST_EDGE,
        linewidth=HIST_EDGELINE,
        label=r"$N_{\mathrm{in\,bin}}$ (all classes)",
    )
    ax.bar(
        x0,
        n_sig,
        width=w,
        align="edge",
        color=COLOR_N_SIGNAL,
        alpha=HIST_ALPHA_SIGNAL,
        edgecolor=HIST_EDGE,
        linewidth=HIST_EDGELINE,
        label=r"$N_{\mathrm{signal\,in\,bin}}$",
    )
    if n_bg_global is not None and n_bg_global > 0:
        ax.axhline(
            n_bg_global,
            color="#424242",
            linestyle="--",
            linewidth=1.4,
            label=rf"$N_{{\mathrm{{bkg}}}}^{{\mathrm{{glob}}}} = {n_bg_global}$ (ROC negatives)",
        )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events per bin")
    ax.set_title(r"Kinematic bin counts (ROC uses global $N_{\mathrm{bkg}}$)")
    ax.grid(True, axis="y", alpha=0.35)
    if leg_title is not None:
        leg = ax.legend(fontsize=7, title=leg_title)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(8)


def plot_sb_stairs(
    ax,
    bin_edges: np.ndarray,
    sb: np.ndarray,
    xlabel: str,
    log_x: bool = False,
    leg_title: str | None = None,
    y_title: str | None = None,
) -> None:
    COLOR_SB = "#2ca02c"
    edges = np.asarray(bin_edges, dtype=float)
    sb = np.asarray(sb, dtype=float)
    sb_plot = ma.masked_invalid(sb)
    ax.stairs(sb_plot, edges, color=COLOR_SB, linewidth=1.6, label=r"$S/B$")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Signal / background ($S/B$)")
    if y_title is None:
        y_title = r"$S/B$ ($B = N_{\mathrm{in\,bin}} - N_{\mathrm{signal}}$)"
    ax.set_title(y_title)
    ax.grid(True, alpha=0.35)
    if leg_title is not None:
        leg = ax.legend(fontsize=7, title=leg_title)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(8)


def plot_tpr_fixed_fpr_two_panel(
    metrics,
    x_left,
    x_right,
    left_key,
    right_key,
    fixed_fpr,
    x_label_left,
    x_label_right,
    reco_baseline_tpr,
    reco_label,
    title,
    playlist: str,
    task_legend_line2: str,
    log_x_left=False,
    colors=None,
):
    def _legend_title_playlist_task(_playlist: str, task_line2: str) -> str:
        return f"Minerva Open Data Playlist 1A/1B\n{task_line2}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), tight_layout=True)
    ax_l, ax_r = axes[0], axes[1]
    leg_title = _legend_title_playlist_task(playlist, task_legend_line2)

    for model_name, model_metrics in metrics.items():
        clr = {} if colors is None else {"color": colors.get(model_name)}
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            _plot_metric_line(
                ax_l,
                x_left,
                model_metrics[left_key][key],
                f"{plot_model_label(model_name)} (FPR={fpr_val:.0%})",
                True,
                **clr,
            )
            _plot_metric_line(
                ax_r,
                x_right,
                model_metrics[right_key][key],
                f"{plot_model_label(model_name)} (FPR={fpr_val:.0%})",
                True,
                **clr,
            )

    ax_l.plot(
        x_left, reco_baseline_tpr[left_key], "s--", color="black", label=reco_label
    )
    ax_r.plot(
        x_right, reco_baseline_tpr[right_key], "s--", color="black", label=reco_label
    )

    ax_l.set_xlabel(x_label_left)
    ax_r.set_xlabel(x_label_right)
    ax_l.set_ylabel("Efficiency (TPR)")
    ax_r.set_ylabel("Efficiency (TPR)")
    ax_l.set_title("TPR @ fixed FPR")
    ax_r.set_title("TPR @ fixed FPR")

    if log_x_left:
        ax_l.set_xscale("log")

    for ax in (ax_l, ax_r):
        ax.grid(True)
        leg = ax.legend(fontsize=7, title=leg_title)
        if leg.get_title() is not None:
            leg.get_title().set_fontsize(8)

    fig.suptitle(title, fontsize=14)
    return fig


def plot_counts_sb_two_panel(
    bin_edges_left: np.ndarray,
    bin_edges_right: np.ndarray,
    n_total_signal_left: tuple[np.ndarray, np.ndarray],
    n_total_signal_right: tuple[np.ndarray, np.ndarray],
    x_label_left: str,
    x_label_right: str,
    title: str,
    playlist: str,
    task_legend_line2: str,
    log_x_left: bool = False,
    n_bg_global: int | None = None,
):
    def _legend_title_playlist_task(_playlist: str, task_line2: str) -> str:
        return f"Minerva Open Data Playlist 1A/1B\n{task_line2}"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), tight_layout=True)
    leg_title = _legend_title_playlist_task(playlist, task_legend_line2)

    n_tot_l, n_sig_l = n_total_signal_left
    n_tot_r, n_sig_r = n_total_signal_right
    if n_bg_global is not None:
        sb_l = sb_ratio_vs_global_bkg(n_sig_l, n_bg_global)
        sb_r = sb_ratio_vs_global_bkg(n_sig_r, n_bg_global)
        sb_title = (
            r"$S/B$ with $B = N_{\mathrm{bkg}}^{\mathrm{glob}}$ (all non-signal, ROC)"
        )
    else:
        sb_l = sb_ratio(n_tot_l, n_sig_l)
        sb_r = sb_ratio(n_tot_r, n_sig_r)
        sb_title = None

    plot_histogram_counts(
        axes[0, 0],
        bin_edges_left,
        n_tot_l,
        n_sig_l,
        x_label_left,
        log_x=log_x_left,
        leg_title=leg_title,
        n_bg_global=n_bg_global,
    )
    plot_sb_stairs(
        axes[0, 1],
        bin_edges_left,
        sb_l,
        x_label_left,
        log_x=log_x_left,
        leg_title=leg_title,
        y_title=sb_title,
    )
    plot_histogram_counts(
        axes[1, 0],
        bin_edges_right,
        n_tot_r,
        n_sig_r,
        x_label_right,
        log_x=False,
        leg_title=leg_title,
        n_bg_global=n_bg_global,
    )
    plot_sb_stairs(
        axes[1, 1],
        bin_edges_right,
        sb_r,
        x_label_right,
        log_x=False,
        leg_title=leg_title,
        y_title=sb_title,
    )

    fig.suptitle(title, fontsize=14)
    return fig
