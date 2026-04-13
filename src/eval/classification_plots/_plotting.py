"""Matplotlib figures for classification evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from ._constants import (
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    DEFAULT_FIXED_FPR,
    MC_INT_TYPE,
    W_METRICS_XLIM_GEV,
    _baseline_legend_with_global_fpr,
    _classification_legend_kw,
    _default_signal_label,
    _global_reco_baseline_fpr,
    _reco_baseline_fpr_on_mask,
    _tpr_column_title_vs_kinematics,
    _tpr_line_legend_label,
)
from ._metrics_binned import get_signal_probabilities, mc_value_in_bin
from ._metrics_tasks import (
    compute_all_metrics,
    compute_all_metrics_q3,
    compute_all_metrics_W,
    compute_signal_baseline,
    compute_signal_baseline_W,
)
from ._reco_baseline import (
    compute_reco_baseline_fpr_per_bin,
    compute_reco_baseline_recall_per_bin,
)


def _set_xlim_w_metrics(ax: plt.Axes) -> None:
    """Set a consistent *W* [GeV] axis span on metric / histogram panels."""
    lo, hi = W_METRICS_XLIM_GEV
    ax.set_xlim(lo, hi)


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
    (line,) = ax.plot(x, mean, "o-", label=label, **kwargs)
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
                ax.set_title(
                    f"{title_prefix}{col_labels[col]} vs. {xlabel.split('[')[0].strip()}"
                )
            else:
                ax.set_title(
                    f"{title_prefix}{tpr_title} vs. {xlabel.split('[')[0].strip()}"
                )
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
        raise ValueError(
            "results and signal_classes must both be set or both be omitted"
        )

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
            y_tb = get_signal_probabilities(results[fm][0], signal_classes, playlist)[
                "ytrue"
            ]
            if len(reco_baseline_pred) == len(y_tb):
                fpr_for_legend = _global_reco_baseline_fpr(reco_baseline_pred, y_tb)
        bl_tpr_label = _baseline_legend_with_global_fpr(
            reco_baseline_label, fpr_for_legend
        )

    # Random baseline (circle markers like models; dashed + gray to distinguish)
    axes[0, 0].plot(E_mid, baseline["E"], "o--", color="gray", label="Random baseline")
    axes[1, 0].plot(
        theta_mid, baseline["theta"], "o--", color="gray", label="Random baseline"
    )

    for model_name, metrics in sorted(all_metrics.items(), key=lambda kv: kv[0]):
        agg_E = metrics["E"]
        agg_theta = metrics["theta"]
        clr = {} if colors is None else {"color": colors.get(model_name)}

        _plot_metric_line(
            axes[0, 0], E_mid, agg_E["auprc"], model_name, uncertainties, **clr
        )
        _plot_metric_line(
            axes[1, 0], theta_mid, agg_theta["auprc"], model_name, uncertainties, **clr
        )
        _plot_metric_line(
            axes[0, 1], E_mid, agg_E["auroc"], model_name, uncertainties, **clr
        )
        _plot_metric_line(
            axes[1, 1], theta_mid, agg_theta["auroc"], model_name, uncertainties, **clr
        )

        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            lbl = _tpr_line_legend_label(model_name, fpr_val, use_global_fpr)
            _plot_metric_line(
                axes[0, 2],
                E_mid,
                agg_E[key],
                lbl,
                uncertainties,
                **clr,
            )
            _plot_metric_line(
                axes[1, 2],
                theta_mid,
                agg_theta[key],
                lbl,
                uncertainties,
                **clr,
            )

    if reco_baseline_tpr is not None:
        if "E" in reco_baseline_tpr:
            axes[0, 2].plot(
                E_mid,
                reco_baseline_tpr["E"],
                "s--",
                color="black",
                label=bl_tpr_label,
            )
        if "theta" in reco_baseline_tpr:
            axes[1, 2].plot(
                theta_mid,
                reco_baseline_tpr["theta"],
                "s--",
                color="black",
                label=bl_tpr_label,
            )

    if n_cols == 4 and y_true_binary is not None:
        has_pion = data["has_pion"]
        hp_bin = has_pion if pion_bins_require_has_pion else None
        plot_mask_e = (
            has_pion
            if pion_bins_require_has_pion
            else np.ones(len(data["pion_E_MC"]), dtype=bool)
        )
        plot_mask_th = (
            has_pion
            if pion_bins_require_has_pion
            else np.isfinite(data["pion_theta_MC"])
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
        suptitle = rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}"
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
        _plot_metric_line(
            axes[0], q3_mid, agg["auprc"], model_name, uncertainties, **clr
        )
        _plot_metric_line(
            axes[1], q3_mid, agg["auroc"], model_name, uncertainties, **clr
        )
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            _plot_metric_line(
                axes[2],
                q3_mid,
                agg[key],
                _tpr_line_legend_label(model_name, fpr_val, use_global_fpr),
                uncertainties,
                **clr,
            )

    if reco_baseline_tpr_q3 is not None:
        axes[2].plot(
            q3_mid,
            reco_baseline_tpr_q3,
            "s--",
            color="black",
            label=reco_baseline_label,
        )

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
        title = rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}"
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
        raise ValueError(
            "results and signal_classes must both be set or both be omitted"
        )

    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    n_cols = 4 if results is not None else 3
    fig_w = 22.0 if n_cols == 4 else 17.0
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 5), tight_layout=True)

    w_mid = data["W_bin_mids"]

    axes[0].plot(w_mid, baseline_W, "o--", color="gray", label="Random baseline")

    for model_name, agg in sorted(all_metrics_W.items(), key=lambda kv: kv[0]):
        clr = {} if colors is None else {"color": colors.get(model_name)}
        _plot_metric_line(
            axes[0], w_mid, agg["auprc"], model_name, uncertainties, **clr
        )
        _plot_metric_line(
            axes[1], w_mid, agg["auroc"], model_name, uncertainties, **clr
        )
        for fpr_val in fixed_fpr:
            key = f"tpr@{fpr_val}"
            _plot_metric_line(
                axes[2],
                w_mid,
                agg[key],
                _tpr_line_legend_label(model_name, fpr_val, use_global_fpr),
                uncertainties,
                **clr,
            )

    if reco_baseline_tpr_W is not None:
        fpr_for_legend = reco_baseline_global_fpr
        if (
            fpr_for_legend is None
            and reco_baseline_pred is not None
            and results is not None
            and signal_classes is not None
        ):
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
        title = rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}"
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
    label_for_signal = (
        signal_label
        if signal_label is not None
        else _default_signal_label(signal_classes)
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
        hist_pion_mask = (
            has_pion
            if pion_bins_require_has_pion
            else np.ones(len(hist_var), dtype=bool)
        )
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
                    0.5,
                    0.5,
                    "No data",
                    transform=axes[row_idx, col].transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="gray",
                )
                axes[row_idx, col].set_title(f"{int_name} (N=0)")
            if x_var == "W":
                for col in range(4):
                    _set_xlim_w_metrics(axes[row_idx, col])
            continue

        n_signal = int(((y_true_binary == 1) & int_mask).sum())
        no_signal_msg = f"No {label_for_signal} signal in this interaction type"

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
            ax_h.set_title(
                f"{int_name} (N={n_events:,}) — events (orange = signal, bottom)"
            )
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
                results,
                data,
                signal_classes,
                int_mask,
                playlist,
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
        axes[row_idx, 0].plot(
            x_mid, bl_values, "o--", color="gray", label="Random baseline"
        )

        for model_name, agg in sorted(all_agg.items(), key=lambda kv: kv[0]):
            clr = {} if colors is None else {"color": colors.get(model_name)}
            _plot_metric_line(
                axes[row_idx, 0], x_mid, agg["auprc"], model_name, uncertainties, **clr
            )
            _plot_metric_line(
                axes[row_idx, 1], x_mid, agg["auroc"], model_name, uncertainties, **clr
            )
            for fpr_val in fixed_fpr:
                key = f"tpr@{fpr_val}"
                _plot_metric_line(
                    axes[row_idx, 2],
                    x_mid,
                    agg[key],
                    _tpr_line_legend_label(model_name, fpr_val, use_global_fpr),
                    uncertainties,
                    **clr,
                )

        # Reconstruction baseline on TPR panel
        if reco_baseline_pred is not None:
            is_signal_masked = (y_true_binary == 1) & int_mask
            if x_var == "q3":
                reco_bl = compute_reco_baseline_recall_per_bin(
                    reco_baseline_pred,
                    is_signal_masked,
                    data["q3_GeV"],
                    data["q3_bin_edges"],
                )
            elif x_var == "W":
                reco_bl = compute_reco_baseline_recall_per_bin(
                    reco_baseline_pred,
                    is_signal_masked,
                    data["W_GeV"],
                    data["W_bin_edges"],
                )
            else:
                var_key = {"pion_E": "pion_E_MC", "pion_theta": "pion_theta_MC"}[x_var]
                edges_key = {
                    "pion_E": "pion_E_MC_bins",
                    "pion_theta": "pion_theta_MC_bins",
                }[x_var]
                reco_bl = compute_reco_baseline_recall_per_bin(
                    reco_baseline_pred,
                    is_signal_masked,
                    data[var_key],
                    data[edges_key],
                    has_pion=data["has_pion"] if pion_bins_require_has_pion else None,
                )
            # FPR in legend: same interaction-type (and pion/hist) slice as this row's histogram
            fpr_row = _reco_baseline_fpr_on_mask(
                reco_baseline_pred, y_true_binary, plot_mask
            )
            bl_lbl = _baseline_legend_with_global_fpr(reco_baseline_label, fpr_row)
            axes[row_idx, 2].plot(x_mid, reco_bl, "s--", color="black", label=bl_lbl)

        # --- Metric columns ---
        col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
        for col, metric in enumerate(col_labels):
            ax = axes[row_idx, col]
            ax.set_xlabel(xlabel)
            ax.set_ylabel(metric)
            if col < 2:
                ax.set_title(
                    f"{int_name} (N={n_events:,}) — {metric} vs. {xlabel.split('[')[0].strip()}"
                )
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
        ax_h.set_title(
            f"{int_name} (N={n_events:,}) — events (orange = signal, bottom)"
        )
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
        kin_mask = (
            data["has_pion"]
            if pion_bins_require_has_pion
            else np.ones(len(var), dtype=bool)
        )
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
        ax.bar(
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
                ax.text(
                    bin_edges[i] + widths[i] / 2,
                    c,
                    str(c),
                    ha="center",
                    va="bottom",
                    fontsize=7,
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
        ax.axhline(
            signal_frac,
            color="gray",
            linestyle="--",
            linewidth=1,
            label="Random baseline",
        )

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

            (line,) = ax.plot(r, p, "-", label=label, **clr)
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
            # ax.set_xscale("log")
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
