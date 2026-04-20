"""Matplotlib figures for classification evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from sklearn.metrics import auc, precision_recall_curve

from ._binning import data_with_signal_pion_bins
from ._constants import (
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    DEFAULT_FIXED_FPR,
    INT_TYPE_COLORS,
    MC_INT_TYPE_MERGED,
    W_METRICS_XLIM_GEV,
    _LABEL_FS,
    _LEGEND_FS,
    _TICK_FS,
    _baseline_legend_with_global_fpr,
    _default_signal_label,
    _reco_baseline_fpr_on_mask,
    _tpr_column_title_vs_kinematics,
    _tpr_line_legend_label,
    merge_int_type_arr,
)
from ._metrics_binned import get_signal_probabilities
from ._metrics_tasks import (
    compute_all_metrics,
    compute_all_metrics_q3,
    compute_all_metrics_W,
    compute_signal_baseline,
    compute_signal_baseline_W,
)
from ._reco_baseline import compute_reco_baseline_recall_per_bin


def _set_xlim_w_metrics(ax: plt.Axes) -> None:
    """Set a consistent *W* [GeV] axis span on metric / histogram panels."""
    lo, hi = W_METRICS_XLIM_GEV
    ax.set_xlim(lo, hi)


_PER_AXES_LEGEND_MARKER = " (FPR "


def _shared_light_legend(fig: plt.Figure, axes: Iterable[plt.Axes]) -> None:
    """One shared legend below the figure; first-seen label order, one handle per label.

    Labels that contain a per-panel annotation (``" (FPR "``, e.g.
    ``"Baseline (FPR 3.5%)"``) are intentionally excluded from the shared legend
    because the same visual marker/color is reused across panels with different
    numeric annotations. For those labels, a compact per-axes legend is placed
    on each axis where they appear.
    """
    axes_list = list(axes)

    # Per-axes legend for FPR-annotated labels (plain strings like "Baseline"
    # continue to be aggregated into the shared legend below).
    for ax in axes_list:
        h_ax, l_ax = ax.get_legend_handles_labels()
        per_h: list[plt.Artist] = []
        per_l: list[str] = []
        seen: set[str] = set()
        for hi, li in zip(h_ax, l_ax):
            if _PER_AXES_LEGEND_MARKER not in li:
                continue
            if li in seen:
                continue
            seen.add(li)
            per_h.append(hi)
            per_l.append(li)
        if per_h:
            ax.legend(
                per_h,
                per_l,
                loc="best",
                fontsize=_LEGEND_FS - 1,
                frameon=True,
                fancybox=True,
                facecolor="white",
                edgecolor="0.4",
                framealpha=0.85,
                handletextpad=0.4,
                borderaxespad=0.3,
            )

    by_label: dict[str, plt.Artist] = {}
    labels_order: list[str] = []
    for ax in axes_list:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if _PER_AXES_LEGEND_MARKER in li:
                continue
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
    """Apply labels, titles, and grid to an (n_rows, 3) axes array."""
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    for row in range(n_rows):
        for col in range(3):
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
            ax.set_ylabel(col_labels[col], fontsize=_LABEL_FS)
            ax.tick_params(axis="both", labelsize=_TICK_FS)
            title_prefix = f"{row_titles[row]} — " if row_titles else ""
            if col < 2:
                ax.set_title(
                    f"{title_prefix}{col_labels[col]} vs. {xlabel.split('[')[0].strip()}"
                )
            else:
                ax.set_title(
                    f"{title_prefix}{tpr_title} vs. {xlabel.split('[')[0].strip()}"
                )
            ax.grid(True, alpha=0.35)
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
    colors: dict[str, str] | None = None,
    legend_title: str | None = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    suptitle: str | None = None,
    use_global_fpr: bool = True,
    playlist: str = "1A",
) -> plt.Figure:
    """2×3 figure: pion *E* (top row) and pion *θ* (bottom row).

    Columns: AUPRC, AUROC, TPR vs kinematics (global or per-bin FPR; see ``use_global_fpr``).

    Parameters
    ----------
    reco_baseline_tpr : optional dict with keys ``"E"`` and ``"theta"``,
        each a per-bin recall array for a reconstruction-level baseline.
        Plotted on the TPR panels (column index 2).
    reco_baseline_global_fpr : optional scalar FPR for the baseline legend on the TPR panels.
    reco_baseline_label : label for the reconstruction baseline in the legend.
    legend_title : optional legend title (e.g. dataset line); ``None`` to omit.
    suptitle : figure super-title; default
        ``$CC1\\pi^\\pm$ tagging - MINERvA Open Data Playlist {playlist}``.
    playlist : playlist id for the default *suptitle*.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR

    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)

    E_mid = data["pion_E_MC_bins_mid"]
    theta_mid = data["pion_theta_MC_bins_mid"]

    bl_tpr_label = reco_baseline_label
    if reco_baseline_tpr is not None:
        bl_tpr_label = _baseline_legend_with_global_fpr(
            reco_baseline_label, reco_baseline_global_fpr
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

    col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
    xlabels = [r"True $E_\pi$ [GeV]", r"True $\theta_\pi$ [rad]"]
    for row, kinematic in enumerate(xlabels):
        for col, metric in enumerate(col_labels):
            ax = axes[row, col]
            ax.set_xlabel(kinematic, fontsize=_LABEL_FS)
            ax.set_ylabel(metric, fontsize=_LABEL_FS)
            ax.tick_params(axis="both", labelsize=_TICK_FS)
            x_short = kinematic.split("[")[0].strip()
            if col < 2:
                ax.set_title(f"{metric} vs. {x_short}")
            else:
                ax.set_title(f"{tpr_title} vs. {x_short}")
            ax.grid(True, alpha=0.35)
            if row == 0:
                ax.set_xlim(E_mid[0] * 0.8, E_mid[-1] * 1.2)
                ax.set_xscale("log")

    _shared_light_legend(fig, axes.ravel())
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
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

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
        ax.set_xlabel(r"True $q_3$ [GeV]", fontsize=_LABEL_FS)
        ax.set_ylabel(metric, fontsize=_LABEL_FS)
        ax.tick_params(axis="both", labelsize=_TICK_FS)
        x_short = r"True $q_3$"
        if col < 2:
            ax.set_title(f"{metric} vs. {x_short}")
        else:
            ax.set_title(f"{tpr_title} vs. {x_short}")
        ax.grid(True, alpha=0.35)

    _shared_light_legend(fig, axes.ravel())
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
    colors: dict[str, str] | None = None,
    title: str | None = None,
    legend_title: str | None = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    use_global_fpr: bool = True,
    playlist: str = "1A",
) -> plt.Figure:
    """1×3 figure: AUPRC / AUROC / TPR vs *W* (global or per-bin FPR).

    Same layout as :func:`plot_multi_pion_vs_q3` but with *W* on the *x* axis.

    Pass *reco_baseline_global_fpr* (scalar FP/(FP+TN) on the test set, same
    convention as the notebook) to show it in the column-3 baseline legend,
    e.g. ``Baseline (FPR 3.5%)``.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR

    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)

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
        bl_lbl = _baseline_legend_with_global_fpr(
            reco_baseline_label, reco_baseline_global_fpr
        )
        axes[2].plot(w_mid, reco_baseline_tpr_W, "s--", color="black", label=bl_lbl)

    col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
    for col, metric in enumerate(col_labels):
        ax = axes[col]
        ax.set_xlabel(r"$W$ [GeV]", fontsize=_LABEL_FS)
        ax.set_ylabel(metric, fontsize=_LABEL_FS)
        ax.tick_params(axis="both", labelsize=_TICK_FS)
        x_short = r"$W$"
        if col < 2:
            ax.set_title(f"{metric} vs. {x_short}")
        else:
            ax.set_title(f"{tpr_title} vs. {x_short}")
        ax.grid(True, alpha=0.35)
        _set_xlim_w_metrics(ax)

    _shared_light_legend(fig, axes.ravel())
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
    """One row per merged interaction type (``{DIS, RES, Other}`` by default),
    3 columns: AUPRC, AUROC, TPR (global or per-bin FPR).

    Parameters
    ----------
    x_var : ``"pion_E"``, ``"pion_theta"``, ``"q3"``, or ``"W"`` (hadronic
        invariant mass; requires ``W_GeV`` / ``W_bin_edges`` on *data*).
    int_types : mapping ``{code -> label}`` where *code* is compared against
        :func:`merge_int_type_arr(data["int_type_arr"])`. Defaults to
        :data:`MC_INT_TYPE_MERGED` (``{3: "DIS", 2: "RES", 0: "Other"}``).
    reco_baseline_pred : optional binary prediction array (same length as
        test set). When provided, the per-bin recall is overlaid on the
        TPR panel for each interaction type and the baseline legend **FPR**
        is computed on the **same row mask** as the metrics
        (interaction type ∩ pion / finiteness rules).
    use_global_fpr : if True, one global score cut per target FPR; if False,
        TPR is taken from each bin's local ROC (and plot titles/legends match).
    reco_baseline_label : legend label for the reconstruction baseline.
    signal_label : optional name for the signal class definition (e.g.
        ``r"$CC\\pi^0$"``). Used when there are events in an interaction
        type but no signal positives; defaults from *signal_classes*.
    pion_bins_require_has_pion : if False, pion E/θ binned metrics include
        all events (θ requires finite MC angle).
    legend_title : optional legend title on metric panels; ``None`` to omit.
    """
    if fixed_fpr is None:
        fixed_fpr = DEFAULT_FIXED_FPR
    if int_types is None:
        int_types = MC_INT_TYPE_MERGED
    tpr_title = _tpr_column_title_vs_kinematics(use_global_fpr)

    int_type_arr = merge_int_type_arr(data["int_type_arr"])
    n_int = len(int_types)
    fig, axes = plt.subplots(
        n_int, 3, figsize=(14.5, 4.0 * n_int), constrained_layout=True
    )
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

    # Resolve the plot mask per x-variable (same convention as binned metrics)
    if x_var == "q3":
        hist_pion_mask = np.ones(len(int_type_arr), dtype=bool)
    elif x_var == "W":
        hist_pion_mask = np.ones(len(int_type_arr), dtype=bool)
    elif x_var == "pion_E":
        hist_pion_mask = (
            has_pion
            if pion_bins_require_has_pion
            else np.ones(len(int_type_arr), dtype=bool)
        )
    elif x_var == "pion_theta":
        if pion_bins_require_has_pion:
            hist_pion_mask = has_pion
        else:
            hist_pion_mask = np.isfinite(data["pion_theta_MC"])
    else:
        raise ValueError(f"Unknown x_var: {x_var!r}")

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
            for col in range(3):
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
                for col in range(3):
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

        col_labels = ["AUPRC", "AUROC", "Efficiency (TPR)"]
        for col, metric in enumerate(col_labels):
            ax = axes[row_idx, col]
            ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
            ax.set_ylabel(metric, fontsize=_LABEL_FS)
            ax.tick_params(axis="both", labelsize=_TICK_FS)
            if col < 2:
                ax.set_title(
                    f"{int_name} (N={n_events:,}) — {metric} vs. {xlabel.split('[')[0].strip()}"
                )
            else:
                ax.set_title(
                    f"{int_name} (N={n_events:,}) — {tpr_title} vs. {xlabel.split('[')[0].strip()}"
                )
            ax.grid(True, alpha=0.35)
            if x_var == "W":
                _set_xlim_w_metrics(ax)
            elif log_x:
                ax.set_xlim(x_mid[0] * 0.8, x_mid[-1] * 1.2)
                ax.set_xscale("log")

    _shared_light_legend(fig, axes.ravel())
    fig.suptitle(title, fontsize=14, y=1.005)
    return fig


# ---------------------------------------------------------------------------
# Composition plots (signal / background / S/B by merged interaction type)
# ---------------------------------------------------------------------------


def _composition_counts_by_inttype(
    merged_int_arr: np.ndarray,
    plot_mask: np.ndarray,
    hist_var: np.ndarray,
    bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-bin event counts split by {DIS, RES, Other} within ``plot_mask``.

    Returns a mapping label → array of length ``len(bin_edges) - 1``.
    """
    out: dict[str, np.ndarray] = {}
    for code, label in MC_INT_TYPE_MERGED.items():
        m = plot_mask & (merged_int_arr == code)
        counts, _ = np.histogram(hist_var[m], bins=bin_edges)
        out[label] = np.asarray(counts, dtype=np.int64)
    return out


def _stack_bars_by_inttype(
    ax: plt.Axes,
    bin_edges: np.ndarray,
    counts_by_label: dict[str, np.ndarray],
    *,
    log_x: bool = False,
    annotate_total: bool = True,
    stack_order: tuple[str, ...] = ("DIS", "RES", "Other"),
) -> None:
    """Stacked bar plot with consistent DIS/RES/Other colors."""
    edges = np.asarray(bin_edges, dtype=float)
    widths = np.diff(edges)
    x0 = edges[:-1]
    bottoms = np.zeros(len(widths), dtype=float)
    totals = np.zeros(len(widths), dtype=float)
    for label in stack_order:
        counts = counts_by_label.get(label)
        if counts is None:
            continue
        counts_f = np.asarray(counts, dtype=float)
        ax.bar(
            x0,
            counts_f,
            width=widths,
            align="edge",
            bottom=bottoms,
            color=INT_TYPE_COLORS[label],
            edgecolor="0.25",
            linewidth=0.35,
            alpha=0.85,
            label=label,
        )
        bottoms = bottoms + counts_f
        totals = totals + counts_f

    if annotate_total:
        for i, tot in enumerate(totals):
            if tot > 0:
                ax.text(
                    x0[i] + widths[i] / 2,
                    float(tot),
                    str(int(tot)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.grid(True, axis="y", alpha=0.35)
    if log_x:
        ax.set_xscale("log")


def _single_bkg_bar(
    ax: plt.Axes,
    merged_int_arr: np.ndarray,
    bkg_mask: np.ndarray,
    *,
    stack_order: tuple[str, ...] = ("DIS", "RES", "Other"),
) -> None:
    """Single stacked bar labelled 'Background' split by {DIS, RES, Other}."""
    totals_per_label: dict[str, int] = {}
    for code, label in MC_INT_TYPE_MERGED.items():
        totals_per_label[label] = int((bkg_mask & (merged_int_arr == code)).sum())

    bottom = 0.0
    total = 0
    for label in stack_order:
        count = totals_per_label.get(label, 0)
        ax.bar(
            [0.0],
            [float(count)],
            width=0.7,
            align="center",
            bottom=[bottom],
            color=INT_TYPE_COLORS[label],
            edgecolor="0.25",
            linewidth=0.35,
            alpha=0.85,
            label=label,
        )
        bottom += float(count)
        total += int(count)

    if total > 0:
        ax.text(0.0, float(total), f"{total:,}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks([0.0])
    ax.set_xticklabels(["Background"])
    ax.set_xlim(-0.7, 0.7)
    ax.grid(True, axis="y", alpha=0.35)


def _sb_stairs_plot(
    ax: plt.Axes,
    bin_edges: np.ndarray,
    n_sig: np.ndarray,
    n_bkg: np.ndarray,
    *,
    log_x: bool = False,
) -> None:
    """Step line of S/B per bin (``S / B``; B = non-signal events in bin)."""
    n_sig = np.asarray(n_sig, dtype=float)
    n_bkg = np.asarray(n_bkg, dtype=float)
    sb = np.divide(n_sig, n_bkg, out=np.full_like(n_sig, np.nan), where=n_bkg > 0)
    sb_plot = ma.masked_invalid(sb)
    ax.stairs(sb_plot, np.asarray(bin_edges, dtype=float), color="#2ca02c", linewidth=1.8, label=r"$S/B$")
    ax.grid(True, alpha=0.35)
    if log_x:
        ax.set_xscale("log")


def _pion_signal_bins(
    data: dict, pid: np.ndarray, signal_classes: list[int]
) -> dict:
    """Shallow copy of *data* with equal-frequency pion-E/θ edges on signal."""
    return data_with_signal_pion_bins(
        data,
        pid,
        signal_classes,
        pion_quantile_require_has_pion=False,
        pion_bin_edge_method="equal_frequency",
    )


def plot_signal_composition_single_pion(
    data: dict,
    pid: np.ndarray,
    playlist: str = "1A",
) -> plt.Figure:
    """2-row × 3-col event-composition figure for single-pion signals.

    Rows: CC1π± (signal class 0) and CC1π⁰ (signal class 2).
    Columns:
      0. N_signal events vs pion *E* stacked by {DIS, RES, Other}.
      1. Same vs pion *θ*.
      2. Single stacked bar of N_background events split by {DIS, RES, Other}.

    *data* must contain ``int_type_arr``, ``pion_E_MC``, ``pion_theta_MC``,
    ``has_pion`` (along with the default pion-bin edges — these are replaced
    per signal definition using equal-frequency edges on signal events).
    """
    signals = [
        (r"$CC1\pi^\pm$", [0]),
        (r"$CC1\pi^0$", [2]),
    ]
    merged_int = merge_int_type_arr(data["int_type_arr"])
    pion_E = np.asarray(data["pion_E_MC"])
    pion_theta = np.asarray(data["pion_theta_MC"])

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0), constrained_layout=True)

    for row_idx, (row_label, classes) in enumerate(signals):
        data_sp = _pion_signal_bins(data, pid, classes)
        sig_mask = np.isin(pid, classes)
        bkg_mask = ~sig_mask

        finite_E = np.isfinite(pion_E)
        finite_th = np.isfinite(pion_theta)

        e_edges = data_sp["pion_E_MC_bins"]
        th_edges = data_sp["pion_theta_MC_bins"]

        counts_E = _composition_counts_by_inttype(
            merged_int, sig_mask & finite_E, pion_E, e_edges
        )
        counts_th = _composition_counts_by_inttype(
            merged_int, sig_mask & finite_th, pion_theta, th_edges
        )

        ax_e = axes[row_idx, 0]
        _stack_bars_by_inttype(ax_e, e_edges, counts_E, log_x=True)
        ax_e.set_xlabel(r"True $E_\pi$ [GeV]", fontsize=_LABEL_FS)
        ax_e.set_ylabel(f"{row_label} signal events", fontsize=_LABEL_FS)
        ax_e.tick_params(axis="both", labelsize=_TICK_FS)
        ax_e.set_title(rf"{row_label}: signal composition vs. True $E_\pi$")
        e_mid = data_sp["pion_E_MC_bins_mid"]
        if len(e_mid) > 0 and np.all(np.isfinite(e_mid[[0, -1]])):
            ax_e.set_xlim(float(e_mid[0]) * 0.8, float(e_mid[-1]) * 1.2)

        ax_th = axes[row_idx, 1]
        _stack_bars_by_inttype(ax_th, th_edges, counts_th, log_x=False)
        ax_th.set_xlabel(r"True $\theta_\pi$ [rad]", fontsize=_LABEL_FS)
        ax_th.set_ylabel(f"{row_label} signal events", fontsize=_LABEL_FS)
        ax_th.tick_params(axis="both", labelsize=_TICK_FS)
        ax_th.set_title(rf"{row_label}: signal composition vs. True $\theta_\pi$")

        ax_bkg = axes[row_idx, 2]
        _single_bkg_bar(ax_bkg, merged_int, bkg_mask)
        ax_bkg.set_ylabel(f"{row_label} background events", fontsize=_LABEL_FS)
        ax_bkg.tick_params(axis="both", labelsize=_TICK_FS)
        ax_bkg.set_title(f"{row_label}: background composition")

    _shared_light_legend(fig, axes.ravel())
    fig.suptitle(
        rf"Event composition (single-pion) - MINERvA Open Data Playlist {playlist}",
        fontsize=14,
    )
    return fig


def plot_composition_vs_kinematic(
    data: dict,
    pid: np.ndarray,
    x_var: str,
    playlist: str = "1A",
) -> plt.Figure:
    """3-row × 3-col event-composition figure for CC1π±, CC1π⁰, CCNπ±.

    Rows: CC1π± (class 0), CC1π⁰ (class 2), CCNπ± (classes [0, 1]).
    Columns:
      0. N_signal events per *x* bin, stacked by {DIS, RES, Other}.
      1. N_background events per *x* bin, stacked by {DIS, RES, Other}.
      2. S/B ratio per *x* bin (green step line; B = non-signal events in bin).

    *x_var* must be ``"W"`` (requires ``W_GeV`` / ``W_bin_edges``) or ``"q3"``.
    """
    if x_var == "W":
        hist_var = np.asarray(data["W_GeV"])
        bin_edges = np.asarray(data["W_bin_edges"])
        xlabel = r"True hadronic $W$ [GeV]"
        x_is_W = True
    elif x_var == "q3":
        hist_var = np.asarray(data["q3_GeV"])
        bin_edges = np.asarray(data["q3_bin_edges"])
        xlabel = r"True $q_3$ [GeV]"
        x_is_W = False
    else:
        raise ValueError(f"Unknown x_var: {x_var!r}; expected 'W' or 'q3'")

    signals = [
        (r"$CC1\pi^\pm$", [0]),
        (r"$CC1\pi^0$", [2]),
        (r"$CCN\pi^\pm$ ($N \geq 1$)", [0, 1]),
    ]
    merged_int = merge_int_type_arr(data["int_type_arr"])
    all_mask = np.ones(len(hist_var), dtype=bool)

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 11.0), constrained_layout=True)

    for row_idx, (row_label, classes) in enumerate(signals):
        sig_mask = np.isin(pid, classes)
        bkg_mask = ~sig_mask

        counts_sig = _composition_counts_by_inttype(
            merged_int, sig_mask & all_mask, hist_var, bin_edges
        )
        counts_bkg = _composition_counts_by_inttype(
            merged_int, bkg_mask & all_mask, hist_var, bin_edges
        )
        n_sig_tot = sum(counts_sig.values())
        n_bkg_tot = sum(counts_bkg.values())

        ax_sig = axes[row_idx, 0]
        _stack_bars_by_inttype(ax_sig, bin_edges, counts_sig)
        ax_sig.set_xlabel(xlabel, fontsize=_LABEL_FS)
        ax_sig.set_ylabel(f"{row_label} signal events", fontsize=_LABEL_FS)
        ax_sig.tick_params(axis="both", labelsize=_TICK_FS)
        ax_sig.set_title(rf"{row_label}: signal composition")

        ax_bkg = axes[row_idx, 1]
        _stack_bars_by_inttype(ax_bkg, bin_edges, counts_bkg)
        ax_bkg.set_xlabel(xlabel, fontsize=_LABEL_FS)
        ax_bkg.set_ylabel(f"{row_label} background events", fontsize=_LABEL_FS)
        ax_bkg.tick_params(axis="both", labelsize=_TICK_FS)
        ax_bkg.set_title(rf"{row_label}: background composition")

        ax_sb = axes[row_idx, 2]
        _sb_stairs_plot(ax_sb, bin_edges, n_sig_tot, n_bkg_tot)
        ax_sb.set_xlabel(xlabel, fontsize=_LABEL_FS)
        ax_sb.set_ylabel(r"Signal / background ($S/B$)", fontsize=_LABEL_FS)
        ax_sb.tick_params(axis="both", labelsize=_TICK_FS)
        ax_sb.set_title(rf"{row_label}: $S/B$")

        if x_is_W:
            for col in range(3):
                _set_xlim_w_metrics(axes[row_idx, col])

    _shared_light_legend(fig, axes.ravel())
    kin_title = r"$W$" if x_is_W else r"$q_3$"
    fig.suptitle(
        rf"Event composition vs. true {kin_title} - MINERvA Open Data Playlist {playlist}",
        fontsize=14,
    )
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

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
        ax.set_xlabel(r"Recall (TPR)", fontsize=_LABEL_FS)
        ax.set_ylabel(r"Precision (purity)", fontsize=_LABEL_FS)
        ax.tick_params(axis="both", labelsize=_TICK_FS)
        ax.set_title(f"{title} ({scale} scale)")
        ax.grid(True, alpha=0.35)
        ax.set_xlim(0, 1)
        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylim(bottom=signal_frac * 0.5, top=1.05)
        else:
            ax.set_ylim(0, 1)

    _shared_light_legend(fig, axes.ravel())
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
