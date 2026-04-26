"""Residual histograms vs energy and vs q3."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.eval._constants import plot_model_label

from ._grouped import _resolve_color_map, _SEED_SEP
from ._constants import DEFAULT_BASELINE_KEY, SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES
from ._load import _build_event_mask, load_eval_data
from ._titles import _Etrue_bin_title, _q3_bin_title

# Match ``plot_regression`` / ``plot_rms_iqr_with_uncertainty`` (legend 9; axes ≈ mpl defaults).
_REGRESSION_LEGEND_FS = 10
# Slightly smaller in-panel legend for the compact two-panel ratio figure.
_SMALL_PAPER_RATIO_LEGEND_FS = 8
_REGRESSION_AXIS_FS = 12
_REGRESSION_TITLE_FS = 12
_REGRESSION_TICK_FS = 12


def plot_residuals_by_energy(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    dataset_to_plot: str = "1A",
    energy_bins: list[float] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    baseline_key: str = DEFAULT_BASELINE_KEY,
    use_cc_selection: int = 2,
    verbose: bool = True,
    data: dict | None = None,
    transform=None,
    suppress_errors: bool = False,
) -> plt.Figure:
    """Per-energy-bin residual and ratio histograms (notebook cell 9)."""
    if energy_bins is None:
        energy_bins = [0, 2, 5, 10, 100]

    if data is None:
        data = load_eval_data(
            CKPT_DIR,
            training_names,
            playlists=playlists,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            verbose=verbose,
            transform=transform,
            suppress_errors=suppress_errors,
        )

    results = data["results"]
    E_pred_dict = data["E_pred_dict"]
    Enu_baselines = data["Enu_baselines"]
    Enu_filters = data["Enu_filters"]
    mc_E = data["mc_E"]
    dp = dataset_to_plot

    has_baselines = (
        dp in Enu_filters
        and "muon_filter_CC_paper" in Enu_filters[dp]
        and dp in Enu_baselines
        and baseline_key in Enu_baselines[dp]
    )

    n_cols = len(energy_bins) - 1
    fig, ax = plt.subplots(3, n_cols, figsize=(4.3 * n_cols, 10.5))
    if n_cols == 1:
        ax = ax[:, np.newaxis]

    residual_bins_list = [
        np.linspace(-1, 1, 100),
        np.linspace(-5, 5, 100),
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
    ]
    ratio_bins = np.linspace(0, 3, 300)
    ratio_bins_wide = np.linspace(0, 10, 300)

    if has_baselines:
        mask_sel = _build_event_mask(
            dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection
        )

    for i in range(min(n_cols, len(residual_bins_list))):
        elow, ehigh = energy_bins[i], energy_bins[i + 1]
        bin_title = _Etrue_bin_title(elow, ehigh)
        if has_baselines:
            mask_e = (mc_E[dp] > elow) & (mc_E[dp] < ehigh)
            mask = mask_e & mask_sel
            true = mc_E[dp][mask]
            baseline = Enu_baselines[dp][baseline_key][mask]
            valid = true > 0
            ratio_bl = baseline[valid] / true[valid]
            ax[0, i].hist(
                baseline - true,
                bins=residual_bins_list[i],
                histtype="step",
                label="Baseline",
            )
            ax[1, i].hist(ratio_bl, bins=ratio_bins, histtype="step", label="Baseline")
            ax[2, i].hist(
                ratio_bl, bins=ratio_bins_wide, histtype="step", label="Baseline"
            )

        for loss in results:
            for model in results[loss]:
                if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                    continue
                if has_baselines:
                    reco = E_pred_dict[dp][loss][model][mask]
                    ratio_model = reco[valid] / true[valid]
                    mlab = f"{plot_model_label(model)} ({loss})"
                    ax[0, i].hist(
                        reco - true,
                        bins=residual_bins_list[i],
                        histtype="step",
                        label=mlab,
                    )
                    ax[1, i].hist(
                        ratio_model, bins=ratio_bins, histtype="step", label=mlab
                    )
                    ax[2, i].hist(
                        ratio_model, bins=ratio_bins_wide, histtype="step", label=mlab
                    )

        ax[0, i].set(
            xlabel=r"$E_{\mathrm{available}}^{\mathrm{reco}} - E_{\mathrm{available}}^{\mathrm{true}}$ [GeV]",
            ylabel="Counts",
            title=bin_title,
        )
        ax[0, i].grid(True)
        ax[1, i].set(
            xlabel=r"$E_{\mathrm{available}}^{\mathrm{reco}} / E_{\mathrm{available}}^{\mathrm{true}}$",
            ylabel="Counts",
            title=bin_title,
        )
        ax[1, i].grid(True)
        ax[2, i].set(
            xlabel=r"$E_{\mathrm{available}}^{\mathrm{reco}} / E_{\mathrm{available}}^{\mathrm{true}}$",
            ylabel="Counts",
            title=bin_title + r" ($0$–$10$, log)",
        )
        ax[2, i].set_yscale("log")
        ax[2, i].grid(True)

    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    handles, labels = ax[1, 0].get_legend_handles_labels()
    if handles:
        ncol = min(len(handles), 5)
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            frameon=False,
            fontsize=8,
            handlelength=1.6,
            columnspacing=1.0,
        )
    return fig


# ---------------------------------------------------------------------------
# (new) Residuals and ratios by q3 bin
# ---------------------------------------------------------------------------


def plot_residuals_by_q3(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    dataset_to_plot: str = "1A",
    q3_bins: list[float] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    baseline_key: str = DEFAULT_BASELINE_KEY,
    use_cc_selection: int = 2,
    legend_title: str | None = None,
    colors: dict[str, Any] | None = None,
    baseline_color: Any = "black",
    verbose: bool = True,
    data: dict | None = None,
    transform=None,
    suppress_errors: bool = False,
) -> plt.Figure:
    """Per-q3-bin residual and ratio histograms (E_reco−E_true and E_reco/E_true).

    The top row uses :math:`E_{\mathrm{reco}} - E_{\mathrm{true}}` on :math:`[-1, 1]` GeV;
    the bottom row uses :math:`E_{\mathrm{reco}} / E_{\mathrm{true}}` on :math:`[0, 2]`.

    Each column is titled with its :math:`q_3` range. A three-line figure title
    (main + playlist + selection when applicable) sits above the axes; the model
    legend is drawn only on the first panel (top-left).

    *colors* is an optional ``{model_name: matplotlib_color}`` dict (same as
    :func:`plot_rms_iqr_with_uncertainty`). Seed suffixes (``§0``, …) are
    stripped for lookup so notebook palettes like ``clrs_dict_full`` match.

    *legend_title* (when not *None*) is appended after the main title line; use
    ``""`` for no extra lines. When *None* and ``use_cc_selection >= 2``, the
    second and third lines are the playlist and selection strings.
    """
    if q3_bins is None:
        q3_bins = [0, 0.3, 0.6, 1.2, 1.8, 2.4, 3.0, 100]

    if data is None:
        data = load_eval_data(
            CKPT_DIR,
            training_names,
            playlists=playlists,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            verbose=verbose,
            transform=transform,
            suppress_errors=suppress_errors,
        )

    results = data["results"]
    E_pred_dict = data["E_pred_dict"]
    Enu_baselines = data["Enu_baselines"]
    Enu_filters = data["Enu_filters"]
    mc_E = data["mc_E"]
    dp = dataset_to_plot

    has_baselines = (
        dp in Enu_filters
        and "muon_filter_CC_paper" in Enu_filters[dp]
        and dp in Enu_baselines
        and baseline_key in Enu_baselines[dp]
    )

    if not has_baselines:
        warnings.warn(
            f"No baselines / filters for dataset '{dp}' – plot_residuals_by_q3 will be empty."
        )

    q3 = (
        Enu_filters[dp]["q3"] if dp in Enu_filters and "q3" in Enu_filters[dp] else None
    )
    if q3 is None:
        warnings.warn(f"No q3 information for dataset '{dp}'.")
        return plt.figure()

    if has_baselines:
        mask_sel = _build_event_mask(
            dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection
        )
    else:
        # Fallback: just select events with positive truth energy
        mask_sel = mc_E[dp] > 0

    n_cols = len(q3_bins) - 1
    # Width per column matches previous style; extra total height reserves space for the
    # three-line suptitle while keeping ~the same subplot height as figsize (..., 5) full-axes.
    fig, ax = plt.subplots(2, n_cols, figsize=(4 * n_cols, 6.4))
    if n_cols == 1:
        ax = ax[:, np.newaxis]

    residual_bins = np.linspace(-1, 1, 160)
    ratio_bins = np.linspace(0, 2, 50)

    def _model_color_key(model: str) -> str:
        return model.split("§", 1)[0] if "§" in model else model

    model_bases_ordered: list[str] = []
    for loss in results:
        for model in results[loss]:
            if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                continue
            key = _model_color_key(model)
            if key not in model_bases_ordered:
                model_bases_ordered.append(key)
    if colors:
        color_by_model_base = _resolve_color_map(model_bases_ordered, colors)
    else:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        cycle_cols = prop_cycle.by_key().get(
            "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        )
        color_by_model_base = {
            m: cycle_cols[i % len(cycle_cols)]
            for i, m in enumerate(model_bases_ordered)
        }

    if legend_title is None:
        if use_cc_selection >= 2:
            title_text = ""
        else:
            title_text = ""
    else:
        title_text = ""
        if legend_title:
            title_text += "\n" + legend_title

    for i in range(n_cols):
        qlow, qhigh = q3_bins[i], q3_bins[i + 1]
        mask_q = (q3 > qlow) & (q3 <= qhigh)
        mask = mask_q & mask_sel

        true = mc_E[dp][mask]
        valid = true > 0

        if has_baselines:
            baseline = Enu_baselines[dp][baseline_key][mask]
            ratio_bl = baseline[valid] / true[valid]
            ax[0, i].hist(
                baseline[valid] - true[valid],
                bins=residual_bins,
                histtype="step",
                label="Baseline",
                color=baseline_color,
            )
            ax[1, i].hist(
                ratio_bl,
                bins=ratio_bins,
                histtype="step",
                label="Baseline",
                color=baseline_color,
            )

        for loss in results:
            for model in results[loss]:
                if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                    continue
                reco = E_pred_dict[dp][loss][model][mask]
                ratio_model = reco[valid] / true[valid]
                mlab = _model_color_key(model)
                mcol = color_by_model_base.get(mlab, "tab:gray")
                mlab_disp = plot_model_label(mlab)
                ax[0, i].hist(
                    reco[valid] - true[valid],
                    bins=residual_bins,
                    histtype="step",
                    label=mlab_disp,
                    color=mcol,
                )
                ax[1, i].hist(
                    ratio_model,
                    bins=ratio_bins,
                    histtype="step",
                    label=mlab_disp,
                    color=mcol,
                )

        col_title = _q3_bin_title(qlow, qhigh)
        fs_a = _REGRESSION_AXIS_FS
        fs_t = _REGRESSION_TITLE_FS
        fs_k = _REGRESSION_TICK_FS
        ax[0, i].set_xlabel(
            r"$E_{\mathrm{available}}^{\mathrm{reco}} - E_{\mathrm{available}}^{\mathrm{true}}$ [GeV]",
            fontsize=fs_a,
        )
        ax[0, i].set_ylabel("Counts", fontsize=fs_a)
        ax[0, i].set_title(col_title, fontsize=fs_t)
        ax[1, i].set_xlabel(
            r"$E_{\mathrm{available}}^{\mathrm{reco}} / E_{\mathrm{available}}^{\mathrm{true}}$",
            fontsize=fs_a,
        )
        ax[1, i].set_ylabel("Counts", fontsize=fs_a)
        ax[0, i].tick_params(axis="both", which="major", labelsize=fs_k)
        ax[1, i].tick_params(axis="both", which="major", labelsize=fs_k)
        ax[0, i].set_xlim(-1, 1)
        ax[0, i].grid(True)
        ax[1, i].grid(True)
        # Headroom so step histogram peaks (and overlapping series) are not flush with the top spine.
        for row in (0, 1):
            lo, hi = ax[row, i].get_ylim()
            ax[row, i].set_ylim(lo, hi * 1.10)

    #fig.suptitle(title_text, fontsize=11, y=0.995)

    # Reserve lower margin for a shared fig.legend (same idea as ``plot_residuals_by_energy``).
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))

    h0, lab0 = ax[0, 0].get_legend_handles_labels()
    if h0:
        by_label: dict[str, Any] = {}
        for h, lab in zip(h0, lab0):
            if lab not in by_label:
                by_label[lab] = h
        sorted_labs = sorted(by_label.keys())
        sorted_handles = [by_label[lab] for lab in sorted_labs]
        ncol = min(len(sorted_handles), 5)
        fig.legend(
            sorted_handles,
            sorted_labs,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            fontsize=_REGRESSION_LEGEND_FS,
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            facecolor="1.0",
            framealpha=0.95,
            handlelength=1.6,
            handletextpad=0.45,
            columnspacing=1.0,
            borderpad=0.35,
            labelspacing=0.35,
        )

    return fig


def plot_ratio_histogram_q3_two_panels(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    dataset_to_plot: str = "1A",
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    baseline_key: str = DEFAULT_BASELINE_KEY,
    use_cc_selection: int = 2,
    colors: dict[str, Any] | None = None,
    verbose: bool = True,
    data: dict | None = None,
    transform=None,
    suppress_errors: bool = False,
    *,
    legend_fontsize: float | None = None,
) -> plt.Figure:
    """Two side-by-side :math:`E_{\\mathrm{reco}}/E_{\\mathrm{true}}` histograms for *q₃* slices.

    Left: :math:`q_3 \\in [0, 1)` GeV; right: :math:`q_3 \\in [1, 2)` GeV. Figure height equals
    :data:`SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES` width; ``wspace`` is small so the
    panels sit close together; ``set_box_aspect(1)`` keeps each subplot ~square. Legend
    on the left panel only (``loc="best"``); the right panel has no legend.
    """
    if data is None:
        data = load_eval_data(
            CKPT_DIR,
            training_names,
            playlists=playlists,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            verbose=verbose,
            transform=transform,
            suppress_errors=suppress_errors,
        )

    results = data["results"]
    E_pred_dict = data["E_pred_dict"]
    Enu_baselines = data["Enu_baselines"]
    Enu_filters = data["Enu_filters"]
    mc_E = data["mc_E"]
    dp = dataset_to_plot

    has_baselines = (
        dp in Enu_filters
        and "muon_filter_CC_paper" in Enu_filters[dp]
        and dp in Enu_baselines
        and baseline_key in Enu_baselines[dp]
    )

    q3 = (
        Enu_filters[dp]["q3"] if dp in Enu_filters and "q3" in Enu_filters[dp] else None
    )
    if q3 is None:
        warnings.warn(f"No q3 information for dataset '{dp}'.")
        return plt.figure()

    if has_baselines:
        mask_sel = _build_event_mask(
            dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection
        )
    else:
        mask_sel = mc_E[dp] > 0

    def _model_color_key(model: str) -> str:
        return model.split("§", 1)[0] if "§" in model else model

    model_bases_ordered: list[str] = []
    for loss in results:
        for model in results[loss]:
            if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                continue
            key = _model_color_key(model)
            if key not in model_bases_ordered:
                model_bases_ordered.append(key)
    if colors:
        color_by_model_base = _resolve_color_map(model_bases_ordered, colors)
    else:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        cycle_cols = prop_cycle.by_key().get(
            "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        )
        color_by_model_base = {
            m: cycle_cols[i % len(cycle_cols)]
            for i, m in enumerate(model_bases_ordered)
        }

    ratio_bins = np.linspace(0, 2, 50)
    baseline_color_kw: Any = "black"

    def _fill_ratio_panel(
        ax: plt.Axes,
        mask_q3: np.ndarray,
        *,
        use_legend_labels: bool,
        panel_title: str,
    ) -> None:
        mask = mask_q3 & mask_sel
        true = mc_E[dp][mask]
        valid = true > 0
        if has_baselines:
            baseline = Enu_baselines[dp][baseline_key][mask]
            ratio_bl = baseline[valid] / true[valid]
            bl_lab = "Baseline" if use_legend_labels else "_nolegend_"
            ax.hist(
                ratio_bl,
                bins=ratio_bins,
                histtype="step",
                label=bl_lab,
                color=baseline_color_kw,
            )
        for loss in results:
            for model in results[loss]:
                if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                    continue
                reco = E_pred_dict[dp][loss][model][mask]
                ratio_model = reco[valid] / true[valid]
                mlab = _model_color_key(model)
                mcol = color_by_model_base.get(mlab, "tab:gray")
                lab = (
                    plot_model_label(mlab) if use_legend_labels else "_nolegend_"
                )
                ax.hist(
                    ratio_model,
                    bins=ratio_bins,
                    histtype="step",
                    label=lab,
                    color=mcol,
                )
        fs_a = _REGRESSION_AXIS_FS
        fs_k = _REGRESSION_TICK_FS
        ax.set_xlabel(
            r"$E_{\mathrm{available}}^{\mathrm{reco}} / E_{\mathrm{available}}^{\mathrm{true}}$",
            fontsize=fs_a,
        )
        ax.set_ylabel("Counts", fontsize=fs_a)
        ax.tick_params(axis="both", which="major", labelsize=fs_k)
        ax.grid(True)
        ax.set_title(panel_title, fontsize=_REGRESSION_TITLE_FS, pad=8)

    _iw = SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES[0]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(1.78 * _iw, _iw),
        sharey=True,
        constrained_layout=False,
        gridspec_kw={"wspace": 0.08},
    )
    mask_01 = (q3 >= 0.0) & (q3 < 1.0)
    mask_12 = (q3 >= 1.0) & (q3 < 2.0)
    _fill_ratio_panel(
        axes[0],
        mask_01,
        use_legend_labels=True,
        panel_title=r"$q_3 \in [0, 1)$ GeV",
    )
    _fill_ratio_panel(
        axes[1],
        mask_12,
        use_legend_labels=False,
        panel_title=r"$q_3 \in [1, 2)$ GeV",
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)

    y_hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0.0, y_hi * 1.08)
    for ax in axes:
        ax.set_box_aspect(1)

    h0, l0 = axes[0].get_legend_handles_labels()
    by_label: dict[str, Any] = {}
    for hi, li in zip(h0, l0):
        if li and li != "_nolegend_" and li not in by_label:
            by_label[li] = hi
    sorted_labs = sorted(by_label.keys())
    sorted_handles = [by_label[x] for x in sorted_labs]
    if sorted_handles:
        leg_fs = (
            float(legend_fontsize)
            if legend_fontsize is not None
            else float(_SMALL_PAPER_RATIO_LEGEND_FS)
        )
        axes[0].legend(
            sorted_handles,
            sorted_labs,
            loc="best",
            fontsize=leg_fs,
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            facecolor="1.0",
            framealpha=0.95,
            handlelength=1.5,
            handletextpad=0.45,
            borderpad=0.35,
            labelspacing=0.35,
        )

    fig.tight_layout()

    return fig
