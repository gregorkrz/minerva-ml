"""RMS / IQR vs q3 summary plot."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np

from src.eval._constants import plot_model_label

from ._constants import DEFAULT_BASELINE_KEY
from ._load import _build_event_mask, load_eval_data


def plot_rms_iqr(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    dataset_to_plot: str | list[str] = "1A",
    dataset_to_linestyle: dict[str, str] | None = None,
    q3_bins: list[float] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    baseline_key: str = DEFAULT_BASELINE_KEY,
    use_cc_selection: int = 2,
    rms_clip: float = 0.6,
    show_q3_histograms: bool = False,
    verbose: bool = True,
    data: dict | None = None,
    transform=None,
    return_hist_fig: bool = False,
    suppress_errors: bool = False,
) -> plt.Figure | tuple[plt.Figure, plt.Figure | None]:
    """Produce the RMS / IQR vs *q3* summary plot.

    Parameters
    ----------
    CKPT_DIR : checkpoint root (ignored when *data* is provided).
    training_names : ``{loss: {model: run_name}}``.
    playlists : default ``["1A"]``.
    dataset_to_plot : which playlist(s) to plot. If a list, *dataset_to_linestyle*
        is required: colors are shared across datasets per (loss, model), and
        linestyle varies by dataset.
    dataset_to_linestyle : required when *dataset_to_plot* is a list. Maps each
        dataset name to a matplotlib linestyle (e.g. ``{"1A": "-", "1B": "--"}``).
    q3_bins : bin edges in GeV (default ``[0, 0.3, 0.6, 1.2, 100]``).
    baseline_ref : ``(loss, model)`` for physics baseline lookup.
    baseline_run : standalone run name that has ``settings.json`` /
        ``best_model.pt``, used to load baselines when none of the
        models in *training_names* carry those files.
    baseline_key : which baseline to plot.  Default ``"blob_recoil_E_scaled"``
        (blob_recoil_E × 1.17, MINERvA's official E_available estimator).
        Pass ``"E_recoil_CCinc_only"`` to use the CCInc hadron recoil instead.
    rms_clip : unused (kept for backwards compatibility).
    show_q3_histograms : if *True* also show the per-q3-bin
        ``E reco / E true`` histograms (the intermediate diagnostic plot).
    verbose : print warnings during data loading.
    data : pre-loaded data dict from :func:`load_eval_data`.  When
        supplied, *CKPT_DIR* and *playlists* are ignored.

    Returns
    -------
    matplotlib.figure.Figure – the RMS / IQR summary figure.
    """
    if q3_bins is None:
        q3_bins = [0, 0.3, 0.6, 1.2, 1.8, 2.4, 3.0, 100]

    datasets = (
        [dataset_to_plot] if isinstance(dataset_to_plot, str) else list(dataset_to_plot)
    )
    if len(datasets) > 1:
        if dataset_to_linestyle is None:
            raise ValueError(
                "dataset_to_linestyle is required when dataset_to_plot is a list"
            )
        missing = [d for d in datasets if d not in dataset_to_linestyle]
        if missing:
            raise ValueError(
                f"dataset_to_linestyle must contain an entry for each dataset; missing: {missing}"
            )

    # -- load data if not provided ------------------------------------------
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

    # Per-dataset: which have q3/baselines, and storage for RMS/IQR
    def _has_q3(dp: str) -> bool:
        return (
            dp in Enu_filters
            and "muon_filter_CC_paper" in Enu_filters[dp]
            and "q3" in Enu_filters[dp]
            and dp in mc_E
        )

    def _has_baselines(dp: str) -> bool:
        return _has_q3(dp) and dp in Enu_baselines and baseline_key in Enu_baselines[dp]

    datasets_with_q3 = [dp for dp in datasets if _has_q3(dp)]
    for dp in datasets:
        if not _has_q3(dp):
            warnings.warn(
                f"No q3 / filter data available for dataset '{dp}'. Skipping. "
                "Pass baseline_run='<run_with_settings.json>' to load them."
            )

    if not datasets_with_q3:
        return plt.figure()

    # -- helper: log1p-style Huber loss (matches training/eval convention) --
    def _log1p_huber_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute mean Huber loss between log1p(y_pred) and log1p(y_true)."""
        y_pred = np.asarray(y_pred, dtype=float)
        y_true = np.asarray(y_true, dtype=float)
        mask_pos = y_true > 0
        if not np.any(mask_pos):
            return float("nan")
        y_pred = y_pred[mask_pos]
        y_true = y_true[mask_pos]
        if y_pred.size == 0:
            return float("nan")
        lp_pred = np.log1p(y_pred)
        lp_true = np.log1p(y_true)
        diff = lp_pred - lp_true
        abs_diff = np.abs(diff)
        loss = np.where(abs_diff <= 1.0, 0.5 * diff**2, abs_diff - 0.5)
        return float(loss.mean())

    # -- compute RMS / IQR per q3 bin, per dataset --------------------------
    q3_arr = np.asarray(q3_bins)
    n_plot_bins = len(q3_arr) - 2
    q3_bin_mids = ((q3_arr[:-1] + q3_arr[1:]) / 2)[:n_plot_bins]

    # dp -> loss -> model -> list (or dp -> list for baseline)
    IQR_models_by_dp: dict[str, dict[str, dict[str, list[float]]]] = {}
    RMS_models_by_dp: dict[str, dict[str, dict[str, list[float]]]] = {}
    PCT_models_by_dp: dict[str, dict[str, dict[str, list[float]]]] = {}
    IQR_baseline_by_dp: dict[str, list[float]] = {}
    RMS_baseline_by_dp: dict[str, list[float]] = {}
    PCT_baseline_by_dp: dict[str, list[float]] = {}
    LOG1P_LOSS_models_by_dp: dict[str, dict[str, dict[str, float]]] = {}
    LOG1P_LOSS_baseline_by_dp: dict[str, float] = {}

    hist_fig = None
    if show_q3_histograms and datasets_with_q3:
        hist_fig, hist_ax = plt.subplots(2, n_plot_bins, figsize=(14, 7))
        if n_plot_bins == 1:
            hist_ax = hist_ax[:, np.newaxis]

    for dp in datasets_with_q3:
        has_baselines = _has_baselines(dp)
        q3 = Enu_filters[dp]["q3"]
        mask_sel = _build_event_mask(
            dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection
        )

        IQR_models_by_dp[dp] = {}
        RMS_models_by_dp[dp] = {}
        PCT_models_by_dp[dp] = {}
        IQR_baseline_by_dp[dp] = []
        RMS_baseline_by_dp[dp] = []
        PCT_baseline_by_dp[dp] = []
        LOG1P_LOSS_models_by_dp[dp] = {}
        LOG1P_LOSS_baseline_by_dp[dp] = float("nan")

        # -- dataset-level log1p-style loss (same convention as training) ----
        true_all = mc_E[dp][mask_sel]
        if has_baselines:
            baseline_all = Enu_baselines[dp][baseline_key][mask_sel]
            LOG1P_LOSS_baseline_by_dp[dp] = _log1p_huber_loss(baseline_all, true_all)
            print(
                f"[plot_rms_iqr] dp={dp}, baseline log1p-Huber loss = "
                f"{LOG1P_LOSS_baseline_by_dp[dp]:.6f}"
            )

        for loss_name in results:
            LOG1P_LOSS_models_by_dp[dp].setdefault(loss_name, {})
            for model_name in results[loss_name]:
                if model_name not in E_pred_dict.get(dp, {}).get(loss_name, {}):
                    continue
                reco_all = E_pred_dict[dp][loss_name][model_name][mask_sel]
                lval = _log1p_huber_loss(reco_all, true_all)
                LOG1P_LOSS_models_by_dp[dp][loss_name][model_name] = lval
                print(
                    f"[plot_rms_iqr] dp={dp}, loss={loss_name}, model={model_name}: "
                    f"log1p-Huber loss = {lval:.6f}"
                )

        for i in range(n_plot_bins):
            mask_q3_orig = (q3 > q3_bins[i]) & (q3 <= q3_bins[i + 1])
            mask = mask_q3_orig & mask_sel
            efficiency = mask.sum() / mask_q3_orig.sum() * 100
            true = mc_E[dp][mask]

            if has_baselines:
                baseline = Enu_baselines[dp][baseline_key][mask]
                valid_true_bl = true > 0
                ratio_bl = baseline[valid_true_bl] / true[valid_true_bl]
                if ratio_bl.size > 0:
                    in_range_bl = (ratio_bl > 0) & (ratio_bl <= 20)
                    pct_in_range_bl = float(in_range_bl.sum() / ratio_bl.size * 100.0)
                    ratio_bl_clipped = ratio_bl[in_range_bl]
                else:
                    pct_in_range_bl = float("nan")
                    ratio_bl_clipped = ratio_bl

                if ratio_bl_clipped.size > 0:
                    iqr = float(
                        np.percentile(ratio_bl_clipped, 75)
                        - np.percentile(ratio_bl_clipped, 25)
                    )
                    rms = float(np.sqrt(np.mean((ratio_bl_clipped - 1.0) ** 2)))
                else:
                    iqr = float("nan")
                    rms = float("nan")

                IQR_baseline_by_dp[dp].append(iqr)
                RMS_baseline_by_dp[dp].append(rms)
                PCT_baseline_by_dp[dp].append(pct_in_range_bl)

                if show_q3_histograms and datasets_with_q3:
                    bins = np.linspace(0.0, 3.0, 301)
                    ratio_bl_hist = ratio_bl[(ratio_bl >= 0.0) & (ratio_bl <= 3.0)]
                    frac_plotted_bl = (
                        float(ratio_bl_hist.size) / float(ratio_bl_clipped.size) * 100.0
                        if ratio_bl_clipped.size > 0
                        else float("nan")
                    )
                    label_bl = (
                        f"baseline (in[0,3]={frac_plotted_bl:.1f}% of IQR sample)"
                        if not np.isnan(frac_plotted_bl)
                        else "baseline"
                    )
                    hist_ax[0, i].hist(
                        ratio_bl_hist,
                        bins=bins,
                        histtype="step",
                        label=label_bl,
                        color="black",
                    )
                    hist_ax[1, i].hist(
                        ratio_bl_hist,
                        bins=bins,
                        histtype="step",
                        label=label_bl,
                        color="black",
                    )

            for loss in results:
                IQR_models_by_dp[dp].setdefault(loss, {})
                RMS_models_by_dp[dp].setdefault(loss, {})
                PCT_models_by_dp[dp].setdefault(loss, {})
                for model in results[loss]:
                    if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                        continue
                    IQR_models_by_dp[dp][loss].setdefault(model, [])
                    RMS_models_by_dp[dp][loss].setdefault(model, [])
                    PCT_models_by_dp[dp][loss].setdefault(model, [])

                    reco = E_pred_dict[dp][loss][model][mask]
                    valid_true = true > 0
                    ratio = reco[valid_true] / true[valid_true]
                    if ratio.size > 0:
                        in_range = (ratio > 0) & (ratio <= 20)
                        pct_in_range = float(in_range.sum() / ratio.size * 100.0)
                        ratio_clipped = ratio[in_range]
                    else:
                        pct_in_range = float("nan")
                        ratio_clipped = ratio

                    if ratio_clipped.size > 0:
                        model_iqr = float(
                            np.percentile(ratio_clipped, 75)
                            - np.percentile(ratio_clipped, 25)
                        )
                        model_rms = float(np.sqrt(np.mean((ratio_clipped - 1.0) ** 2)))
                    else:
                        model_iqr = float("nan")
                        model_rms = float("nan")

                    IQR_models_by_dp[dp][loss][model].append(model_iqr)
                    RMS_models_by_dp[dp][loss][model].append(model_rms)
                    PCT_models_by_dp[dp][loss][model].append(pct_in_range)

                    if show_q3_histograms and datasets_with_q3:
                        bins = np.linspace(0.0, 3.0, 301)
                        ratio_hist = ratio[(ratio >= 0.0) & (ratio <= 3.0)]
                        frac_plotted = (
                            float(ratio_hist.size) / float(ratio_clipped.size) * 100.0
                            if ratio_clipped.size > 0
                            else float("nan")
                        )
                        label = (
                            f"{plot_model_label(model)}-{loss} (RMS={model_rms:.3f}, "
                            f"plotted={frac_plotted:.1f}% of IQR sample)"
                            if not np.isnan(frac_plotted)
                            else f"{plot_model_label(model)}-{loss} (RMS={model_rms:.3f})"
                        )
                        hist_ax[0, i].hist(
                            ratio_hist,
                            bins=bins,
                            histtype="step",
                            label=label,
                        )
                        hist_ax[1, i].hist(
                            ratio_hist,
                            bins=bins,
                            histtype="step",
                            label=label,
                        )

            if show_q3_histograms and datasets_with_q3:
                q3_label = f"{q3_bins[i]}-{q3_bins[i+1]}"
                eff_str = f" (eff: {efficiency:.1f}%)" if has_baselines else ""
                hist_ax[0, i].set(
                    xlabel="E reco / E true",
                    ylabel="Counts",
                    title=f"q3: {q3_label} GeV{eff_str}",
                )
                hist_ax[0, i].grid(True)
                hist_ax[1, i].set(
                    xlabel="E reco / E true", ylabel="Counts", title=f"q3: {q3_label}"
                )
                hist_ax[1, i].set_yscale("log")
                hist_ax[1, i].legend(loc="lower left", fontsize=7)
                hist_ax[1, i].grid(True)

    if show_q3_histograms and hist_fig is not None:
        hist_fig.tight_layout()

    # -- summary RMS / IQR figure -------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    single_dataset = len(datasets_with_q3) == 1
    default_linestyle = ".--"
    # One color per (loss, model); cycle through default prop_cycle
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key().get(
        "color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    )
    lm_pairs = [(loss, model) for loss in results for model in results[loss]]
    color_by_lm = {lm: colors[i % len(colors)] for i, lm in enumerate(lm_pairs)}

    for loss, model in lm_pairs:
        for dp in datasets_with_q3:
            if model not in RMS_models_by_dp[dp].get(loss, {}):
                continue
            rms_vals = np.array(RMS_models_by_dp[dp][loss][model])
            iqr_vals = np.array(IQR_models_by_dp[dp][loss][model])
            pct_vals = np.array(PCT_models_by_dp[dp][loss][model])
            pct_mean = np.nanmean(pct_vals) if pct_vals.size > 0 else float("nan")
            train_loss = (
                LOG1P_LOSS_models_by_dp.get(dp, {})
                .get(loss, {})
                .get(model, float("nan"))
            )
            color = color_by_lm[(loss, model)]
            if single_dataset:
                ls = default_linestyle
                lab = plot_model_label(model)
            else:
                ls = dataset_to_linestyle.get(dp, "-")
                lab = plot_model_label(model)
            ax[0].plot(q3_bin_mids, rms_vals, ls, color=color, label=lab)
            ax[1].plot(q3_bin_mids, iqr_vals, ls, color=color, label=lab)

    for dp in datasets_with_q3:
        rms_bl = RMS_baseline_by_dp.get(dp)
        iqr_bl = IQR_baseline_by_dp.get(dp)
        pct_bl = PCT_baseline_by_dp.get(dp)
        if not rms_bl or not iqr_bl or not pct_bl:
            continue
        pct_bl_arr = np.array(pct_bl)
        # pct_bl_mean = np.nanmean(pct_bl_arr) if pct_bl_arr.size > 0 else float("nan")
        # bl_loss = LOG1P_LOSS_baseline_by_dp.get(dp, float("nan"))
        if single_dataset:
            ls = default_linestyle
            lab = "baseline"
        else:
            ls = dataset_to_linestyle.get(dp, "-")
            lab = "baseline"
        ax[0].plot(q3_bin_mids, np.array(rms_bl), ls, label=lab, color="black")
        ax[1].plot(q3_bin_mids, np.array(iqr_bl), ls, label=lab, color="black")

    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    ax[0].set(
        xlabel=r"True $q_3$ [GeV]",
        ylabel="RMS of $E_{\\mathrm{available}}^{\\mathrm{reco}}/E_{\\mathrm{available}}^{\\mathrm{true}}$",
    )
    ax[1].set(
        xlabel=r"True $q_3$ [GeV]",
        ylabel="25-75 IQR of $E_{\\mathrm{available}}^{\\mathrm{reco}}/E_{\\mathrm{available}}^{\\mathrm{true}}$",
    )
    ax[0].grid(True)
    ax[1].grid(True)
    fig.tight_layout()

    if return_hist_fig:
        return fig, hist_fig
    return fig
