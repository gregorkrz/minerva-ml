"""RMS/IQR vs q3 with seed uncertainty bands."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

from src.eval._constants import plot_model_label

from ._constants import DEFAULT_BASELINE_KEY, SMALL_PAPER_COMPACT_IQR_MPV_LEGEND_FS
from ._grouped import (
    _SEED_SEP,
    _resolve_color_map,
    flatten_grouped_training_names,
    load_eval_data_grouped,
)
from ._load import _build_event_mask
from ._plot_rms import plot_rms_iqr


def plot_rms_iqr_with_uncertainty(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]] | dict[str, dict[str, list[str]]],
    playlists: list[str] | None = None,
    dataset_to_plot: str | list[str] = "1A",
    dataset_to_linestyle: dict[str, str] | None = None,
    q3_bins: list[float] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    baseline_key: str = DEFAULT_BASELINE_KEY,
    use_cc_selection: int = 2,
    rms_clip: float = 0.6,
    show_q3_histograms: bool = False,  # accepted for API parity, ignored
    colors: dict[str, Any] | None = None,
    verbose: bool = True,
    data: dict | None = None,
    transform=None,
    return_hist_fig: bool = False,  # accepted for API parity, ignored
    return_values: bool = False,
    suppress_errors: bool = False,
    text: str = "",
    iqr_only: bool = True,
    label_fn: Callable[[str], str] | None = None,
    compact_figsize: tuple[float, float] | None = None,
    compact_style: bool = False,
) -> plt.Figure | tuple[plt.Figure, dict]:
    """RMS / IQR vs *q3* with ±1 std-dev uncertainty bands across seeds.

    Parameters
    ----------
    training_names : either ``{loss: {model: run_name}}`` (like :func:`plot_rms_iqr`)
        or a grouped dict ``{loss: {config_label: [run_name, …]}}``.  In the
        former case each (loss, model) is treated as a single-seed config.
    colors : dict mapping sample-count tokens (e.g. ``"6M"``, ``"500k"``)
        to matplotlib colours.  Each config label is matched to the first
        token found as a substring.  Unmatched labels default to grey.
    return_values : if *True*, return ``(fig, values)`` where *values* is
        a dict with q3 bin midpoints and per-config / baseline arrays.
    iqr_only : if *True* (default), use a **2×1** column: top = IQR/MPV vs $q_3$,
        bottom = MPV vs $q_3$ (no left RMS panel).  If *False*, use the full 2×2
        layout (RMS, IQR, and duplicated MPV row); the MPV row has no legend
        (paper-style).
    compact_figsize : optional ``(width, height)`` inches when ``iqr_only`` is *True*;
        defaults to ``(4.8, 5.27)``.
    compact_style : if *True* with ``iqr_only``, use short y-axis labels (``IQR / MPV``,
        ``MPV``) and a single shared x-axis label on the bottom panel only.
    Other parameters match :func:`plot_rms_iqr`.  Only a single
    ``dataset_to_plot`` is supported; ``dataset_to_linestyle``,
    ``show_q3_histograms`` and ``return_hist_fig`` are accepted for API
    compatibility but are ignored.
    """
    # If multiple datasets / linestyles are requested, fall back to mean-only plot.
    # We still accept grouped training_names but just flatten them.
    if isinstance(dataset_to_plot, list):
        # Flatten grouped form, if present
        try:
            flat_training = flatten_grouped_training_names(
                training_names  # type: ignore[arg-type]
            )
        except Exception:
            flat_training = training_names  # already flat
        return plot_rms_iqr(
            CKPT_DIR=CKPT_DIR,
            training_names=flat_training,  # type: ignore[arg-type]
            playlists=playlists,
            dataset_to_plot=dataset_to_plot,
            dataset_to_linestyle=dataset_to_linestyle,
            q3_bins=q3_bins,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            baseline_key=baseline_key,
            use_cc_selection=use_cc_selection,
            rms_clip=rms_clip,
            show_q3_histograms=show_q3_histograms,
            verbose=verbose,
            data=data,
            transform=transform,
            return_hist_fig=return_hist_fig,
            suppress_errors=suppress_errors,
        )

    # Normalise training_names to grouped form expected by load_eval_data_grouped
    training_names_grouped: dict[str, dict[str, list[str]]] = {}
    for loss, models in training_names.items():
        training_names_grouped.setdefault(loss, {})
        for key, val in models.items():
            if isinstance(val, list):
                # Already grouped: key is a config label
                runs = val
                if not runs:
                    continue
                training_names_grouped[loss][key] = runs
            else:
                # Flat form: single run per (loss, model) → treat as 1‑seed config
                run_name = val
                if run_name is None:
                    continue
                config_label = str(key)
                training_names_grouped[loss].setdefault(config_label, [])
                training_names_grouped[loss][config_label].append(run_name)

    if q3_bins is None:
        q3_bins = [0, 0.3, 0.6, 1.2, 1.8, 2.4, 100]

    if data is None:
        data = load_eval_data_grouped(
            CKPT_DIR,
            training_names_grouped,
            playlists=playlists,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            verbose=verbose,
            transform=transform,
            suppress_errors=suppress_errors,
        )

    E_pred_dict = data["E_pred_dict"]
    E_true_dict = data["E_true_dict"]
    Enu_baselines = data["Enu_baselines"]
    Enu_filters = data["Enu_filters"]
    mc_E = data["mc_E"]
    dp = dataset_to_plot

    has_q3 = (
        dp in Enu_filters
        and "muon_filter_CC_paper" in Enu_filters[dp]
        and "q3" in Enu_filters[dp]
        and dp in mc_E
    )
    has_baselines = has_q3 and dp in Enu_baselines and baseline_key in Enu_baselines[dp]

    if not has_q3:
        warnings.warn(
            f"No q3 / filter data for dataset '{dp}'. "
            "Pass baseline_run='<run_with_settings.json>'."
        )
        empty_fig = plt.figure()
        return (empty_fig, {}) if return_values else empty_fig

    q3_arr = np.asarray(q3_bins)
    n_plot_bins = len(q3_arr) - 2
    q3_bin_mids = ((q3_arr[:-1] + q3_arr[1:]) / 2)[:n_plot_bins]

    q3 = Enu_filters[dp]["q3"]
    mask_sel = _build_event_mask(
        dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection
    )

    bin_masks = []
    for i in range(n_plot_bins):
        mask_q3 = (q3 > q3_bins[i]) & (q3 <= q3_bins[i + 1])
        bin_masks.append(mask_q3 & mask_sel)

    all_labels: list[str] = []
    for loss in training_names_grouped:
        for label in training_names_grouped[loss]:
            if label not in all_labels:
                all_labels.append(label)
    color_map = _resolve_color_map(all_labels, colors)

    values: dict[str, Any] = {"q3_bin_mids": q3_bin_mids}

    ax_mpv_panel: plt.Axes | None = None
    if iqr_only:
        # One column: IQR/MPV (top), MPV (bottom); same height ratio as full 2×2 bottom row.
        _fs = compact_figsize if compact_figsize is not None else (4.8, 5.27)
        fig, axes_col = plt.subplots(
            2,
            1,
            figsize=_fs,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax_iqr = axes_col[0]
        ax_mpv_panel = axes_col[1]
        ax_rms = None
        ax_bottom = None
    else:
        # 2×2 – top: RMS/MPV and IQR/MPV; bottom: MPV (duplicated);
        # bottom row has half the height of the top row.
        fig, ax = plt.subplots(
            2,
            2,
            figsize=(9, 7),
            gridspec_kw={"height_ratios": [2, 1]},
        )
        ax_top = ax[0]
        ax_bottom = ax[1]
        ax_rms = ax_top[0]
        ax_iqr = ax_top[1]

    if has_baselines:
        rms_bl: list[float] = []
        iqr_bl: list[float] = []
        mpv_bl: list[float] = []
        pct_bl: list[float] = []
        for i in range(n_plot_bins):
            true = mc_E[dp][bin_masks[i]]
            bl = Enu_baselines[dp][baseline_key][bin_masks[i]]
            valid = true > 0
            ratio_bl = bl[valid] / true[valid]
            if ratio_bl.size > 0:
                # Global in-range for RMS/IQR
                in_range_bl = (ratio_bl >= 0) & (ratio_bl <= 20)
                pct_in_range_bl = float(in_range_bl.sum() / ratio_bl.size * 100.0)
                ratio_bl_clipped = ratio_bl[in_range_bl]
                # Tighter MPV window (0, 2] for MPV via histogram mode
                mpv_mask_bl = (ratio_bl > 0) & (ratio_bl <= 2.0)
                ratio_bl_mpv = ratio_bl[mpv_mask_bl]
            else:
                pct_in_range_bl = float("nan")
                ratio_bl_clipped = ratio_bl
                ratio_bl_mpv = ratio_bl
            if ratio_bl_clipped.size > 0:
                iqr_val = float(
                    np.percentile(ratio_bl_clipped, 75)
                    - np.percentile(ratio_bl_clipped, 25)
                )
                rms_val = float(np.sqrt(np.mean((ratio_bl_clipped - 1.0) ** 2)))
                if ratio_bl_mpv.size > 0:
                    hist, edges = np.histogram(ratio_bl_mpv, bins=50, range=(0.0, 2.0))
                    max_idx = int(np.argmax(hist))
                    mpv_val = float(0.5 * (edges[max_idx] + edges[max_idx + 1]))
                else:
                    mpv_val = float("nan")
            else:
                iqr_val = float("nan")
                rms_val = float("nan")
                mpv_val = float("nan")
            iqr_bl.append(iqr_val)
            rms_bl.append(rms_val)
            mpv_bl.append(mpv_val)
            pct_bl.append(pct_in_range_bl)

        print(f"  baseline: RMS = {rms_bl}, IQR = {iqr_bl}, MPV = {mpv_bl}")
        values["baseline"] = {
            "rms": np.array(rms_bl),
            "iqr": np.array(iqr_bl),
            "mpv": np.array(mpv_bl),
            "pct_in_range": np.array(pct_bl),
        }

    print(
        f"E_pred_dict keys for '{dp}': { {l: list(m.keys()) for l, m in E_pred_dict.get(dp, {}).items()} }"
    )
    for loss in training_names_grouped:
        for config_label, runs in training_names_grouped[loss].items():
            seed_rms = np.empty((len(runs), n_plot_bins))
            seed_iqr = np.empty((len(runs), n_plot_bins))
            seed_mpv = np.empty((len(runs), n_plot_bins))
            for s in range(len(runs)):
                flat_key = f"{config_label}{_SEED_SEP}{s}"
                if flat_key not in E_pred_dict.get(dp, {}).get(loss, {}):
                    print(
                        f"  WARNING: '{flat_key}' not found in E_pred_dict['{dp}']['{loss}']"
                    )
                    continue
                # Use per-run truth (same NPZ as predictions). Using global mc_E[dp] here
                # caused IndexError when another run had a different #events than the first
                # model used to build mc_E.
                true_vec = E_true_dict[dp][loss][flat_key]
                pred_vec = E_pred_dict[dp][loss][flat_key]
                n_ref = len(bin_masks[0])
                if len(true_vec) != n_ref or len(pred_vec) != n_ref:
                    raise ValueError(
                        f"Event count mismatch for {dp}/{loss}/{flat_key}: "
                        f"len(truth)={len(true_vec)}, len(pred)={len(pred_vec)}, "
                        f"but q3 / bin masks length is {n_ref} (from baseline split). "
                        f"Re-evaluate all models on the same playlist or use runs trained "
                        f"on the same data split."
                    )
                for i in range(n_plot_bins):
                    true = true_vec[bin_masks[i]]
                    reco = pred_vec[bin_masks[i]]
                    valid = true > 0
                    ratio = reco[valid] / true[valid]
                    if ratio.size > 0:
                        # Global in-range for RMS/IQR
                        in_range = (ratio > 0) & (ratio <= 20)
                        ratio_clipped = ratio[in_range]
                        # Tighter MPV window (0, 2] for MPV via histogram mode
                        mpv_mask = (ratio > 0) & (ratio <= 2.0)
                        ratio_mpv = ratio[mpv_mask]
                    else:
                        ratio_clipped = ratio
                        ratio_mpv = ratio
                    if ratio_clipped.size > 0:
                        seed_iqr[s, i] = float(
                            np.percentile(ratio_clipped, 75)
                            - np.percentile(ratio_clipped, 25)
                        )
                        seed_rms[s, i] = float(
                            np.sqrt(np.mean((ratio_clipped - 1.0) ** 2))
                        )
                        if ratio_mpv.size > 0:
                            hist, edges = np.histogram(
                                ratio_mpv, bins=100, range=(0.0, 2.0)
                            )
                            max_idx = int(np.argmax(hist))
                            seed_mpv[s, i] = float(
                                0.5 * (edges[max_idx] + edges[max_idx + 1])
                            )
                        else:
                            seed_mpv[s, i] = float("nan")
                    else:
                        seed_iqr[s, i] = float("nan")
                        seed_rms[s, i] = float("nan")
                        seed_mpv[s, i] = float("nan")

            mean_rms = seed_rms.mean(axis=0)
            std_rms = seed_rms.std(axis=0)
            mean_iqr = seed_iqr.mean(axis=0)
            std_iqr = seed_iqr.std(axis=0)
            mean_mpv = seed_mpv.mean(axis=0)
            std_mpv = seed_mpv.std(axis=0)

            print(f"  {config_label} ({loss}):")
            print(f"    RMS = {mean_rms.tolist()}")
            print(f"    IQR = {mean_iqr.tolist()}")
            print(f"    MPV = {mean_mpv.tolist()}")

            values.setdefault(loss, {})[config_label] = {
                "rms_mean": mean_rms,
                "rms_std": std_rms,
                "iqr_mean": mean_iqr,
                "iqr_std": std_iqr,
                "mpv_mean": mean_mpv,
                "mpv_std": std_mpv,
                "rms_per_seed": seed_rms,
                "iqr_per_seed": seed_iqr,
                "mpv_per_seed": seed_mpv,
            }

            color = color_map[config_label]
            lbl = (
                label_fn(config_label)
                if label_fn is not None
                else plot_model_label(config_label)
            )

            with np.errstate(divide="ignore", invalid="ignore"):
                rms_over_mpv = mean_rms / mean_mpv
                iqr_over_mpv = mean_iqr / mean_mpv

            if not iqr_only:
                ax_rms.plot(q3_bin_mids, rms_over_mpv, ".--", color=color, label=lbl)
                ax_rms.fill_between(
                    q3_bin_mids,
                    rms_over_mpv - (std_rms / mean_mpv),
                    rms_over_mpv + (std_rms / mean_mpv),
                    alpha=0.25,
                    color=color,
                )
            ax_iqr.plot(q3_bin_mids, iqr_over_mpv, ".--", color=color, label=lbl)
            ax_iqr.fill_between(
                q3_bin_mids,
                iqr_over_mpv - (std_iqr / mean_mpv),
                iqr_over_mpv + (std_iqr / mean_mpv),
                alpha=0.25,
                color=color,
            )

            if iqr_only:
                ax_mpv_panel.plot(q3_bin_mids, mean_mpv, ".--", color=color, label=lbl)
                ax_mpv_panel.fill_between(
                    q3_bin_mids,
                    mean_mpv - std_mpv,
                    mean_mpv + std_mpv,
                    alpha=0.25,
                    color=color,
                )
            else:
                ax_bottom[0].plot(q3_bin_mids, mean_mpv, ".--", color=color, label=lbl)
                ax_bottom[0].fill_between(
                    q3_bin_mids,
                    mean_mpv - std_mpv,
                    mean_mpv + std_mpv,
                    alpha=0.25,
                    color=color,
                )
                ax_bottom[1].plot(q3_bin_mids, mean_mpv, ".--", color=color, label=lbl)
                ax_bottom[1].fill_between(
                    q3_bin_mids,
                    mean_mpv - std_mpv,
                    mean_mpv + std_mpv,
                    alpha=0.25,
                    color=color,
                )

    # Baseline curves for normalised metrics and MPV
    if "baseline" in values:
        bl = values["baseline"]
        bl_rms = bl["rms"]
        bl_iqr = bl["iqr"]
        bl_mpv = bl["mpv"]
        with np.errstate(divide="ignore", invalid="ignore"):
            bl_rms_over_mpv = bl_rms / bl_mpv
            bl_iqr_over_mpv = bl_iqr / bl_mpv
        if not iqr_only:
            ax_rms.plot(
                q3_bin_mids, bl_rms_over_mpv, ".--", color="black", label="Baseline"
            )
        ax_iqr.plot(
            q3_bin_mids, bl_iqr_over_mpv, ".--", color="black", label="Baseline"
        )
        if iqr_only:
            ax_mpv_panel.plot(
                q3_bin_mids, bl_mpv, ".--", color="black", label="Baseline"
            )
        else:
            ax_bottom[0].plot(
                q3_bin_mids, bl_mpv, ".--", color="black", label="Baseline"
            )
            ax_bottom[1].plot(
                q3_bin_mids, bl_mpv, ".--", color="black", label="Baseline"
            )

    # Legend placement; optional *text* is used as legend title if provided.
    legend_fs = (
        SMALL_PAPER_COMPACT_IQR_MPV_LEGEND_FS if (iqr_only and compact_style) else 9
    )
    legend_kwargs: dict[str, Any] = {"fontsize": legend_fs, "loc": "upper right"}
    if text:
        legend_kwargs["title"] = text
        legend_kwargs["title_fontsize"] = 10

    if iqr_only:
        ax_iqr.legend(**legend_kwargs)
    else:
        ax_rms.legend(**legend_kwargs)
        ax_iqr.legend(**legend_kwargs)

    ax_iqr.set(
        xlabel="" if (iqr_only and compact_style) else r"True $q_3$ [GeV]",
        ylabel=(
            "IQR / MPV"
            if (iqr_only and compact_style)
            else "IQR / MPV of $E_{\\mathrm{available}}^{\\mathrm{reco}}/E_{\\mathrm{available}}^{\\mathrm{true}}$"
        ),
    )
    ax_iqr.grid(True)
    if iqr_only and compact_style:
        ax_iqr.tick_params(labelbottom=False)
    if iqr_only:
        ax_mpv_panel.set(
            xlabel=r"True $q_3$ [GeV]",
            ylabel=(
                "MPV"
                if compact_style
                else "MPV of $E_{\\mathrm{available}}^{\\mathrm{reco}}/E_{\\mathrm{available}}^{\\mathrm{true}}$"
            ),
        )
        ax_mpv_panel.grid(True)
    else:
        ax_rms.set(
            xlabel=r"True $q_3$ [GeV]",
            ylabel="RMS / MPV of $E_{\\mathrm{available}}^{\\mathrm{reco}}/E_{\\mathrm{available}}^{\\mathrm{true}}$",
        )
        ax_rms.grid(True)
        ax_bottom[0].set(
            xlabel=r"True $q_3$ [GeV]",
            ylabel="MPV of $E_{\\mathrm{available}}^{\\mathrm{reco}}/E_{\\mathrm{available}}^{\\mathrm{true}}$",
        )
        ax_bottom[1].set(
            xlabel=r"True $q_3$ [GeV]",
            ylabel="MPV of $E_{\\mathrm{available}}^{\\mathrm{reco}}/E_{\\mathrm{available}}^{\\mathrm{true}}$",
        )
        ax_bottom[0].grid(True)
        ax_bottom[1].grid(True)

    fig.tight_layout()

    if return_values:
        return fig, values
    return fig
