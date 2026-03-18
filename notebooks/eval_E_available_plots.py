"""
Evaluation plotting utilities for E_available models.

Loads evaluation results from checkpoint directories and produces
RMS / IQR vs q3 summary plots (and optional intermediate diagnostics).

Usage example:

    from eval_E_available_plots import plot_rms_iqr

    training_names = {
        "LogMSE": {
            "Transformer": "E_avail_LogMSE_20260224_062703",
            "OmniLearned_Small_Pretrained": "E_avail_LogMSE_PT_1A_20260225_102544",
            "OmniLearned_Small": "E_avail_LogMSE_1A_20260225_102540",
        },
    }

    fig = plot_rms_iqr(
        CKPT_DIR="/global/cfs/cdirs/m3246/gregork/checkpoints",
        training_names=training_names,
    )
"""

from __future__ import annotations

import json
import os
import pickle
import re
from tkinter import NONE
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

EAVAILABLE_SCALE = 1.17 # The  E_available scale factor for blob recoil energy (https://github.com/MinervaExpt/MAT-MINERvA/blob/main/calculators/LowRecoilFunctions.h)
DEFAULT_BASELINE_KEY = "blob_recoil_E_scaled"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_data(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    verbose: bool = True,
    transform = None,
    suppress_errors: bool = False,
) -> dict[str, Any]:
    """Load evaluation results, configs, baselines and derived arrays.

    Parameters
    ----------
    CKPT_DIR : path to the checkpoints root directory.
    training_names : ``{loss: {model: run_name}}``
    playlists : evaluation playlists (default ``["1A"]``).
    baseline_ref : ``(loss, model)`` pair whose config / data_path is used
        to locate the physics baselines.  If *None* the first model that
        has a ``settings.json`` is used automatically.
    baseline_run : standalone run name (checkpoint folder) that has
        ``settings.json`` and/or ``best_model.pt``.  Use this when none
        of the models in *training_names* carry those files (e.g. the
        SmallDataset Transformer sweeps).  The baselines and filters
        (q3, muon filter, …) will be loaded from this run's data path.
    verbose : print warnings about missing files.

    Returns
    -------
    dict with keys: ``results``, ``configs``, ``data_paths``,
    ``E_true_dict``, ``E_pred_dict``, ``Enu_baselines``, ``Enu_filters``,
    ``mc_E``, ``playlists``.
    """
    CKPT_DIR = Path(CKPT_DIR)
    if playlists is None:
        playlists = ["1A"]

    # --- results & configs (cell 3) ----------------------------------------
    results: dict = {}
    configs: dict = {}
    for loss in training_names:
        results[loss] = {}
        configs[loss] = {}
        for model in training_names[loss]:
            results[loss][model] = {}
            configs[loss][model] = {}
            for playlist in playlists:
                run = training_names[loss][model]
                p1 = CKPT_DIR / run / "test_results" / f"outputs_{run}_minerva_{playlist}_0.npz"
                p2 = CKPT_DIR / run / "test_results" / f"outputs_best_model_minerva_{playlist}_0.npz"
                if p1.exists():
                    results[loss][model][playlist] = dict(np.load(p1))
                elif p2.exists():
                    results[loss][model][playlist] = dict(np.load(p2))
                else:
                    if suppress_errors:
                        if verbose:
                            print(f"Skipping {run} on playlist {playlist} (no eval results found)")
                        continue
                    raise FileNotFoundError(
                        f"No results for {run} on playlist {playlist} "
                        f"– checked {p1} and {p2}"
                    )
                pred = results[loss][model][playlist]["prediction"]
                pred[pred < 0] = 0

                settings_path = CKPT_DIR / run / "settings.json"
                if settings_path.exists():
                    with open(settings_path, "r") as f:
                        configs[loss][model][playlist] = json.load(f)
                else:
                    #if verbose:
                    #    print(f"No settings found for {run} on playlist {playlist}")
                    configs[loss][model][playlist] = {}

    # --- data_paths from best_model.pt (cell 4) ----------------------------
    data_paths: dict = {}
    for loss in training_names:
        data_paths[loss] = {}
        for model in training_names[loss]:
            data_paths[loss][model] = {}
            for playlist in playlists:
                mp = CKPT_DIR / training_names[loss][model] / "best_model.pt"
                if mp.exists():
                    ckpt = torch.load(mp, weights_only=False, map_location="cpu")
                    data_paths[loss][model][playlist] = ckpt["args"]["data_path"]
                else:
                    if verbose:
                        print(f"No best model found for {training_names[loss][model]} on {playlist}")

    # --- E_true / E_pred dicts (cell 6) ------------------------------------
    E_true_dict: dict = {}
    E_pred_dict: dict = {}
    for dataset in playlists:
        E_true_dict[dataset] = {}
        E_pred_dict[dataset] = {}
        for loss in data_paths:
            E_true_dict[dataset][loss] = {}
            E_pred_dict[dataset][loss] = {}
            for model in data_paths[loss]:
                if dataset not in results[loss][model]:
                    continue
                E_true_dict[dataset][loss][model] = results[loss][model][dataset]["pid"].flatten()
                E_pred_dict[dataset][loss][model] = results[loss][model][dataset]["prediction"].flatten()
    # --- baselines & filters (cell 7) --------------------------------------
    # Resolve where the baseline / filter data lives.  Three strategies:
    #   1. baseline_run  – a standalone run name with settings.json / best_model.pt
    #   2. baseline_ref  – (loss, model) pair already in training_names
    #   3. auto-detect   – first model in training_names that has settings.json
    baseline_data_path: str | None = None
    if baseline_run is not None:
        bl_settings = CKPT_DIR / baseline_run / "settings.json"
        bl_model_pt = CKPT_DIR / baseline_run / "best_model.pt"
        if bl_settings.exists():
            with open(bl_settings, "r") as f:
                bl_cfg = json.load(f)
            baseline_data_path = bl_cfg.get("path")
        if baseline_data_path is None and bl_model_pt.exists():
            ckpt = torch.load(bl_model_pt, weights_only=False, map_location="cpu")
            baseline_data_path = ckpt["args"]["data_path"]
        if baseline_data_path is None and verbose:
            print(f"baseline_run '{baseline_run}' has no settings.json or best_model.pt")

    ref_loss, ref_model = _resolve_baseline_ref(
        baseline_ref, configs, data_paths, training_names
    )

    Enu_baselines: dict = {}
    Enu_filters: dict = {}
    mc_E: dict = {}

    # Pick the data_path source: explicit baseline_run, or reference model
    _bl_data_paths: dict[str, str] = {}  # playlist -> data_path
    if baseline_data_path is not None:
        for pl in playlists:
            _bl_data_paths[pl] = baseline_data_path
    elif ref_loss is not None:
        for pl in playlists:
            cfg = configs.get(ref_loss, {}).get(ref_model, {}).get(pl, {})
            if "path" in cfg:
                _bl_data_paths[pl] = cfg["path"]
            elif pl in data_paths.get(ref_loss, {}).get(ref_model, {}):
                _bl_data_paths[pl] = data_paths[ref_loss][ref_model][pl]

    # mc_E: use the first available model's truth values
    first_loss = next(iter(E_true_dict.get(playlists[0], {})), None) if playlists else None
    first_model = next(iter(E_true_dict[playlists[0]][first_loss]), None) if first_loss else None

    for eval_dataset in playlists:
        if eval_dataset not in _bl_data_paths:
            continue
        dp_path = _bl_data_paths[eval_dataset]

        # mc_E from any available model
        for _l in E_true_dict.get(eval_dataset, {}):
            for _m in E_true_dict[eval_dataset][_l]:
                mc_E[eval_dataset] = E_true_dict[eval_dataset][_l][_m]
                break
            if eval_dataset in mc_E:
                break

        Enu_baselines.setdefault(eval_dataset, {})
        Enu_filters.setdefault(eval_dataset, {})

        split_idx_path = os.path.join(dp_path, "result.pkl")
        if not os.path.exists(split_idx_path):
            if verbose:
                print(f"result.pkl not found at {split_idx_path}, skipping baselines for {eval_dataset}")
            continue
        with open(split_idx_path, "rb") as f:
            split_idx = pickle.load(f)
        bl_path = os.path.join(dp_path, "baselines", f"{eval_dataset}_enu_baselines.npz")
        if not os.path.exists(bl_path):
            if verbose:
                print(f"Baselines file not found: {bl_path}")
            continue
        current_baselines = np.load(bl_path, mmap_mode="r")
        print("keys: ", split_idx.keys())
        test_idx = split_idx[eval_dataset]["test_idx"]
        for key in current_baselines:
            if key in ["muon_filter_CC_paper", "mc_current", "q0", "q3"]:
                Enu_filters[eval_dataset][key] = current_baselines[key][test_idx]
                if key in ["q0", "q3"]:
                    Enu_filters[eval_dataset][key] = Enu_filters[eval_dataset][key] / 1000
            elif key == "E_recoil_CCinc_only":
                bl = current_baselines[key][test_idx] / 1000
                bl[bl == 0] = -1
                Enu_baselines[eval_dataset][key] = bl
            elif key == "blob_recoil_E":
                bl_raw = current_baselines[key][test_idx] / 1000
                bl_scaled = bl_raw * EAVAILABLE_SCALE
                bl_scaled[bl_raw == 0] = -1
                Enu_baselines[eval_dataset]["blob_recoil_E_scaled"] = bl_scaled

    return {
        "results": results,
        "configs": configs,
        "data_paths": data_paths,
        "E_true_dict": E_true_dict,
        "E_pred_dict": E_pred_dict,
        "Enu_baselines": Enu_baselines,
        "Enu_filters": Enu_filters,
        "mc_E": mc_E,
        "playlists": playlists,
    }


def _resolve_baseline_ref(baseline_ref, configs, data_paths, training_names):
    """Pick the (loss, model) to use for loading baselines."""
    if baseline_ref is not None:
        return baseline_ref

    for loss in training_names:
        for model in training_names[loss]:
            for pl in configs.get(loss, {}).get(model, {}):
                cfg = configs[loss][model][pl]
                has_path = "path" in cfg or pl in data_paths.get(loss, {}).get(model, {})
                if cfg and has_path:
                    return loss, model
    return None, None


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

def _build_event_mask(
    dp: str,
    Enu_filters: dict,
    Enu_baselines: dict,
    baseline_key: str,
    use_cc_selection: int,
) -> np.ndarray:
    """Build a boolean event-selection mask.

    Parameters
    ----------
    use_cc_selection :
        0 – only require ``baseline_key >= 0`` (minimal, baseline-only cut).
        1 – muon_filter_CC_paper only.
        2 – muon_filter_CC_paper AND ``E_recoil_CCinc_only >= 0`` (default,
            full CC-inclusive analysis selection).
    """
    if use_cc_selection == 0:
        if dp in Enu_baselines and baseline_key in Enu_baselines[dp]:
            return Enu_baselines[dp][baseline_key] >= 0
        n = len(next(iter(Enu_filters[dp].values())))
        return np.ones(n, dtype=bool)

    mask = Enu_filters[dp]["muon_filter_CC_paper"]
    if use_cc_selection >= 2:
        if dp in Enu_baselines and "E_recoil_CCinc_only" in Enu_baselines[dp]:
            mask = mask & (Enu_baselines[dp]["E_recoil_CCinc_only"] >= 0)
    return mask


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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
    transform = None,
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
        q3_bins = [0, 0.3, 0.6, 1.2, 1.8, 2.4, 3.0,100]

    datasets = [dataset_to_plot] if isinstance(dataset_to_plot, str) else list(dataset_to_plot)
    if len(datasets) > 1:
        if dataset_to_linestyle is None:
            raise ValueError("dataset_to_linestyle is required when dataset_to_plot is a list")
        missing = [d for d in datasets if d not in dataset_to_linestyle]
        if missing:
            raise ValueError(f"dataset_to_linestyle must contain an entry for each dataset; missing: {missing}")

    # -- load data if not provided ------------------------------------------
    if data is None:
        data = load_eval_data(
            CKPT_DIR, training_names,
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
        return (
            _has_q3(dp)
            and dp in Enu_baselines
            and baseline_key in Enu_baselines[dp]
        )

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
        loss = np.where(abs_diff <= 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
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
        mask_sel = _build_event_mask(dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection)

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
                    iqr = float(np.percentile(ratio_bl_clipped, 75) - np.percentile(ratio_bl_clipped, 25))
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
                        model_iqr = float(np.percentile(ratio_clipped, 75) - np.percentile(ratio_clipped, 25))
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
                            f"{model}-{loss} (RMS={model_rms:.3f}, "
                            f"plotted={frac_plotted:.1f}% of IQR sample)"
                            if not np.isnan(frac_plotted)
                            else f"{model}-{loss} (RMS={model_rms:.3f})"
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
                hist_ax[0, i].set(xlabel="E reco / E true", ylabel="Counts",
                                  title=f"q3: {q3_label} GeV{eff_str}")
                hist_ax[0, i].grid(True)
                hist_ax[1, i].set(xlabel="E reco / E true", ylabel="Counts",
                                  title=f"q3: {q3_label}")
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
    colors = prop_cycle.by_key().get("color", ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
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
            train_loss = LOG1P_LOSS_models_by_dp.get(dp, {}).get(loss, {}).get(model, float("nan"))
            color = color_by_lm[(loss, model)]
            if single_dataset:
                ls = default_linestyle
                lab = (
                    f"{model}"
                )
            else:
                ls = dataset_to_linestyle.get(dp, "-")
                lab = (
                    f"{model}"
                )
            ax[0].plot(q3_bin_mids, rms_vals, ls, color=color, label=lab)
            ax[1].plot(q3_bin_mids, iqr_vals, ls, color=color, label=lab)

    for dp in datasets_with_q3:
        rms_bl = RMS_baseline_by_dp.get(dp)
        iqr_bl = IQR_baseline_by_dp.get(dp)
        pct_bl = PCT_baseline_by_dp.get(dp)
        if not rms_bl or not iqr_bl or not pct_bl:
            continue
        pct_bl_arr = np.array(pct_bl)
        #pct_bl_mean = np.nanmean(pct_bl_arr) if pct_bl_arr.size > 0 else float("nan")
        #bl_loss = LOG1P_LOSS_baseline_by_dp.get(dp, float("nan"))
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
    ax[0].set(xlabel="MC truth $q_3$ [GeV]", ylabel="RMS of $E_{\\mathrm{reco}}/E_{\\mathrm{true}}$")
    ax[1].set(xlabel="MC truth $q_3$ [GeV]", ylabel="25-75 IQR of $E_{\\mathrm{reco}}/E_{\\mathrm{true}}$")
    ax[0].grid(True)
    ax[1].grid(True)
    fig.tight_layout()

    if return_hist_fig:
        return fig, hist_fig
    return fig


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
            CKPT_DIR, training_names,
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

    residual_bins_list = [np.linspace(-1, 1, 100), np.linspace(-5, 5, 100), np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)]
    ratio_bins = np.linspace(0, 3, 300)
    ratio_bins_wide = np.linspace(0, 10, 300)

    if has_baselines:
        mask_sel = _build_event_mask(dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection)

    for i in range(min(n_cols, len(residual_bins_list))):
        elow, ehigh = energy_bins[i], energy_bins[i + 1]
        if has_baselines:
            mask_e = (mc_E[dp] > elow) & (mc_E[dp] < ehigh)
            mask = mask_e & mask_sel
            true = mc_E[dp][mask]
            baseline = Enu_baselines[dp][baseline_key][mask]
            valid = true > 0
            ratio_bl = baseline[valid] / true[valid]
            ax[0, i].hist(baseline - true, bins=residual_bins_list[i], histtype="step", label="baseline")
            ax[1, i].hist(ratio_bl, bins=ratio_bins, histtype="step", label="baseline")
            ax[2, i].hist(ratio_bl, bins=ratio_bins_wide, histtype="step", label="baseline")

        for loss in results:
            for model in results[loss]:
                if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                    continue
                if has_baselines:
                    reco = E_pred_dict[dp][loss][model][mask]
                    ratio_model = reco[valid] / true[valid]
                    ax[0, i].hist(reco - true, bins=residual_bins_list[i], histtype="step",
                                  label=f"{model}-{loss}")
                    ax[1, i].hist(ratio_model, bins=ratio_bins, histtype="step",
                                  label=f"{model}-{loss}")
                    ax[2, i].hist(ratio_model, bins=ratio_bins_wide, histtype="step",
                                  label=f"{model}-{loss}")

        elabel = f"{elow}-{ehigh}" if ehigh < 100 else f"{elow}+"
        ax[0, i].set(xlabel="E reco − E true", ylabel="Counts", title=f"E true: {elabel} GeV")
        #ax[0, i].legend(loc="lower left", fontsize=7)
        ax[0, i].grid(True)
        ax[1, i].set(xlabel="E reco / E true", ylabel="Counts", title=f"E true: {elabel} GeV")
        ax[1, i].legend(loc="lower left", fontsize=7)
        ax[1, i].grid(True)
        ax[2, i].set(xlabel="E reco / E true", ylabel="Counts", title=f"E true: {elabel} GeV (0–10, log)")
        ax[2, i].set_yscale("log")
        ax[2, i].legend(loc="lower left", fontsize=7)
        ax[2, i].grid(True)
        ax[0, i].legend(loc="lower left", fontsize=7)

    fig.tight_layout()
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
    verbose: bool = True,
    data: dict | None = None,
    transform=None,
    suppress_errors: bool = False,
) -> plt.Figure:
    """Per-q3-bin residual and ratio histograms (E_reco−E_true and E_reco/E_true)."""
    if q3_bins is None:
        q3_bins = [0, 0.3, 0.6, 1.2, 1.8, 2.4, 3.0, 100]

    if data is None:
        data = load_eval_data(
            CKPT_DIR, training_names,
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

    q3 = Enu_filters[dp]["q3"] if dp in Enu_filters and "q3" in Enu_filters[dp] else None
    if q3 is None:
        warnings.warn(f"No q3 information for dataset '{dp}'.")
        return plt.figure()

    if has_baselines:
        mask_sel = _build_event_mask(dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection)
    else:
        # Fallback: just select events with positive truth energy
        mask_sel = mc_E[dp] > 0

    n_cols = len(q3_bins) - 1
    fig, ax = plt.subplots(2, n_cols, figsize=(4.3 * n_cols, 7.0))
    if n_cols == 1:
        ax = ax[:, np.newaxis]

    residual_bins = np.linspace(-2, 2, 160)
    ratio_bins = np.linspace(0, 2, 50)

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
                label="baseline",
            )
            ax[1, i].hist(
                ratio_bl,
                bins=ratio_bins,
                histtype="step",
                label="baseline",
            )

        for loss in results:
            for model in results[loss]:
                if model not in E_pred_dict.get(dp, {}).get(loss, {}):
                    continue
                reco = E_pred_dict[dp][loss][model][mask]
                ratio_model = reco[valid] / true[valid]
                ax[0, i].hist(
                    reco[valid] - true[valid],
                    bins=residual_bins,
                    histtype="step",
                    label=f"{model}-{loss}",
                )
                ax[1, i].hist(
                    ratio_model,
                    bins=ratio_bins,
                    histtype="step",
                    label=f"{model}-{loss}",
                )

        qlabel = f"{qlow}-{qhigh}" if qhigh < 100 else f"{qlow}+"
        ax[0, i].set(
            xlabel="E reco − E true [GeV]",
            ylabel="Counts",
            title=f"$q_3$: {qlabel} GeV",
        )
        ax[1, i].set(
            xlabel="E reco / E true",
            ylabel="Counts",
            title=f"$q_3$: {qlabel} GeV",
        )
        ax[0, i].legend(loc="lower left", fontsize=7)
        ax[1, i].legend(loc="lower left", fontsize=7)
        ax[0, i].grid(True)
        ax[1, i].grid(True)

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Grouped training_names utilities (seeds as arrays)
# ---------------------------------------------------------------------------

_SEED_SEP = "§"


def _extract_sample_count(label: str) -> float:
    """Extract numeric sample count from labels like 'Transformer 6M' or '500k'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*([kKmM])\b", label)
    if match:
        num = float(match.group(1))
        suffix = match.group(2).upper()
        return num * (1e6 if suffix == "M" else 1e3)
    match = re.search(r"(\d+)", label)
    return float(match.group(1)) if match else 0.0


def _resolve_color_map(
    config_labels: list[str],
    colors: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a label → colour mapping from a user-supplied *colors* dict.

    *colors* maps human-readable sample-count tokens (e.g. ``"6M"``,
    ``"500k"``) to any matplotlib-compatible colour.  Each config label
    is matched to the first token that appears as a substring.  Labels
    without a match get a default grey.
    """
    if not config_labels:
        return {}
    if colors is None:
        colors = {}
    out: dict[str, Any] = {}
    for label in config_labels:
        if label in colors:
            out[label] = colors[label]
        else:
            out[label] = "tab:gray"
    return out


def flatten_grouped_training_names(
    training_names_grouped: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, str]]:
    """Flatten ``{loss: {label: [run, …]}}`` → ``{loss: {label§i: run, …}}``."""
    flat: dict[str, dict[str, str]] = {}
    for loss, configs in training_names_grouped.items():
        flat[loss] = {}
        for label, runs in configs.items():
            for i, run in enumerate(runs):
                flat[loss][f"{label}{_SEED_SEP}{i}"] = run
    return flat


def load_eval_data_grouped(
    CKPT_DIR: str | Path,
    training_names_grouped: dict[str, dict[str, list[str]]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Load evaluation data for grouped training names (seeds as lists)."""
    return load_eval_data(
        CKPT_DIR,
        flatten_grouped_training_names(training_names_grouped),
        **kwargs,
    )


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
    transform = NONE,
    return_hist_fig: bool = False,  # accepted for API parity, ignored
    return_values: bool = False,
    suppress_errors: bool = False,
    text: str = "",
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
            CKPT_DIR, training_names_grouped,
            playlists=playlists,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            verbose=verbose,
            transform=transform,
            suppress_errors=suppress_errors,
        )

    E_pred_dict = data["E_pred_dict"]
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
    has_baselines = (
        has_q3
        and dp in Enu_baselines
        and baseline_key in Enu_baselines[dp]
    )

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
    mask_sel = _build_event_mask(dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection)

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

    # Layout: 2×2 – top: RMS/MPV and IQR/MPV; bottom: MPV (duplicated);
    # bottom row has half the height of the top row.
    fig, ax = plt.subplots(
        2,
        2,
        figsize=(9, 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_top = ax[0]
    ax_bottom = ax[1]

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
                    np.percentile(ratio_bl_clipped, 75) - np.percentile(ratio_bl_clipped, 25)
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

    print(f"E_pred_dict keys for '{dp}': { {l: list(m.keys()) for l, m in E_pred_dict.get(dp, {}).items()} }")
    for loss in training_names_grouped:
        for config_label, runs in training_names_grouped[loss].items():
            seed_rms = np.empty((len(runs), n_plot_bins))
            seed_iqr = np.empty((len(runs), n_plot_bins))
            seed_mpv = np.empty((len(runs), n_plot_bins))
            for s in range(len(runs)):
                flat_key = f"{config_label}{_SEED_SEP}{s}"
                if flat_key not in E_pred_dict.get(dp, {}).get(loss, {}):
                    print(f"  WARNING: '{flat_key}' not found in E_pred_dict['{dp}']['{loss}']")
                    continue
                for i in range(n_plot_bins):
                    true = mc_E[dp][bin_masks[i]]
                    reco = E_pred_dict[dp][loss][flat_key][bin_masks[i]]
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
                            np.percentile(ratio_clipped, 75) - np.percentile(ratio_clipped, 25)
                        )
                        seed_rms[s, i] = float(
                            np.sqrt(np.mean((ratio_clipped - 1.0) ** 2))
                        )
                        if ratio_mpv.size > 0:
                            hist, edges = np.histogram(ratio_mpv, bins=100, range=(0.0, 2.0))
                            max_idx = int(np.argmax(hist))
                            seed_mpv[s, i] = float(0.5 * (edges[max_idx] + edges[max_idx + 1]))
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
            lbl = f"{config_label}"

            with np.errstate(divide="ignore", invalid="ignore"):
                rms_over_mpv = mean_rms / mean_mpv
                iqr_over_mpv = mean_iqr / mean_mpv

            ax_top[0].plot(q3_bin_mids, rms_over_mpv, ".--", color=color, label=lbl)
            ax_top[0].fill_between(
                q3_bin_mids,
                rms_over_mpv - (std_rms / mean_mpv),
                rms_over_mpv + (std_rms / mean_mpv),
                alpha=0.25,
                color=color,
            )
            ax_top[1].plot(q3_bin_mids, iqr_over_mpv, ".--", color=color, label=lbl)
            ax_top[1].fill_between(
                q3_bin_mids,
                iqr_over_mpv - (std_iqr / mean_mpv),
                iqr_over_mpv + (std_iqr / mean_mpv),
                alpha=0.25,
                color=color,
            )

            # MPV curves with ±1σ bands (bottom row, duplicated on both axes)
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
        ax_top[0].plot(q3_bin_mids, bl_rms_over_mpv, ".--", color="black", label="baseline")
        ax_top[1].plot(q3_bin_mids, bl_iqr_over_mpv, ".--", color="black", label="baseline")
        ax_bottom[0].plot(q3_bin_mids, bl_mpv, ".--", color="black", label="baseline")
        ax_bottom[1].plot(q3_bin_mids, bl_mpv, ".--", color="black", label="baseline")

    # Legend placement; optional *text* is used as legend title if provided.
    # Slightly larger font for better readability in notebook/figures.
    legend_kwargs: dict[str, Any] = {"fontsize": 9, "loc": "upper right"}
    if text:
        legend_kwargs["title"] = text
        legend_kwargs["title_fontsize"] = 10

    ax_top[0].legend(**legend_kwargs)
    ax_top[1].legend(**legend_kwargs)
    ax_bottom[0].legend(fontsize=7)
    ax_bottom[1].legend(fontsize=7)

    ax_top[0].set(
        xlabel="MC truth $q_3$ [GeV]",
        ylabel="RMS / MPV of $E_{\\mathrm{reco}}/E_{\\mathrm{true}}$",
    )
    ax_top[1].set(
        xlabel="MC truth $q_3$ [GeV]",
        ylabel="IQR / MPV of $E_{\\mathrm{reco}}/E_{\\mathrm{true}}$",
    )
    ax_bottom[0].set(
        xlabel="MC truth $q_3$ [GeV]",
        ylabel="MPV of $E_{\\mathrm{reco}}/E_{\\mathrm{true}}$",
    )
    ax_bottom[1].set(
        xlabel="MC truth $q_3$ [GeV]",
        ylabel="MPV of $E_{\\mathrm{reco}}/E_{\\mathrm{true}}$",
    )

    for a in ax_top:
        a.grid(True)
    for a in ax_bottom:
        a.grid(True)

    fig.tight_layout()

    if return_values:
        return fig, values
    return fig


def plot_scaling_law(
    values: dict[str, Any],
    metric: str = "iqr",
    q3_bin_index: int = -1,
    colors: dict[str, Any] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a scaling-law curve: IQR (or RMS) of the highest q3 bin vs training samples.

    Parameters
    ----------
    values : the *values* dict returned by
        ``plot_rms_iqr_with_uncertainty(..., return_values=True)``.
    metric : ``"iqr"`` or ``"rms"``.
    q3_bin_index : which q3 bin to use.  ``-1`` (default) picks the last
        plotted bin (highest q3).
    colors : optional ``{substring_token: colour}`` mapping applied to
        config labels (same convention as *plot_rms_iqr_with_uncertainty*).
    ax : optional axes to draw on; a new figure is created if *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    q3_mids = values["q3_bin_mids"]
    bin_idx = q3_bin_index if q3_bin_index >= 0 else len(q3_mids) + q3_bin_index

    all_labels: list[str] = []
    for loss in values:
        if loss in ("q3_bin_mids", "baseline"):
            continue
        for label in values[loss]:
            if label not in all_labels:
                all_labels.append(label)
    color_map = _resolve_color_map(all_labels, colors)

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    for loss in values:
        if loss in ("q3_bin_mids", "baseline"):
            continue
        for config_label, v in values[loss].items():
            n_samples = _extract_sample_count(config_label)
            if n_samples == 0:
                continue
            y_mean = v[mean_key][bin_idx]
            y_std = v[std_key][bin_idx]
            color = color_map.get(config_label, "tab:gray")
            ax.errorbar(
                n_samples, y_mean, yerr=y_std, fmt="o",
                color=color, capsize=4,
            )
            ax.annotate(
                config_label, (n_samples, y_mean),
                textcoords="offset points", xytext=(6, 4), fontsize=7,
                color=color,
            )

    if "baseline" in values:
        bl_val = values["baseline"][metric][bin_idx]
        ax.axhline(bl_val, color="black", ls="--", lw=1, label="baseline")

    ax.set_xscale("log")
    ax.set_xlabel("Number of training samples")
    metric_label = "IQR" if metric == "iqr" else "RMS"
    q3_lo = q3_mids[bin_idx] - (q3_mids[1] - q3_mids[0]) / 2 if len(q3_mids) > 1 else 0
    ax.set_ylabel(f"{metric_label} [GeV]  (q$_3$ bin {bin_idx})")
    ax.set_title(f"Scaling law – {metric_label} vs training set size")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    return fig

