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
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


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
                    if verbose:
                        print(f"No settings found for {run} on playlist {playlist}")
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
        test_idx = split_idx[eval_dataset]["test_idx"]
        for key in current_baselines:
            if key in ["muon_filter_CC_paper", "q0", "q3"]:
                Enu_filters[eval_dataset][key] = current_baselines[key][test_idx]
                if key in ["q0", "q3"]:
                    Enu_filters[eval_dataset][key] = Enu_filters[eval_dataset][key] / 1000
            elif key == "E_recoil_CCinc_only":
                bl = current_baselines[key][test_idx] / 1000
                bl[bl == 0] = -1
                Enu_baselines[eval_dataset][key] = bl

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
# Plotting
# ---------------------------------------------------------------------------

def plot_rms_iqr(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    dataset_to_plot: str = "1A",
    q3_bins: list[float] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    rms_clip: float = 0.6,
    show_q3_histograms: bool = False,
    verbose: bool = True,
    data: dict | None = None,
) -> plt.Figure:
    """Produce the RMS / IQR vs *q3* summary plot.

    Parameters
    ----------
    CKPT_DIR : checkpoint root (ignored when *data* is provided).
    training_names : ``{loss: {model: run_name}}``.
    playlists : default ``["1A"]``.
    dataset_to_plot : which playlist to plot.
    q3_bins : bin edges in GeV (default ``[0, 0.3, 0.6, 1.2, 100]``).
    baseline_ref : ``(loss, model)`` for physics baseline lookup.
    baseline_run : standalone run name that has ``settings.json`` /
        ``best_model.pt``, used to load baselines when none of the
        models in *training_names* carry those files.
    rms_clip : events with ``|reco-true| > rms_clip`` are excluded from
        the RMS calculation (matches the notebook convention).
    show_q3_histograms : if *True* also show the per-q3-bin residual
        histograms (the intermediate diagnostic plot).
    verbose : print warnings during data loading.
    data : pre-loaded data dict from :func:`load_eval_data`.  When
        supplied, *CKPT_DIR* and *playlists* are ignored.

    Returns
    -------
    matplotlib.figure.Figure – the RMS / IQR summary figure.
    """
    if q3_bins is None:
        q3_bins = [0, 0.3, 0.6, 1.2, 100]

    # -- load data if not provided ------------------------------------------
    if data is None:
        data = load_eval_data(
            CKPT_DIR, training_names,
            playlists=playlists,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            verbose=verbose,
        )

    results = data["results"]
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
        and "E_recoil_CCinc_only" in Enu_baselines[dp]
    )

    if not has_q3:
        warnings.warn(
            f"No q3 / filter data available for dataset '{dp}'. "
            "Cannot produce q3-binned plots. "
            "Pass baseline_run='<run_with_settings.json>' to load them."
        )
        return plt.figure()

    # -- compute RMS / IQR per q3 bin (cell 10 logic) ----------------------
    q3_arr = np.asarray(q3_bins)
    n_plot_bins = len(q3_arr) - 2  # last bin is overflow, excluded from plot
    q3_bin_mids = ((q3_arr[:-1] + q3_arr[1:]) / 2)[:n_plot_bins]

    mask_filter = Enu_filters[dp]["muon_filter_CC_paper"]
    q3 = Enu_filters[dp]["q3"]
    if has_baselines:
        mask_reco_bl = Enu_baselines[dp]["E_recoil_CCinc_only"] >= 0
        mask_sel = mask_filter & mask_reco_bl
    else:
        mask_sel = mask_filter

    IQR_models: dict[str, dict[str, list[float]]] = {}
    RMS_models: dict[str, dict[str, list[float]]] = {}
    IQR_baseline: list[float] = []
    RMS_baseline: list[float] = []

    hist_fig = None
    if show_q3_histograms:
        hist_fig, hist_ax = plt.subplots(2, n_plot_bins, figsize=(14, 7))
        if n_plot_bins == 1:
            hist_ax = hist_ax[:, np.newaxis]

    for i in range(n_plot_bins):
        mask_q3_orig = (q3 > q3_bins[i]) & (q3 <= q3_bins[i + 1])
        mask = mask_q3_orig & mask_sel
        efficiency = mask.sum() / mask_q3_orig.sum() * 100
        true = mc_E[dp][mask]

        if has_baselines:
            baseline = Enu_baselines[dp]["E_recoil_CCinc_only"][mask]
            bl_residual = baseline - true
            iqr = float(np.percentile(bl_residual, 75) - np.percentile(bl_residual, 25))
            rms = float(np.sqrt(np.mean(bl_residual[np.abs(bl_residual) < rms_clip] ** 2)))
            IQR_baseline.append(iqr)
            RMS_baseline.append(rms)

            if show_q3_histograms:
                bins = np.linspace(-rms_clip, rms_clip, 100)
                hist_ax[0, i].hist(bl_residual, bins=bins, histtype="step", label="baseline", color="black")
                hist_ax[1, i].hist(bl_residual, bins=bins, histtype="step", label="baseline", color="black")

        for loss in results:
            IQR_models.setdefault(loss, {})
            RMS_models.setdefault(loss, {})
            for model in results[loss]:
                IQR_models[loss].setdefault(model, [])
                RMS_models[loss].setdefault(model, [])

                reco = E_pred_dict[dp][loss][model][mask]
                reco_minus_true = reco - true

                model_iqr = float(np.percentile(reco_minus_true, 75) - np.percentile(reco_minus_true, 25))
                model_rms = float(np.sqrt(np.mean(reco_minus_true[np.abs(reco_minus_true) < rms_clip] ** 2)))
                IQR_models[loss][model].append(model_iqr)
                RMS_models[loss][model].append(model_rms)

                if show_q3_histograms:
                    bins = np.linspace(-rms_clip, rms_clip, 100)
                    n_in = int(np.sum(np.abs(reco_minus_true) < rms_clip))
                    label = f"{model}-{loss} (RMS={model_rms:.2f}, N={n_in})"
                    hist_ax[0, i].hist(reco_minus_true, bins=bins, histtype="step", label=label)
                    hist_ax[1, i].hist(reco_minus_true, bins=bins, histtype="step", label=label)

        if show_q3_histograms:
            q3_label = f"{q3_bins[i]}-{q3_bins[i+1]}"
            eff_str = f" (eff: {efficiency:.1f}%)" if has_baselines else ""
            hist_ax[0, i].set(xlabel="E reco − E true", ylabel="Counts",
                              title=f"q3: {q3_label} GeV{eff_str}")
            hist_ax[0, i].grid(True)
            hist_ax[1, i].set(xlabel="E reco − E true", ylabel="Counts",
                              title=f"q3: {q3_label}")
            hist_ax[1, i].set_yscale("log")
            hist_ax[1, i].legend(loc="lower left", fontsize=7)
            hist_ax[1, i].grid(True)

    if show_q3_histograms and hist_fig is not None:
        hist_fig.tight_layout()

    # -- summary RMS / IQR figure (cell 11) ---------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    for loss in results:
        for model in results[loss]:
            ax[0].plot(q3_bin_mids, np.array(RMS_models[loss][model]), ".--",
                       label=f"{model}-{loss}")
            ax[1].plot(q3_bin_mids, np.array(IQR_models[loss][model]), ".--",
                       label=f"{model}-{loss}")

    if IQR_baseline:
        ax[0].plot(q3_bin_mids, np.array(RMS_baseline), ".--", label="baseline", color="black")
        ax[1].plot(q3_bin_mids, np.array(IQR_baseline), ".--", label="baseline", color="black")

    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    ax[0].set(xlabel="Bin middle point $q_3$ [GeV]", ylabel="RMS [GeV]")
    ax[1].set(xlabel="Bin middle point $q_3$ [GeV]", ylabel="IQR [GeV]")
    ax[0].grid(True)
    ax[1].grid(True)
    fig.tight_layout()

    return fig


def plot_residuals_by_energy(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    dataset_to_plot: str = "1A",
    energy_bins: list[float] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    verbose: bool = True,
    data: dict | None = None,
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
        and "E_recoil_CCinc_only" in Enu_baselines[dp]
    )

    n_cols = len(energy_bins) - 1
    fig, ax = plt.subplots(2, n_cols, figsize=(4.3 * n_cols, 7))
    if n_cols == 1:
        ax = ax[:, np.newaxis]

    residual_bins_list = [np.linspace(-1, 1, 100), np.linspace(-5, 5, 100), np.linspace(-10, 10, 100)]
    ratio_bins = np.linspace(0, 2, 100)

    if has_baselines:
        mask_filter = Enu_filters[dp]["muon_filter_CC_paper"]
        mask_reco_bl = Enu_baselines[dp]["E_recoil_CCinc_only"] >= 0
        mask_sel = mask_filter & mask_reco_bl

    for i in range(min(n_cols, len(residual_bins_list))):
        elow, ehigh = energy_bins[i], energy_bins[i + 1]
        if has_baselines:
            mask_e = (mc_E[dp] > elow) & (mc_E[dp] < ehigh)
            mask = mask_e & mask_sel
            true = mc_E[dp][mask]
            baseline = Enu_baselines[dp]["E_recoil_CCinc_only"][mask]
            valid = true > 0
            ax[0, i].hist(baseline - true, bins=residual_bins_list[i], histtype="step", label="baseline")
            ax[1, i].hist(baseline[valid] / true[valid], bins=ratio_bins, histtype="step", label="baseline")

        for loss in results:
            for model in results[loss]:
                if has_baselines:
                    reco = E_pred_dict[dp][loss][model][mask]
                    ax[0, i].hist(reco - true, bins=residual_bins_list[i], histtype="step",
                                  label=f"{model}-{loss}")
                    ax[1, i].hist(reco[valid] / true[valid], bins=ratio_bins, histtype="step",
                                  label=f"{model}-{loss}")

        elabel = f"{elow}-{ehigh}" if ehigh < 100 else f"{elow}+"
        ax[0, i].set(xlabel="E reco − E true", ylabel="Counts", title=f"E true: {elabel} GeV")
        ax[0, i].legend(loc="lower left", fontsize=7)
        ax[0, i].grid(True)
        ax[1, i].set(xlabel="E reco / E true", ylabel="Counts", title=f"E true: {elabel} GeV")
        ax[1, i].legend(loc="lower left", fontsize=7)
        ax[1, i].grid(True)

    fig.tight_layout()
    return fig

