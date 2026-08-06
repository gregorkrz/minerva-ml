"""Load regression evaluation arrays from checkpoints."""

from __future__ import annotations

import json
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ._constants import DEFAULT_BASELINE_KEY, EAVAILABLE_SCALE, MIN_EPRED_GEV


def load_eval_data(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]],
    playlists: list[str] | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_run: str | None = None,
    verbose: bool = True,
    transform=None,
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
                p1 = (
                    CKPT_DIR
                    / run
                    / "test_results"
                    / f"outputs_{run}_minerva_{playlist}_0.npz"
                )
                p2 = (
                    CKPT_DIR
                    / run
                    / "test_results"
                    / f"outputs_best_model_minerva_{playlist}_0.npz"
                )
                if p1.exists():
                    results[loss][model][playlist] = dict(np.load(p1))
                elif p2.exists():
                    results[loss][model][playlist] = dict(np.load(p2))
                else:
                    if suppress_errors:
                        if verbose:
                            print(
                                f"Skipping {run} on playlist {playlist} (no eval results found)"
                            )
                        continue
                    raise FileNotFoundError(
                        f"No results for {run} on playlist {playlist} "
                        f"– checked {p1} and {p2}"
                    )
                pred = results[loss][model][playlist]["prediction"]
                pred[pred < MIN_EPRED_GEV] = 0

                settings_path = CKPT_DIR / run / "settings.json"
                if settings_path.exists():
                    with open(settings_path, "r") as f:
                        configs[loss][model][playlist] = json.load(f)
                else:
                    # if verbose:
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
                        print(
                            f"No best model found for {training_names[loss][model]} on {playlist}"
                        )

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
                E_true_dict[dataset][loss][model] = results[loss][model][dataset][
                    "pid"
                ].flatten()
                E_pred_dict[dataset][loss][model] = results[loss][model][dataset][
                    "prediction"
                ].flatten()
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
            print(
                f"baseline_run '{baseline_run}' has no settings.json or best_model.pt"
            )

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
    first_loss = (
        next(iter(E_true_dict.get(playlists[0], {})), None) if playlists else None
    )
    first_model = (
        next(iter(E_true_dict[playlists[0]][first_loss]), None) if first_loss else None
    )

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
                print(
                    f"result.pkl not found at {split_idx_path}, skipping baselines for {eval_dataset}"
                )
            continue
        with open(split_idx_path, "rb") as f:
            split_idx = pickle.load(f)
        bl_path = os.path.join(
            dp_path, "baselines", f"{eval_dataset}_enu_baselines.npz"
        )
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
                    Enu_filters[eval_dataset][key] = (
                        Enu_filters[eval_dataset][key] / 1000
                    )
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
                has_path = "path" in cfg or pl in data_paths.get(loss, {}).get(
                    model, {}
                )
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
