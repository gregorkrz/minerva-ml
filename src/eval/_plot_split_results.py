"""Load evaluation results from non-default splits for plot-config overlays."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.eval._plot_config import PlotConfig
from src.eval.classification_plots import add_hadronic_W_to_classification_data
from src.eval.classification_plots._io import (
    load_results,
    load_truth_and_baselines,
    run_has_test_results_npz,
)
from src.eval.classification_plots._metrics_tasks import metrics_W_by_model_data


def overlay_entries(cfg: PlotConfig) -> list[tuple[str, str, str]]:
    """Return ``(plot_name, source_name, split)`` for models not on ``eval_split``."""
    out: list[tuple[str, str, str]] = []
    for entry in cfg.models:
        split = entry.split or cfg.eval_split
        source = entry.source_name or entry.name
        if split != cfg.eval_split or source != entry.name:
            out.append((entry.name, source, split))
    return out


def load_split_data_by_playlist(
    clf: dict[str, Any],
    split: str,
    event_indices: np.ndarray | None = None,
    playlists: list[str] | None = None,
) -> dict[str, dict]:
    """Truth/baseline arrays aligned to *split* (no hadronic *W* augmentation)."""
    ckpt_dir = clf["ckpt_dir"]
    training_names = clf["training_names"]
    if playlists is None:
        playlists = clf["playlists"]
    out: dict[str, dict] = {}
    for pl in playlists:
        out[pl] = load_truth_and_baselines(
            ckpt_dir,
            training_names,
            playlists=[pl],
            split=split,
            event_indices=event_indices,
        )
    return out


def load_split_data_w_by_playlist(
    clf: dict[str, Any],
    split: str,
    event_indices: np.ndarray | None = None,
    playlists: list[str] | None = None,
) -> dict[str, dict]:
    """Truth/baseline/W arrays aligned to *split* indices."""
    ckpt_dir = clf["ckpt_dir"]
    training_names = clf["training_names"]
    if playlists is None:
        playlists = clf["playlists"]
    out: dict[str, dict] = {}
    for pl in playlists:
        data = load_truth_and_baselines(
            ckpt_dir,
            training_names,
            playlists=[pl],
            split=split,
            event_indices=event_indices,
        )
        out[pl] = add_hadronic_W_to_classification_data(data, pl)
    return out


def load_overlay_results(
    clf: dict[str, Any],
    cfg: PlotConfig,
    playlists: list[str] | None = None,
) -> tuple[dict[str, list], dict[str, dict[str, dict]], dict[str, dict[str, dict]]]:
    """Load overlay model results and split-specific truth / ``data_w`` dicts.

    Returns
    -------
    overlay_results
        ``{plot_name: run_list}`` keyed by plot-config name (not source).
    data_by_split
        ``{split: {playlist: data}}`` for kinematic metrics (pion, q3).
    data_w_by_split
        ``{split: {playlist: data_w}}`` for hadronic *W* metrics.
    """
    overlays = overlay_entries(cfg)
    if not overlays:
        return {}, {}

    ckpt_dir = clf["ckpt_dir"]
    training_names = clf["training_names"]
    if playlists is None:
        playlists = clf["playlists"]

    overlay_results: dict[str, list] = {}
    split_data_cache: dict[tuple, dict[str, dict[str, dict]]] = {}
    splits_needed: set[str] = set()

    for plot_name, source, split in overlays:
        if source not in training_names:
            print(
                f"[{plot_name}] skipping overlay: source {source!r} "
                f"not in training_names"
            )
            continue
        splits_needed.add(split)
        run_names = training_names[source]
        chosen_run = None
        for run_name in reversed(run_names):
            if run_has_test_results_npz(
                ckpt_dir, run_name, playlists=playlists, split=split
            ):
                chosen_run = run_name
                break
        if chosen_run is None:
            print(
                f"[{plot_name}] skipping overlay: no {split} results for "
                f"{source!r} (tried {len(run_names)} run(s))"
            )
            continue
        run_results = load_results(
            ckpt_dir,
            {source: [chosen_run]},
            playlists=playlists,
            split=split,
            verbose=False,
        )
        overlay_results[plot_name] = run_results[source]
        pl0 = playlists[0]
        ev_idx = run_results[source][0][pl0].get("eval_indices")
        cache_key = (split, ev_idx.tobytes() if ev_idx is not None else None)
        if cache_key not in split_data_cache:
            split_data_cache[cache_key] = {
                "data": load_split_data_by_playlist(
                    clf, split, event_indices=ev_idx, playlists=playlists
                ),
                "data_w": load_split_data_w_by_playlist(
                    clf, split, event_indices=ev_idx, playlists=playlists
                ),
            }

    data_by_split: dict[str, dict[str, dict]] = {}
    data_w_by_split: dict[str, dict[str, dict]] = {}
    for (split, _), cached in split_data_cache.items():
        data_by_split.setdefault(split, cached["data"])
        data_w_by_split.setdefault(split, cached["data_w"])

    return overlay_results, data_by_split, data_w_by_split


def apply_plot_config_results(
    clf: dict[str, Any],
    cfg: PlotConfig | None,
    playlists: list[str] | None = None,
) -> tuple[dict, dict, dict[str, dict[str, dict]]]:
    """Merge test pickle results with config split overlays.

    Returns ``(results, data_w_by_playlist, data_w_by_split)``.
    """
    results = dict(clf["results"])
    data_w_by_playlist = clf["data_w_by_playlist"]
    data_w_by_split: dict[str, dict[str, dict]] = {}

    if cfg is not None:
        overlay_results, _data_by_split, data_w_by_split = load_overlay_results(
            clf, cfg, playlists=playlists
        )
        results.update(overlay_results)
        results = cfg.filter_dict(results)

    return results, data_w_by_playlist, data_w_by_split


def overlay_model_names(cfg: PlotConfig | None) -> list[str]:
    if cfg is None:
        return []
    return [name for name, _, _ in overlay_entries(cfg)]


def build_data_by_model(
    results: dict,
    cfg: PlotConfig | None,
    data_w_by_split: dict[str, dict[str, dict]],
    playlist: str,
) -> dict[str, dict] | None:
    """Per-model ``data_w`` for models evaluated on a non-default split."""
    if cfg is None:
        return None
    out: dict[str, dict] = {}
    for name in results:
        split = cfg.model_split(name)
        if split != cfg.eval_split:
            data_w = data_w_by_split.get(split, {}).get(playlist)
            if data_w is not None:
                out[name] = data_w
    return out or None


def compute_all_metrics_W_for_config(
    results: dict,
    data_w: dict,
    data_w_by_split: dict[str, dict[str, dict]],
    cfg: PlotConfig | None,
    signal_classes: list[int],
    *,
    playlist: str = "1A",
    **kwargs,
) -> dict:
    """W metrics respecting per-model eval splits from *cfg*."""
    from src.eval.classification_plots import compute_all_metrics_W

    data_by_model = build_data_by_model(results, cfg, data_w_by_split, playlist)
    if data_by_model is None:
        return compute_all_metrics_W(
            results, data_w, signal_classes, playlist=playlist, **kwargs
        )
    return metrics_W_by_model_data(
        results,
        data_w,
        data_by_model,
        signal_classes,
        playlist=playlist,
        **kwargs,
    )


def merge_inttype_agg_overlays(
    pre_agg: dict,
    results: dict,
    data_w: dict,
    data_w_by_split: dict[str, dict[str, dict]],
    cfg: PlotConfig | None,
    signal_classes: list[int],
    *,
    fixed_fpr: list[float],
    playlist: str,
    use_global_fpr: bool,
) -> dict:
    """Append int-type W metrics for split-overlay models to *pre_agg*."""
    from src.eval._classification_metrics_cache import _int_masks

    overlay_names = overlay_model_names(cfg)
    if not overlay_names or cfg is None:
        return pre_agg

    data_by_model = build_data_by_model(results, cfg, data_w_by_split, playlist)
    if not data_by_model:
        return pre_agg

    merged = {code: dict(agg) for code, agg in pre_agg.items()}
    overlay_results = {k: results[k] for k in overlay_names if k in results}

    for int_code in merged:
        for model_name in overlay_names:
            if model_name not in overlay_results:
                continue
            data_m = data_by_model[model_name]
            n_overlay = len(overlay_results[model_name][0][playlist]["pid"])
            masks_m = _int_masks(data_m, playlist, n_events=n_overlay)
            mask_m = masks_m.get(int_code)
            if mask_m is None:
                continue
            extra = metrics_W_by_model_data(
                {model_name: overlay_results[model_name]},
                data_m,
                None,
                signal_classes,
                fixed_fpr=fixed_fpr,
                event_mask=mask_m,
                playlist=playlist,
                use_global_fpr=use_global_fpr,
            )
            if model_name in extra:
                merged[int_code][model_name] = extra[model_name]

    return merged


def compute_overlay_light_metrics(
    panel: str,
    signal_classes: list[int],
    fixed_fpr: list[float],
    overlay_results: dict,
    data_by_split: dict[str, dict[str, dict]],
    data_w_by_split: dict[str, dict[str, dict]],
    cfg: PlotConfig,
    playlist: str = "1A",
) -> dict[str, dict]:
    """Light-plot metrics for split-overlay models only (``pion_E`` or ``w`` panel)."""
    from src.eval.classification_plots import (
        compute_all_metrics,
        compute_all_metrics_W,
        data_with_signal_pion_bins,
    )

    out: dict[str, dict] = {}
    for model_name, run_list in overlay_results.items():
        split = cfg.model_split(model_name)
        if panel == "w":
            data_w = data_w_by_split[split][playlist]
            out.update(
                compute_all_metrics_W(
                    {model_name: run_list},
                    data_w,
                    signal_classes,
                    fixed_fpr=fixed_fpr,
                    playlist=playlist,
                    use_global_fpr=True,
                )
            )
        elif panel == "pion_E":
            data = data_by_split[split][playlist]
            pid = run_list[0][playlist]["pid"]
            data_sp = data_with_signal_pion_bins(
                data,
                pid,
                signal_classes,
                pion_quantile_require_has_pion=False,
                pion_bin_edge_method="equal_frequency",
            )
            out.update(
                compute_all_metrics(
                    {model_name: run_list},
                    data_sp,
                    signal_classes,
                    fixed_fpr=fixed_fpr,
                    playlist=playlist,
                    pion_bins_require_has_pion=False,
                )
            )
        else:
            raise ValueError(f"Unsupported light TPR panel: {panel!r}")
    return out


def merge_overlay_light_metrics(
    all_metrics: dict[str, dict],
    panel: str,
    signal_classes: list[int],
    fixed_fpr: list[float],
    overlay_results: dict,
    data_by_split: dict[str, dict[str, dict]],
    data_w_by_split: dict[str, dict[str, dict]],
    cfg: PlotConfig,
    playlist: str = "1A",
) -> dict[str, dict]:
    """Append overlay-model entries to a light-plot ``all_metrics`` dict."""
    if not overlay_results:
        return all_metrics
    extra = compute_overlay_light_metrics(
        panel,
        signal_classes,
        fixed_fpr,
        overlay_results,
        data_by_split,
        data_w_by_split,
        cfg,
        playlist=playlist,
    )
    merged = dict(all_metrics)
    for name in cfg.ordered_model_names():
        if name in extra:
            merged[name] = extra[name]
    return merged
