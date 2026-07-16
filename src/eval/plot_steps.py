#!/usr/bin/env python3
"""Training curves: log10(FLOPs) vs validation loss and log10(steps) vs validation loss.

BDT models (``BDT*``) appear as **horizontal dashed lines** (with seed error bands)
on classification and regression FLOPs and steps panels at mean final validation loss.
The cut-based reco baseline (``Reco-baseline`` / Cut baseline) is never plotted.

Pickles default to ``<repo>/<--out-dir>/``; PDFs default to
``<repo>/<--plots-dir>/``.

Writes **four** combined 1×2 panels (classification | regression, shared legend)
plus single-task PDFs (no title; y-axis ``Validation loss (classification|regression)``):

- ``steps_combined/`` — base models only (no BERT, no HyperScale)
- ``steps_combined_BERT/`` — base + BERT
- ``steps_combined_HYPERSCALE/`` — base + HyperScale
- ``steps_combined_small/`` — small/medium architecture comparison (HyperScale small+medium, OLS small, BERT)

Use ``--separate-panels`` to also emit per-task PDFs under
``{classification,regression}/steps[<suffix>]/``.

Config / plots-only
-------------------
Pass ``--config JSON`` to plot a custom model subset (colors, optional display
names, optional step cutoff) as a single combined variant written to
``steps_combined/``.  The four standard variants are not produced in this mode.

Pass ``--plots-only`` to read the loss histories from the plots cache
(``plots/tmp_results/steps.pkl``) instead of the eval pickles. On a normal run
the cache is written automatically so subsequent ``--plots-only`` runs are fast.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.eval._legend import (
    DEFAULT_LEGEND_FS as _LEGEND_FS,
    layout_legend_with_column_stacks as _layout_legend_with_column_stacks,
    order_legend_handles_labels as _order_legend_handles_labels,
    shared_figure_legend as _shared_figure_legend,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._constants import (
    CANONICAL_CLASSIFICATION_PICKLE,
    CANONICAL_REGRESSION_PICKLE,
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    REGRESSION_PICKLE_STEM,
    STEPS_SMALL_MODEL_COLORS,
    STEPS_SMALL_MODELS,
    is_base_steps_model,
    is_bdt_model,
    is_horizontal_reference_model,
    is_bert_model,
    is_hyperscale_model,
    is_steps_plot_excluded_model,
    plot_model_label,
    repo_output_path,
)
from src.eval._plot_config import CurveEndConfig, ModelCurveCuts, PlotConfig
from src.eval._steps_cache import build_steps_cache_from_pickles, save_steps_cache

_REGRESSION_STEPS_ZOOM_LOG_XMIN = 4.0
_REGRESSION_STEPS_ZOOM_LOG_XMAX = 5.0
# Auto y-limits: extra headroom below curves (asymmetric padding).
_VAL_LOSS_YLIM_PAD_BOTTOM = 0.12
_VAL_LOSS_YLIM_PAD_TOP = 0.05
_LABEL_FS = 12
_TICK_FS = 11
_TITLE_FS = 13

# Single-task steps_combined panels: square, 10% shorter than the legacy (8, 5) height.
_SINGLE_PANEL_HEIGHT = 5 * 0.9
_SINGLE_PANEL_FIGSIZE = (_SINGLE_PANEL_HEIGHT, _SINGLE_PANEL_HEIGHT)

_STEPS_CACHE_NAME = "steps.pkl"
# Baseline models shown last in combined-panel legends (config mode).
_LEGEND_TAIL_MODELS = ("Transformer-xsmall", "Transformer-small", "MLP")


@dataclass(frozen=True)
class StepsPlotVariant:
    """One output bundle: subdirectory suffix and model filter."""

    suffix: str
    model_filter: Callable[[str], bool]
    colors_override: dict[str, str] | None = None


STEPS_PLOT_VARIANTS: tuple[StepsPlotVariant, ...] = (
    StepsPlotVariant("", is_base_steps_model),
    StepsPlotVariant("_BERT", lambda m: not is_hyperscale_model(m)),
    StepsPlotVariant("_HYPERSCALE", lambda m: not is_bert_model(m)),
    StepsPlotVariant(
        "_small",
        lambda m: m in STEPS_SMALL_MODELS,
        STEPS_SMALL_MODEL_COLORS,
    ),
)


def _pickle_paths(out_dir: Path, flag: str) -> tuple[Path, Path]:
    return (
        out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl",
        out_dir / f"{REGRESSION_PICKLE_STEM}_{flag}.pkl",
    )


def _subset_for_plot(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    model_filter: Callable[[str], bool],
    colors_override: dict[str, str] | None = None,
) -> tuple[dict, dict, dict[str, str]]:
    """Keep models present in both *loss_histories* and *flops_per_step*."""
    keys = sorted(
        k
        for k in flops_per_step
        if (
            model_filter(k)
            and not is_steps_plot_excluded_model(k)
            and k in loss_histories
            and loss_histories[k]
        )
    )
    lh = {k: loss_histories[k] for k in keys}
    flops = {k: flops_per_step[k] for k in keys}
    clrs = {k: colors.get(k, "tab:gray") for k in keys}
    if colors_override:
        for k in keys:
            if k in colors_override:
                clrs[k] = colors_override[k]
    return lh, flops, clrs


def _config_horizontal_ref_names(cfg: PlotConfig, task: str) -> set[str]:
    """Horizontal-reference models for classification or regression val-loss panels."""
    return cfg.horizontal_ref_names(task)


def _merge_config_horizontal_refs(
    lh: dict,
    flops: dict,
    colors: dict[str, str],
    *,
    ref_names: set[str],
    source_lh: dict,
    source_flops: dict,
    source_colors: dict[str, str],
) -> tuple[dict, dict, dict[str, str]]:
    """Ensure config horizontal references are present for val-loss panels."""
    if not ref_names:
        return lh, flops, colors
    out_lh = dict(lh)
    out_flops = dict(flops)
    out_colors = dict(colors)
    for name in sorted(ref_names):
        if is_steps_plot_excluded_model(name):
            continue
        if name in out_lh and out_lh[name]:
            continue
        if name not in source_lh or not source_lh.get(name):
            continue
        if name not in source_flops:
            continue
        out_lh[name] = source_lh[name]
        out_flops[name] = source_flops[name]
        out_colors[name] = source_colors.get(name, out_colors.get(name, "tab:gray"))
    return out_lh, out_flops, out_colors


def _runs_per_model(
    loss_histories: dict, flops_per_step: dict
) -> OrderedDict[str, list]:
    runs: OrderedDict[str, list] = OrderedDict()
    for model in sorted(flops_per_step):
        if model not in loss_histories or is_steps_plot_excluded_model(model):
            continue
        runs[model] = loss_histories[model]
    return runs


def _merged_colors(
    colors_a: dict[str, str], colors_b: dict[str, str]
) -> dict[str, str]:
    out = dict(colors_a)
    for k, v in colors_b.items():
        out.setdefault(k, v)
    return out


def _plot_flops_vs_loss(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    panel_title: str,
    ylim: tuple[float, float] | None,
    out_pdf: Path,
    *,
    legend_outside: bool = False,
    legend_label_order: list[str] | None = None,
    label_fn: Callable[[str], str] | None = None,
    ylabel: str = "Validation loss",
    figsize: tuple[float, float] = (8, 5),
    flops_xmin: float | None = None,
    step_cutoff: int | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts: ModelCurveCuts | None = None,
    horizontal_ref_models: set[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _draw_flops_curves(
        ax,
        loss_histories,
        flops_per_step,
        colors,
        ylim,
        panel_title,
        label_fn=label_fn,
        ylabel=ylabel,
        flops_xmin=flops_xmin,
        step_cutoff=step_cutoff,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts,
        horizontal_ref_models=horizontal_ref_models,
    )
    _apply_ax_legend(
        ax,
        legend_outside=legend_outside,
        legend_label_order=legend_label_order,
    )
    ax.grid(True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _mean_final_loss_per_seed(series_list: list) -> tuple[float, float]:
    """Mean and std of the final validation loss across seed runs."""
    finals = [float(lo[-1]) for st, lo in series_list if len(lo) > 0]
    if not finals:
        return float("nan"), 0.0
    mean_loss = float(np.mean(finals))
    sigma_loss = float(np.std(finals)) if len(finals) > 1 else 0.0
    return mean_loss, sigma_loss


def _align_mean_val_loss(
    series_list: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed histories onto a common step grid; return mean ± std loss."""
    all_steps, all_losses = [], []
    for st, lo in series_list:
        if len(st) > 0 and len(lo) > 0:
            all_steps.append(st)
            all_losses.append(lo)
    if not all_steps:
        return None
    steps_grid = np.unique(np.concatenate(all_steps)).astype(float)
    if len(steps_grid) == 0:
        return None
    losses_aligned = np.array(
        [np.interp(steps_grid, st, lo) for st, lo in zip(all_steps, all_losses)]
    )
    mean_loss = np.mean(losses_aligned, axis=0)
    sigma_loss = (
        np.std(losses_aligned, axis=0)
        if losses_aligned.shape[0] > 1
        else np.zeros_like(mean_loss)
    )
    return steps_grid, mean_loss, sigma_loss


def _step_at_min_val_loss(steps_grid: np.ndarray, mean_loss: np.ndarray) -> float:
    """Training step where the mean validation loss reaches its minimum."""
    return float(steps_grid[int(np.argmin(mean_loss))])


def _resolve_model_step_cutoffs(
    curve_runs: OrderedDict[str, list],
    *,
    model_curve_cuts: ModelCurveCuts | None = None,
    curve_end: CurveEndConfig | None = None,
    global_step_cutoff: int | None = None,
) -> dict[str, float]:
    """Max training step (inclusive) to plot per model."""
    if not curve_runs:
        return {}

    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model, series_list in curve_runs.items():
        packed = _align_mean_val_loss(series_list)
        if packed is None:
            continue
        steps_grid, mean_loss, _sigma = packed
        aligned[model] = (steps_grid, mean_loss)

    cuts = model_curve_cuts or ModelCurveCuts()
    if curve_end is not None and curve_end.policy == "min_val_loss":
        cuts = ModelCurveCuts(
            step_cut=dict(cuts.step_cut),
            flop_cut=dict(cuts.flop_cut),
            step_cut_policy=dict(cuts.step_cut_policy),
            step_cut_match=dict(cuts.step_cut_match),
            legacy_global_min_val_loss=True,
            legacy_match={**cuts.legacy_match, **dict(curve_end.match)},
        )

    cutoffs: dict[str, float] = {}
    for model, (steps_grid, mean_loss) in aligned.items():
        policy = cuts.step_cut_policy.get(model)
        use_min_loss = policy == "min_val_loss" or (
            policy is None
            and model not in cuts.step_cut
            and model not in cuts.step_cut_match
            and cuts.legacy_global_min_val_loss
        )
        if use_min_loss:
            cutoffs[model] = _step_at_min_val_loss(steps_grid, mean_loss)
        else:
            cutoffs[model] = float(np.max(steps_grid))

    for model, explicit in cuts.step_cut.items():
        if model in cutoffs:
            cutoffs[model] = min(cutoffs[model], float(explicit))

    for model, ref in cuts.step_cut_match.items():
        if model in cutoffs and ref in cutoffs:
            cutoffs[model] = cutoffs[ref]

    for model, ref in cuts.legacy_match.items():
        if model in cutoffs and ref in cutoffs:
            cutoffs[model] = cutoffs[ref]

    if global_step_cutoff is not None:
        for model in cutoffs:
            cutoffs[model] = min(cutoffs[model], float(global_step_cutoff))

    return cutoffs


def _apply_step_cutoff_mask(
    steps_grid: np.ndarray,
    *arrays: np.ndarray,
    max_step: float | None,
) -> tuple[np.ndarray, ...]:
    """Keep points with ``steps_grid <= max_step``."""
    if max_step is None:
        return (steps_grid, *arrays)
    mask = steps_grid <= max_step
    if not np.any(mask):
        return (steps_grid[:0], *(a[:0] for a in arrays))
    return (steps_grid[mask], *(a[mask] for a in arrays))


def _apply_flop_xmin_mask(
    log10_cum_flops: np.ndarray,
    *arrays: np.ndarray,
    min_log10_flop: float | None,
) -> tuple[np.ndarray, ...]:
    """Keep points with ``log10(cum_flops + 1) >= min_log10_flop``."""
    if min_log10_flop is None:
        return (log10_cum_flops, *arrays)
    mask = log10_cum_flops >= min_log10_flop
    if not np.any(mask):
        return (log10_cum_flops[:0], *(a[:0] for a in arrays))
    return (log10_cum_flops[mask], *(a[mask] for a in arrays))


def _apply_flop_cut_mask(
    log10_cum_flops: np.ndarray,
    *arrays: np.ndarray,
    max_log10_flop: float | None,
) -> tuple[np.ndarray, ...]:
    """Keep points with ``log10(cum_flops + 1) <= max_log10_flop``."""
    if max_log10_flop is None:
        return (log10_cum_flops, *arrays)
    mask = log10_cum_flops <= max_log10_flop
    if not np.any(mask):
        return (log10_cum_flops[:0], *(a[:0] for a in arrays))
    return (log10_cum_flops[mask], *(a[mask] for a in arrays))


def _apply_flop_cut_step_mask(
    steps_grid: np.ndarray,
    *arrays: np.ndarray,
    flops_per_step: float,
    max_log10_flop: float | None,
) -> tuple[np.ndarray, ...]:
    """Keep step points within the same cumulative-FLOP cap used on FLOPs panels."""
    if max_log10_flop is None:
        return (steps_grid, *arrays)
    log10_cum_flops = np.log10(steps_grid * flops_per_step + 1.0)
    mask = log10_cum_flops <= max_log10_flop
    if not np.any(mask):
        return (steps_grid[:0], *(a[:0] for a in arrays))
    return (steps_grid[mask], *(a[mask] for a in arrays))


def _apply_flop_xmin_step_mask(
    steps_grid: np.ndarray,
    *arrays: np.ndarray,
    flops_per_step: float,
    min_log10_flop: float | None,
) -> tuple[np.ndarray, ...]:
    """Keep step points within the same cumulative-FLOP floor used on FLOPs panels."""
    if min_log10_flop is None:
        return (steps_grid, *arrays)
    log10_cum_flops = np.log10(steps_grid * flops_per_step + 1.0)
    mask = log10_cum_flops >= min_log10_flop
    if not np.any(mask):
        return (steps_grid[:0], *(a[:0] for a in arrays))
    return (steps_grid[mask], *(a[mask] for a in arrays))


def _apply_log_step_cut_mask(
    steps_grid: np.ndarray,
    *arrays: np.ndarray,
    max_log10_step: float | None,
) -> tuple[np.ndarray, ...]:
    """Keep step points with ``log10(step + 1) <= max_log10_step``."""
    if max_log10_step is None:
        return (steps_grid, *arrays)
    log10_steps = np.log10(steps_grid + 1.0)
    mask = log10_steps <= max_log10_step
    if not np.any(mask):
        return (steps_grid[:0], *(a[:0] for a in arrays))
    return (steps_grid[mask], *(a[mask] for a in arrays))


def _split_bdt_runs(
    loss_histories: dict,
    flops_per_step: dict,
    *,
    horizontal_ref_models: set[str] | None = None,
) -> tuple[OrderedDict[str, list], OrderedDict[str, list]]:
    runs = _runs_per_model(loss_histories, flops_per_step)
    if horizontal_ref_models is None:

        def _is_ref(name: str) -> bool:
            return is_horizontal_reference_model(name)

    else:
        ref_set = horizontal_ref_models

        def _is_ref(name: str) -> bool:
            return name in ref_set

    curve_runs = OrderedDict((m, s) for m, s in runs.items() if not _is_ref(m))
    ref_runs = OrderedDict((m, s) for m, s in runs.items() if _is_ref(m))
    return curve_runs, ref_runs


def _ylim_including_bdt(
    ylim: tuple[float, float] | None,
    ref_runs: OrderedDict[str, list],
) -> tuple[float, float] | None:
    """Expand a fixed y-limit so horizontal reference baselines are not clipped."""
    if ylim is None or not ref_runs:
        return ylim
    ymin, ymax = ylim
    for series_list in ref_runs.values():
        mean_loss, sigma_loss = _mean_final_loss_per_seed(series_list)
        if not np.isfinite(mean_loss):
            continue
        pad = sigma_loss if sigma_loss > 0 else 0.01 * abs(mean_loss)
        ymin = min(ymin, mean_loss - pad)
        ymax = max(ymax, mean_loss + pad)
    return ymin, ymax


def _loss_values_in_log_steps_window(
    steps: np.ndarray,
    losses: np.ndarray,
    *,
    log_steps_xmin: float | None,
    log_steps_xmax: float | None = None,
) -> np.ndarray:
    """Return loss values whose step maps to the requested log-step window."""
    if len(losses) == 0:
        return losses
    if log_steps_xmin is None and log_steps_xmax is None:
        return losses
    log_steps = np.log10(steps.astype(float) + 1.0)
    mask = np.ones(len(log_steps), dtype=bool)
    if log_steps_xmin is not None:
        mask &= log_steps >= log_steps_xmin
    if log_steps_xmax is not None:
        mask &= log_steps <= log_steps_xmax
    if not np.any(mask):
        return losses
    return losses[mask]


def _mask_log_steps_series(
    log_steps_plot: np.ndarray,
    *arrays: np.ndarray,
    log_steps_xmin: float | None = None,
    log_steps_xmax: float | None = None,
) -> tuple[np.ndarray, ...]:
    """Keep aligned step/loss arrays inside ``[log_steps_xmin, log_steps_xmax]``."""
    if log_steps_xmin is None and log_steps_xmax is None:
        return (log_steps_plot, *arrays)
    mask = np.ones(len(log_steps_plot), dtype=bool)
    if log_steps_xmin is not None:
        mask &= log_steps_plot >= log_steps_xmin
    if log_steps_xmax is not None:
        mask &= log_steps_plot <= log_steps_xmax
    if not np.any(mask):
        empty = log_steps_plot[:0]
        return (empty, *(a[:0] for a in arrays))
    return (log_steps_plot[mask], *(a[mask] for a in arrays))


def _validation_loss_y_limits(
    curve_runs: OrderedDict[str, list],
    ref_runs: OrderedDict[str, list],
    *,
    ylim: tuple[float, float] | None = None,
    log_steps_xmin: float | None = None,
    log_steps_xmax: float | None = None,
    model_step_cutoffs: dict[str, float] | None = None,
    flop_cuts: dict[str, float] | None = None,
    flop_xmins: dict[str, float] | None = None,
    global_flops_xmin: float | None = None,
    log_step_cuts: dict[str, float] | None = None,
    flops_per_step: dict[str, float] | None = None,
    pad_fraction_bottom: float = _VAL_LOSS_YLIM_PAD_BOTTOM,
    pad_fraction_top: float = _VAL_LOSS_YLIM_PAD_TOP,
) -> tuple[float, float] | None:
    """Y limits for val-loss panels: fixed *ylim* or auto from data (+ reference lines)."""
    if ylim is not None:
        return _ylim_including_bdt(ylim, ref_runs)

    ymin, ymax = np.inf, -np.inf
    for model, series_list in curve_runs.items():
        max_step = (
            model_step_cutoffs.get(model) if model_step_cutoffs is not None else None
        )
        for st, lo in series_list:
            if len(lo) == 0:
                continue
            st_arr = np.asarray(st, dtype=float)
            lo_arr = np.asarray(lo, dtype=float)
            if max_step is not None:
                step_mask = st_arr <= max_step
                st_arr = st_arr[step_mask]
                lo_arr = lo_arr[step_mask]
            if (
                flop_cuts
                and flops_per_step
                and model in flops_per_step
                and flop_cuts.get(model) is not None
            ):
                log10_cum_flops = np.log10(st_arr * flops_per_step[model] + 1.0)
                flop_mask = log10_cum_flops <= flop_cuts[model]
                st_arr = st_arr[flop_mask]
                lo_arr = lo_arr[flop_mask]
            if (
                flop_xmins
                and flops_per_step
                and model in flops_per_step
                and flop_xmins.get(model) is not None
            ):
                log10_cum_flops = np.log10(st_arr * flops_per_step[model] + 1.0)
                flop_mask = log10_cum_flops >= flop_xmins[model]
                st_arr = st_arr[flop_mask]
                lo_arr = lo_arr[flop_mask]
            if (
                global_flops_xmin is not None
                and flops_per_step
                and model in flops_per_step
            ):
                log10_cum_flops = np.log10(st_arr * flops_per_step[model] + 1.0)
                flop_mask = log10_cum_flops >= global_flops_xmin
                st_arr = st_arr[flop_mask]
                lo_arr = lo_arr[flop_mask]
            if log_step_cuts and log_step_cuts.get(model) is not None:
                log10_steps = np.log10(st_arr + 1.0)
                step_mask = log10_steps <= log_step_cuts[model]
                st_arr = st_arr[step_mask]
                lo_arr = lo_arr[step_mask]
            if len(lo_arr) == 0:
                continue
            lo_win = _loss_values_in_log_steps_window(
                st_arr,
                lo_arr,
                log_steps_xmin=log_steps_xmin,
                log_steps_xmax=log_steps_xmax,
            )
            if len(lo_win) == 0:
                continue
            ymin = min(ymin, float(np.min(lo_win)))
            ymax = max(ymax, float(np.max(lo_win)))
    for series_list in ref_runs.values():
        mean_loss, sigma_loss = _mean_final_loss_per_seed(series_list)
        if not np.isfinite(mean_loss):
            continue
        pad = sigma_loss if sigma_loss > 0 else 0.01 * abs(mean_loss)
        ymin = min(ymin, mean_loss - pad)
        ymax = max(ymax, mean_loss + pad)

    if not np.isfinite(ymin):
        return None

    span = ymax - ymin
    if span <= 0:
        span = max(0.01, 0.01 * abs(ymax))
    pad_bottom = pad_fraction_bottom * span
    pad_top = pad_fraction_top * span
    return ymin - pad_bottom, ymax + pad_top


def _autoscale_x_from_plotted_data(
    ax: plt.Axes,
    *,
    flops_xmin: float | None = None,
    log_steps_xmin: float | None = None,
) -> None:
    """Recompute x limits from plotted data; keep the current y range."""
    ymin, ymax = ax.get_ylim()
    ax.relim()
    ax.autoscale_view(scalex=True, scaley=False)
    if flops_xmin is not None:
        _, xmax = ax.get_xlim()
        ax.set_xlim(flops_xmin, xmax)
    if log_steps_xmin is not None:
        _, xmax = ax.get_xlim()
        ax.set_xlim(log_steps_xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _global_flops_xmin_as_log_steps(
    global_flops_xmin: float,
    curve_runs: OrderedDict[str, list],
    flops_per_step: dict[str, float],
) -> float | None:
    """Map a global ``log10(FLOPs)`` floor to the tightest matching ``log10(steps)``."""
    xmin: float | None = None
    for model in curve_runs:
        fps = flops_per_step.get(model)
        if fps is None:
            continue
        step_at = (10.0**global_flops_xmin - 1.0) / fps
        if step_at < 0:
            continue
        log_step = float(np.log10(step_at + 1.0))
        xmin = log_step if xmin is None else min(xmin, log_step)
    return xmin


def _draw_bdt_baseline_hlines(
    ax: plt.Axes,
    ref_runs: OrderedDict[str, list],
    colors: dict[str, str],
    label_fn: Callable[[str], str],
) -> None:
    """Horizontal dashed reference lines for non-training baselines (BDT, cut baseline)."""
    if not ref_runs:
        return
    xmin, xmax = ax.get_xlim()
    xmid = 0.5 * (xmin + xmax)
    for model, series_list in ref_runs.items():
        color = colors.get(model, "tab:gray")
        mean_loss, sigma_loss = _mean_final_loss_per_seed(series_list)
        if not np.isfinite(mean_loss):
            continue
        label = label_fn(model)
        if sigma_loss > 0:
            ax.fill_between(
                [xmin, xmax],
                mean_loss - sigma_loss,
                mean_loss + sigma_loss,
                alpha=0.25,
                color=color,
                zorder=1,
            )
            ax.errorbar(
                [xmid],
                [mean_loss],
                yerr=[[sigma_loss], [sigma_loss]],
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                zorder=3,
            )
        ax.plot(
            [xmin, xmax],
            [mean_loss, mean_loss],
            linestyle="--",
            color=color,
            linewidth=1.8,
            label=label,
            zorder=4,
        )


def _draw_flops_curves(
    ax: plt.Axes,
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    ylim: tuple[float, float] | None,
    panel_title: str,
    label_fn: Callable[[str], str] | None = None,
    *,
    ylabel: str = "Validation loss",
    flops_xmin: float | None = None,
    step_cutoff: int | None = None,
    model_curve_cuts: ModelCurveCuts | None = None,
    curve_end: CurveEndConfig | None = None,
    horizontal_ref_models: set[str] | None = None,
) -> None:
    if label_fn is None:
        label_fn = plot_model_label
    curve_runs, ref_runs = _split_bdt_runs(
        loss_histories,
        flops_per_step,
        horizontal_ref_models=horizontal_ref_models,
    )
    model_step_cutoffs = _resolve_model_step_cutoffs(
        curve_runs,
        model_curve_cuts=model_curve_cuts,
        curve_end=curve_end,
        global_step_cutoff=step_cutoff,
    )
    flop_cuts = model_curve_cuts.flop_cut if model_curve_cuts is not None else {}
    flop_xmins = model_curve_cuts.flop_xmin if model_curve_cuts is not None else {}

    for model, series_list in curve_runs.items():
        color = colors.get(model, "tab:gray")
        flops = flops_per_step[model]
        packed = _align_mean_val_loss(series_list)
        if packed is None:
            continue
        steps_grid, mean_loss, sigma_loss = packed
        steps_grid, mean_loss, sigma_loss = _apply_step_cutoff_mask(
            steps_grid,
            mean_loss,
            sigma_loss,
            max_step=model_step_cutoffs.get(model),
        )
        if len(steps_grid) == 0:
            continue
        cum_flops = steps_grid * flops
        x = np.log10(cum_flops + 1)
        x, mean_loss, sigma_loss = _apply_flop_xmin_mask(
            x,
            mean_loss,
            sigma_loss,
            min_log10_flop=flop_xmins.get(model),
        )
        x, mean_loss, sigma_loss = _apply_flop_cut_mask(
            x,
            mean_loss,
            sigma_loss,
            max_log10_flop=flop_cuts.get(model),
        )
        if len(x) == 0:
            continue
        ax.plot(x, mean_loss, color=color, label=label_fn(model))
        if np.any(sigma_loss > 0):
            ax.fill_between(
                x,
                mean_loss - sigma_loss,
                mean_loss + sigma_loss,
                alpha=0.25,
                color=color,
            )

    if panel_title:
        ax.set_title(panel_title, fontsize=_TITLE_FS, pad=10)
    ax.set_xlabel(r"$log_{10}$(Training FLOPs)", fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    effective_ylim = _validation_loss_y_limits(
        curve_runs,
        ref_runs,
        ylim=ylim,
        model_step_cutoffs=model_step_cutoffs,
        flop_cuts=flop_cuts,
        flop_xmins=flop_xmins,
        global_flops_xmin=flops_xmin,
        flops_per_step=flops_per_step,
    )
    if effective_ylim:
        ax.set_ylim(effective_ylim)

    _draw_bdt_baseline_hlines(ax, ref_runs, colors, label_fn)
    _autoscale_x_from_plotted_data(ax, flops_xmin=flops_xmin)


def _plot_steps_vs_loss(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    panel_title: str,
    ylim: tuple[float, float] | None,
    step_cutoff: int | None,
    out_pdf: Path,
    *,
    legend_outside: bool = False,
    legend_label_order: list[str] | None = None,
    label_fn: Callable[[str], str] | None = None,
    ylabel: str = "Validation loss",
    figsize: tuple[float, float] = (8, 5),
    global_flops_xmin: float | None = None,
    log_steps_xmin: float | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts: ModelCurveCuts | None = None,
    horizontal_ref_models: set[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _draw_steps_curves(
        ax,
        loss_histories,
        flops_per_step,
        colors,
        ylim,
        panel_title,
        step_cutoff,
        label_fn=label_fn,
        ylabel=ylabel,
        global_flops_xmin=global_flops_xmin,
        log_steps_xmin=log_steps_xmin,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts,
        horizontal_ref_models=horizontal_ref_models,
    )
    _apply_ax_legend(
        ax,
        legend_outside=legend_outside,
        legend_label_order=legend_label_order,
    )
    ax.grid(True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _draw_steps_curves(
    ax: plt.Axes,
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    ylim: tuple[float, float] | None,
    panel_title: str,
    step_cutoff: int | None = None,
    *,
    label_fn: Callable[[str], str] | None = None,
    ylabel: str = "Validation loss",
    log_steps_xmin: float | None = None,
    log_steps_xmax: float | None = None,
    global_flops_xmin: float | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts: ModelCurveCuts | None = None,
    horizontal_ref_models: set[str] | None = None,
) -> None:
    if label_fn is None:
        label_fn = plot_model_label
    curve_runs, ref_runs = _split_bdt_runs(
        loss_histories,
        flops_per_step,
        horizontal_ref_models=horizontal_ref_models,
    )
    model_step_cutoffs = _resolve_model_step_cutoffs(
        curve_runs,
        model_curve_cuts=model_curve_cuts,
        curve_end=curve_end,
        global_step_cutoff=step_cutoff,
    )
    flop_cuts = model_curve_cuts.flop_cut if model_curve_cuts is not None else {}
    log_step_cuts = (
        model_curve_cuts.log_step_cut if model_curve_cuts is not None else {}
    )

    for model, series_list in curve_runs.items():
        color = colors.get(model, "tab:gray")
        fps = flops_per_step.get(model)
        packed = _align_mean_val_loss(series_list)
        if packed is None:
            continue
        steps_grid, mean_loss, sigma_loss = packed
        steps_grid, mean_loss, sigma_loss = _apply_step_cutoff_mask(
            steps_grid,
            mean_loss,
            sigma_loss,
            max_step=model_step_cutoffs.get(model),
        )
        # flop_xmin / flops_xmin apply on FLOPs panels only; steps use log_steps_xmin.
        if fps is not None:
            if flop_cuts.get(model) is not None:
                steps_grid, mean_loss, sigma_loss = _apply_flop_cut_step_mask(
                    steps_grid,
                    mean_loss,
                    sigma_loss,
                    flops_per_step=fps,
                    max_log10_flop=flop_cuts.get(model),
                )
            elif log_step_cuts.get(model) is not None:
                steps_grid, mean_loss, sigma_loss = _apply_log_step_cut_mask(
                    steps_grid,
                    mean_loss,
                    sigma_loss,
                    max_log10_step=log_step_cuts.get(model),
                )
        elif log_step_cuts.get(model) is not None:
            steps_grid, mean_loss, sigma_loss = _apply_log_step_cut_mask(
                steps_grid,
                mean_loss,
                sigma_loss,
                max_log10_step=log_step_cuts.get(model),
            )
        if len(steps_grid) == 0:
            continue
        log_steps_plot = np.log10(steps_grid + 1)
        log_steps_plot, mean_loss, sigma_loss = _mask_log_steps_series(
            log_steps_plot,
            mean_loss,
            sigma_loss,
            log_steps_xmin=log_steps_xmin,
            log_steps_xmax=log_steps_xmax,
        )
        if len(log_steps_plot) == 0:
            continue
        ax.plot(log_steps_plot, mean_loss, color=color, label=label_fn(model))
        if np.any(sigma_loss > 0):
            ax.fill_between(
                log_steps_plot,
                mean_loss - sigma_loss,
                mean_loss + sigma_loss,
                alpha=0.25,
                color=color,
            )

    if panel_title:
        ax.set_title(panel_title, fontsize=_TITLE_FS, pad=10)
    ax.set_xlabel(r"$log_{10}$(Training steps)", fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    effective_ylim = _validation_loss_y_limits(
        curve_runs,
        ref_runs,
        ylim=ylim,
        log_steps_xmin=log_steps_xmin,
        log_steps_xmax=log_steps_xmax,
        model_step_cutoffs=model_step_cutoffs,
        flop_cuts=flop_cuts,
        flop_xmins=None,
        global_flops_xmin=None,
        log_step_cuts=log_step_cuts,
        flops_per_step=flops_per_step,
    )
    if effective_ylim:
        ax.set_ylim(effective_ylim)
    _draw_bdt_baseline_hlines(ax, ref_runs, colors, label_fn)
    steps_xmin = log_steps_xmin
    if steps_xmin is None and global_flops_xmin is not None:
        steps_xmin = _global_flops_xmin_as_log_steps(
            global_flops_xmin, curve_runs, flops_per_step
        )
    _autoscale_x_from_plotted_data(ax, log_steps_xmin=steps_xmin)


def _strip_inset_axis_decorations(ax: plt.Axes) -> None:
    """Remove axis titles and tick labels from a zoom inset."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        axis="both",
        which="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )


def _parent_axes_size_inches(ax: plt.Axes) -> tuple[float, float]:
    """Return parent axes width and height in inches (requires draw)."""
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    dpi = ax.figure.dpi
    return bbox.width / dpi, bbox.height / dpi


def _inset_axes_upper_right(
    ax: plt.Axes,
    *,
    side_inches: float = 1.05,
    width: float | None = None,
    height: float | None = None,
    right: float = 0.98,
    top: float = 0.96,
    min_left: float = 0.05,
    min_bottom: float = 0.08,
    square: bool = True,
) -> plt.Axes:
    """Place a zoom inset in the upper-right of a steps panel."""
    ax_w_in, ax_h_in = _parent_axes_size_inches(ax)
    if square:
        max_side_in = min(
            (right - min_left) * ax_w_in,
            (top - min_bottom) * ax_h_in,
        )
        side = min(side_inches, max_side_in)
        width = side / ax_w_in
        height = side / ax_h_in
    else:
        if width is None:
            width = 0.708
        if height is None:
            height = width
    inset_left = max(min_left, right - width)
    inset_bottom = max(min_bottom, top - height)
    axins = ax.inset_axes([inset_left, inset_bottom, width, height])
    if square:
        axins.set_box_aspect(1)
    return axins


def _style_inset_top_corner_connectors(indicator) -> None:
    """Hide default inset connectors; use :func:`_draw_inset_top_connectors` instead."""
    connectors = getattr(indicator, "connectors", None)
    if not connectors:
        return
    for conn in connectors:
        if conn is not None:
            conn.set_visible(False)


def _draw_inset_top_connectors(
    ax: plt.Axes,
    axins: plt.Axes,
    *,
    log_xmin: float,
    log_xmax: float,
) -> None:
    """Draw dashed lines from the inset top corners to the zoom region below."""
    from matplotlib.patches import ConnectionPatch

    _, ymax = ax.get_ylim()
    fig = ax.figure
    for x_data, inset_x in ((log_xmin, 0.0), (log_xmax, 1.0)):
        con = ConnectionPatch(
            xyA=(x_data, ymax),
            coordsA=ax.transData,
            xyB=(inset_x, 1.0),
            coordsB=axins.transAxes,
            linestyle="--",
            color="0.45",
            linewidth=0.9,
        )
        fig.add_artist(con)


def _add_steps_zoom_inset(
    ax: plt.Axes,
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    ylim: tuple[float, float] | None,
    step_cutoff: int | None,
    *,
    log_steps_xmin: float,
    log_steps_xmax: float,
    label_fn: Callable[[str], str] | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts: ModelCurveCuts | None = None,
    horizontal_ref_models: set[str] | None = None,
) -> None:
    """Add an upper-right zoom box for ``log10(steps)`` in ``[xmin, xmax]``."""
    axins = _inset_axes_upper_right(ax)
    _draw_steps_curves(
        axins,
        loss_histories,
        flops_per_step,
        colors,
        ylim,
        "",
        step_cutoff,
        label_fn=label_fn,
        ylabel="",
        log_steps_xmin=log_steps_xmin,
        log_steps_xmax=log_steps_xmax,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts,
        horizontal_ref_models=horizontal_ref_models,
    )
    _strip_inset_axis_decorations(axins)
    axins.set_xlim(log_steps_xmin, log_steps_xmax)
    axins.set_box_aspect(1)
    indicator = ax.indicate_inset_zoom(
        axins,
        edgecolor="0.45",
        linestyle="--",
        linewidth=0.9,
    )
    _style_inset_top_corner_connectors(indicator)
    _draw_inset_top_connectors(
        ax,
        axins,
        log_xmin=log_steps_xmin,
        log_xmax=log_steps_xmax,
    )


def _inset_axes_below_legend(
    ax: plt.Axes,
    *,
    width: float = 0.48,
    height: float = 0.384,
    gap: float = 0.02,
) -> plt.Axes:
    """Place a zoom inset directly under the axes legend (upper-right stack)."""
    legend = ax.get_legend()
    if legend is None:
        return ax.inset_axes([0.50, 0.48, width, height])

    fig = ax.figure
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(ax.transAxes.inverted())
    inset_left = max(0.05, legend_bbox.x1 - width)
    inset_bottom = max(0.06, legend_bbox.y0 - gap - height)
    return ax.inset_axes([inset_left, inset_bottom, width, height])


def _plot_steps_vs_loss_with_zoom(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    panel_title: str,
    ylim: tuple[float, float] | None,
    step_cutoff: int | None,
    out_pdf: Path,
    *,
    log_steps_xmin: float = _REGRESSION_STEPS_ZOOM_LOG_XMIN,
    log_steps_xmax: float = _REGRESSION_STEPS_ZOOM_LOG_XMAX,
    legend_outside: bool = False,
    legend_label_order: list[str] | None = None,
    label_fn: Callable[[str], str] | None = None,
    ylabel: str = "Validation loss",
    figsize: tuple[float, float] = (8, 5),
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts: ModelCurveCuts | None = None,
    horizontal_ref_models: set[str] | None = None,
) -> None:
    """Steps vs val-loss with an inset zoomed to a log-step window."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _draw_steps_curves(
        ax,
        loss_histories,
        flops_per_step,
        colors,
        ylim,
        panel_title,
        step_cutoff,
        label_fn=label_fn,
        ylabel=ylabel,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts,
        horizontal_ref_models=horizontal_ref_models,
    )
    _apply_ax_legend(
        ax,
        legend_outside=legend_outside,
        legend_label_order=legend_label_order,
    )
    _add_steps_zoom_inset(
        ax,
        loss_histories,
        flops_per_step,
        colors,
        ylim,
        step_cutoff,
        log_steps_xmin=log_steps_xmin,
        log_steps_xmax=log_steps_xmax,
        label_fn=label_fn,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts,
        horizontal_ref_models=horizontal_ref_models,
    )
    ax.grid(True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _apply_ax_legend(
    ax: plt.Axes,
    *,
    legend_outside: bool,
    legend_label_order: list[str] | None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = _order_legend_handles_labels(handles, labels, legend_label_order)
    legend_kw = dict(fontsize=_LEGEND_FS, frameon=True)
    if legend_outside:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            **legend_kw,
        )
    else:
        ax.legend(handles, labels, loc="upper right", **legend_kw)


def _legend_labels_from_cfg(
    cfg: PlotConfig,
    tail: tuple[str, ...] = (
        "BDT",
        "Transformer-xsmall",
        "Transformer-small",
        "MLP",
    ),
) -> list[str]:
    """Display labels in config order, with *tail* model names moved to the end."""
    return cfg.legend_labels(tail)


def _plot_combined_flops_row(
    lh_c: dict,
    flops_c: dict,
    colors_c: dict[str, str],
    ylim_c: tuple[float, float] | None,
    lh_r: dict,
    flops_r: dict,
    colors_r: dict[str, str],
    ylim_r: tuple[float, float] | None,
    out_pdf: Path,
    label_fn: Callable[[str], str] | None = None,
    flops_xmin: float | None = None,
    legend_label_order: list[str] | None = None,
    legend_column_stacks: list[list[str]] | None = None,
    step_cutoff: int | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts_c: ModelCurveCuts | None = None,
    model_curve_cuts_r: ModelCurveCuts | None = None,
    horizontal_ref_models_c: set[str] | None = None,
    horizontal_ref_models_r: set[str] | None = None,
) -> None:
    if not lh_c and not lh_r:
        print("Skip (no models):", out_pdf)
        return
    colors = _merged_colors(colors_c, colors_r)
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        constrained_layout=True,
        sharey=False,
    )
    _draw_flops_curves(
        ax0,
        lh_c,
        flops_c,
        colors,
        ylim_c,
        "Classification",
        label_fn,
        flops_xmin=None,
        step_cutoff=step_cutoff,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_c,
        horizontal_ref_models=horizontal_ref_models_c,
    )
    _draw_flops_curves(
        ax1,
        lh_r,
        flops_r,
        colors,
        ylim_r,
        "Regression",
        label_fn,
        flops_xmin=flops_xmin,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_r,
        horizontal_ref_models=horizontal_ref_models_r,
    )
    ax1.set_ylabel("")
    for ax in (ax0, ax1):
        ax.grid(True)
    _shared_figure_legend(
        fig,
        (ax0, ax1),
        label_order=legend_label_order,
        column_stack_labels=legend_column_stacks,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _plot_combined_steps_row(
    lh_c: dict,
    flops_c: dict,
    colors_c: dict[str, str],
    ylim_c: tuple[float, float] | None,
    step_cutoff_c: int | None,
    lh_r: dict,
    flops_r: dict,
    colors_r: dict[str, str],
    ylim_r: tuple[float, float] | None,
    out_pdf: Path,
    label_fn: Callable[[str], str] | None = None,
    flops_xmin: float | None = None,
    log_steps_xmin: float | None = None,
    legend_label_order: list[str] | None = None,
    legend_column_stacks: list[list[str]] | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts_c: ModelCurveCuts | None = None,
    model_curve_cuts_r: ModelCurveCuts | None = None,
    horizontal_ref_models_c: set[str] | None = None,
    horizontal_ref_models_r: set[str] | None = None,
    *,
    regression_zoom_inset: bool = True,
) -> None:
    if not lh_c and not lh_r:
        print("Skip (no models):", out_pdf)
        return
    colors = _merged_colors(colors_c, colors_r)
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        constrained_layout=True,
        sharey=False,
    )
    _draw_steps_curves(
        ax0,
        lh_c,
        flops_c,
        colors,
        ylim_c,
        "Classification",
        step_cutoff_c,
        label_fn=label_fn,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_c,
        horizontal_ref_models=horizontal_ref_models_c,
    )
    _draw_steps_curves(
        ax1,
        lh_r,
        flops_r,
        colors,
        ylim_r,
        "Regression",
        None,
        label_fn=label_fn,
        log_steps_xmin=log_steps_xmin,
        global_flops_xmin=flops_xmin,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_r,
        horizontal_ref_models=horizontal_ref_models_r,
    )
    ax1.set_ylabel("")
    for ax in (ax0, ax1):
        ax.grid(True)
    _shared_figure_legend(
        fig,
        (ax0, ax1),
        label_order=legend_label_order,
        column_stack_labels=legend_column_stacks,
    )
    if regression_zoom_inset:
        _add_steps_zoom_inset(
            ax1,
            lh_r,
            flops_r,
            colors,
            ylim_r,
            None,
            log_steps_xmin=_REGRESSION_STEPS_ZOOM_LOG_XMIN,
            log_steps_xmax=_REGRESSION_STEPS_ZOOM_LOG_XMAX,
            label_fn=label_fn,
            curve_end=curve_end,
            model_curve_cuts=model_curve_cuts_r,
            horizontal_ref_models=horizontal_ref_models_r,
        )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _plot_variant_bundle(
    variant: StepsPlotVariant,
    lh_c: dict,
    flops_c: dict,
    colors_c: dict[str, str],
    ylim_c: tuple[float, float] | None,
    lh_r: dict,
    flops_r: dict,
    colors_r: dict[str, str],
    ylim_r: tuple[float, float] | None,
    plots_root: Path,
    *,
    separate_panels: bool,
    step_cutoff: int | None = None,
    flops_xmin: float | None = None,
    log_steps_xmin: float | None = None,
    label_fn: Callable[[str], str] | None = None,
    legend_label_order: list[str] | None = None,
    legend_column_stacks: list[list[str]] | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts_c: ModelCurveCuts | None = None,
    model_curve_cuts_r: ModelCurveCuts | None = None,
    config_horizontal_refs_c: set[str] | None = None,
    config_horizontal_refs_r: set[str] | None = None,
) -> None:
    lh_c_v, flops_c_v, colors_c_v = _subset_for_plot(
        lh_c, flops_c, colors_c, variant.model_filter, variant.colors_override
    )
    lh_r_v, flops_r_v, colors_r_v = _subset_for_plot(
        lh_r, flops_r, colors_r, variant.model_filter, variant.colors_override
    )
    horizontal_ref_models_c = config_horizontal_refs_c
    horizontal_ref_models_r = config_horizontal_refs_r
    if config_horizontal_refs_c:
        merged_colors_c = _merged_colors(colors_c, colors_c_v)
        lh_c_v, flops_c_v, colors_c_v = _merge_config_horizontal_refs(
            lh_c_v,
            flops_c_v,
            colors_c_v,
            ref_names=config_horizontal_refs_c,
            source_lh=lh_c,
            source_flops=flops_c,
            source_colors=merged_colors_c,
        )
    if config_horizontal_refs_r:
        merged_colors_r = _merged_colors(colors_r, colors_r_v)
        lh_r_v, flops_r_v, colors_r_v = _merge_config_horizontal_refs(
            lh_r_v,
            flops_r_v,
            colors_r_v,
            ref_names=config_horizontal_refs_r,
            source_lh=lh_r,
            source_flops=flops_r,
            source_colors=merged_colors_r,
        )

    tag = variant.suffix or " (base)"
    models_c = ", ".join(sorted(flops_c_v)) or "(none)"
    models_r = ", ".join(sorted(flops_r_v)) or "(none)"
    print(f"Variant{tag}: clf=[{models_c}] reg=[{models_r}]")

    combined_out = plots_root / f"steps_combined{variant.suffix}"
    _plot_combined_flops_row(
        lh_c_v,
        flops_c_v,
        colors_c_v,
        ylim_c,
        lh_r_v,
        flops_r_v,
        colors_r_v,
        ylim_r,
        combined_out / "log_flops_vs_val_loss.pdf",
        label_fn,
        flops_xmin=flops_xmin,
        step_cutoff=step_cutoff,
        curve_end=curve_end,
        model_curve_cuts_c=model_curve_cuts_c,
        model_curve_cuts_r=model_curve_cuts_r,
        legend_label_order=legend_label_order,
        legend_column_stacks=legend_column_stacks,
        horizontal_ref_models_c=horizontal_ref_models_c,
        horizontal_ref_models_r=horizontal_ref_models_r,
    )
    _plot_combined_steps_row(
        lh_c_v,
        flops_c_v,
        colors_c_v,
        ylim_c,
        step_cutoff,
        lh_r_v,
        flops_r_v,
        colors_r_v,
        ylim_r,
        combined_out / "log_steps_vs_val_loss.pdf",
        label_fn,
        flops_xmin=flops_xmin,
        log_steps_xmin=log_steps_xmin,
        legend_label_order=legend_label_order,
        legend_column_stacks=legend_column_stacks,
        curve_end=curve_end,
        model_curve_cuts_c=model_curve_cuts_c,
        model_curve_cuts_r=model_curve_cuts_r,
        horizontal_ref_models_c=horizontal_ref_models_c,
        horizontal_ref_models_r=horizontal_ref_models_r,
        regression_zoom_inset=False,
    )

    if lh_c_v:
        _plot_flops_vs_loss(
            lh_c_v,
            flops_c_v,
            colors_c_v,
            "",
            ylim_c,
            combined_out / "log_flops_vs_val_loss_classification.pdf",
            label_fn=label_fn,
            legend_label_order=legend_label_order,
            ylabel="Validation loss (classification)",
            figsize=_SINGLE_PANEL_FIGSIZE,
            flops_xmin=None,
            step_cutoff=step_cutoff,
            curve_end=curve_end,
            model_curve_cuts=model_curve_cuts_c,
            horizontal_ref_models=horizontal_ref_models_c,
        )
        _plot_steps_vs_loss(
            lh_c_v,
            flops_c_v,
            colors_c_v,
            "",
            ylim_c,
            step_cutoff,
            combined_out / "log_steps_vs_val_loss_classification.pdf",
            label_fn=label_fn,
            legend_label_order=legend_label_order,
            ylabel="Validation loss (classification)",
            figsize=_SINGLE_PANEL_FIGSIZE,
            curve_end=curve_end,
            model_curve_cuts=model_curve_cuts_c,
            horizontal_ref_models=horizontal_ref_models_c,
        )
    if lh_r_v:
        _plot_flops_vs_loss(
            lh_r_v,
            flops_r_v,
            colors_r_v,
            "",
            ylim_r,
            combined_out / "log_flops_vs_val_loss_regression.pdf",
            label_fn=label_fn,
            legend_label_order=legend_label_order,
            ylabel="Validation loss (regression)",
            figsize=_SINGLE_PANEL_FIGSIZE,
            flops_xmin=flops_xmin,
            curve_end=curve_end,
            model_curve_cuts=model_curve_cuts_r,
            horizontal_ref_models=horizontal_ref_models_r,
        )
        _plot_steps_vs_loss(
            lh_r_v,
            flops_r_v,
            colors_r_v,
            "",
            ylim_r,
            None,
            combined_out / "log_steps_vs_val_loss_regression.pdf",
            label_fn=label_fn,
            legend_label_order=legend_label_order,
            ylabel="Validation loss (regression)",
            figsize=_SINGLE_PANEL_FIGSIZE,
            global_flops_xmin=flops_xmin,
            log_steps_xmin=log_steps_xmin,
            curve_end=curve_end,
            model_curve_cuts=model_curve_cuts_r,
            horizontal_ref_models=horizontal_ref_models_r,
        )

    if not separate_panels:
        return

    clf_out = plots_root / "classification" / f"steps{variant.suffix}"
    reg_out = plots_root / "regression" / f"steps{variant.suffix}"
    _plot_flops_vs_loss(
        lh_c_v,
        flops_c_v,
        colors_c_v,
        "Classification",
        ylim_c,
        clf_out / "log_flops_vs_val_loss.pdf",
        legend_outside=True,
        flops_xmin=None,
        step_cutoff=step_cutoff,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_c,
        horizontal_ref_models=horizontal_ref_models_c,
    )
    _plot_steps_vs_loss(
        lh_c_v,
        flops_c_v,
        colors_c_v,
        "Classification",
        ylim_c,
        step_cutoff,
        clf_out / "log_steps_vs_val_loss.pdf",
        legend_outside=True,
        label_fn=label_fn,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_c,
        horizontal_ref_models=horizontal_ref_models_c,
    )
    _plot_flops_vs_loss(
        lh_r_v,
        flops_r_v,
        colors_r_v,
        "Regression",
        ylim_r,
        reg_out / "log_flops_vs_val_loss.pdf",
        legend_outside=False,
        flops_xmin=flops_xmin,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_r,
        horizontal_ref_models=horizontal_ref_models_r,
    )
    _plot_steps_vs_loss(
        lh_r_v,
        flops_r_v,
        colors_r_v,
        "Regression",
        ylim_r,
        None,
        reg_out / "log_steps_vs_val_loss.pdf",
        legend_outside=False,
        label_fn=label_fn,
        global_flops_xmin=flops_xmin,
        log_steps_xmin=log_steps_xmin,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts_r,
        horizontal_ref_models=horizontal_ref_models_r,
    )


def write_classification_val_loss_log_flops_steps_pdf(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    ylim: tuple[float, float] | None,
    out_pdf: Path,
    *,
    model_names: set[str] | None = None,
    label_fn: Callable[[str], str] | None = None,
    step_cutoff: int | None = None,
    flops_xmin: float | None = None,
    curve_end: CurveEndConfig | None = None,
    model_curve_cuts: ModelCurveCuts | None = None,
    legend_label_order: list[str] | None = None,
    legend_column_stacks: list[list[str]] | None = None,
    config_horizontal_refs: set[str] | None = None,
    horizontal_ref_models: set[str] | None = None,
) -> None:
    """One row: log10(FLOPs) | log10(steps) vs classification validation loss."""
    if model_names is None:
        model_filter = lambda m: True  # noqa: E731
    else:
        model_filter = lambda m: m in model_names  # noqa: E731
    lh, flops, clrs = _subset_for_plot(
        loss_histories,
        flops_per_step,
        colors,
        model_filter,
    )
    if config_horizontal_refs:
        lh, flops, clrs = _merge_config_horizontal_refs(
            lh,
            flops,
            clrs,
            ref_names=config_horizontal_refs,
            source_lh=loss_histories,
            source_flops=flops_per_step,
            source_colors=colors,
        )
    if horizontal_ref_models is None:
        horizontal_ref_models = config_horizontal_refs
    if not lh:
        print("Skip (no classification models):", out_pdf)
        return

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        constrained_layout=True,
        sharey=True,
    )
    _draw_flops_curves(
        ax0,
        lh,
        flops,
        clrs,
        ylim,
        "",
        label_fn=label_fn,
        ylabel="Validation loss (classification)",
        flops_xmin=None,
        step_cutoff=step_cutoff,
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts,
        horizontal_ref_models=horizontal_ref_models,
    )
    _draw_steps_curves(
        ax1,
        lh,
        flops,
        clrs,
        ylim,
        "",
        step_cutoff,
        label_fn=label_fn,
        ylabel="",
        curve_end=curve_end,
        model_curve_cuts=model_curve_cuts,
        horizontal_ref_models=horizontal_ref_models,
    )
    for ax in (ax0, ax1):
        ax.grid(True)
    _shared_figure_legend(
        fig,
        (ax0, ax1),
        label_order=legend_label_order,
        column_stack_labels=legend_column_stacks,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory containing classification_<flag>.pkl and regression_<flag>.pkl "
        "(default: out/ under repo)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Root for PDF output (default: <out-dir>/plots)",
    )
    ap.add_argument(
        "--separate-panels",
        action="store_true",
        help="Also write classification/steps[<suffix>]/*.pdf and "
        "regression/steps[<suffix>]/*.pdf per variant.",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument("--regression-pickle", type=Path, default=None)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="JSON",
        help="Plot config JSON. When given, produces a single combined variant "
        "with the listed models instead of the four standard variants.",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Read loss histories from the plots cache "
        "(plots/tmp_results/steps.pkl) instead of the eval pickles.",
    )
    ap.add_argument(
        "--plots-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Override the steps plots cache path (default: plots/tmp_results/steps.pkl).",
    )
    args = ap.parse_args(argv)
    if args.plots_dir is None:
        args.plots_dir = Path(args.out_dir or DEFAULT_OUT_DIR) / "plots"

    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    plots_cache_path = args.plots_cache or (cache_root / _STEPS_CACHE_NAME)

    cfg: PlotConfig | None = PlotConfig.load(args.config) if args.config else None

    if args.plots_only:
        print(f"Loading steps cache from {plots_cache_path} …")
        with open(plots_cache_path, "rb") as f:
            cached = pickle.load(f)
        lh_c = cached["lh_c"]
        flops_c = cached["flops_c"]
        colors_c = cached["colors_c"]
        lh_r = cached["lh_r"]
        flops_r = cached["flops_r"]
        colors_r = cached["colors_r"]
    else:
        data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
        clf_p = args.classification_pickle or _pickle_paths(data_root, args.flag)[0]
        reg_p = args.regression_pickle or _pickle_paths(data_root, args.flag)[1]

        payload = build_steps_cache_from_pickles(clf_p, reg_p)
        lh_c = payload["lh_c"]
        flops_c = payload["flops_c"]
        colors_c = payload["colors_c"]
        lh_r = payload["lh_r"]
        flops_r = payload["flops_r"]
        colors_r = payload["colors_r"]
        save_steps_cache(payload, plots_cache_path)

    ylim_c = None
    ylim_r = None

    if cfg is not None:
        # Single custom variant with config models.
        model_names = set(cfg.model_names())
        custom_filter = lambda m: m in model_names  # noqa: E731
        colors_override = cfg.colors()
        variant = StepsPlotVariant("", custom_filter, colors_override)
        label_fn = cfg.label_for
        legend_label_order = _legend_labels_from_cfg(cfg)
        legend_column_stacks = (
            [[cfg.label_for(n) for n in stack] for stack in cfg.legend_column_stacks]
            if cfg.legend_column_stacks
            else None
        )
        config_horizontal_refs_c = _config_horizontal_ref_names(cfg, "classification")
        config_horizontal_refs_r = _config_horizontal_ref_names(cfg, "regression")
        _plot_variant_bundle(
            variant,
            lh_c,
            flops_c,
            colors_c,
            ylim_c,
            lh_r,
            flops_r,
            colors_r,
            ylim_r,
            plots_root,
            separate_panels=args.separate_panels,
            step_cutoff=cfg.step_cutoff,
            flops_xmin=cfg.flops_xmin,
            log_steps_xmin=cfg.log_steps_xmin,
            model_curve_cuts_c=cfg.model_curve_cuts("classification"),
            model_curve_cuts_r=cfg.model_curve_cuts("regression"),
            label_fn=label_fn,
            legend_label_order=legend_label_order,
            legend_column_stacks=legend_column_stacks,
            config_horizontal_refs_c=config_horizontal_refs_c,
            config_horizontal_refs_r=config_horizontal_refs_r,
        )
    else:
        for variant in STEPS_PLOT_VARIANTS:
            _plot_variant_bundle(
                variant,
                lh_c,
                flops_c,
                colors_c,
                ylim_c,
                lh_r,
                flops_r,
                colors_r,
                ylim_r,
                plots_root,
                separate_panels=args.separate_panels,
            )


if __name__ == "__main__":
    main()
