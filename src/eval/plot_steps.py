#!/usr/bin/env python3
"""Training curves: log10(FLOPs) vs validation loss and log10(steps) vs validation loss.

Pickles default to ``<repo>/<--out-dir>/``; PDFs default to
``<repo>/<--plots-dir>/``.

Writes **four** combined 1×2 panels (classification | regression, shared legend):

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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._constants import (
    CANONICAL_CLASSIFICATION_PICKLE,
    CANONICAL_REGRESSION_PICKLE,
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_WANDB_TAG,
    REGRESSION_PICKLE_STEM,
    STEPS_SMALL_MODEL_COLORS,
    STEPS_SMALL_MODELS,
    is_base_steps_model,
    is_bert_model,
    is_hyperscale_model,
    plot_model_label,
    repo_output_path,
)
from src.eval._plot_config import PlotConfig

# Paper-style typography: axis labels 12; legend slightly smaller; ticks 11.
_LABEL_FS = 12
_LEGEND_FS = 10
_TICK_FS = 11
_TITLE_FS = 13

_STEPS_CACHE_NAME = "steps.pkl"


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
        if model_filter(k) and k in loss_histories and loss_histories[k]
    )
    lh = {k: loss_histories[k] for k in keys}
    flops = {k: flops_per_step[k] for k in keys}
    clrs = {k: colors.get(k, "tab:gray") for k in keys}
    if colors_override:
        for k in keys:
            if k in colors_override:
                clrs[k] = colors_override[k]
    return lh, flops, clrs


def _runs_per_model(
    loss_histories: dict, flops_per_step: dict
) -> OrderedDict[str, list]:
    runs: OrderedDict[str, list] = OrderedDict()
    for model in sorted(flops_per_step):
        if model not in loss_histories:
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
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _draw_flops_curves(ax, loss_histories, flops_per_step, colors, ylim, panel_title)
    if legend_outside:
        ax.legend(
            fontsize=_LEGEND_FS,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
    else:
        ax.legend(fontsize=_LEGEND_FS, loc="upper right", frameon=True)
    ax.grid(True)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _draw_flops_curves(
    ax: plt.Axes,
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    ylim: tuple[float, float] | None,
    panel_title: str,
    label_fn: Callable[[str], str] | None = None,
) -> None:
    if label_fn is None:
        label_fn = plot_model_label
    for model, series_list in _runs_per_model(loss_histories, flops_per_step).items():
        color = colors.get(model, "tab:gray")
        flops = flops_per_step[model]
        all_steps, all_losses = [], []
        for st, lo in series_list:
            if len(st) > 0 and len(lo) > 0:
                all_steps.append(st)
                all_losses.append(lo)
        if not all_steps:
            continue
        steps_grid = np.unique(np.concatenate(all_steps)).astype(float)
        if len(steps_grid) == 0:
            continue
        losses_aligned = np.array(
            [np.interp(steps_grid, st, lo) for st, lo in zip(all_steps, all_losses)]
        )
        mean_loss = np.mean(losses_aligned, axis=0)
        sigma_loss = (
            np.std(losses_aligned, axis=0)
            if losses_aligned.shape[0] > 1
            else np.zeros_like(mean_loss)
        )
        cum_flops = steps_grid * flops
        x = np.log10(cum_flops + 1)
        ax.plot(x, mean_loss, color=color, label=label_fn(model))
        if losses_aligned.shape[0] > 1:
            ax.fill_between(
                x,
                mean_loss - sigma_loss,
                mean_loss + sigma_loss,
                alpha=0.25,
                color=color,
            )

    ax.set_title(panel_title, fontsize=_TITLE_FS, pad=10)
    ax.set_xlabel(r"$log_{10}$(Training FLOPs)", fontsize=_LABEL_FS)
    ax.set_ylabel("Validation loss", fontsize=_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    if ylim:
        ax.set_ylim(ylim)


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
    label_fn: Callable[[str], str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _draw_steps_curves(
        ax, loss_histories, flops_per_step, colors, ylim, panel_title, step_cutoff,
        label_fn=label_fn,
    )
    if legend_outside:
        ax.legend(
            fontsize=_LEGEND_FS,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
    else:
        ax.legend(fontsize=_LEGEND_FS, loc="upper right", frameon=True)
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
) -> None:
    if label_fn is None:
        label_fn = plot_model_label
    for model, series_list in _runs_per_model(loss_histories, flops_per_step).items():
        color = colors.get(model, "tab:gray")
        all_steps, all_losses = [], []
        for st, lo in series_list:
            if len(st) > 0 and len(lo) > 0:
                all_steps.append(st)
                all_losses.append(lo)
        if not all_steps:
            continue
        steps_grid = np.unique(np.concatenate(all_steps)).astype(float)
        if len(steps_grid) == 0:
            continue
        losses_aligned = np.array(
            [np.interp(steps_grid, st, lo) for st, lo in zip(all_steps, all_losses)]
        )
        mean_loss = np.mean(losses_aligned, axis=0)
        sigma_loss = (
            np.std(losses_aligned, axis=0)
            if losses_aligned.shape[0] > 1
            else np.zeros_like(mean_loss)
        )
        step_mask = (
            steps_grid <= step_cutoff if step_cutoff is not None
            else np.ones(len(steps_grid), dtype=bool)
        )
        steps_plot = steps_grid[step_mask]
        mean_loss_plot = mean_loss[step_mask]
        sigma_loss_plot = sigma_loss[step_mask]
        log_steps_plot = np.log10(steps_plot + 1)
        ax.plot(log_steps_plot, mean_loss_plot, color=color, label=label_fn(model))
        if losses_aligned.shape[0] > 1:
            ax.fill_between(
                log_steps_plot,
                mean_loss_plot - sigma_loss_plot,
                mean_loss_plot + sigma_loss_plot,
                alpha=0.25,
                color=color,
            )

    ax.set_title(panel_title, fontsize=_TITLE_FS, pad=10)
    ax.set_xlabel(r"$log_{10}$(Training steps)", fontsize=_LABEL_FS)
    ax.set_ylabel("Validation loss", fontsize=_LABEL_FS)
    ax.tick_params(axis="both", labelsize=_TICK_FS)
    if ylim:
        ax.set_ylim(ylim)


def _shared_figure_legend(fig: plt.Figure, axes: tuple[plt.Axes, ...]) -> None:
    """One legend for all *axes*, de-duplicated by model name."""
    by_label: dict[str, plt.Artist] = {}
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            by_label.setdefault(li, hi)
    labels = sorted(by_label.keys())
    handles = [by_label[k] for k in labels]
    if not handles:
        return
    n = len(labels)
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
        fig.legend(handles, labels, loc="outside lower center", **legend_kw)
    except (TypeError, ValueError):
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            **legend_kw,
        )


def _plot_combined_flops_row(
    lh_c: dict,
    flops_c: dict,
    colors_c: dict[str, str],
    ylim_c: tuple[float, float],
    lh_r: dict,
    flops_r: dict,
    colors_r: dict[str, str],
    ylim_r: tuple[float, float],
    out_pdf: Path,
    label_fn: Callable[[str], str] | None = None,
) -> None:
    if not lh_c and not lh_r:
        print("Skip (no models):", out_pdf)
        return
    colors = _merged_colors(colors_c, colors_r)
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), constrained_layout=True, sharey=False,
    )
    _draw_flops_curves(ax0, lh_c, flops_c, colors, ylim_c, "Classification", label_fn)
    _draw_flops_curves(ax1, lh_r, flops_r, colors, ylim_r, "Regression", label_fn)
    ax1.set_ylabel("")
    for ax in (ax0, ax1):
        ax.grid(True)
    _shared_figure_legend(fig, (ax0, ax1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _plot_combined_steps_row(
    lh_c: dict,
    flops_c: dict,
    colors_c: dict[str, str],
    ylim_c: tuple[float, float],
    step_cutoff_c: int | None,
    lh_r: dict,
    flops_r: dict,
    colors_r: dict[str, str],
    ylim_r: tuple[float, float],
    out_pdf: Path,
    label_fn: Callable[[str], str] | None = None,
) -> None:
    if not lh_c and not lh_r:
        print("Skip (no models):", out_pdf)
        return
    colors = _merged_colors(colors_c, colors_r)
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(11.5, 4.6), constrained_layout=True, sharey=False,
    )
    _draw_steps_curves(
        ax0, lh_c, flops_c, colors, ylim_c, "Classification", step_cutoff_c,
        label_fn=label_fn,
    )
    _draw_steps_curves(
        ax1, lh_r, flops_r, colors, ylim_r, "Regression", None, label_fn=label_fn,
    )
    ax1.set_ylabel("")
    for ax in (ax0, ax1):
        ax.grid(True)
    _shared_figure_legend(fig, (ax0, ax1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _plot_variant_bundle(
    variant: StepsPlotVariant,
    lh_c: dict,
    flops_c: dict,
    colors_c: dict[str, str],
    ylim_c: tuple[float, float],
    lh_r: dict,
    flops_r: dict,
    colors_r: dict[str, str],
    ylim_r: tuple[float, float],
    plots_root: Path,
    *,
    separate_panels: bool,
    step_cutoff: int | None = None,
    label_fn: Callable[[str], str] | None = None,
) -> None:
    lh_c_v, flops_c_v, colors_c_v = _subset_for_plot(
        lh_c, flops_c, colors_c, variant.model_filter, variant.colors_override
    )
    lh_r_v, flops_r_v, colors_r_v = _subset_for_plot(
        lh_r, flops_r, colors_r, variant.model_filter, variant.colors_override
    )

    tag = variant.suffix or " (base)"
    models_c = ", ".join(sorted(flops_c_v)) or "(none)"
    models_r = ", ".join(sorted(flops_r_v)) or "(none)"
    print(f"Variant{tag}: clf=[{models_c}] reg=[{models_r}]")

    combined_out = plots_root / f"steps_combined{variant.suffix}"
    _plot_combined_flops_row(
        lh_c_v, flops_c_v, colors_c_v, ylim_c,
        lh_r_v, flops_r_v, colors_r_v, ylim_r,
        combined_out / "log_flops_vs_val_loss.pdf",
        label_fn,
    )
    _plot_combined_steps_row(
        lh_c_v, flops_c_v, colors_c_v, ylim_c, step_cutoff,
        lh_r_v, flops_r_v, colors_r_v, ylim_r,
        combined_out / "log_steps_vs_val_loss.pdf",
        label_fn,
    )

    if not separate_panels:
        return

    clf_out = plots_root / "classification" / f"steps{variant.suffix}"
    reg_out = plots_root / "regression" / f"steps{variant.suffix}"
    _plot_flops_vs_loss(
        lh_c_v, flops_c_v, colors_c_v, "Classification", ylim_c,
        clf_out / "log_flops_vs_val_loss.pdf", legend_outside=True,
    )
    _plot_steps_vs_loss(
        lh_c_v, flops_c_v, colors_c_v, "Classification", ylim_c, step_cutoff,
        clf_out / "log_steps_vs_val_loss.pdf", legend_outside=True, label_fn=label_fn,
    )
    _plot_flops_vs_loss(
        lh_r_v, flops_r_v, colors_r_v, "Regression", ylim_r,
        reg_out / "log_flops_vs_val_loss.pdf", legend_outside=False,
    )
    _plot_steps_vs_loss(
        lh_r_v, flops_r_v, colors_r_v, "Regression", ylim_r, None,
        reg_out / "log_steps_vs_val_loss.pdf", legend_outside=False, label_fn=label_fn,
    )


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
        default=DEFAULT_PLOTS_DIR,
        help="Root for PDF output (default: plots/ under repo)",
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

        with open(clf_p, "rb") as f:
            clf = pickle.load(f)
        lh_c = clf["loss_histories"]
        flops_c = clf["flops_per_step"]
        colors_c = clf["clrs_dict_full"]

        with open(reg_p, "rb") as f:
            print("- reading regression pickle from", reg_p)
            reg = pickle.load(f)
            print(
                "- keys in regression pickle:",
                reg["training_names_full"]["Log1p"].keys(),
            )
        lh_r = reg["loss_histories"]
        flops_r = reg["flops_per_step"]
        colors_r = reg["clrs_dict_full"]

        # Save plots cache for --plots-only reuse.
        plots_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plots_cache_path, "wb") as f:
            pickle.dump(
                {
                    "lh_c": lh_c,
                    "flops_c": flops_c,
                    "colors_c": colors_c,
                    "lh_r": lh_r,
                    "flops_r": flops_r,
                    "colors_r": colors_r,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"Saved steps cache → {plots_cache_path}")

    ylim_c = (1.075, 1.25)
    ylim_r = (0.03, 0.06)

    if cfg is not None:
        # Single custom variant with config models.
        model_names = set(cfg.model_names())
        custom_filter = lambda m: m in model_names  # noqa: E731
        colors_override = cfg.colors()
        variant = StepsPlotVariant("", custom_filter, colors_override)
        label_fn = cfg.label_for
        _plot_variant_bundle(
            variant, lh_c, flops_c, colors_c, ylim_c,
            lh_r, flops_r, colors_r, ylim_r,
            plots_root,
            separate_panels=args.separate_panels,
            step_cutoff=cfg.step_cutoff,
            label_fn=label_fn,
        )
    else:
        for variant in STEPS_PLOT_VARIANTS:
            _plot_variant_bundle(
                variant, lh_c, flops_c, colors_c, ylim_c,
                lh_r, flops_r, colors_r, ylim_r,
                plots_root,
                separate_panels=args.separate_panels,
            )


if __name__ == "__main__":
    main()
