#!/usr/bin/env python3
"""Training curves: log10(FLOPs) vs validation loss and log10(steps) vs validation loss.

Pickles default to ``<repo>/<--out-dir>/``; PDFs default to
``<repo>/<--plots-dir>/``.

By default writes **one row of two panels** (classification | regression) with a
**single shared legend** under ``steps_combined/`` — suitable for side-by-side
paper figures without duplicating the legend. BERT-tiny* (legend BERT-small*)
appears on both FLOPs and steps panels; Transformer2-DIS is omitted.

Use ``--separate-panels`` to also emit the legacy per-task PDFs under
``classification/steps/`` and ``regression/steps/``.

:func:`write_classification_val_loss_log_flops_steps_pdf` is used by
``plot_small_paper.py`` for the classification-only small-paper PDF.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_OUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_WANDB_TAG,
    REGRESSION_PICKLE_STEM,
    model_key_excluded_from_training_curve_flops,
    model_key_excluded_from_training_curve_steps,
    plot_model_label,
    repo_output_path,
)

# Paper-style typography: axis labels 12; legend slightly smaller; ticks 11.
_LABEL_FS = 12
_LEGEND_FS = 10
_TICK_FS = 11
_TITLE_FS = 13


def _pickle_paths(out_dir: Path, flag: str) -> tuple[Path, Path]:
    return (
        out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl",
        out_dir / f"{REGRESSION_PICKLE_STEM}_{flag}.pkl",
    )


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
) -> None:
    for model, series_list in _runs_per_model(loss_histories, flops_per_step).items():
        if model_key_excluded_from_training_curve_flops(model):
            continue
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
        ax.plot(x, mean_loss, color=color, label=plot_model_label(model))
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
    olm_step_cap: int | None,
    out_pdf: Path,
    *,
    legend_outside: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _draw_steps_curves(
        ax, loss_histories, flops_per_step, colors, ylim, panel_title, olm_step_cap
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
    olm_step_cap: int | None,
) -> None:
    for model, series_list in _runs_per_model(loss_histories, flops_per_step).items():
        if model_key_excluded_from_training_curve_steps(model):
            continue
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
        #if olm_step_cap is not None and "OmniLearned-medium" in model:
        #    step_mask = steps_grid <= olm_step_cap
        #else:
        step_mask = np.full_like(steps_grid, True, dtype=bool)
        steps_plot = steps_grid[step_mask]
        mean_loss_plot = mean_loss[step_mask]
        sigma_loss_plot = sigma_loss[step_mask]
        log_steps_plot = np.log10(steps_plot + 1)
        ax.plot(
            log_steps_plot, mean_loss_plot, color=color, label=plot_model_label(model)
        )
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
        # Older matplotlib: no "outside" loc keyword
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            **legend_kw,
        )


def write_classification_val_loss_log_flops_steps_pdf(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    ylim: tuple[float, float],
    out_pdf: Path,
    *,
    olm_step_cap: int = 25_000,
) -> None:
    """One row: log10(FLOPs) vs val loss | log10(steps) vs val loss (classification)."""
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        constrained_layout=True,
        sharey=False,
    )
    _draw_flops_curves(ax0, loss_histories, flops_per_step, colors, ylim, "Classification")
    _draw_steps_curves(
        ax1,
        loss_histories,
        flops_per_step,
        colors,
        ylim,
        "Classification",
        olm_step_cap,
    )
    ax1.set_ylabel("")
    for ax in (ax0, ax1):
        ax.grid(True)
    _shared_figure_legend(fig, (ax0, ax1))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


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
) -> None:
    colors = _merged_colors(colors_c, colors_r)
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        constrained_layout=True,
        sharey=False,
    )
    _draw_flops_curves(ax0, lh_c, flops_c, colors, ylim_c, "Classification")
    _draw_flops_curves(ax1, lh_r, flops_r, colors, ylim_r, "Regression")
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
    olm_cap_c: int,
    lh_r: dict,
    flops_r: dict,
    colors_r: dict[str, str],
    ylim_r: tuple[float, float],
    out_pdf: Path,
) -> None:
    colors = _merged_colors(colors_c, colors_r)
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 4.6),
        constrained_layout=True,
        sharey=False,
    )
    _draw_steps_curves(ax0, lh_c, flops_c, colors, ylim_c, "Classification", olm_cap_c)
    _draw_steps_curves(ax1, lh_r, flops_r, colors, ylim_r, "Regression", None)
    ax1.set_ylabel("")
    for ax in (ax0, ax1):
        ax.grid(True)
    _shared_figure_legend(fig, (ax0, ax1))
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
        help="Directory containing classification_<flag>.pkl and regression_<flag>.pkl (default: out/ under repo)",
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
        help="Also write classification/steps/*.pdf and regression/steps/*.pdf (each with its own legend).",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument("--regression-pickle", type=Path, default=None)
    args = ap.parse_args(argv)

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    clf_p = args.classification_pickle or _pickle_paths(data_root, args.flag)[0]
    reg_p = args.regression_pickle or _pickle_paths(data_root, args.flag)[1]

    combined_out = plots_root / "steps_combined"
    clf_out = plots_root / "classification" / "steps"
    reg_out = plots_root / "regression" / "steps"

    with open(clf_p, "rb") as f:
        clf = pickle.load(f)
    lh_c = clf["loss_histories"]
    flops_c = clf["flops_per_step"]
    colors_c = clf["clrs_dict_full"]

    with open(reg_p, "rb") as f:
        reg = pickle.load(f)
    lh_r = reg["loss_histories"]
    flops_r = reg["flops_per_step"]
    colors_r = reg["clrs_dict_full"]

    ylim_c = (1.075, 1.25)
    ylim_r = (0.03, 0.06)

    _plot_combined_flops_row(
        lh_c,
        flops_c,
        colors_c,
        ylim_c,
        lh_r,
        flops_r,
        colors_r,
        ylim_r,
        combined_out / "log_flops_vs_val_loss.pdf",
    )
    _plot_combined_steps_row(
        lh_c,
        flops_c,
        colors_c,
        ylim_c,
        25_000,
        lh_r,
        flops_r,
        colors_r,
        ylim_r,
        combined_out / "log_steps_vs_val_loss.pdf",
    )

    if args.separate_panels:
        _plot_flops_vs_loss(
            lh_c,
            flops_c,
            colors_c,
            "Classification",
            ylim_c,
            clf_out / "log_flops_vs_val_loss.pdf",
            legend_outside=True,
        )
        _plot_steps_vs_loss(
            lh_c,
            flops_c,
            colors_c,
            "Classification",
            ylim_c,
            25_000,
            clf_out / "log_steps_vs_val_loss.pdf",
            legend_outside=True,
        )
        _plot_flops_vs_loss(
            lh_r,
            flops_r,
            colors_r,
            "Regression",
            ylim_r,
            reg_out / "log_flops_vs_val_loss.pdf",
            legend_outside=False,
        )
        _plot_steps_vs_loss(
            lh_r,
            flops_r,
            colors_r,
            "Regression",
            ylim_r,
            None,
            reg_out / "log_steps_vs_val_loss.pdf",
            legend_outside=False,
        )


if __name__ == "__main__":
    main()
