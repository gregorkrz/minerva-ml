#!/usr/bin/env python3
"""Training curves: log10(FLOPs) vs validation loss and log10(steps) vs validation loss.

Pickles default to ``<repo>/<--out-dir>/eval_data/``; PDFs default to
``<repo>/<--plots-dir>/classification|regression/``.
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
    EVAL_DATA_SUBDIR,
    FLOPS_PER_STEP,
    REGRESSION_PICKLE_STEM,
    repo_output_path,
)


def _pickle_paths(out_dir: Path, flag: str) -> tuple[Path, Path]:
    ed = out_dir / EVAL_DATA_SUBDIR
    return ed / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl", ed / f"{REGRESSION_PICKLE_STEM}_{flag}.pkl"


def _plot_flops_vs_loss(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    title: str,
    ylim: tuple[float, float] | None,
    out_pdf: Path,
) -> None:
    runs_per_model = OrderedDict()
    for model in sorted(flops_per_step):
        if model not in loss_histories:
            continue
        runs_per_model[model] = loss_histories[model]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, series_list in sorted(runs_per_model.items(), key=lambda kv: kv[0]):
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
        losses_aligned = np.array([np.interp(steps_grid, st, lo) for st, lo in zip(all_steps, all_losses)])
        mean_loss = np.mean(losses_aligned, axis=0)
        sigma_loss = np.std(losses_aligned, axis=0) if losses_aligned.shape[0] > 1 else np.zeros_like(mean_loss)
        cum_flops = steps_grid * flops
        x = np.log10(cum_flops + 1)
        ax.plot(x, mean_loss, color=color, label=model)
        if losses_aligned.shape[0] > 1:
            ax.fill_between(x, mean_loss - sigma_loss, mean_loss + sigma_loss, alpha=0.25, color=color)

    ax.set_xlabel(r"$log_{10}$(Training FLOPs)")
    ax.set_ylabel("Validation loss")
    if ylim:
        ax.set_ylim(ylim)
    leg = ax.legend(title=title, fontsize=9, loc="upper right")
    leg.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _plot_steps_vs_loss(
    loss_histories: dict,
    flops_per_step: dict,
    colors: dict[str, str],
    title: str,
    ylim: tuple[float, float] | None,
    olm_step_cap: int | None,
    out_pdf: Path,
) -> None:
    runs_per_model = OrderedDict()
    for model in sorted(flops_per_step):
        if model not in loss_histories:
            continue
        runs_per_model[model] = loss_histories[model]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, series_list in sorted(runs_per_model.items(), key=lambda kv: kv[0]):
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
        losses_aligned = np.array([np.interp(steps_grid, st, lo) for st, lo in zip(all_steps, all_losses)])
        mean_loss = np.mean(losses_aligned, axis=0)
        sigma_loss = np.std(losses_aligned, axis=0) if losses_aligned.shape[0] > 1 else np.zeros_like(mean_loss)
        if olm_step_cap is not None and "OmniLearned-medium" in model:
            step_mask = steps_grid <= olm_step_cap
        else:
            step_mask = np.full_like(steps_grid, True, dtype=bool)
        steps_plot = steps_grid[step_mask]
        mean_loss_plot = mean_loss[step_mask]
        sigma_loss_plot = sigma_loss[step_mask]
        log_steps_plot = np.log10(steps_plot + 1)
        ax.plot(log_steps_plot, mean_loss_plot, color=color, label=model)
        if losses_aligned.shape[0] > 1:
            ax.fill_between(
                log_steps_plot, mean_loss_plot - sigma_loss_plot, mean_loss_plot + sigma_loss_plot,
                alpha=0.25, color=color,
            )

    ax.set_xlabel(r"$log_{10}$(Training steps)")
    ax.set_ylabel("Validation loss")
    if ylim:
        ax.set_ylim(ylim)
    leg = ax.legend(title=title, fontsize=9, loc="upper right")
    leg.set_title(title)
    ax.grid(True)
    fig.tight_layout()
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
        help="Root containing eval_data/ pickles (default: out/ under repo)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Root for PDF output (default: plots/ under repo)",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument("--regression-pickle", type=Path, default=None)
    args = ap.parse_args(argv)

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    clf_p = args.classification_pickle or _pickle_paths(data_root, args.flag)[0]
    reg_p = args.regression_pickle or _pickle_paths(data_root, args.flag)[1]

    clf_out = plots_root / "classification" / "steps"
    reg_out = plots_root / "regression" / "steps"

    with open(clf_p, "rb") as f:
        clf = pickle.load(f)
    lh_c = clf["loss_histories"]
    flops_c = clf["flops_per_step"]
    colors_c = clf["clrs_dict_full"]
    _plot_flops_vs_loss(
        lh_c, flops_c, colors_c,
        "MINERvA Open Data Playlist 1A\nTask: Classification",
        (1.075, 1.25),
        clf_out / "log_flops_vs_val_loss.pdf",
    )
    _plot_steps_vs_loss(
        lh_c, flops_c, colors_c,
        "MINERvA Open Data Playlist 1A\nTask: Classification",
        (1.075, 1.25),
        25_000,
        clf_out / "log_steps_vs_val_loss.pdf",
    )

    with open(reg_p, "rb") as f:
        reg = pickle.load(f)
    lh_r = reg["loss_histories"]
    flops_r = reg["flops_per_step"]
    colors_r = reg["clrs_dict_full"]
    _plot_flops_vs_loss(
        lh_r, flops_r, colors_r,
        "Minerva Open Data Playlist 1A\nTask: Regression",
        (0.03, 0.06),
        reg_out / "log_flops_vs_val_loss.pdf",
    )
    _plot_steps_vs_loss(
        lh_r, flops_r, colors_r,
        "Minerva Open Data Playlist 1A\nTask: Regression",
        (0.03, 0.06),
        None,
        reg_out / "log_steps_vs_val_loss.pdf",
    )


if __name__ == "__main__":
    main()
