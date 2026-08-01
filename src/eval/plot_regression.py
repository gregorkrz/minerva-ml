#!/usr/bin/env python3
"""Regression evaluation PDFs (IQR vs q3, residuals, scaling) from cached pickle.

Pickles default under ``<repo>/<--out-dir>/``; PDFs under
``<repo>/<--plots-dir>/regression/`` and small-paper ratio histograms under
``<repo>/<--plots-dir>/small_paper/``.

Pass ``--config JSON`` to restrict plots to a model subset with custom colors.
Pass ``--plots-only`` to read from the regression plots cache
(``plots/tmp_results/regression.pkl``) instead of the eval pickle; on a normal
run the cache is written automatically.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.e_available_plots import (
    plot_example_E_pred_true,
    plot_ratio_histogram_q3_two_panels,
    plot_residuals_by_energy,
    plot_residuals_by_q3,
    plot_rms_iqr_with_uncertainty,
)
from src.eval._constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    REGRESSION_PICKLE_STEM,
    plot_model_label,
    repo_output_path,
)
from src.eval._plot_config import PlotConfig

_REG_CACHE_NAME = "regression.pkl"


def _maybe_register_arial() -> None:
    arial_path = Path.home() / ".local/share/fonts/ARIAL.TTF"
    if not arial_path.exists():
        return
    from matplotlib import font_manager

    font_manager.fontManager.addfont(str(arial_path))
    matplotlib.rcParams["font.family"] = ["Arial", "sans-serif"]


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / f"{REGRESSION_PICKLE_STEM}_{flag}.pkl"


def _cap_to_label(cap: int) -> str:
    if cap == -1:
        return "6M"
    if cap >= 1_000_000:
        return f"{cap // 1_000_000}M"
    return f"{cap // 1000}k"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory containing regression_<flag>.pkl (default: out/ under repo)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Root for PDF output (default: <out-dir>/plots)",
    )
    ap.add_argument("--regression-pickle", type=Path, default=None)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="JSON",
        help="Plot config JSON (models, colors). When given, only listed models "
        "appear in all regression figures.",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Read from the regression plots cache "
        "(plots/tmp_results/regression.pkl) instead of the eval pickle.",
    )
    ap.add_argument(
        "--plots-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Override the regression plots cache path "
        "(default: plots/tmp_results/regression.pkl).",
    )
    args = ap.parse_args(argv)
    if args.plots_dir is None:
        args.plots_dir = Path(args.out_dir or DEFAULT_OUT_DIR) / "plots"

    _maybe_register_arial()

    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    out_dir = plots_root / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    small_paper_dir = plots_root / "small_paper"
    small_paper_dir.mkdir(parents=True, exist_ok=True)

    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    plots_cache_path = args.plots_cache or (cache_root / _REG_CACHE_NAME)

    cfg: PlotConfig | None = PlotConfig.load(args.config) if args.config else None

    if args.plots_only:
        print(f"Loading regression cache from {plots_cache_path} …")
        with open(plots_cache_path, "rb") as f:
            reg = pickle.load(f)
    else:
        data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
        pkl = args.regression_pickle or _pickle_path(data_root, args.flag)
        with open(pkl, "rb") as f:
            reg = pickle.load(f)
        # Save plots cache for --plots-only reuse.
        plots_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plots_cache_path, "wb") as f:
            pickle.dump(reg, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved regression cache → {plots_cache_path}")

    CKPT_DIR = Path(reg["ckpt_dir"])
    suppress = bool(reg["suppress_errors"])
    training_names_full_no_rw = reg["training_names_full_no_rw"]
    training_names_full_first_only = reg["training_names_full_first_only"]
    baseline_run = reg["baseline_run"]
    clrs = reg["clrs_dict_full"]
    runs_by_model_cap = reg["runs_by_model_cap"]
    # Pre-loaded dicts from collect_eval_data (schema_version >= 2); else reload from disk.
    data_no_rw = reg.get("eval_data_no_rw")
    data_first = reg.get("eval_data_first_only")
    data_scaling = reg.get("eval_data_scaling")

    # Apply config filtering.
    if cfg is not None:
        clrs = {**clrs, **cfg.colors()}
        training_names_full_no_rw = cfg.filter_nested(training_names_full_no_rw)
        training_names_full_first_only = cfg.filter_nested(
            training_names_full_first_only
        )
        config_model_names = set(cfg.model_names())
        runs_by_model_cap = {
            k: v for k, v in runs_by_model_cap.items() if k in config_model_names
        }

    label_fn = cfg.label_for if cfg is not None else None
    # IQR legends: flat reading order (OL ±rw, HS ±rw, …, MLP, Baseline).
    # Column stacks are for shared multi-panel figure legends, not the in-axes IQR box.
    iqr_legend_order = (
        cfg.legend_labels(include_baseline=True) if cfg is not None else None
    )
    ratio_legend_order = cfg.legend_labels() if cfg is not None else None
    ratio_legend_stacks = (
        [[cfg.label_for(n) for n in stack] for stack in cfg.legend_column_stacks]
        if cfg is not None and cfg.legend_column_stacks
        else None
    )
    iqr_legend_kw = dict(
        label_fn=label_fn,
        legend_label_order=iqr_legend_order,
        legend_column_stacks=None,
    )
    ratio_legend_kw = dict(
        label_fn=label_fn,
        legend_label_order=ratio_legend_order,
        legend_column_stacks=ratio_legend_stacks,
    )
    residuals_legend_kw = dict(
        label_fn=label_fn,
        legend_label_order=iqr_legend_order,
        legend_column_stacks=ratio_legend_stacks,
    )

    # Shared q3 edges for the 1A paper IQR/MPV figure and the 1A+1B overlay
    # (must match so the 1A baseline/curves coincide on both PDFs).
    q3_bins_paper = [0, 0.6, 1.2, 1.8, 2.4, 3.0, 100]

    fig_i = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        q3_bins=q3_bins_paper,
        show_q3_histograms=True,
        return_hist_fig=False,
        suppress_errors=suppress,
        use_cc_selection=2,
        colors=clrs,
        text="",
        data=data_no_rw,
        **iqr_legend_kw,
    )
    fig_i.savefig(out_dir / "q3_vs_iqr_rms_full_1A.pdf", bbox_inches="tight")
    plt.close(fig_i)
    print("Saved:", out_dir / "q3_vs_iqr_rms_full_1A.pdf")

    if data_first is not None and "mc_E" in data_first:
        ratio_playlists = sorted(data_first["mc_E"].keys())
    else:
        ratio_playlists = ["1A"]
    for pl in ratio_playlists:
        fig_q = plot_ratio_histogram_q3_two_panels(
            CKPT_DIR=CKPT_DIR,
            training_names=training_names_full_first_only,
            playlists=[pl],
            dataset_to_plot=pl,
            baseline_run=baseline_run,
            use_cc_selection=2,
            colors=clrs,
            suppress_errors=suppress,
            data=data_first,
            **ratio_legend_kw,
        )
        out_q = small_paper_dir / f"regression_e_ratio_hist_q3_0_1_and_1_2_{pl}.pdf"
        fig_q.savefig(out_q, bbox_inches="tight")
        plt.close(fig_q)
        print("Saved:", out_q)

    fig_i = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        q3_bins=q3_bins_paper,
        show_q3_histograms=True,
        return_hist_fig=False,
        suppress_errors=suppress,
        use_cc_selection=1,
        colors=clrs,
        text="",
        data=data_no_rw,
        **iqr_legend_kw,
    )
    fig_i.savefig(
        out_dir / "q3_vs_iqr_rms_full_muon_selection_only.pdf", bbox_inches="tight"
    )
    plt.close(fig_i)
    print("Saved:", out_dir / "q3_vs_iqr_rms_full_muon_selection_only.pdf")

    fig_i = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1B"],
        dataset_to_plot="1B",
        baseline_run=baseline_run,
        show_q3_histograms=True,
        return_hist_fig=True,
        suppress_errors=suppress,
        colors=clrs,
        text="",
        data=data_no_rw,
        **iqr_legend_kw,
    )
    fig_i.savefig(out_dir / "q3_vs_iqr_rms_full_1B.pdf", bbox_inches="tight")
    plt.close(fig_i)
    print("Saved:", out_dir / "q3_vs_iqr_rms_full_1B.pdf")

    fig_dbg = plot_example_E_pred_true(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        use_cc_selection=2,
        n_examples=10,
        seed=42,
        suppress_errors=suppress,
        colors=clrs,
        data=data_no_rw,
    )
    fig_dbg.savefig(
        out_dir / "debug_E_pred_vs_true_examples_1A.pdf", bbox_inches="tight"
    )
    plt.close(fig_dbg)
    print("Saved:", out_dir / "debug_E_pred_vs_true_examples_1A.pdf")

    fig_1A, vals_1A = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        q3_bins=q3_bins_paper,
        use_cc_selection=2,
        suppress_errors=suppress,
        return_values=True,
        colors=clrs,
        data=data_no_rw,
        **iqr_legend_kw,
    )
    plt.close(fig_1A)

    fig_1B, vals_1B = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1B"],
        dataset_to_plot="1B",
        baseline_run=baseline_run,
        q3_bins=q3_bins_paper,
        use_cc_selection=2,
        suppress_errors=suppress,
        return_values=True,
        colors=clrs,
        data=data_no_rw,
        **iqr_legend_kw,
    )
    plt.close(fig_1B)

    q3 = vals_1A["q3_bin_mids"]
    fig_both, ax = plt.subplots(figsize=(4.8, 3.9))
    # Color = model; linestyle = playlist (1A dashed, 1B solid). Separate legends.
    # Curves use seed-averaged IQR/MPV (same as plot_rms_iqr_with_uncertainty).
    model_proxies: dict[str, Line2D] = {}
    for loss in vals_1A:
        if loss in ("q3_bin_mids", "baseline"):
            continue
        for cfg_label, vA in vals_1A[loss].items():
            method = cfg_label.split()[0]
            color = clrs.get(method, "tab:gray")
            lab = label_fn(method) if label_fn is not None else plot_model_label(method)
            with np.errstate(divide="ignore", invalid="ignore"):
                yA = vA["iqr_mean"] / vA["mpv_mean"]
            ax.plot(q3, yA, "--", color=color)
            vB = vals_1B.get(loss, {}).get(cfg_label)
            if vB is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    yB = vB["iqr_mean"] / vB["mpv_mean"]
                ax.plot(q3, yB, "-", color=color)
            if lab not in model_proxies:
                model_proxies[lab] = Line2D(
                    [], [], color=color, linestyle="-", linewidth=2.0
                )

    if "baseline" in vals_1A:
        blA = vals_1A["baseline"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(q3, blA["iqr"] / blA["mpv"], "k--")
    if "baseline" in vals_1B:
        blB = vals_1B["baseline"]
        with np.errstate(divide="ignore", invalid="ignore"):
            ax.plot(q3, blB["iqr"] / blB["mpv"], "k-")
    if "baseline" in vals_1A or "baseline" in vals_1B:
        model_proxies.setdefault(
            "Baseline", Line2D([], [], color="k", linestyle="-", linewidth=2.0)
        )

    if iqr_legend_order:
        model_labels = [lab for lab in iqr_legend_order if lab in model_proxies]
        model_labels.extend(
            lab for lab in model_proxies if lab not in model_labels
        )
    else:
        model_labels = list(model_proxies.keys())
    model_handles = [model_proxies[lab] for lab in model_labels]

    ax.set(
        xlabel=r"True $q_3$ [GeV]",
        ylabel=(
            r"IQR / MPV of "
            r"$E_{\mathrm{available}}^{\mathrm{reco}}/"
            r"E_{\mathrm{available}}^{\mathrm{true}}$"
        ),
    )
    style_handles = [
        Line2D([], [], color="0.45", linestyle="--", linewidth=2.0),
        Line2D([], [], color="0.45", linestyle="-", linewidth=2.0),
    ]
    # Stack: 1A/1B style box on top, model colors directly below (upper right).
    leg_style = ax.legend(
        style_handles,
        ["1A", "1B"],
        fontsize=9,
        loc="upper right",
        ncol=2,
        framealpha=0.9,
    )
    ax.add_artist(leg_style)
    ax.grid(True)
    fig_both.tight_layout()
    fig_both.canvas.draw()
    renderer = fig_both.canvas.get_renderer()
    style_bb = leg_style.get_window_extent(renderer)
    # ~8 pt clear air between the two legend frames (display pixels).
    gap_px = 8.0 * fig_both.dpi / 72.0
    x1, y_anchor = ax.transAxes.inverted().transform(
        (style_bb.x1, style_bb.y0 - gap_px)
    )
    ax.legend(
        model_handles,
        model_labels,
        fontsize=9,
        loc="upper right",
        bbox_to_anchor=(x1, y_anchor),
        borderaxespad=0.0,
        framealpha=0.9,
    )
    fig_both.savefig(out_dir / "q3_vs_iqr_rms_full_1A_1B.pdf", bbox_inches="tight")
    plt.close(fig_both)
    print("Saved:", out_dir / "q3_vs_iqr_rms_full_1A_1B.pdf")

    fig_ii = plot_residuals_by_energy(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_first_only,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        suppress_errors=suppress,
        data=data_first,
    )
    fig_ii.savefig(out_dir / "residuals_by_energy_full.pdf", bbox_inches="tight")
    plt.close(fig_ii)
    print("Saved:", out_dir / "residuals_by_energy_full.pdf")

    fig_ii_q3 = plot_residuals_by_q3(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_first_only,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        q3_bins=[0, 0.6, 1.2, 1.8, 2.4],
        colors=clrs,
        suppress_errors=suppress,
        data=data_first,
        **residuals_legend_kw,
    )
    fig_ii_q3.savefig(
        out_dir / "residuals_by_q3_select_events_by_E_recoil_CCinc.pdf",
        bbox_inches="tight",
    )
    plt.close(fig_ii_q3)
    print("Saved:", out_dir / "residuals_by_q3_select_events_by_E_recoil_CCinc.pdf")

    fig_ii_q3 = plot_residuals_by_q3(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_first_only,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        q3_bins=[0, 0.6, 1.2, 1.8, 2.4],
        colors=clrs,
        suppress_errors=suppress,
        use_cc_selection=1,
        data=data_first,
        **residuals_legend_kw,
    )
    fig_ii_q3.savefig(
        out_dir / "residuals_by_q3_select_events_by_muons.pdf", bbox_inches="tight"
    )
    plt.close(fig_ii_q3)
    print("Saved:", out_dir / "residuals_by_q3_select_events_by_muons.pdf")

    training_names_grouped: dict = {"Log1p": {}}
    for model, caps in runs_by_model_cap.items():
        for cap, run_list in sorted(
            caps.items(), key=lambda x: (x[0] == -1, -x[0] if x[0] > 0 else 0)
        ):
            if not run_list:
                continue
            label = f"{model} {_cap_to_label(cap)}"
            training_names_grouped["Log1p"][label] = run_list

    if not training_names_grouped["Log1p"]:
        raise SystemExit("No runs for scaling plot.")

    _, values = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_grouped,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        return_values=True,
        suppress_errors=suppress,
        data=data_scaling,
    )
    plt.close("all")

    Q3_BIN_IDX = 2

    def _n_samples(label: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)\s*([kKmM])\b", label)
        if m:
            return float(m.group(1)) * (1e6 if m.group(2).upper() == "M" else 1e3)
        return 0.0

    def _method(label: str) -> str:
        return re.sub(r"\s+\d+(?:\.\d+)?[kKmM]$", "", label)

    methods: dict[str, list] = {}
    for loss in values:
        if loss in ("q3_bin_mids", "baseline"):
            continue
        for cfg_label, v in values[loss].items():
            methods.setdefault(_method(cfg_label), []).append(
                (
                    _n_samples(cfg_label),
                    v["iqr_mean"][Q3_BIN_IDX],
                    v["iqr_std"][Q3_BIN_IDX],
                )
            )

    for m in methods:
        methods[m].sort()

    fig_iii, axs = plt.subplots(figsize=(7, 5))
    for method, pts in methods.items():
        xs, ys, yerrs = zip(*pts)
        axs.errorbar(
            xs, ys, yerr=yerrs, marker="o", capsize=4, label=plot_model_label(method)
        )

    if "baseline" in values:
        axs.axhline(
            values["baseline"]["iqr"][Q3_BIN_IDX],
            color="black",
            linestyle=":",
            lw=1,
            label="Baseline",
        )

    axs.set_xscale("log")
    axs.set_xlabel("Number of training samples")
    axs.set_ylabel("IQR [GeV]")
    q3_mid = values["q3_bin_mids"][Q3_BIN_IDX]
    axs.set_title(
        f"IQR vs Training Samples (q$_3$ bin {Q3_BIN_IDX}, mid = {q3_mid:.2f} GeV)"
    )
    axs.legend(fontsize=9)
    axs.grid(True)
    fig_iii.tight_layout()
    fig_iii.savefig(out_dir / "iqr_vs_n_samples_bin2.pdf", bbox_inches="tight")
    plt.close(fig_iii)
    print("Saved:", out_dir / "iqr_vs_n_samples_bin2.pdf")


if __name__ == "__main__":
    main()
