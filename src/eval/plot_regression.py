#!/usr/bin/env python3
"""Regression evaluation PDFs (IQR vs q3, residuals, scaling) from cached pickle.

Pickles default under ``<repo>/<--out-dir>/``; PDFs under
``<repo>/<--plots-dir>/regression/``.
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.e_available_plots import (
    plot_example_E_pred_true,
    plot_residuals_by_energy,
    plot_residuals_by_q3,
    plot_rms_iqr_with_uncertainty,
)
from src.eval._constants import (
    DEFAULT_OUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_WANDB_TAG,
    REGRESSION_PICKLE_STEM,
    repo_output_path,
)


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
        default=DEFAULT_PLOTS_DIR,
        help="Root for PDF output (default: plots/ under repo)",
    )
    ap.add_argument("--regression-pickle", type=Path, default=None)
    args = ap.parse_args(argv)

    _maybe_register_arial()

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    out_dir = plots_root / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl = args.regression_pickle or _pickle_path(data_root, args.flag)
    with open(pkl, "rb") as f:
        reg = pickle.load(f)

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

    fig_i = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        q3_bins=[0, 0.6, 1.2, 1.8, 2.4, 3.0, 100],
        show_q3_histograms=True,
        return_hist_fig=False,
        suppress_errors=suppress,
        use_cc_selection=2,
        colors=clrs,
        text="",
        data=data_no_rw,
    )
    fig_i.savefig(out_dir / "q3_vs_iqr_rms_full_1A.pdf", bbox_inches="tight")
    plt.close(fig_i)
    print("Saved:", out_dir / "q3_vs_iqr_rms_full_1A.pdf")

    fig_i = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=baseline_run,
        q3_bins=[0, 0.6, 1.2, 1.8, 2.4, 3.0, 100],
        show_q3_histograms=True,
        return_hist_fig=False,
        suppress_errors=suppress,
        use_cc_selection=1,
        colors=clrs,
        text="",
        data=data_no_rw,
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
        suppress_errors=suppress,
        return_values=True,
        data=data_no_rw,
    )
    plt.close(fig_1A)

    fig_1B, vals_1B = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_full_no_rw,
        playlists=["1B"],
        dataset_to_plot="1B",
        baseline_run=baseline_run,
        suppress_errors=suppress,
        return_values=True,
        data=data_no_rw,
    )
    plt.close(fig_1B)

    q3 = vals_1A["q3_bin_mids"]
    fig_both, ax = plt.subplots(figsize=(5.5, 4.5))
    for loss in vals_1A:
        if loss in ("q3_bin_mids", "baseline"):
            continue
        for cfg_label, vA in vals_1A[loss].items():
            method = cfg_label.split()[0]
            color = clrs.get(method, "tab:gray")
            ax.plot(q3, vA["iqr_mean"], "--", color=color, label=f"{method} (1A)")
            vB = vals_1B.get(loss, {}).get(cfg_label)
            if vB is not None:
                ax.plot(q3, vB["iqr_mean"], "-", color=color, label=f"{method} (1B)")

    if "baseline" in vals_1A:
        bl = vals_1A["baseline"]
        ax.plot(q3, bl["iqr"], "k:", label="Baseline (1A)")
    if "baseline" in vals_1B:
        bl = vals_1B["baseline"]
        ax.plot(q3, bl["iqr"], "k--", label="Baseline (1B)")

    ax.set(
        xlabel="MC truth $q_3$ [GeV]",
        ylabel=r"25-75 IQR of $E_{\mathrm{available}}^{\mathrm{reco}}/E_{\mathrm{available}}^{\mathrm{true}}$",
    )
    selection_text = ""
    leg1 = ax.legend(title=selection_text, fontsize=9, loc="upper right")
    leg1.set_title(selection_text)
    ax.grid(True)
    fig_both.tight_layout()
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
    )
    fig_ii_q3.savefig(
        out_dir / "residuals_by_q3_select_events_by_muons.pdf", bbox_inches="tight"
    )
    plt.close(fig_ii_q3)
    print("Saved:", out_dir / "residuals_by_q3_select_events_by_muons.pdf")

    training_names_grouped = {"Log1p": {}}
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
        for cfg, v in values[loss].items():
            methods.setdefault(_method(cfg), []).append(
                (_n_samples(cfg), v["iqr_mean"][Q3_BIN_IDX], v["iqr_std"][Q3_BIN_IDX])
            )

    for m in methods:
        methods[m].sort()

    fig_iii, axs = plt.subplots(figsize=(7, 5))
    for method, pts in methods.items():
        xs, ys, yerrs = zip(*pts)
        axs.errorbar(xs, ys, yerr=yerrs, marker="o", capsize=4, label=method)

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
