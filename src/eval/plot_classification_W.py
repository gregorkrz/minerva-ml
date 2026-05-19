#!/usr/bin/env python3
"""Classification vs hadronic W (``Eval_Classification_W`` + ``W_Npi_Ng1`` parity).

Pickles default under ``<repo>/<--out-dir>/``; PDFs under
``<repo>/<--plots-dir>/classification/w_bins/``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.classification_plots import (
    compute_all_metrics_W,
    compute_reco_baseline_recall_per_bin,
    compute_signal_baseline_W,
    plot_binned_by_inttype,
    plot_composition_vs_kinematic,
    plot_multi_classification_vs_W,
    save_figures_to_pdf,
)
from src.eval._bootstrap import silence_classification_empty_bin_warnings
from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_OUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_WANDB_TAG,
    filter_classification_results_for_standard_plots,
    repo_output_path,
)


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl"


def _run_w_notebook_style(
    results: dict,
    data_w_by_playlist: dict,
    clrs: dict[str, str],
    use_global_fpr: bool,
    out_dir: Path,
    playlists: list[str],
    tag_suffix: str,
) -> None:
    _PDF_FPR_TAG = "_global_fpr" if use_global_fpr else "_per_bin_fpr"
    cc1pi_classes = [0]
    cc1pi0_classes = [2]
    PI0_MASS_MEV = 134.977
    DELTA_M_MEV = PI0_MASS_MEV
    multi_pi_classes = [0, 1]

    for playlist in playlists:
        data_w = data_w_by_playlist[playlist]

        # ----- CC1pi -----
        figs_cc1pi_W = []
        test_idx = data_w["test_idx"][playlist]
        baselines_pl = data_w["baselines"][playlist]
        n_muons = baselines_pl["n_muons"][test_idx]
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        first_model = next(iter(results))
        run0 = results[first_model][0][playlist]
        pid = run0["pid"]

        y_true_cc1pi = np.isin(pid, cc1pi_classes).astype(int)
        y_pred_cc1pi = (
            (n_muons == 1) & (n_charged_prongs == 1) & (improved_nmichel == 1)
        ).astype(int)
        tp = np.sum((y_pred_cc1pi == 1) & (y_true_cc1pi == 1))
        fp = np.sum((y_pred_cc1pi == 1) & (y_true_cc1pi == 0))
        fn = np.sum((y_pred_cc1pi == 0) & (y_true_cc1pi == 1))
        tn = np.sum((y_pred_cc1pi == 0) & (y_true_cc1pi == 0))
        baseline_fpr_cc1pi = fp / (fp + tn)

        metrics_W_cc1pi = compute_all_metrics_W(
            results,
            data_w,
            signal_classes=cc1pi_classes,
            fixed_fpr=[baseline_fpr_cc1pi],
            playlist=playlist,
            use_global_fpr=use_global_fpr,
        )
        baseline_W_cc1pi = compute_signal_baseline_W(
            results,
            data_w,
            signal_classes=cc1pi_classes,
            playlist=playlist,
        )
        is_signal_cc1pi = y_true_cc1pi == 1
        reco_baseline_tpr_W_cc1pi = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi,
            is_signal_cc1pi,
            data_w["W_GeV"],
            data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W_cc1pi,
            data_w,
            baseline_W_cc1pi,
            fixed_fpr=[baseline_fpr_cc1pi],
            uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_cc1pi,
            reco_baseline_global_fpr=baseline_fpr_cc1pi,
            colors=clrs,
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr,
            playlist=playlist,
        )
        figs_cc1pi_W.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data_w,
            signal_classes=cc1pi_classes,
            x_var="W",
            xlabel=r"$W$ [GeV]",
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_pred=y_pred_cc1pi,
            playlist=playlist,
            colors=clrs,
            use_global_fpr=use_global_fpr,
        )
        figs_cc1pi_W.append(fig)
        plt.close(fig)
        save_figures_to_pdf(
            figs_cc1pi_W,
            out_dir / f"eval_cc1pi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf",
        )
        print(
            "Saved:",
            out_dir / f"eval_cc1pi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf",
        )

        # ----- CC1pi0 -----
        figs_cc1pi0_W = []
        is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
        two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
        improved_nmichel0 = baselines_pl["improved_nmichel"][test_idx]
        y_true_cc1pi0 = np.isin(pid, cc1pi0_classes).astype(int)
        y_pred_cc1pi0 = (
            (n_muons == 1)
            & (is_pizero_signal == 2)
            & (np.abs(two_gamma_inv_mass - PI0_MASS_MEV) < DELTA_M_MEV)
            & (improved_nmichel0 == 0)
        ).astype(int)
        tp0 = np.sum((y_pred_cc1pi0 == 1) & (y_true_cc1pi0 == 1))
        fp0 = np.sum((y_pred_cc1pi0 == 1) & (y_true_cc1pi0 == 0))
        fn0 = np.sum((y_pred_cc1pi0 == 0) & (y_true_cc1pi0 == 1))
        tn0 = np.sum((y_pred_cc1pi0 == 0) & (y_true_cc1pi0 == 0))
        baseline_fpr_cc1pi0 = fp0 / (fp0 + tn0) if (fp0 + tn0) > 0 else float("nan")

        metrics_W_cc1pi0 = compute_all_metrics_W(
            results,
            data_w,
            signal_classes=cc1pi0_classes,
            fixed_fpr=[baseline_fpr_cc1pi0],
            playlist=playlist,
            use_global_fpr=use_global_fpr,
        )
        baseline_W_cc1pi0 = compute_signal_baseline_W(
            results,
            data_w,
            signal_classes=cc1pi0_classes,
            playlist=playlist,
        )
        is_signal_cc1pi0 = y_true_cc1pi0 == 1
        reco_baseline_tpr_W_cc1pi0 = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi0,
            is_signal_cc1pi0,
            data_w["W_GeV"],
            data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W_cc1pi0,
            data_w,
            baseline_W_cc1pi0,
            fixed_fpr=[baseline_fpr_cc1pi0],
            uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_cc1pi0,
            reco_baseline_global_fpr=baseline_fpr_cc1pi0,
            colors=clrs,
            title=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr,
            playlist=playlist,
        )
        figs_cc1pi0_W.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data_w,
            signal_classes=cc1pi0_classes,
            x_var="W",
            xlabel=r"$W$ [GeV]",
            title=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr_cc1pi0],
            reco_baseline_pred=y_pred_cc1pi0,
            playlist=playlist,
            colors=clrs,
            use_global_fpr=use_global_fpr,
        )
        figs_cc1pi0_W.append(fig)
        plt.close(fig)
        save_figures_to_pdf(
            figs_cc1pi0_W,
            out_dir / f"eval_cc1pi0_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf",
        )
        print(
            "Saved:",
            out_dir / f"eval_cc1pi0_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf",
        )

        # ----- CCNpi N>=1 -----
        figs_npi_W = []
        y_true_ccnpi = np.isin(pid, multi_pi_classes).astype(int)
        y_pred_ccnpi = (
            (n_muons == 1) & (n_charged_prongs >= 1) & (improved_nmichel >= 1)
        ).astype(int)
        tpn = np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 1))
        fpn = np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 0))
        fnn = np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 1))
        tnn = np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 0))
        baseline_fpr_ccnpi = fpn / (fpn + tnn)

        metrics_W = compute_all_metrics_W(
            results,
            data_w,
            signal_classes=multi_pi_classes,
            fixed_fpr=[baseline_fpr_ccnpi],
            playlist=playlist,
            use_global_fpr=use_global_fpr,
        )
        baseline_W_multi = compute_signal_baseline_W(
            results,
            data_w,
            signal_classes=multi_pi_classes,
            playlist=playlist,
        )
        is_signal_ccnpi = y_true_ccnpi == 1
        reco_baseline_tpr_W = compute_reco_baseline_recall_per_bin(
            y_pred_ccnpi,
            is_signal_ccnpi,
            data_w["W_GeV"],
            data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W,
            data_w,
            baseline_W_multi,
            fixed_fpr=[baseline_fpr_ccnpi],
            uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W,
            reco_baseline_global_fpr=baseline_fpr_ccnpi,
            colors=clrs,
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr,
            playlist=playlist,
        )
        figs_npi_W.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data_w,
            signal_classes=multi_pi_classes,
            x_var="W",
            xlabel=r"$W$ [GeV]",
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr_ccnpi],
            reco_baseline_pred=y_pred_ccnpi,
            playlist=playlist,
            colors=clrs,
            use_global_fpr=use_global_fpr,
        )
        figs_npi_W.append(fig)
        plt.close(fig)
        save_figures_to_pdf(
            figs_npi_W,
            out_dir / f"eval_Npi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf",
        )
        print(
            "Saved:",
            out_dir / f"eval_Npi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf",
        )


def _run_w_ngt1(
    results: dict,
    data_w_by_playlist: dict,
    clrs: dict[str, str],
    use_global_fpr: bool,
    out_dir: Path,
    playlists: list[str],
) -> None:
    _PDF_FPR_TAG = "_global_fpr" if use_global_fpr else "_per_bin_fpr"
    multi_pi_classes = [1]

    for playlist in playlists:
        data_w = data_w_by_playlist[playlist]
        figs_npi_W = []
        test_idx = data_w["test_idx"][playlist]
        baselines_pl = data_w["baselines"][playlist]
        n_muons = baselines_pl["n_muons"][test_idx]
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        first_model = next(iter(results))
        run0 = results[first_model][0][playlist]
        pid = run0["pid"]

        y_true_ccnpi = np.isin(pid, multi_pi_classes).astype(int)
        y_pred_ccnpi = (
            (n_muons == 1) & (n_charged_prongs >= 2) & (improved_nmichel >= 1)
        ).astype(int)
        tp = np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 1))
        fp = np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 0))
        fn = np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 1))
        tn = np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 0))
        baseline_fpr_ccnpi = fp / (fp + tn)

        metrics_W = compute_all_metrics_W(
            results,
            data_w,
            signal_classes=multi_pi_classes,
            fixed_fpr=[baseline_fpr_ccnpi],
            playlist=playlist,
            use_global_fpr=use_global_fpr,
        )
        baseline_W_multi = compute_signal_baseline_W(
            results,
            data_w,
            signal_classes=multi_pi_classes,
            playlist=playlist,
        )
        is_signal_ccnpi = y_true_ccnpi == 1
        reco_baseline_tpr_W = compute_reco_baseline_recall_per_bin(
            y_pred_ccnpi,
            is_signal_ccnpi,
            data_w["W_GeV"],
            data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W,
            data_w,
            baseline_W_multi,
            fixed_fpr=[baseline_fpr_ccnpi],
            uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W,
            reco_baseline_global_fpr=baseline_fpr_ccnpi,
            colors=clrs,
            title=rf"$CCN\pi^\pm$ tagging ($N > 1$) - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr,
            playlist=playlist,
        )
        figs_npi_W.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data_w,
            signal_classes=multi_pi_classes,
            x_var="W",
            xlabel=r"$W$ [GeV]",
            title=rf"$CCN\pi^\pm$ tagging ($N > 1$) - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr_ccnpi],
            reco_baseline_pred=y_pred_ccnpi,
            playlist=playlist,
            colors=clrs,
            use_global_fpr=use_global_fpr,
        )
        figs_npi_W.append(fig)
        plt.close(fig)
        save_figures_to_pdf(
            figs_npi_W,
            out_dir / f"eval_Npi_Ngt1_tagging_W_{playlist}{_PDF_FPR_TAG}.pdf",
        )
        print(
            "Saved:", out_dir / f"eval_Npi_Ngt1_tagging_W_{playlist}{_PDF_FPR_TAG}.pdf"
        )


def main(argv: list[str] | None = None) -> None:
    silence_classification_empty_bin_warnings()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory containing classification_<flag>.pkl (default: out/ under repo)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Root for PDF output (default: plots/ under repo)",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    args = ap.parse_args(argv)

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    out_dir = plots_root / "classification" / "w_bins"
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl = args.classification_pickle or _pickle_path(data_root, args.flag)
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    results = filter_classification_results_for_standard_plots(clf["results"])
    data_w_by_playlist = clf["data_w_by_playlist"]
    clrs = clf["clrs_dict_full"]
    use_global_fpr = clf.get("use_global_fpr_W", True)
    playlists = ["1A"]  # match Eval_Classification_W notebook default

    _run_w_notebook_style(
        results,
        data_w_by_playlist,
        clrs,
        use_global_fpr,
        out_dir,
        playlists,
        tag_suffix="",
    )
    _run_w_ngt1(results, data_w_by_playlist, clrs, use_global_fpr, out_dir, playlists)

    # Event-composition (signal + background + S/B vs W) for all three signal
    # definitions — one single-page PDF per playlist.
    for playlist in playlists:
        data_w = data_w_by_playlist[playlist]
        first_model = next(iter(results))
        pid = results[first_model][0][playlist]["pid"]
        fig_comp = plot_composition_vs_kinematic(
            data=data_w,
            pid=pid,
            x_var="W",
            playlist=playlist,
        )
        fp = out_dir / f"event_composition_W_{playlist}.pdf"
        fig_comp.savefig(fp, bbox_inches="tight")
        plt.close(fig_comp)
        print("Saved:", fp)


if __name__ == "__main__":
    main()
