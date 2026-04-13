#!/usr/bin/env python3
"""Classification vs pion kinematics (E, θ), CC1π⁰, π⁰ Δm scan; plus light appendix PDFs.

Pickles default under ``<repo>/<--out-dir>/eval_data/``; PDFs under
``<repo>/<--plots-dir>/classification/pions/`` and ``.../classification/light/``
(per playlist: ``eval_classification_light_{cc1pi,cc1pi0}_{q3,W,pion_kinematics}_<pl>.pdf``).
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
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    add_hadronic_W_to_classification_data,
    compute_all_metrics,
    compute_all_metrics_W,
    compute_reco_baseline_recall_per_bin,
    compute_signal_baseline,
    compute_signal_baseline_W,
    data_with_signal_pion_bins,
    plot_binned_by_inttype,
    plot_cc1pi_vs_pion_kinematics,
    plot_multi_classification_vs_W,
    plot_prc_curves,
    save_figures_to_pdf,
)
from src.eval._bootstrap import silence_classification_empty_bin_warnings
from src.eval._classification_light import save_light_classification_pdfs
from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_OUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_WANDB_TAG,
    EVAL_DATA_SUBDIR,
    repo_output_path,
)


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / EVAL_DATA_SUBDIR / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl"


PI0_MASS = 134.977


def get_pi0_baseline_pred(is_pizero_signal, two_gamma_inv_mass, delta_m, n_muons, n_michel):
    has_candidate = is_pizero_signal == 2
    in_mass_window = np.abs(two_gamma_inv_mass - PI0_MASS) < delta_m
    return ((n_muons == 1) & has_candidate & in_mass_window & (n_michel == 0)).astype(int)


def precision_recall_fpr_vs_deltam(
    y_true, is_pizero_signal, two_gamma_inv_mass, delta_m_values, n_muons, n_michel,
):
    precisions, recalls, fprs = [], [], []
    for dm in delta_m_values:
        y_pred = get_pi0_baseline_pred(is_pizero_signal, two_gamma_inv_mass, dm, n_muons, n_michel)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        fprs.append(fpr)
    return np.array(precisions), np.array(recalls), np.array(fprs)


def main(argv: list[str] | None = None) -> None:
    silence_classification_empty_bin_warnings()
    _ = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE
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
    args = ap.parse_args(argv)

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    out_dir = plots_root / "classification" / "pions"
    out_dir.mkdir(parents=True, exist_ok=True)
    light_dir = plots_root / "classification" / "light"
    light_dir.mkdir(parents=True, exist_ok=True)

    pkl = args.classification_pickle or _pickle_path(data_root, args.flag)
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    results = clf["results"]
    data_by_playlist = clf["data_by_playlist"]
    data_w_by_playlist = clf.get("data_w_by_playlist")
    clrs = clf["clrs_dict_full"]
    playlists = clf["playlists"]

    cc1pi_classes = [0]
    cc1pi0_classes = [2]
    DELTA_M = PI0_MASS

    for playlist in playlists:
        data = data_by_playlist[playlist]
        figs_cc1pi = []
        test_idx = data["test_idx"][playlist]
        baselines_pl = data["baselines"][playlist]
        n_muons = baselines_pl["n_muons"][test_idx]
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        first_model = next(iter(results))
        run0 = results[first_model][0][playlist]
        pid = run0["pid"]
        data_cc1pi = data_with_signal_pion_bins(
            data, pid, cc1pi_classes,
            pion_quantile_require_has_pion=False,
            pion_bin_edge_method="equal_frequency",
        )
        y_true_cc1pi = np.isin(pid, cc1pi_classes).astype(int)
        y_pred_cc1pi = (
            (n_muons == 1) & (n_charged_prongs == 1) & (improved_nmichel == 1)
        ).astype(int)
        tp = np.sum((y_pred_cc1pi == 1) & (y_true_cc1pi == 1))
        fp = np.sum((y_pred_cc1pi == 1) & (y_true_cc1pi == 0))
        fn = np.sum((y_pred_cc1pi == 0) & (y_true_cc1pi == 1))
        tn = np.sum((y_pred_cc1pi == 0) & (y_true_cc1pi == 0))
        baseline_fpr_cc1pi = fp / (fp + tn)

        metrics_cc1pi = compute_all_metrics(
            results, data_cc1pi, signal_classes=cc1pi_classes, fixed_fpr=[baseline_fpr_cc1pi],
            playlist=playlist, pion_bins_require_has_pion=False,
        )
        baseline_cc1pi = compute_signal_baseline(
            results, data_cc1pi, signal_classes=cc1pi_classes, playlist=playlist,
            pion_bins_require_has_pion=False,
        )
        is_signal_cc1pi = y_true_cc1pi == 1
        reco_baseline_tpr_cc1pi = {
            "E": compute_reco_baseline_recall_per_bin(
                y_pred_cc1pi, is_signal_cc1pi,
                data_cc1pi["pion_E_MC"], data_cc1pi["pion_E_MC_bins"], has_pion=None,
            ),
            "theta": compute_reco_baseline_recall_per_bin(
                y_pred_cc1pi, is_signal_cc1pi,
                data_cc1pi["pion_theta_MC"], data_cc1pi["pion_theta_MC_bins"],
                has_pion=None, finite_bin_var=True,
            ),
        }
        data_cc1pi_w = add_hadronic_W_to_classification_data(data_cc1pi, playlist)
        metrics_W_cc1pi = compute_all_metrics_W(
            results, data_cc1pi_w, signal_classes=cc1pi_classes, fixed_fpr=[baseline_fpr_cc1pi],
            playlist=playlist,
        )
        baseline_W_cc1pi = compute_signal_baseline_W(
            results, data_cc1pi_w, signal_classes=cc1pi_classes, playlist=playlist,
        )
        reco_baseline_tpr_W_cc1pi = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi, is_signal_cc1pi, data_cc1pi_w["W_GeV"], data_cc1pi_w["W_bin_edges"],
        )
        fig_w_cc1pi = plot_multi_classification_vs_W(
            metrics_W_cc1pi, data_cc1pi_w, baseline_W_cc1pi,
            fixed_fpr=[baseline_fpr_cc1pi], uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_cc1pi,
            reco_baseline_global_fpr=baseline_fpr_cc1pi,
            reco_baseline_pred=y_pred_cc1pi,
            colors=clrs,
            title=fr"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist, use_global_fpr=True, results=results, signal_classes=cc1pi_classes,
        )
        figs_cc1pi.append(fig_w_cc1pi)
        plt.close(fig_w_cc1pi)

        fig = plot_cc1pi_vs_pion_kinematics(
            metrics_cc1pi, data_cc1pi, baseline_cc1pi, uncertainties=True,
            fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_tpr=reco_baseline_tpr_cc1pi,
            reco_baseline_pred=y_pred_cc1pi,
            reco_baseline_global_fpr=baseline_fpr_cc1pi,
            results=results, signal_classes=cc1pi_classes, colors=clrs, playlist=playlist,
        )
        figs_cc1pi.append(fig)
        plt.close(fig)

        fig = plot_prc_curves(
            results, signal_classes=cc1pi_classes,
            title=fr"PRC — $CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist, colors=clrs, uncertainties=True,
        )
        figs_cc1pi.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data_cc1pi, signal_classes=cc1pi_classes, x_var="pion_E", xlabel=r"True $E_\pi$ [GeV]",
            title=fr"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            log_x=True, uncertainties=True, fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_pred=y_pred_cc1pi, playlist=playlist, colors=clrs,
            pion_bins_require_has_pion=False,
        )
        figs_cc1pi.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data_cc1pi, signal_classes=cc1pi_classes, x_var="pion_theta", xlabel=r"True $\theta_\pi$ [rad]",
            title=fr"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_cc1pi], reco_baseline_pred=y_pred_cc1pi,
            playlist=playlist, colors=clrs, pion_bins_require_has_pion=False,
        )
        figs_cc1pi.append(fig)
        plt.close(fig)
        save_figures_to_pdf(figs_cc1pi, out_dir / f"eval_cc1pi_tagging_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_cc1pi_tagging_{playlist}.pdf")

    for playlist in playlists:
        data = data_by_playlist[playlist]
        figs_pi0 = []
        test_idx = data["test_idx"][playlist]
        baselines_pl = data["baselines"][playlist]
        n_muons = baselines_pl["n_muons"][test_idx]
        is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
        two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
        n_michel = baselines_pl["improved_nmichel"][test_idx]
        first_model = next(iter(results))
        run0 = results[first_model][0][playlist]
        pid = run0["pid"]
        data_pi0 = data_with_signal_pion_bins(
            data, pid, cc1pi0_classes,
            pion_quantile_require_has_pion=False,
            pion_bin_edge_method="equal_frequency",
        )
        y_true_pi0 = np.isin(pid, cc1pi0_classes).astype(int)
        y_pred_baseline = (
            (n_muons == 1) & (is_pizero_signal == 2)
            & (np.abs(two_gamma_inv_mass - PI0_MASS) < DELTA_M)
            & (n_michel == 0)
        ).astype(int)
        tp_g = np.sum((y_pred_baseline == 1) & (y_true_pi0 == 1))
        fp_g = np.sum((y_pred_baseline == 1) & (y_true_pi0 == 0))
        fn_g = np.sum((y_pred_baseline == 0) & (y_true_pi0 == 1))
        tn_g = np.sum((y_pred_baseline == 0) & (y_true_pi0 == 0))
        baseline_fpr = fp_g / (fp_g + tn_g)

        metrics_cc1pi0 = compute_all_metrics(
            results, data_pi0, signal_classes=cc1pi0_classes, fixed_fpr=[baseline_fpr],
            playlist=playlist, pion_bins_require_has_pion=False,
        )
        baseline_cc1pi0 = compute_signal_baseline(
            results, data_pi0, signal_classes=cc1pi0_classes, playlist=playlist,
            pion_bins_require_has_pion=False,
        )
        is_signal_pi0 = y_true_pi0 == 1
        reco_baseline_tpr = {
            "E": compute_reco_baseline_recall_per_bin(
                y_pred_baseline, is_signal_pi0,
                data_pi0["pion_E_MC"], data_pi0["pion_E_MC_bins"], has_pion=None,
            ),
            "theta": compute_reco_baseline_recall_per_bin(
                y_pred_baseline, is_signal_pi0,
                data_pi0["pion_theta_MC"], data_pi0["pion_theta_MC_bins"],
                has_pion=None, finite_bin_var=True,
            ),
        }
        data_pi0_w = add_hadronic_W_to_classification_data(data_pi0, playlist)
        metrics_W_pi0 = compute_all_metrics_W(
            results, data_pi0_w, signal_classes=cc1pi0_classes, fixed_fpr=[baseline_fpr],
            playlist=playlist,
        )
        baseline_W_pi0 = compute_signal_baseline_W(
            results, data_pi0_w, signal_classes=cc1pi0_classes, playlist=playlist,
        )
        reco_baseline_tpr_W_pi0 = compute_reco_baseline_recall_per_bin(
            y_pred_baseline, is_signal_pi0, data_pi0_w["W_GeV"], data_pi0_w["W_bin_edges"],
        )
        fig_w_pi0 = plot_multi_classification_vs_W(
            metrics_W_pi0, data_pi0_w, baseline_W_pi0,
            fixed_fpr=[baseline_fpr], uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_pi0,
            reco_baseline_global_fpr=baseline_fpr,
            reco_baseline_pred=y_pred_baseline,
            colors=clrs,
            title=fr"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist, use_global_fpr=True, results=results, signal_classes=cc1pi0_classes,
        )
        figs_pi0.append(fig_w_pi0)
        plt.close(fig_w_pi0)

        fig = plot_cc1pi_vs_pion_kinematics(
            metrics_cc1pi0, data_pi0, baseline_cc1pi0, uncertainties=True,
            fixed_fpr=[baseline_fpr], reco_baseline_tpr=reco_baseline_tpr,
            reco_baseline_pred=y_pred_baseline, reco_baseline_global_fpr=baseline_fpr,
            results=results, signal_classes=cc1pi0_classes, colors=clrs,
            suptitle=fr"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs_pi0.append(fig)
        plt.close(fig)

        fig2 = plot_binned_by_inttype(
            results, data_pi0, signal_classes=cc1pi0_classes, x_var="pion_E", xlabel=r"True $E_\pi$ [GeV]",
            title=fr"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            log_x=True, uncertainties=True, fixed_fpr=[baseline_fpr],
            reco_baseline_pred=y_pred_baseline, playlist=playlist, colors=clrs,
            pion_bins_require_has_pion=False,
        )
        figs_pi0.append(fig2)
        plt.close(fig2)

        fig3 = plot_binned_by_inttype(
            results, data_pi0, signal_classes=cc1pi0_classes, x_var="pion_theta", xlabel=r"True $\theta_\pi$ [rad]",
            title=fr"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr], reco_baseline_pred=y_pred_baseline,
            playlist=playlist, colors=clrs, pion_bins_require_has_pion=False,
        )
        figs_pi0.append(fig3)
        plt.close(fig3)

        fig4 = plot_prc_curves(
            results, signal_classes=cc1pi0_classes,
            title=fr"PRC — $CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist, uncertainties=True, colors=clrs,
        )
        figs_pi0.append(fig4)
        plt.close(fig4)
        save_figures_to_pdf(figs_pi0, out_dir / f"eval_cc1pi0_tagging_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_cc1pi0_tagging_{playlist}.pdf")

    delta_m_values = np.linspace(1, 509, 20)
    for playlist in playlists:
        data = data_by_playlist[playlist]
        test_idx = data["test_idx"][playlist]
        baselines_pl = data["baselines"][playlist]
        is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
        two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
        n_muons_b = baselines_pl["n_muons"][test_idx]
        n_michel = baselines_pl["improved_nmichel"][test_idx]
        first_model = next(iter(results))
        run0 = results[first_model][0][playlist]
        pid = run0["pid"]
        y_true = np.isin(pid, cc1pi0_classes).astype(int)
        precisions, recalls, fprs = precision_recall_fpr_vs_deltam(
            y_true, is_pizero_signal, two_gamma_inv_mass, delta_m_values, n_muons_b, n_michel,
        )
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(delta_m_values, precisions, ".-", color="steelblue", linewidth=1.5, label="Precision")
        ax1.plot(delta_m_values, recalls, ".-", color="darkorange", linewidth=1.5, label="Recall")
        ax1.plot(delta_m_values, fprs, ".-", color="firebrick", linewidth=1.5, label="FPR")
        ax1.set_xlabel(r"$\Delta m$ window [MeV]")
        ax1.set_ylabel(r"Fraction")
        ax1.legend(loc="center right")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(r"$CC\pi^0$ baseline vs. $\Delta m$ window")
        fig.tight_layout()
        fp = out_dir / f"pi0_baseline_deltam_{playlist}.pdf"
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", fp)

    save_light_classification_pdfs(
        light_dir,
        results,
        data_by_playlist,
        clrs,
        playlists,
        components=("pion",),
        data_w_by_playlist=data_w_by_playlist,
    )


if __name__ == "__main__":
    main()
