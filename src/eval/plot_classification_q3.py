#!/usr/bin/env python3
"""Classification vs q3, confusion matrices, CCNπ q₃ panels; plus light q₃ appendix PDFs.

Pickles default under ``<repo>/<--out-dir>/eval_data/``; PDFs under
``<repo>/<--plots-dir>/classification/q3/`` and ``.../classification/light/``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.classification_plots import (
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    compute_all_metrics_q3,
    compute_signal_baseline,
    compute_reco_baseline_recall_per_bin,
    plot_binned_by_inttype,
    plot_multi_pion_vs_q3,
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
    out_dir = plots_root / "classification" / "q3"
    out_dir.mkdir(parents=True, exist_ok=True)
    light_dir = plots_root / "classification" / "light"
    light_dir.mkdir(parents=True, exist_ok=True)

    pkl = args.classification_pickle or _pickle_path(data_root, args.flag)
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    results = clf["results"]
    data_by_playlist = clf["data_by_playlist"]
    clrs = clf["clrs_dict_full"]
    playlists = clf["playlists"]

    class_names = ["CC1π±", "CCNπ", "CC1π0", "Other-CC", "Other-NC"]
    model_names = sorted(results.keys())
    n_models = len(model_names)

    for playlist in playlists:
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), tight_layout=True)
        if n_models == 1:
            axes = [axes]
        for ax, model_name in zip(axes, model_names):
            run0 = results[model_name][0][playlist]
            y_true = run0["pid"]
            y_pred = run0["prediction"].argmax(axis=1)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names,
            )
            ax.set_xlabel(r"Predicted class")
            ax.set_ylabel(r"True class")
            ax.set_title(model_name)
        fig.suptitle(f"Confusion matrices (first run per model) — {playlist}", fontsize=14)
        fp = out_dir / f"confusion_matrices_{playlist}.pdf"
        fig.savefig(fp)
        plt.close(fig)
        print("Saved:", fp)

    cc1pi_classes = [0]
    for playlist in playlists:
        data = data_by_playlist[playlist]
        figs_cc1pi_q3 = []
        test_idx = data["test_idx"][playlist]
        baselines_pl = data["baselines"][playlist]
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

        metrics_q3_cc1pi = compute_all_metrics_q3(
            results, data, signal_classes=cc1pi_classes, fixed_fpr=[baseline_fpr_cc1pi],
            playlist=playlist,
        )
        baseline_cc1pi = compute_signal_baseline(results, data, signal_classes=cc1pi_classes, playlist=playlist)
        is_signal_cc1pi = y_true_cc1pi == 1
        reco_baseline_tpr_q3_cc1pi = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi, is_signal_cc1pi, data["q3_GeV"], data["q3_bin_edges"],
        )
        fig = plot_multi_pion_vs_q3(
            metrics_q3_cc1pi, data, baseline_cc1pi["q3"],
            fixed_fpr=[baseline_fpr_cc1pi], uncertainties=True,
            reco_baseline_tpr_q3=reco_baseline_tpr_q3_cc1pi,
            colors=clrs,
            title=fr"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs_cc1pi_q3.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data, signal_classes=cc1pi_classes, x_var="q3", xlabel=r"True $q_3$ [GeV]",
            title=fr"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_cc1pi], reco_baseline_pred=y_pred_cc1pi,
            playlist=playlist, colors=clrs,
        )
        figs_cc1pi_q3.append(fig)
        plt.close(fig)
        save_figures_to_pdf(figs_cc1pi_q3, out_dir / f"eval_cc1pi_tagging_q3_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_cc1pi_tagging_q3_{playlist}.pdf")

    multi_pi_classes = [0, 1]
    for playlist in playlists:
        data = data_by_playlist[playlist]
        figs_npi = []
        test_idx = data["test_idx"][playlist]
        baselines_pl = data["baselines"][playlist]
        n_muons = baselines_pl["n_muons"][test_idx]
        n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
        improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
        first_model = next(iter(results))
        run0 = results[first_model][0][playlist]
        pid = run0["pid"]

        y_true_ccnpi = np.isin(pid, multi_pi_classes).astype(int)
        y_pred_ccnpi = (
            (n_muons == 1) & (n_charged_prongs >= 1) & (improved_nmichel >= 1)
        ).astype(int)
        tp = np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 1))
        fp = np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 0))
        fn = np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 1))
        tn = np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 0))
        baseline_fpr_ccnpi = fp / (fp + tn)

        metrics_q3 = compute_all_metrics_q3(
            results, data, signal_classes=multi_pi_classes, fixed_fpr=[baseline_fpr_ccnpi],
            playlist=playlist,
        )
        baseline_multi = compute_signal_baseline(results, data, signal_classes=multi_pi_classes, playlist=playlist)
        is_signal_ccnpi = y_true_ccnpi == 1
        reco_baseline_tpr_q3 = compute_reco_baseline_recall_per_bin(
            y_pred_ccnpi, is_signal_ccnpi, data["q3_GeV"], data["q3_bin_edges"],
        )
        fig = plot_multi_pion_vs_q3(
            metrics_q3, data, baseline_multi["q3"],
            fixed_fpr=[baseline_fpr_ccnpi], uncertainties=True,
            reco_baseline_tpr_q3=reco_baseline_tpr_q3,
            colors=clrs,
            title=fr"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs_npi.append(fig)
        plt.close(fig)

        fig = plot_prc_curves(
            results, signal_classes=multi_pi_classes,
            title=fr"PRC — $CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            playlist=playlist, uncertainties=True, colors=clrs,
        )
        figs_npi.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data, signal_classes=multi_pi_classes, x_var="q3", xlabel=r"True $q_3$ [GeV]",
            title=fr"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_ccnpi], reco_baseline_pred=y_pred_ccnpi,
            playlist=playlist, colors=clrs,
        )
        figs_npi.append(fig)
        plt.close(fig)
        save_figures_to_pdf(figs_npi, out_dir / f"eval_Npi_tagging_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_Npi_tagging_{playlist}.pdf")

    save_light_classification_pdfs(
        light_dir, results, data_by_playlist, clrs, playlists, components=("q3",),
    )


if __name__ == "__main__":
    main()
