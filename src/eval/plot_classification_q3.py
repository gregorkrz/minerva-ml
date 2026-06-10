#!/usr/bin/env python3
"""Classification vs q3, confusion matrices, CCNπ q₃ panels; plus light appendix PDFs.

Pickles default under ``<repo>/<--out-dir>/``; PDFs under
``<repo>/<--plots-dir>/classification/q3/`` and ``.../classification/light/``
(``eval_classification_light_ccnpi_q3_<pl>.pdf`` and ``..._W_<pl>.pdf`` when *W* is in the pickle).

Use ``--plots-only`` to skip the 16 GB source pickle and read from the
pre-computed metrics cache (``plots/tmp_results/classification_metrics.pkl``)
instead.  The cache is written automatically on a normal (non-``--plots-only``)
run, or with ``python -m src.eval.build_classification_cache``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval.classification_plots import (
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    compute_all_metrics_q3,
    compute_signal_baseline,
    compute_reco_baseline_recall_per_bin,
    plot_binned_by_inttype,
    plot_composition_vs_kinematic,
    plot_multi_pion_vs_q3,
    plot_prc_curves,
    save_figures_to_pdf,
)
from src.eval._bootstrap import silence_classification_empty_bin_warnings
from src.eval._classification_light import (
    draw_light_classification_from_cache,
    save_light_classification_pdfs,
)
from src.eval._classification_metrics_cache import (
    CLF_METRICS_CACHE_NAME,
    build_metrics_cache,
    load_metrics_cache,
    precomputed_bl_from_signal_baseline,
    precomputed_inttype_agg,
    save_metrics_cache,
)
from src.eval._constants import (
    CANONICAL_CLASSIFICATION_PICKLE,
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    plot_model_label,
    repo_output_path,
)
from src.eval._plot_config import PlotConfig

_LIGHT_CACHE_NAME = "classification_light.pkl"

_CC1PI_CLASSES = [0]
_CCNPI_GE1_CLASSES = [0, 1]


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl"


def main(argv: list[str] | None = None) -> None:
    silence_classification_empty_bin_warnings()
    _ = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE
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
        default=None,
        help="Root for PDF output (default: <out-dir>/plots)",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="JSON",
        help="Plot config JSON (models, colors, optional display_name). "
        "When given, only listed models are drawn.",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Load from the pre-computed metrics cache instead of the source "
        "classification pickle (fast path — requires the cache to have been "
        "built by a previous normal run or build_classification_cache).",
    )
    ap.add_argument(
        "--plots-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Override the light-classification plots cache path used by "
        "--plots-only (default: plots/tmp_results/classification_light.pkl).",
    )
    ap.add_argument(
        "--metrics-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Override the metrics cache path "
        "(default: plots/tmp_results/classification_metrics.pkl).",
    )
    args = ap.parse_args(argv)
    if args.plots_dir is None:
        args.plots_dir = Path(args.out_dir or DEFAULT_OUT_DIR) / "plots"

    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    out_dir = plots_root / "classification" / "q3"
    out_dir.mkdir(parents=True, exist_ok=True)
    light_dir = plots_root / "classification" / "light"
    light_dir.mkdir(parents=True, exist_ok=True)

    metrics_cache_path = args.metrics_cache or (cache_root / CLF_METRICS_CACHE_NAME)
    cfg: PlotConfig | None = PlotConfig.load(args.config) if args.config else None

    if args.plots_only:
        _run_plots_only(args, cfg, cache_root, out_dir, light_dir, metrics_cache_path)
        return

    # --- Normal (compute + save cache + draw) path ---
    pkl = args.classification_pickle or _pickle_path(data_root, args.flag)
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    results = clf["results"]
    data_by_playlist = clf["data_by_playlist"]
    data_w_by_playlist = clf.get("data_w_by_playlist")
    clrs = clf["clrs_dict_full"]
    playlists = clf["playlists"]

    if cfg is not None:
        results = cfg.filter_dict(results)
        clrs = {**clrs, **cfg.colors()}
        clrs = cfg.filter_dict(clrs)

    label_fn = cfg.label_for if cfg is not None else plot_model_label

    # Build and save the metrics cache on every normal run (config-independent).
    print("Building classification metrics cache …")
    cache = build_metrics_cache(clf)
    save_metrics_cache(cache, metrics_cache_path)

    class_names = ["CC1π±", "CCNπ", "CC1π0", "Other-CC", "Other-NC"]
    model_names = sorted(results.keys())
    n_models = len(model_names)

    for playlist in playlists:
        fig, axes = plt.subplots(
            1, n_models, figsize=(7 * n_models, 6), tight_layout=True
        )
        if n_models == 1:
            axes = [axes]
        for ax, model_name in zip(axes, model_names):
            run0 = results[model_name][0][playlist]
            y_true = run0["pid"]
            y_pred = run0["prediction"].argmax(axis=1)
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
            )
            ax.set_xlabel(r"Predicted class")
            ax.set_ylabel(r"True class")
            ax.set_title(label_fn(model_name))
        fig.suptitle(
            f"Confusion matrices (first run per model) — {playlist}", fontsize=14
        )
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
            results,
            data,
            signal_classes=cc1pi_classes,
            fixed_fpr=[baseline_fpr_cc1pi],
            playlist=playlist,
        )
        baseline_cc1pi = compute_signal_baseline(
            results, data, signal_classes=cc1pi_classes, playlist=playlist
        )
        is_signal_cc1pi = y_true_cc1pi == 1
        reco_baseline_tpr_q3_cc1pi = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi,
            is_signal_cc1pi,
            data["q3_GeV"],
            data["q3_bin_edges"],
        )
        fig = plot_multi_pion_vs_q3(
            metrics_q3_cc1pi,
            data,
            baseline_cc1pi["q3"],
            fixed_fpr=[baseline_fpr_cc1pi],
            uncertainties=True,
            reco_baseline_tpr_q3=reco_baseline_tpr_q3_cc1pi,
            colors=clrs,
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs_cc1pi_q3.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data,
            signal_classes=cc1pi_classes,
            x_var="q3",
            xlabel=r"True $q_3$ [GeV]",
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_pred=y_pred_cc1pi,
            playlist=playlist,
            colors=clrs,
        )
        figs_cc1pi_q3.append(fig)
        plt.close(fig)
        save_figures_to_pdf(
            figs_cc1pi_q3, out_dir / f"eval_cc1pi_tagging_q3_{playlist}.pdf"
        )
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
            results,
            data,
            signal_classes=multi_pi_classes,
            fixed_fpr=[baseline_fpr_ccnpi],
            playlist=playlist,
        )
        baseline_multi = compute_signal_baseline(
            results, data, signal_classes=multi_pi_classes, playlist=playlist
        )
        is_signal_ccnpi = y_true_ccnpi == 1
        reco_baseline_tpr_q3 = compute_reco_baseline_recall_per_bin(
            y_pred_ccnpi,
            is_signal_ccnpi,
            data["q3_GeV"],
            data["q3_bin_edges"],
        )
        fig = plot_multi_pion_vs_q3(
            metrics_q3,
            data,
            baseline_multi["q3"],
            fixed_fpr=[baseline_fpr_ccnpi],
            uncertainties=True,
            reco_baseline_tpr_q3=reco_baseline_tpr_q3,
            colors=clrs,
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs_npi.append(fig)
        plt.close(fig)

        fig = plot_prc_curves(
            results,
            signal_classes=multi_pi_classes,
            title=rf"PRC — $CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
            uncertainties=True,
            colors=clrs,
        )
        figs_npi.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data,
            signal_classes=multi_pi_classes,
            x_var="q3",
            xlabel=r"True $q_3$ [GeV]",
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr_ccnpi],
            reco_baseline_pred=y_pred_ccnpi,
            playlist=playlist,
            colors=clrs,
        )
        figs_npi.append(fig)
        plt.close(fig)
        save_figures_to_pdf(figs_npi, out_dir / f"eval_Npi_tagging_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_Npi_tagging_{playlist}.pdf")

    # Event-composition plots
    for playlist in playlists:
        data = data_by_playlist[playlist]
        first_model = next(iter(results))
        pid = results[first_model][0][playlist]["pid"]
        fig_comp = plot_composition_vs_kinematic(
            data=data,
            pid=pid,
            x_var="q3",
            playlist=playlist,
        )
        fp = out_dir / f"event_composition_q3_{playlist}.pdf"
        fig_comp.savefig(fp, bbox_inches="tight")
        plt.close(fig_comp)
        print("Saved:", fp)

    save_light_classification_pdfs(
        light_dir,
        results,
        data_by_playlist,
        clrs,
        playlists,
        components=("q3",),
        data_w_by_playlist=data_w_by_playlist,
    )


def _run_plots_only(
    args: argparse.Namespace,
    cfg: PlotConfig | None,
    cache_root: Path,
    out_dir: Path,
    light_dir: Path,
    metrics_cache_path: Path,
) -> None:
    """Fast --plots-only path: load metrics cache, skip the 16 GB source pickle."""
    cache = load_metrics_cache(metrics_cache_path)

    clrs = cache["clrs_dict_full"]
    playlists = cache["playlists"]
    data_by_playlist = cache["data_by_playlist"]

    if cfg is not None:
        clrs = {**clrs, **cfg.colors()}
        clrs = cfg.filter_dict(clrs)

    label_fn = cfg.label_for if cfg is not None else plot_model_label

    class_names = ["CC1π±", "CCNπ", "CC1π0", "Other-CC", "Other-NC"]

    # --- Confusion matrices (precomputed) ---
    confusion_matrices = cache["confusion_matrices"]
    if cfg is not None:
        confusion_matrices = cfg.filter_dict(confusion_matrices)
    model_names = sorted(confusion_matrices.keys())
    n_models = len(model_names)

    for playlist in playlists:
        fig, axes = plt.subplots(
            1, n_models, figsize=(7 * n_models, 6), tight_layout=True
        )
        if n_models == 1:
            axes = [axes]
        for ax, model_name in zip(axes, model_names):
            cm = confusion_matrices[model_name][playlist]
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names,
            )
            ax.set_xlabel(r"Predicted class")
            ax.set_ylabel(r"True class")
            ax.set_title(label_fn(model_name))
        fig.suptitle(
            f"Confusion matrices (first run per model) — {playlist}", fontsize=14
        )
        fp = out_dir / f"confusion_matrices_{playlist}.pdf"
        fig.savefig(fp)
        plt.close(fig)
        print("Saved:", fp)

    # --- CC1π± q3 ---
    for playlist in playlists:
        data = data_by_playlist[playlist]
        figs = []

        baseline_fpr_cc1pi = cache["baseline_fpr"]["cc1pi"][playlist]
        metrics_q3_cc1pi = cache["metrics_q3"]["cc1pi"][playlist]["all"]
        if cfg is not None:
            metrics_q3_cc1pi = cfg.filter_dict(metrics_q3_cc1pi)

        baseline_q3_cc1pi = cache["signal_baseline"]["cc1pi"][playlist]["q3"]
        reco_tpr = cache["reco_tpr"][f"q3_cc1pi_{playlist}"]
        reco_pred = cache["reco_pred"]["cc1pi"][playlist]
        y_true_binary = np.isin(cache["pid"][playlist], _CC1PI_CLASSES).astype(int)

        fig = plot_multi_pion_vs_q3(
            metrics_q3_cc1pi, data, baseline_q3_cc1pi,
            fixed_fpr=[baseline_fpr_cc1pi], uncertainties=True,
            reco_baseline_tpr_q3=reco_tpr, colors=clrs,
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs.append(fig); plt.close(fig)

        pre_agg = precomputed_inttype_agg(cache["metrics_q3"]["cc1pi"][playlist])
        if cfg is not None:
            pre_agg = {code: cfg.filter_dict(agg) for code, agg in pre_agg.items()}
        pre_bl = precomputed_bl_from_signal_baseline(
            cache["signal_baseline_inttype"], "cc1pi", playlist, "q3"
        )
        fig = plot_binned_by_inttype(
            {}, data, _CC1PI_CLASSES, x_var="q3",
            xlabel=r"True $q_3$ [GeV]",
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_pred=reco_pred, playlist=playlist, colors=clrs,
            precomputed_agg=pre_agg, precomputed_bl_values=pre_bl,
            precomputed_y_true_binary=y_true_binary,
        )
        figs.append(fig); plt.close(fig)
        save_figures_to_pdf(figs, out_dir / f"eval_cc1pi_tagging_q3_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_cc1pi_tagging_q3_{playlist}.pdf")

    # --- CCNπ (N≥1) q3 ---
    for playlist in playlists:
        data = data_by_playlist[playlist]
        figs = []

        baseline_fpr_ccnpi = cache["baseline_fpr"]["ccnpi_ge1"][playlist]
        metrics_q3_ccnpi = cache["metrics_q3"]["ccnpi_ge1"][playlist]["all"]
        if cfg is not None:
            metrics_q3_ccnpi = cfg.filter_dict(metrics_q3_ccnpi)

        baseline_q3_ccnpi = cache["signal_baseline"]["ccnpi_ge1"][playlist]["q3"]
        reco_tpr = cache["reco_tpr"][f"q3_ccnpi_ge1_{playlist}"]
        reco_pred = cache["reco_pred"]["ccnpi_ge1"][playlist]
        y_true_binary = np.isin(cache["pid"][playlist], _CCNPI_GE1_CLASSES).astype(int)

        fig = plot_multi_pion_vs_q3(
            metrics_q3_ccnpi, data, baseline_q3_ccnpi,
            fixed_fpr=[baseline_fpr_ccnpi], uncertainties=True,
            reco_baseline_tpr_q3=reco_tpr, colors=clrs,
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs.append(fig); plt.close(fig)

        fig = plot_prc_curves(
            {}, _CCNPI_GE1_CLASSES,
            title=rf"PRC — $CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            playlist=playlist, uncertainties=True, colors=clrs,
            precomputed_curves=cfg.filter_dict(cache["prc"]["ccnpi_ge1"][playlist]) if cfg else cache["prc"]["ccnpi_ge1"][playlist],
            signal_frac=cache["signal_frac"]["ccnpi_ge1"][playlist],
        )
        figs.append(fig); plt.close(fig)

        pre_agg = precomputed_inttype_agg(cache["metrics_q3"]["ccnpi_ge1"][playlist])
        if cfg is not None:
            pre_agg = {code: cfg.filter_dict(agg) for code, agg in pre_agg.items()}
        pre_bl = precomputed_bl_from_signal_baseline(
            cache["signal_baseline_inttype"], "ccnpi_ge1", playlist, "q3"
        )
        fig = plot_binned_by_inttype(
            {}, data, _CCNPI_GE1_CLASSES, x_var="q3",
            xlabel=r"True $q_3$ [GeV]",
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_ccnpi],
            reco_baseline_pred=reco_pred, playlist=playlist, colors=clrs,
            precomputed_agg=pre_agg, precomputed_bl_values=pre_bl,
            precomputed_y_true_binary=y_true_binary,
        )
        figs.append(fig); plt.close(fig)
        save_figures_to_pdf(figs, out_dir / f"eval_Npi_tagging_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_Npi_tagging_{playlist}.pdf")

    # --- Event composition (pid + data needed, both in cache) ---
    for playlist in playlists:
        data = data_by_playlist[playlist]
        pid = cache["pid"][playlist]
        fig_comp = plot_composition_vs_kinematic(
            data=data, pid=pid, x_var="q3", playlist=playlist,
        )
        fp = out_dir / f"event_composition_q3_{playlist}.pdf"
        fig_comp.savefig(fp, bbox_inches="tight")
        plt.close(fig_comp)
        print("Saved:", fp)

    # --- Light classification figures ---
    light_cache_path = args.plots_cache or (cache_root / _LIGHT_CACHE_NAME)
    with open(light_cache_path, "rb") as f:
        light_cached = pickle.load(f)
    draw_light_classification_from_cache(light_cached["specs"], clrs, light_dir, cfg=cfg)


if __name__ == "__main__":
    main()
