#!/usr/bin/env python3
"""Classification vs pion kinematics (E, θ), CC1π⁰, π⁰ Δm scan; plus light appendix PDFs.

Pickles default under ``<repo>/<--out-dir>/``; PDFs under
``<repo>/<--plots-dir>/classification/pions/`` and ``.../classification/light/``
(per playlist: ``eval_classification_light_{cc1pi,cc1pi0}_{q3,W,pion_kinematics}_<pl>.pdf``).

Use ``--plots-only`` to skip the 16 GB source pickle and load from the
pre-computed metrics cache (``plots/tmp_results/classification_metrics.pkl``).
The cache is written automatically on a normal (non-``--plots-only``) run, or
with ``python -m src.eval.build_classification_cache``.
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
    plot_signal_composition_single_pion,
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
    repo_output_path,
)
from src.eval._plot_config import PlotConfig

_LIGHT_CACHE_NAME = "classification_light.pkl"

_CC1PI_CLASSES = [0]
_CC1PI0_CLASSES = [2]

PI0_MASS = 134.977


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl"


def get_pi0_baseline_pred(
    is_pizero_signal, two_gamma_inv_mass, delta_m, n_muons, n_michel
):
    has_candidate = is_pizero_signal == 2
    in_mass_window = np.abs(two_gamma_inv_mass - PI0_MASS) < delta_m
    return ((n_muons == 1) & has_candidate & in_mass_window & (n_michel == 0)).astype(
        int
    )


def precision_recall_fpr_vs_deltam(
    y_true,
    is_pizero_signal,
    two_gamma_inv_mass,
    delta_m_values,
    n_muons,
    n_michel,
):
    precisions, recalls, fprs = [], [], []
    for dm in delta_m_values:
        y_pred = get_pi0_baseline_pred(
            is_pizero_signal, two_gamma_inv_mass, dm, n_muons, n_michel
        )
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


def _draw_pi0_deltam_figs(
    out_dir: Path,
    playlists: list[str],
    pid_by_pl: dict,
    data_by_playlist: dict,
) -> None:
    """Draw the pi0 Δm scan figures (no sklearn — pure numpy)."""
    delta_m_values = np.linspace(1, 509, 20)
    for playlist in playlists:
        data = data_by_playlist[playlist]
        test_idx = data["test_idx"][playlist]
        baselines_pl = data["baselines"][playlist]
        is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
        two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
        n_muons_b = baselines_pl["n_muons"][test_idx]
        n_michel = baselines_pl["improved_nmichel"][test_idx]
        pid = pid_by_pl[playlist]
        y_true = np.isin(pid, _CC1PI0_CLASSES).astype(int)
        precisions, recalls, fprs = precision_recall_fpr_vs_deltam(
            y_true,
            is_pizero_signal,
            two_gamma_inv_mass,
            delta_m_values,
            n_muons_b,
            n_michel,
        )
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(
            delta_m_values,
            precisions,
            ".-",
            color="steelblue",
            linewidth=1.5,
            label="Precision",
        )
        ax1.plot(
            delta_m_values,
            recalls,
            ".-",
            color="darkorange",
            linewidth=1.5,
            label="Recall",
        )
        ax1.plot(
            delta_m_values, fprs, ".-", color="firebrick", linewidth=1.5, label="FPR"
        )
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


def main(argv: list[str] | None = None) -> None:
    silence_classification_empty_bin_warnings()
    _ = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--plots-dir", type=Path, default=None)
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="JSON",
        help="Plot config JSON (models, colors, optional display_name).",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Load from the pre-computed metrics cache (fast path).",
    )
    ap.add_argument("--plots-cache", type=Path, default=None, metavar="PKL")
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
    out_dir = plots_root / "classification" / "pions"
    out_dir.mkdir(parents=True, exist_ok=True)
    light_dir = plots_root / "classification" / "light"
    light_dir.mkdir(parents=True, exist_ok=True)

    metrics_cache_path = args.metrics_cache or (cache_root / CLF_METRICS_CACHE_NAME)
    cfg: PlotConfig | None = PlotConfig.load(args.config) if args.config else None

    if args.plots_only:
        _run_plots_only(args, cfg, cache_root, out_dir, light_dir, metrics_cache_path)
        return

    # --- Normal path ---
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

    # Build and save the metrics cache on every normal run (config-independent).
    print("Building classification metrics cache …")
    cache = build_metrics_cache(clf)
    save_metrics_cache(cache, metrics_cache_path)

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
            data,
            pid,
            cc1pi_classes,
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
            results,
            data_cc1pi,
            signal_classes=cc1pi_classes,
            fixed_fpr=[baseline_fpr_cc1pi],
            playlist=playlist,
            pion_bins_require_has_pion=False,
        )
        baseline_cc1pi = compute_signal_baseline(
            results,
            data_cc1pi,
            signal_classes=cc1pi_classes,
            playlist=playlist,
            pion_bins_require_has_pion=False,
        )
        is_signal_cc1pi = y_true_cc1pi == 1
        reco_baseline_tpr_cc1pi = {
            "E": compute_reco_baseline_recall_per_bin(
                y_pred_cc1pi,
                is_signal_cc1pi,
                data_cc1pi["pion_E_MC"],
                data_cc1pi["pion_E_MC_bins"],
                has_pion=None,
            ),
            "theta": compute_reco_baseline_recall_per_bin(
                y_pred_cc1pi,
                is_signal_cc1pi,
                data_cc1pi["pion_theta_MC"],
                data_cc1pi["pion_theta_MC_bins"],
                has_pion=None,
                finite_bin_var=True,
            ),
        }
        data_cc1pi_w = add_hadronic_W_to_classification_data(data_cc1pi, playlist)
        metrics_W_cc1pi = compute_all_metrics_W(
            results,
            data_cc1pi_w,
            signal_classes=cc1pi_classes,
            fixed_fpr=[baseline_fpr_cc1pi],
            playlist=playlist,
        )
        baseline_W_cc1pi = compute_signal_baseline_W(
            results,
            data_cc1pi_w,
            signal_classes=cc1pi_classes,
            playlist=playlist,
        )
        reco_baseline_tpr_W_cc1pi = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi,
            is_signal_cc1pi,
            data_cc1pi_w["W_GeV"],
            data_cc1pi_w["W_bin_edges"],
        )
        fig_w_cc1pi = plot_multi_classification_vs_W(
            metrics_W_cc1pi,
            data_cc1pi_w,
            baseline_W_cc1pi,
            fixed_fpr=[baseline_fpr_cc1pi],
            uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_cc1pi,
            reco_baseline_global_fpr=baseline_fpr_cc1pi,
            colors=clrs,
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
            use_global_fpr=True,
        )
        figs_cc1pi.append(fig_w_cc1pi)
        plt.close(fig_w_cc1pi)

        fig = plot_cc1pi_vs_pion_kinematics(
            metrics_cc1pi,
            data_cc1pi,
            baseline_cc1pi,
            uncertainties=True,
            fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_tpr=reco_baseline_tpr_cc1pi,
            reco_baseline_global_fpr=baseline_fpr_cc1pi,
            colors=clrs,
            playlist=playlist,
        )
        figs_cc1pi.append(fig)
        plt.close(fig)

        fig = plot_prc_curves(
            results,
            signal_classes=cc1pi_classes,
            title=rf"PRC — $CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
            colors=clrs,
            uncertainties=True,
        )
        figs_cc1pi.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data_cc1pi,
            signal_classes=cc1pi_classes,
            x_var="pion_E",
            xlabel=r"True $E_\pi$ [GeV]",
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            log_x=True,
            uncertainties=True,
            fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_pred=y_pred_cc1pi,
            playlist=playlist,
            colors=clrs,
            pion_bins_require_has_pion=False,
        )
        figs_cc1pi.append(fig)
        plt.close(fig)

        fig = plot_binned_by_inttype(
            results,
            data_cc1pi,
            signal_classes=cc1pi_classes,
            x_var="pion_theta",
            xlabel=r"True $\theta_\pi$ [rad]",
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_pred=y_pred_cc1pi,
            playlist=playlist,
            colors=clrs,
            pion_bins_require_has_pion=False,
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
            data,
            pid,
            cc1pi0_classes,
            pion_quantile_require_has_pion=False,
            pion_bin_edge_method="equal_frequency",
        )
        y_true_pi0 = np.isin(pid, cc1pi0_classes).astype(int)
        y_pred_baseline = (
            (n_muons == 1)
            & (is_pizero_signal == 2)
            & (np.abs(two_gamma_inv_mass - PI0_MASS) < DELTA_M)
            & (n_michel == 0)
        ).astype(int)
        tp_g = np.sum((y_pred_baseline == 1) & (y_true_pi0 == 1))
        fp_g = np.sum((y_pred_baseline == 1) & (y_true_pi0 == 0))
        fn_g = np.sum((y_pred_baseline == 0) & (y_true_pi0 == 1))
        tn_g = np.sum((y_pred_baseline == 0) & (y_true_pi0 == 0))
        baseline_fpr = fp_g / (fp_g + tn_g)

        metrics_cc1pi0 = compute_all_metrics(
            results,
            data_pi0,
            signal_classes=cc1pi0_classes,
            fixed_fpr=[baseline_fpr],
            playlist=playlist,
            pion_bins_require_has_pion=False,
        )
        baseline_cc1pi0 = compute_signal_baseline(
            results,
            data_pi0,
            signal_classes=cc1pi0_classes,
            playlist=playlist,
            pion_bins_require_has_pion=False,
        )
        is_signal_pi0 = y_true_pi0 == 1
        reco_baseline_tpr = {
            "E": compute_reco_baseline_recall_per_bin(
                y_pred_baseline,
                is_signal_pi0,
                data_pi0["pion_E_MC"],
                data_pi0["pion_E_MC_bins"],
                has_pion=None,
            ),
            "theta": compute_reco_baseline_recall_per_bin(
                y_pred_baseline,
                is_signal_pi0,
                data_pi0["pion_theta_MC"],
                data_pi0["pion_theta_MC_bins"],
                has_pion=None,
                finite_bin_var=True,
            ),
        }
        data_pi0_w = add_hadronic_W_to_classification_data(data_pi0, playlist)
        metrics_W_pi0 = compute_all_metrics_W(
            results,
            data_pi0_w,
            signal_classes=cc1pi0_classes,
            fixed_fpr=[baseline_fpr],
            playlist=playlist,
        )
        baseline_W_pi0 = compute_signal_baseline_W(
            results,
            data_pi0_w,
            signal_classes=cc1pi0_classes,
            playlist=playlist,
        )
        reco_baseline_tpr_W_pi0 = compute_reco_baseline_recall_per_bin(
            y_pred_baseline,
            is_signal_pi0,
            data_pi0_w["W_GeV"],
            data_pi0_w["W_bin_edges"],
        )
        fig_w_pi0 = plot_multi_classification_vs_W(
            metrics_W_pi0,
            data_pi0_w,
            baseline_W_pi0,
            fixed_fpr=[baseline_fpr],
            uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_pi0,
            reco_baseline_global_fpr=baseline_fpr,
            colors=clrs,
            title=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
            use_global_fpr=True,
        )
        figs_pi0.append(fig_w_pi0)
        plt.close(fig_w_pi0)

        fig = plot_cc1pi_vs_pion_kinematics(
            metrics_cc1pi0,
            data_pi0,
            baseline_cc1pi0,
            uncertainties=True,
            fixed_fpr=[baseline_fpr],
            reco_baseline_tpr=reco_baseline_tpr,
            reco_baseline_global_fpr=baseline_fpr,
            colors=clrs,
            suptitle=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
        )
        figs_pi0.append(fig)
        plt.close(fig)

        fig2 = plot_binned_by_inttype(
            results,
            data_pi0,
            signal_classes=cc1pi0_classes,
            x_var="pion_E",
            xlabel=r"True $E_\pi$ [GeV]",
            title=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            log_x=True,
            uncertainties=True,
            fixed_fpr=[baseline_fpr],
            reco_baseline_pred=y_pred_baseline,
            playlist=playlist,
            colors=clrs,
            pion_bins_require_has_pion=False,
        )
        figs_pi0.append(fig2)
        plt.close(fig2)

        fig3 = plot_binned_by_inttype(
            results,
            data_pi0,
            signal_classes=cc1pi0_classes,
            x_var="pion_theta",
            xlabel=r"True $\theta_\pi$ [rad]",
            title=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True,
            fixed_fpr=[baseline_fpr],
            reco_baseline_pred=y_pred_baseline,
            playlist=playlist,
            colors=clrs,
            pion_bins_require_has_pion=False,
        )
        figs_pi0.append(fig3)
        plt.close(fig3)

        fig4 = plot_prc_curves(
            results,
            signal_classes=cc1pi0_classes,
            title=rf"PRC — $CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            playlist=playlist,
            uncertainties=True,
            colors=clrs,
        )
        figs_pi0.append(fig4)
        plt.close(fig4)
        save_figures_to_pdf(figs_pi0, out_dir / f"eval_cc1pi0_tagging_{playlist}.pdf")
        print("Saved:", out_dir / f"eval_cc1pi0_tagging_{playlist}.pdf")

    for playlist in playlists:
        data = data_by_playlist[playlist]
        first_model = next(iter(results))
        pid = results[first_model][0][playlist]["pid"]
        fig_comp = plot_signal_composition_single_pion(
            data=data, pid=pid, playlist=playlist
        )
        fp = out_dir / f"event_composition_single_pion_{playlist}.pdf"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fig_comp.savefig(fp, bbox_inches="tight")
        plt.close(fig_comp)
        print("Saved:", fp)

    _draw_pi0_deltam_figs(
        out_dir,
        playlists,
        {pl: results[next(iter(results))][0][pl]["pid"] for pl in playlists},
        data_by_playlist,
    )

    save_light_classification_pdfs(
        light_dir,
        results,
        data_by_playlist,
        clrs,
        playlists,
        components=("pion",),
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
    """Fast --plots-only path using the precomputed metrics cache."""
    cache = load_metrics_cache(metrics_cache_path)

    clrs = cache["clrs_dict_full"]
    playlists = cache["playlists"]
    data_by_playlist = cache["data_by_playlist"]

    if cfg is not None:
        clrs = {**clrs, **cfg.colors()}
        clrs = cfg.filter_dict(clrs)

    def _filter(d: dict) -> dict:
        return cfg.filter_dict(d) if cfg is not None else d

    def _draw_cc1pi_section(
        sig_tag, sig_classes, pdf_stem, title_base, suptitle_base=None
    ):
        for playlist in playlists:
            data_pion = cache[f"data_{sig_tag}"][playlist]
            data_pion_w = cache[f"data_{sig_tag}_w"][playlist]
            figs = []

            fpr = cache["baseline_fpr"][sig_tag][playlist]
            metrics_all_full = cache["metrics_pion"][sig_tag][playlist]["all"]
            metrics_all = _filter(metrics_all_full)
            baseline_pion = cache["signal_baseline"][sig_tag][playlist]
            reco_tpr_E = cache["reco_tpr"][f"pion_E_{sig_tag}_{playlist}"]
            reco_tpr_theta = cache["reco_tpr"][f"pion_theta_{sig_tag}_{playlist}"]
            reco_tpr_W = cache["reco_tpr"][f"W_pion_{sig_tag}_{playlist}"]
            reco_pred = cache["reco_pred"][sig_tag][playlist]
            y_true_binary = np.isin(cache["pid"][playlist], sig_classes).astype(int)

            metrics_W_all = _filter(cache["metrics_W_pion"][sig_tag][playlist]["all"])
            W_baseline_all = cache["W_pion_baseline"][sig_tag][playlist]

            # W plot
            fig_w = plot_multi_classification_vs_W(
                metrics_W_all,
                data_pion_w,
                W_baseline_all,
                fixed_fpr=[fpr],
                uncertainties=True,
                reco_baseline_tpr_W=reco_tpr_W,
                reco_baseline_global_fpr=fpr,
                colors=clrs,
                title=rf"{title_base} - MINERvA Open Data Playlist {playlist}",
                playlist=playlist,
                use_global_fpr=True,
            )
            figs.append(fig_w)
            plt.close(fig_w)

            # Pion E + theta kinematics plot
            reco_tpr = {"E": reco_tpr_E, "theta": reco_tpr_theta}
            kw = {"suptitle": suptitle_base} if suptitle_base else {}
            fig_kin = plot_cc1pi_vs_pion_kinematics(
                metrics_all,
                data_pion,
                baseline_pion,
                uncertainties=True,
                fixed_fpr=[fpr],
                reco_baseline_tpr=reco_tpr,
                reco_baseline_global_fpr=fpr,
                colors=clrs,
                playlist=playlist,
                **kw,
            )
            figs.append(fig_kin)
            plt.close(fig_kin)

            # PRC
            prc_data = _filter(cache["prc"][sig_tag][playlist])
            sig_frac = cache["signal_frac"][sig_tag][playlist]
            fig_prc = plot_prc_curves(
                {},
                sig_classes,
                title=rf"PRC — {title_base} - MINERvA Open Data Playlist {playlist}",
                playlist=playlist,
                uncertainties=True,
                colors=clrs,
                precomputed_curves=prc_data,
                signal_frac=sig_frac,
            )
            figs.append(fig_prc)
            plt.close(fig_prc)

            # pion_E by inttype
            inttype_agg_full = precomputed_inttype_agg(
                cache["metrics_pion"][sig_tag][playlist]
            )
            # extract E sub-key for pion_E x_var
            pre_agg_E = {
                code: _filter({m: v["E"] for m, v in agg.items()})
                for code, agg in inttype_agg_full.items()
            }
            pre_bl_E = precomputed_bl_from_signal_baseline(
                cache["signal_baseline_inttype"], sig_tag, playlist, "pion_E"
            )
            fig_E = plot_binned_by_inttype(
                {},
                data_pion,
                sig_classes,
                x_var="pion_E",
                xlabel=r"True $E_\pi$ [GeV]",
                title=rf"{title_base} - MINERvA Open Data Playlist {playlist} - by interaction type",
                log_x=True,
                uncertainties=True,
                fixed_fpr=[fpr],
                reco_baseline_pred=reco_pred,
                playlist=playlist,
                colors=clrs,
                pion_bins_require_has_pion=False,
                precomputed_agg=pre_agg_E,
                precomputed_bl_values=pre_bl_E,
                precomputed_y_true_binary=y_true_binary,
            )
            figs.append(fig_E)
            plt.close(fig_E)

            # pion_theta by inttype
            pre_agg_theta = {
                code: _filter({m: v["theta"] for m, v in agg.items()})
                for code, agg in inttype_agg_full.items()
            }
            pre_bl_theta = precomputed_bl_from_signal_baseline(
                cache["signal_baseline_inttype"], sig_tag, playlist, "pion_theta"
            )
            fig_theta = plot_binned_by_inttype(
                {},
                data_pion,
                sig_classes,
                x_var="pion_theta",
                xlabel=r"True $\theta_\pi$ [rad]",
                title=rf"{title_base} - MINERvA Open Data Playlist {playlist} - by interaction type",
                uncertainties=True,
                fixed_fpr=[fpr],
                reco_baseline_pred=reco_pred,
                playlist=playlist,
                colors=clrs,
                pion_bins_require_has_pion=False,
                precomputed_agg=pre_agg_theta,
                precomputed_bl_values=pre_bl_theta,
                precomputed_y_true_binary=y_true_binary,
            )
            figs.append(fig_theta)
            plt.close(fig_theta)

            save_figures_to_pdf(figs, out_dir / f"{pdf_stem}_{playlist}.pdf")
            print("Saved:", out_dir / f"{pdf_stem}_{playlist}.pdf")

    _draw_cc1pi_section(
        "cc1pi", _CC1PI_CLASSES, "eval_cc1pi_tagging", r"$CC1\pi^\pm$ tagging"
    )
    _draw_cc1pi_section(
        "cc1pi0",
        _CC1PI0_CLASSES,
        "eval_cc1pi0_tagging",
        r"$CC1\pi^0$ tagging",
        suptitle_base=None,
    )

    # Event composition (single pion)
    for playlist in playlists:
        data = data_by_playlist[playlist]
        pid = cache["pid"][playlist]
        fig_comp = plot_signal_composition_single_pion(
            data=data, pid=pid, playlist=playlist
        )
        fp = out_dir / f"event_composition_single_pion_{playlist}.pdf"
        fig_comp.savefig(fp, bbox_inches="tight")
        plt.close(fig_comp)
        print("Saved:", fp)

    # pi0 delta-m scan (pure numpy, fast)
    _draw_pi0_deltam_figs(out_dir, playlists, cache["pid"], data_by_playlist)

    # Light classification figures
    light_cache_path = args.plots_cache or (cache_root / _LIGHT_CACHE_NAME)
    with open(light_cache_path, "rb") as f:
        light_cached = pickle.load(f)
    draw_light_classification_from_cache(
        light_cached["specs"], clrs, light_dir, cfg=cfg
    )


if __name__ == "__main__":
    main()
