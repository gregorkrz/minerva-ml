#!/usr/bin/env python3
"""Classification vs hadronic W (``Eval_Classification_W`` + ``W_Npi_Ng1`` parity).

Pickles default under ``<repo>/<--out-dir>/``; PDFs under
``<repo>/<--plots-dir>/classification/w_bins/``.

Use ``--plots-only`` to skip the 16 GB source pickle and load from the
pre-computed metrics cache (``plots/tmp_results/classification_metrics.pkl``).
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
    plot_baseline_tpr_fpr_vs_W,
    plot_binned_by_inttype,
    plot_composition_vs_kinematic,
    plot_confusion_matrices_at_threshold_W,
    plot_multi_classification_vs_W,
    save_figures_to_pdf,
)
from src.eval.classification_plots._constants import TRUE_W_XLABEL, w_bin_edges_finer
from src.eval._bootstrap import silence_classification_empty_bin_warnings
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
    DEFAULT_CFS_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    canonical_classification_pickle_paths,
    repo_output_path,
)
from src.eval._constants import plot_model_label
from src.eval._plot_config import PlotConfig
from src.eval._plot_split_results import (
    apply_plot_config_results,
    build_data_by_model,
    compute_all_metrics_W_for_config,
    merge_inttype_agg_overlays,
    overlay_model_names,
)

_CC1PI_CLASSES = [0]
_CC1PI0_CLASSES = [2]
_CCNPI_GE1_CLASSES = [0, 1]
_CCNPI_GT1_CLASSES = [1]


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl"


def _resolve_classification_pickle(
    args: argparse.Namespace,
    repo_root: Path,
    cache_root: Path,
    data_root: Path,
    *,
    plots_only: bool = False,
) -> Path:
    """Pickle path: explicit override, then largest existing candidate, else default."""
    if args.classification_pickle is not None:
        return Path(str(args.classification_pickle).strip())
    candidates = canonical_classification_pickle_paths(
        repo_root, args.flag, data_root
    )
    existing = [p for p in candidates if p.exists()]
    if existing:
        # Avoid truncated copies of the same pickle (cache symlink/cp can be partial).
        return max(existing, key=lambda p: p.stat().st_size)
    if plots_only:
        return candidates[0]
    return _pickle_path(data_root, args.flag)


def _save_event_composition_W(
    data_w: dict,
    pid: np.ndarray,
    out_dir: Path,
    playlist: str,
    *,
    finer_bins: bool = False,
) -> None:
    """Write event-composition / S/B vs *W* PDF for one playlist."""
    if finer_bins:
        edges = w_bin_edges_finer(10)
        stem = f"event_composition_W_{playlist}_finer_bins"
    else:
        edges = None
        stem = f"event_composition_W_{playlist}"
    fig_comp = plot_composition_vs_kinematic(
        data=data_w, pid=pid, x_var="W", playlist=playlist, bin_edges=edges,
    )
    fp = out_dir / f"{stem}.pdf"
    fig_comp.savefig(fp, bbox_inches="tight")
    plt.close(fig_comp)
    print("Saved:", fp)


def _run_w_notebook_style(
    results: dict,
    data_w_by_playlist: dict,
    clrs: dict[str, str],
    use_global_fpr: bool,
    out_dir: Path,
    playlists: list[str],
    tag_suffix: str,
    *,
    cfg: PlotConfig | None = None,
    data_w_by_split: dict | None = None,
) -> None:
    _PDF_FPR_TAG = "_global_fpr" if use_global_fpr else "_per_bin_fpr"
    data_w_by_split = data_w_by_split or {}
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

        metrics_W_cc1pi = compute_all_metrics_W_for_config(
            results, data_w, data_w_by_split, cfg,
            signal_classes=cc1pi_classes,
            fixed_fpr=[baseline_fpr_cc1pi], playlist=playlist, use_global_fpr=use_global_fpr,
        )
        baseline_W_cc1pi = compute_signal_baseline_W(results, data_w, signal_classes=cc1pi_classes, playlist=playlist)
        is_signal_cc1pi = y_true_cc1pi == 1
        reco_baseline_tpr_W_cc1pi = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi, is_signal_cc1pi, data_w["W_GeV"], data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W_cc1pi, data_w, baseline_W_cc1pi,
            fixed_fpr=[baseline_fpr_cc1pi], uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_cc1pi,
            reco_baseline_global_fpr=baseline_fpr_cc1pi, colors=clrs,
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr, playlist=playlist,
        )
        figs_cc1pi_W.append(fig); plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data_w, signal_classes=cc1pi_classes, x_var="W",
            xlabel=TRUE_W_XLABEL,
            title=rf"$CC1\pi^\pm$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_cc1pi],
            reco_baseline_pred=y_pred_cc1pi, playlist=playlist, colors=clrs,
            use_global_fpr=use_global_fpr,
            data_by_model=build_data_by_model(results, cfg, data_w_by_split, playlist),
        )
        figs_cc1pi_W.append(fig); plt.close(fig)
        save_figures_to_pdf(figs_cc1pi_W, out_dir / f"eval_cc1pi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf")
        print("Saved:", out_dir / f"eval_cc1pi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf")

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

        metrics_W_cc1pi0 = compute_all_metrics_W_for_config(
            results, data_w, data_w_by_split, cfg,
            signal_classes=cc1pi0_classes,
            fixed_fpr=[baseline_fpr_cc1pi0], playlist=playlist, use_global_fpr=use_global_fpr,
        )
        baseline_W_cc1pi0 = compute_signal_baseline_W(results, data_w, signal_classes=cc1pi0_classes, playlist=playlist)
        is_signal_cc1pi0 = y_true_cc1pi0 == 1
        reco_baseline_tpr_W_cc1pi0 = compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi0, is_signal_cc1pi0, data_w["W_GeV"], data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W_cc1pi0, data_w, baseline_W_cc1pi0,
            fixed_fpr=[baseline_fpr_cc1pi0], uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W_cc1pi0,
            reco_baseline_global_fpr=baseline_fpr_cc1pi0, colors=clrs,
            title=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr, playlist=playlist,
        )
        figs_cc1pi0_W.append(fig); plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data_w, signal_classes=cc1pi0_classes, x_var="W",
            xlabel=TRUE_W_XLABEL,
            title=rf"$CC1\pi^0$ tagging - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_cc1pi0],
            reco_baseline_pred=y_pred_cc1pi0, playlist=playlist, colors=clrs,
            use_global_fpr=use_global_fpr,
            data_by_model=build_data_by_model(results, cfg, data_w_by_split, playlist),
        )
        figs_cc1pi0_W.append(fig); plt.close(fig)
        save_figures_to_pdf(figs_cc1pi0_W, out_dir / f"eval_cc1pi0_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf")
        print("Saved:", out_dir / f"eval_cc1pi0_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf")

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

        metrics_W = compute_all_metrics_W_for_config(
            results, data_w, data_w_by_split, cfg,
            signal_classes=multi_pi_classes,
            fixed_fpr=[baseline_fpr_ccnpi], playlist=playlist, use_global_fpr=use_global_fpr,
        )
        baseline_W_multi = compute_signal_baseline_W(results, data_w, signal_classes=multi_pi_classes, playlist=playlist)
        is_signal_ccnpi = y_true_ccnpi == 1
        reco_baseline_tpr_W = compute_reco_baseline_recall_per_bin(
            y_pred_ccnpi, is_signal_ccnpi, data_w["W_GeV"], data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W, data_w, baseline_W_multi,
            fixed_fpr=[baseline_fpr_ccnpi], uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W,
            reco_baseline_global_fpr=baseline_fpr_ccnpi, colors=clrs,
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr, playlist=playlist,
        )
        figs_npi_W.append(fig); plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data_w, signal_classes=multi_pi_classes, x_var="W",
            xlabel=TRUE_W_XLABEL,
            title=rf"$CCN\pi^\pm$ tagging ($N \geq 1$) - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_ccnpi],
            reco_baseline_pred=y_pred_ccnpi, playlist=playlist, colors=clrs,
            use_global_fpr=use_global_fpr,
            data_by_model=build_data_by_model(results, cfg, data_w_by_split, playlist),
        )
        figs_npi_W.append(fig); plt.close(fig)
        save_figures_to_pdf(figs_npi_W, out_dir / f"eval_Npi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf")
        print("Saved:", out_dir / f"eval_Npi_tagging_W_{playlist}{tag_suffix}{_PDF_FPR_TAG}.pdf")


def _run_conf_matrix_at_threshold(
    results: dict,
    data_w_by_playlist: dict,
    out_dir: Path,
    playlists: list[str],
    use_global_fpr: bool,
    *,
    baseline_fpr_by_tag: dict[str, dict[str, float]] | None = None,
    reco_pred_by_tag: dict[str, dict[str, np.ndarray]] | None = None,
    cfg: PlotConfig | None = None,
) -> None:
    """One multi-page PDF: binary CMs per *W* bin + global at baseline FPR."""
    _PDF_FPR_TAG = "_global_fpr" if use_global_fpr else "_per_bin_fpr"
    label_fn = cfg.label_for if cfg is not None else None

    _SIGNALS = (
        ("cc1pi", _CC1PI_CLASSES, r"$CC1\pi^\pm$ tagging"),
        ("cc1pi0", _CC1PI0_CLASSES, r"$CC1\pi^0$ tagging"),
        ("ccnpi_ge1", _CCNPI_GE1_CLASSES, r"$CCN\pi^\pm$ tagging ($N \geq 1$)"),
        ("ccnpi_gt1", _CCNPI_GT1_CLASSES, r"$CCN\pi^\pm$ tagging ($N > 1$)"),
    )

    for playlist in playlists:
        data_w = data_w_by_playlist[playlist]
        figs: list[plt.Figure] = []
        for tag, sig_classes, title_base in _SIGNALS:
            if baseline_fpr_by_tag is not None and reco_pred_by_tag is not None:
                baseline_fpr = baseline_fpr_by_tag[tag][playlist]
                reco_pred = reco_pred_by_tag[tag][playlist]
            else:
                test_idx = data_w["test_idx"][playlist]
                baselines_pl = data_w["baselines"][playlist]
                n_muons = baselines_pl["n_muons"][test_idx]
                n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
                improved_nmichel = baselines_pl["improved_nmichel"][test_idx]
                first_model = next(iter(results))
                pid = results[first_model][0][playlist]["pid"]
                y_true = np.isin(pid, sig_classes).astype(int)

                if tag == "cc1pi":
                    reco_pred = (
                        (n_muons == 1)
                        & (n_charged_prongs == 1)
                        & (improved_nmichel == 1)
                    ).astype(int)
                elif tag == "cc1pi0":
                    PI0_MASS_MEV = 134.977
                    is_pizero_signal = baselines_pl["is_pizero_signal"][test_idx]
                    two_gamma_inv_mass = baselines_pl["two_gamma_invariant_mass"][test_idx]
                    reco_pred = (
                        (n_muons == 1)
                        & (is_pizero_signal == 2)
                        & (np.abs(two_gamma_inv_mass - PI0_MASS_MEV) < PI0_MASS_MEV)
                        & (improved_nmichel == 0)
                    ).astype(int)
                elif tag == "ccnpi_ge1":
                    reco_pred = (
                        (n_muons == 1)
                        & (n_charged_prongs >= 1)
                        & (improved_nmichel >= 1)
                    ).astype(int)
                else:
                    reco_pred = (
                        (n_muons == 1)
                        & (n_charged_prongs >= 2)
                        & (improved_nmichel >= 1)
                    ).astype(int)

                fp = int(((reco_pred == 1) & (y_true == 0)).sum())
                tn = int(((reco_pred == 0) & (y_true == 0)).sum())
                baseline_fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

            fig = plot_confusion_matrices_at_threshold_W(
                results,
                sig_classes,
                data_w,
                baseline_fpr,
                reco_pred,
                use_global_fpr=use_global_fpr,
                playlist=playlist,
                title=rf"{title_base} — MINERvA Open Data Playlist {playlist}",
                model_label_fn=label_fn or plot_model_label,
            )
            figs.append(fig)
            plt.close(fig)

        fp = (
            out_dir / "conf_matrix_at_threshold.pdf"
            if use_global_fpr
            else out_dir / f"conf_matrix_at_threshold{_PDF_FPR_TAG}.pdf"
        )
        save_figures_to_pdf(figs, fp)


def _run_w_ngt1(
    results: dict,
    data_w_by_playlist: dict,
    clrs: dict[str, str],
    use_global_fpr: bool,
    out_dir: Path,
    playlists: list[str],
    *,
    cfg: PlotConfig | None = None,
    data_w_by_split: dict | None = None,
) -> None:
    _PDF_FPR_TAG = "_global_fpr" if use_global_fpr else "_per_bin_fpr"
    data_w_by_split = data_w_by_split or {}
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

        metrics_W = compute_all_metrics_W_for_config(
            results, data_w, data_w_by_split, cfg,
            signal_classes=multi_pi_classes,
            fixed_fpr=[baseline_fpr_ccnpi], playlist=playlist, use_global_fpr=use_global_fpr,
        )
        baseline_W_multi = compute_signal_baseline_W(results, data_w, signal_classes=multi_pi_classes, playlist=playlist)
        is_signal_ccnpi = y_true_ccnpi == 1
        reco_baseline_tpr_W = compute_reco_baseline_recall_per_bin(
            y_pred_ccnpi, is_signal_ccnpi, data_w["W_GeV"], data_w["W_bin_edges"],
        )
        fig = plot_multi_classification_vs_W(
            metrics_W, data_w, baseline_W_multi,
            fixed_fpr=[baseline_fpr_ccnpi], uncertainties=True,
            reco_baseline_tpr_W=reco_baseline_tpr_W,
            reco_baseline_global_fpr=baseline_fpr_ccnpi, colors=clrs,
            title=rf"$CCN\pi^\pm$ tagging ($N > 1$) - MINERvA Open Data Playlist {playlist}",
            use_global_fpr=use_global_fpr, playlist=playlist,
        )
        figs_npi_W.append(fig); plt.close(fig)

        fig = plot_binned_by_inttype(
            results, data_w, signal_classes=multi_pi_classes, x_var="W",
            xlabel=TRUE_W_XLABEL,
            title=rf"$CCN\pi^\pm$ tagging ($N > 1$) - MINERvA Open Data Playlist {playlist} - by interaction type",
            uncertainties=True, fixed_fpr=[baseline_fpr_ccnpi],
            reco_baseline_pred=y_pred_ccnpi, playlist=playlist, colors=clrs,
            use_global_fpr=use_global_fpr,
            data_by_model=build_data_by_model(results, cfg, data_w_by_split, playlist),
        )
        figs_npi_W.append(fig); plt.close(fig)
        save_figures_to_pdf(figs_npi_W, out_dir / f"eval_Npi_Ngt1_tagging_W_{playlist}{_PDF_FPR_TAG}.pdf")
        print("Saved:", out_dir / f"eval_Npi_Ngt1_tagging_W_{playlist}{_PDF_FPR_TAG}.pdf")


def _baseline_reco_pred_from_data(
    data_w: dict,
    playlist: str,
    tag: str,
    pid: np.ndarray,
) -> np.ndarray:
    """Binary reco-baseline prediction for signal tag *tag*."""
    from src.eval._classification_metrics_cache import _compute_reco_pred

    cond = {
        "cc1pi": "cc1pi",
        "cc1pi0": "cc1pi0",
        "ccnpi_ge1": "ccnpi_ge1",
        "ccnpi_gt1": "ccnpi_gt1",
    }[tag]
    return _compute_reco_pred(data_w, pid, playlist, cond)


def _run_baseline_tpr_fpr_W(
    data_w_by_playlist: dict,
    out_dir: Path,
    playlists: list[str],
    *,
    reco_pred_by_tag: dict[str, dict[str, np.ndarray]] | None = None,
    pid_by_playlist: dict[str, np.ndarray] | None = None,
) -> None:
    """Per-*W*-bin baseline TPR + FPR diagnostic PDFs (dual *y* axes)."""
    _SIGNALS = (
        ("cc1pi", _CC1PI_CLASSES, "eval_cc1pi_baseline_tpr_fpr_W", r"$CC1\pi^\pm$ tagging"),
        ("cc1pi0", _CC1PI0_CLASSES, "eval_cc1pi0_baseline_tpr_fpr_W", r"$CC1\pi^0$ tagging"),
        ("ccnpi_ge1", _CCNPI_GE1_CLASSES, "eval_Npi_baseline_tpr_fpr_W", r"$CCN\pi^\pm$ tagging ($N \geq 1$)"),
        ("ccnpi_gt1", _CCNPI_GT1_CLASSES, "eval_Npi_Ngt1_baseline_tpr_fpr_W", r"$CCN\pi^\pm$ tagging ($N > 1$)"),
    )
    for playlist in playlists:
        data_w = data_w_by_playlist[playlist]
        if pid_by_playlist is None:
            raise ValueError("pid_by_playlist is required")
        pid = pid_by_playlist[playlist]
        for tag, sig_classes, pdf_stem, title_base in _SIGNALS:
            if reco_pred_by_tag is not None:
                reco_pred = reco_pred_by_tag[tag][playlist]
            else:
                reco_pred = _baseline_reco_pred_from_data(data_w, playlist, tag, pid)
            y_true = np.isin(pid, sig_classes).astype(int)
            fig = plot_baseline_tpr_fpr_vs_W(
                data_w,
                reco_pred,
                y_true,
                title=rf"{title_base} — baseline per-bin TPR & FPR — Playlist {playlist}",
                playlist=playlist,
                by_inttype=True,
            )
            fp = out_dir / f"{pdf_stem}_{playlist}.pdf"
            fig.savefig(fp, bbox_inches="tight")
            plt.close(fig)
            print("Saved:", fp)


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
        default=None,
        help="Root for PDF output (default: <out-dir>/plots)",
    )
    ap.add_argument(
        "--classification-pickle",
        type=Path,
        default=None,
        help="Override classification pickle "
        f"(default: {DEFAULT_CFS_CACHE_DIR / CANONICAL_CLASSIFICATION_PICKLE} "
        "or repo plots/tmp_results/classification.pkl for --plots-only, "
        "else <out-dir>/classification_<flag>.pkl).",
    )
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
        help="Load metrics from plots/tmp_results/classification_metrics.pkl "
        f"and model scores from {DEFAULT_CFS_CACHE_DIR / CANONICAL_CLASSIFICATION_PICKLE} "
        "(fast path).",
    )
    ap.add_argument(
        "--metrics-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Override the metrics cache path "
        "(default: plots/tmp_results/classification_metrics.pkl).",
    )
    ap.add_argument(
        "--per-bin-fpr",
        action="store_true",
        help="Use per-bin FPR for TPR (local ROC cut in each W bin) instead of "
        "a single global score threshold. Output PDFs get a _per_bin_fpr suffix.",
    )
    args = ap.parse_args(argv)
    if args.plots_dir is None:
        args.plots_dir = Path(args.out_dir or DEFAULT_OUT_DIR) / "plots"

    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    out_dir = plots_root / "classification" / "w_bins"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_cache_path = args.metrics_cache or (cache_root / CLF_METRICS_CACHE_NAME)
    cfg: PlotConfig | None = PlotConfig.load(args.config) if args.config else None

    if args.plots_only:
        _run_plots_only(
            args, cfg, out_dir, metrics_cache_path, cache_root, data_root,
            use_global_fpr=not args.per_bin_fpr,
        )
        return

    # --- Normal path ---
    pkl = _resolve_classification_pickle(
        args, _REPO_ROOT, cache_root, data_root
    )
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    results, data_w_by_playlist, data_w_by_split = apply_plot_config_results(
        clf, cfg, playlists=playlists
    )
    clrs = clf["clrs_dict_full"]
    use_global_fpr = False if args.per_bin_fpr else clf.get("use_global_fpr_W", True)
    playlists = ["1A"]

    if cfg is not None:
        clrs = {**clrs, **cfg.colors()}
        clrs = cfg.filter_dict(clrs)

    # Build and save the metrics cache (config-independent).
    if not metrics_cache_path.exists():
        print("Building classification metrics cache …")
        cache = build_metrics_cache(clf)
        save_metrics_cache(cache, metrics_cache_path)

    _run_w_notebook_style(
        results, data_w_by_playlist, clrs, use_global_fpr, out_dir, playlists,
        tag_suffix="", cfg=cfg, data_w_by_split=data_w_by_split,
    )
    _run_w_ngt1(
        results, data_w_by_playlist, clrs, use_global_fpr, out_dir, playlists,
        cfg=cfg, data_w_by_split=data_w_by_split,
    )
    test_results = {
        k: v for k, v in results.items()
        if cfg is None or cfg.model_split(k) == cfg.eval_split
    }
    pid_by_pl = {
        pl: test_results[next(iter(test_results))][0][pl]["pid"] for pl in playlists
    }
    _run_baseline_tpr_fpr_W(
        data_w_by_playlist, out_dir, playlists, pid_by_playlist=pid_by_pl,
    )
    _run_conf_matrix_at_threshold(
        test_results, data_w_by_playlist, out_dir, playlists, use_global_fpr,
    )

    for playlist in playlists:
        data_w = data_w_by_playlist[playlist]
        first_model = next(iter(results))
        pid = results[first_model][0][playlist]["pid"]
        _save_event_composition_W(data_w, pid, out_dir, playlist)
        _save_event_composition_W(data_w, pid, out_dir, playlist, finer_bins=True)


def _load_results_for_conf_matrix(
    args: argparse.Namespace,
    cfg: PlotConfig | None,
    cache_root: Path,
    data_root: Path,
) -> dict | None:
    """Load model results from the classification pickle (needed for score cuts)."""
    pkl = _resolve_classification_pickle(
        args, _REPO_ROOT, cache_root, data_root, plots_only=True
    )
    if not pkl.exists():
        print(
            f"Skipping conf_matrix_at_threshold: classification pickle not found "
            f"({pkl}; pass --classification-pickle)."
        )
        return None
    print(f"Loading classification results for conf matrices from {pkl} …")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)
    results = clf["results"]
    if cfg is not None:
        results = cfg.filter_dict(results)
        results = {
            k: v for k, v in results.items()
            if cfg.model_split(k) == cfg.eval_split
        }
    return results


def _load_overlay_context(
    args: argparse.Namespace,
    cfg: PlotConfig | None,
    cache_root: Path,
    data_root: Path,
) -> tuple[dict | None, dict]:
    """Load split-overlay results and ``data_w_by_split`` when configured."""
    if cfg is None or not overlay_model_names(cfg):
        return None, {}
    pkl = _resolve_classification_pickle(
        args, _REPO_ROOT, cache_root, data_root, plots_only=True
    )
    if not pkl.exists():
        raise SystemExit(
            "Split overlays require the classification pickle "
            f"({pkl}); pass --classification-pickle."
        )
    print(f"Loading split-overlay results from {pkl} …")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)
    results, _, data_w_by_split = apply_plot_config_results(
        clf, cfg, playlists=["1A"]
    )
    return results, data_w_by_split


def _run_plots_only(
    args: argparse.Namespace,
    cfg: PlotConfig | None,
    out_dir: Path,
    metrics_cache_path: Path,
    cache_root: Path,
    data_root: Path,
    *,
    use_global_fpr: bool | None = None,
) -> None:
    """Fast --plots-only path using the precomputed metrics cache."""
    cache = load_metrics_cache(metrics_cache_path)

    clrs = cache["clrs_dict_full"]
    if use_global_fpr is None:
        use_global_fpr = cache["use_global_fpr_W"]
    data_w_by_playlist = cache["data_w_by_playlist"]
    playlists = ["1A"]

    if cfg is not None:
        clrs = {**clrs, **cfg.colors()}
        clrs = cfg.filter_dict(clrs)

    overlay_results, data_w_by_split = _load_overlay_context(
        args, cfg, cache_root, data_root
    )
    results = None
    if not use_global_fpr or overlay_results is not None:
        if overlay_results is not None:
            results = overlay_results
        else:
            results = _load_results_for_conf_matrix(args, cfg, cache_root, data_root)
        if results is None:
            raise SystemExit(
                "Model scores required: pass --classification-pickle "
                "or ensure the classification pickle is available."
            )

    _PDF_FPR_TAG = "_global_fpr" if use_global_fpr else "_per_bin_fpr"

    def _filter(d: dict) -> dict:
        return cfg.filter_dict(d) if cfg is not None else d

    def _draw_one_signal(sig_tag, sig_classes, pdf_stem, title_base):
        for playlist in playlists:
            data_w = data_w_by_playlist[playlist]
            figs = []
            fpr = cache["baseline_fpr"][sig_tag][playlist]
            reco_pred = cache["reco_pred"][sig_tag][playlist]
            y_true_binary = np.isin(cache["pid"][playlist], sig_classes).astype(int)

            if use_global_fpr:
                metrics_all = _filter(cache["metrics_W_clf"][sig_tag][playlist]["all"])
                if overlay_results is not None:
                    overlay_only = {
                        k: overlay_results[k]
                        for k in overlay_model_names(cfg)
                        if k in overlay_results
                    }
                    extra = compute_all_metrics_W_for_config(
                        overlay_only,
                        data_w,
                        data_w_by_split,
                        cfg,
                        sig_classes,
                        fixed_fpr=[fpr],
                        playlist=playlist,
                        use_global_fpr=True,
                    )
                    metrics_all = {**metrics_all, **extra}
                baseline_W = cache["W_clf_baseline"][sig_tag][playlist]
                reco_tpr = cache["W_clf_reco_tpr"][sig_tag][playlist]
            else:
                metrics_all = compute_all_metrics_W_for_config(
                    results,
                    data_w,
                    data_w_by_split,
                    cfg,
                    sig_classes,
                    fixed_fpr=[fpr],
                    playlist=playlist,
                    use_global_fpr=False,
                )
                metrics_all = _filter(metrics_all)
                baseline_W = compute_signal_baseline_W(
                    results, data_w, sig_classes, playlist=playlist,
                )
                is_signal = y_true_binary == 1
                reco_tpr = compute_reco_baseline_recall_per_bin(
                    reco_pred, is_signal, data_w["W_GeV"], data_w["W_bin_edges"],
                )

            fig = plot_multi_classification_vs_W(
                metrics_all, data_w, baseline_W,
                fixed_fpr=[fpr], uncertainties=True,
                reco_baseline_tpr_W=reco_tpr, reco_baseline_global_fpr=fpr,
                colors=clrs, title=rf"{title_base} - MINERvA Open Data Playlist {playlist}",
                use_global_fpr=use_global_fpr, playlist=playlist,
            )
            figs.append(fig); plt.close(fig)

            if use_global_fpr:
                pre_agg = precomputed_inttype_agg(cache["metrics_W_clf"][sig_tag][playlist])
                pre_agg = {code: _filter(agg) for code, agg in pre_agg.items()}
                if overlay_results is not None:
                    pre_agg = merge_inttype_agg_overlays(
                        pre_agg,
                        overlay_results,
                        data_w,
                        data_w_by_split,
                        cfg,
                        sig_classes,
                        fixed_fpr=[fpr],
                        playlist=playlist,
                        use_global_fpr=True,
                    )
                pre_bl = {
                    code: bl
                    for code, bl in cache["W_clf_baseline_inttype"][sig_tag][playlist].items()
                }
                plot_results: dict = overlay_results or {}
            else:
                pre_agg = None
                pre_bl = None
                plot_results = results

            fig = plot_binned_by_inttype(
                plot_results, data_w, sig_classes, x_var="W", xlabel=TRUE_W_XLABEL,
                title=rf"{title_base} - MINERvA Open Data Playlist {playlist} - by interaction type",
                uncertainties=True, fixed_fpr=[fpr],
                reco_baseline_pred=reco_pred, playlist=playlist, colors=clrs,
                use_global_fpr=use_global_fpr,
                precomputed_agg=pre_agg, precomputed_bl_values=pre_bl,
                precomputed_y_true_binary=y_true_binary,
                data_by_model=build_data_by_model(
                    plot_results, cfg, data_w_by_split, playlist
                ),
            )
            figs.append(fig); plt.close(fig)
            fp = out_dir / f"{pdf_stem}_{playlist}{_PDF_FPR_TAG}.pdf"
            save_figures_to_pdf(figs, fp)
            print("Saved:", fp)

    _draw_one_signal("cc1pi", _CC1PI_CLASSES, "eval_cc1pi_tagging_W", r"$CC1\pi^\pm$ tagging")
    _draw_one_signal("cc1pi0", _CC1PI0_CLASSES, "eval_cc1pi0_tagging_W", r"$CC1\pi^0$ tagging")
    _draw_one_signal("ccnpi_ge1", _CCNPI_GE1_CLASSES, "eval_Npi_tagging_W", r"$CCN\pi^\pm$ tagging ($N \geq 1$)")
    _draw_one_signal("ccnpi_gt1", _CCNPI_GT1_CLASSES, "eval_Npi_Ngt1_tagging_W", r"$CCN\pi^\pm$ tagging ($N > 1$)")

    _run_baseline_tpr_fpr_W(
        data_w_by_playlist,
        out_dir,
        playlists,
        reco_pred_by_tag=cache["reco_pred"],
        pid_by_playlist=cache["pid"],
    )

    results_cm = _load_results_for_conf_matrix(args, cfg, cache_root, data_root)
    if results_cm is not None:
        _run_conf_matrix_at_threshold(
            results_cm,
            data_w_by_playlist,
            out_dir,
            playlists,
            use_global_fpr,
            baseline_fpr_by_tag=cache["baseline_fpr"],
            reco_pred_by_tag=cache["reco_pred"],
            cfg=cfg,
        )

    # Event composition
    for playlist in playlists:
        data_w = data_w_by_playlist[playlist]
        pid = cache["pid"][playlist]
        _save_event_composition_W(data_w, pid, out_dir, playlist)
        _save_event_composition_W(data_w, pid, out_dir, playlist, finer_bins=True)


if __name__ == "__main__":
    main()
