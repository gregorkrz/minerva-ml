#!/usr/bin/env python3
"""Figures under ``plots/small_paper/`` that include BERT-small* but not DIS.

Also see ``plot_regression.py`` (ratio histograms omit BERT-small* and DIS).

Outputs:

* ``regression_q3_iqr_mpv_1A_compact.pdf`` — all seeds per model (same bundle as
  ``plot_regression`` main IQR curves: ``eval_data_no_rw`` +
  ``training_names_full_no_rw``), not first-seed-only.
* ``classification_tpr_at_fixed_fpr_baseline_1A.pdf``
* ``classification_val_loss_vs_log10_flops_and_log10_steps.pdf``
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

from src.eval._classification_light import _figure_metrics_1x3, _save_single_fig
from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_OUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_WANDB_TAG,
    REGRESSION_PICKLE_STEM,
    filter_regression_training_names,
    repo_output_path,
)
from src.eval.classification_plots import (
    compute_all_metrics_q3,
    compute_reco_baseline_recall_per_bin,
    compute_signal_baseline,
)
from src.eval.e_available_plots import (
    SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES,
    plot_rms_iqr_with_uncertainty,
)
from src.eval.plot_steps import write_classification_val_loss_log_flops_steps_pdf


def _pickle_paths(out_dir: Path, flag: str) -> tuple[Path, Path]:
    return (
        out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl",
        out_dir / f"{REGRESSION_PICKLE_STEM}_{flag}.pkl",
    )


def _classification_results_allow_bert_drop_dis(results: dict) -> dict:
    return {k: v for k, v in results.items() if k != "Transformer2-DIS"}


def _loss_histories_drop_dis(loss_histories: dict) -> dict:
    return {k: v for k, v in loss_histories.items() if k != "Transformer2-DIS"}


def _save_classification_tpr_ccnpi_q3_baseline_1a(
    *,
    out_pdf: Path,
    results: dict,
    data_by_playlist: dict,
    clrs: dict[str, str],
) -> None:
    """CCNπ 1×3 vs *q₃* at reco baseline FPR (matches light-bundle CCNπ *q₃* panel)."""
    playlist = "1A"
    data = data_by_playlist[playlist]
    test_idx = data["test_idx"][playlist]
    baselines_pl = data["baselines"][playlist]
    first_model = next(iter(results))
    run0 = results[first_model][0][playlist]
    pid = run0["pid"]
    n_muons = baselines_pl["n_muons"][test_idx]
    n_charged_prongs = baselines_pl["n_charged_prongs"][test_idx]
    improved_nmichel = baselines_pl["improved_nmichel"][test_idx]

    multi_pi_classes = [0, 1]
    y_true_ccnpi = np.isin(pid, multi_pi_classes).astype(int)
    y_pred_ccnpi = (
        (n_muons == 1) & (n_charged_prongs >= 1) & (improved_nmichel >= 1)
    ).astype(int)
    fpn = int(np.sum((y_pred_ccnpi == 1) & (y_true_ccnpi == 0)))
    tnn = int(np.sum((y_pred_ccnpi == 0) & (y_true_ccnpi == 0)))
    baseline_fpr_ccnpi = fpn / (fpn + tnn) if (fpn + tnn) > 0 else float("nan")
    if not np.isfinite(baseline_fpr_ccnpi):
        raise SystemExit("Non-finite CCNπ baseline FPR; cannot build TPR figure.")

    fpr_n = [baseline_fpr_ccnpi]
    metrics_q3 = compute_all_metrics_q3(
        results,
        data,
        signal_classes=multi_pi_classes,
        fixed_fpr=fpr_n,
        playlist=playlist,
    )
    bl_q3 = compute_signal_baseline(
        results,
        data,
        signal_classes=multi_pi_classes,
        playlist=playlist,
        pion_bins_require_has_pion=False,
    )["q3"]
    is_signal_ccnpi = y_true_ccnpi == 1
    reco_q3 = compute_reco_baseline_recall_per_bin(
        y_pred_ccnpi,
        is_signal_ccnpi,
        data["q3_GeV"],
        data["q3_bin_edges"],
    )
    fig_n = _figure_metrics_1x3(
        metrics_q3,
        data["q3_bin_mids"],
        r"True $q_3$ [GeV]",
        bl_q3,
        fpr_n,
        reco_q3,
        "Baseline",
        clrs,
        log_x=False,
        reco_baseline_global_fpr=baseline_fpr_ccnpi,
    )
    _save_single_fig(fig_n, out_pdf)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pickle directory (default: out/ under repo)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Plot root (default: plots/ under repo)",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument("--regression-pickle", type=Path, default=None)
    args = ap.parse_args(argv)

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    small_dir = plots_root / "small_paper"
    small_dir.mkdir(parents=True, exist_ok=True)

    clf_p, reg_p = _pickle_paths(data_root, args.flag)
    clf_p = args.classification_pickle or clf_p
    reg_p = args.regression_pickle or reg_p

    with open(clf_p, "rb") as f:
        clf = pickle.load(f)
    with open(reg_p, "rb") as f:
        reg = pickle.load(f)

    results_bert = _classification_results_allow_bert_drop_dis(clf["results"])
    data_by_playlist = clf["data_by_playlist"]
    clrs = clf["clrs_dict_full"]

    _save_classification_tpr_ccnpi_q3_baseline_1a(
        out_pdf=small_dir / "classification_tpr_at_fixed_fpr_baseline_1A.pdf",
        results=results_bert,
        data_by_playlist=data_by_playlist,
        clrs=clrs,
    )

    ylim_c = (1.075, 1.25)
    write_classification_val_loss_log_flops_steps_pdf(
        _loss_histories_drop_dis(clf["loss_histories"]),
        clf["flops_per_step"],
        clrs,
        ylim_c,
        small_dir / "classification_val_loss_vs_log10_flops_and_log10_steps.pdf",
    )

    CKPT_DIR = Path(reg["ckpt_dir"])
    suppress = bool(reg["suppress_errors"])
    baseline_run = reg["baseline_run"]
    # All seeds (grouped eval), same as ``plot_regression`` ``q3_vs_iqr_rms_full_1A`` —
    # not ``*_first_only`` / ``eval_data_first_only``.
    eval_iqr = reg.get("eval_data_no_rw")
    training_names_full = reg.get("training_names_full")
    '''if eval_iqr is not None:
        training_iqr_src = reg["training_names_full_no_rw"]
    else:
        eval_iqr = reg.get("eval_data_first_only")
        training_iqr_src = reg["training_names_full_first_only"]
        print(
            "Warning: regression pickle has no eval_data_no_rw; compact IQR uses "
            "first seed only. Re-run collect_eval_data to cache the multi-seed bundle."
        )'''
    training_names_bert = filter_regression_training_names(
        training_names_full,
        small_paper=True,
        small_paper_include_bert=True,
    )
    fig_i = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=CKPT_DIR,
        training_names=training_names_bert,
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
        data=eval_iqr,
        compact_figsize=SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES,
    )
    out_i = small_dir / "regression_q3_iqr_mpv_1A_compact.pdf"
    out_i.parent.mkdir(parents=True, exist_ok=True)
    fig_i.savefig(out_i, bbox_inches="tight")
    plt.close(fig_i)
    print("Saved:", out_i)


if __name__ == "__main__":
    main()
