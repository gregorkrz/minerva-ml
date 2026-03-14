# Eval Regression – q3 vs IQR/RMS, residuals by energy, IQR vs training samples (bin 2)
# Plots saved as PDFs in out/regression_eval
from sklearn.metrics import confusion_matrix
import seaborn as sns
from eval_classification_plots import (
    load_results,
    load_truth_and_baselines,
    compute_all_metrics,
    compute_all_metrics_q3,
    compute_signal_baseline,
    compute_reco_baseline_recall_per_bin,
    plot_cc1pi_vs_pion_kinematics,
    plot_multi_pion_vs_q3,
    plot_binned_by_inttype,
    plot_prc_curves,
    save_figures_to_pdf,
)
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
if str(ROOT / "notebooks") not in sys.path:
    sys.path.insert(0, str(ROOT / "notebooks"))

CKPT_DIR = Path("/global/cfs/cdirs/m3246/gregork/checkpoints")
OUTPUT_DIR = ROOT / "out" / "regression_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WANDB_TAG = "Run_1203"  # wandb tag to select runs (set as needed)

from src.utils.utils import get_classification_runs_by_model_and_cap

runs_by_model_cap = get_classification_runs_by_model_and_cap(WANDB_TAG)

training_names = {key: value[-1] for key, value in runs_by_model_cap.items()}


PLAYLISTS = ["1A", "1B"]
results = load_results(CKPT_DIR, training_names, playlists=PLAYLISTS)
data_by_playlist = {
    pl: load_truth_and_baselines(CKPT_DIR, training_names, playlists=[pl])
    for pl in PLAYLISTS
}
# For backward compatibility, data is 1A (used in cells that are not yet looped)
data = data_by_playlist["1A"]

cc1pi_classes = [0]

for playlist in PLAYLISTS:
    data = data_by_playlist[playlist]
    figs_cc1pi = []

    # --- Reco baseline: 1 muon, 1 charged prong, 1 Michel ---
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
    baseline_recall_cc1pi = tp / (tp + fn)
    baseline_precision_cc1pi = tp / (tp + fp)
    print(f"[{playlist}] CC1π± reco baseline (1μ + 1 ch.prong + 1 Michel):  "
          f"Precision={baseline_precision_cc1pi:.4f}  "
          f"Recall={baseline_recall_cc1pi:.4f}  "
          f"FPR={baseline_fpr_cc1pi:.4f}")

    reco_label_cc1pi = r"Reco baseline ($1\mu + 1$ ch. prong $+ 1$ Michel)"

    # --- Recompute model metrics with the baseline's FPR ---
    metrics_cc1pi = compute_all_metrics(
        results, data, signal_classes=cc1pi_classes, fixed_fpr=[baseline_fpr_cc1pi],
        playlist=playlist,
    )
    baseline_cc1pi = compute_signal_baseline(results, data, signal_classes=cc1pi_classes, playlist=playlist)

    # --- Per-bin baseline recall for E and theta ---
    is_signal_cc1pi = y_true_cc1pi == 1
    reco_baseline_tpr_cc1pi = {
        "E": compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi, is_signal_cc1pi,
            data["pion_E_MC"], data["pion_E_MC_bins"],
            has_pion=data["has_pion"],
        ),
        "theta": compute_reco_baseline_recall_per_bin(
            y_pred_cc1pi, is_signal_cc1pi,
            data["pion_theta_MC"], data["pion_theta_MC_bins"],
            has_pion=data["has_pion"],
        ),
    }

    # --- Standard kinematic plots (with reco baseline on TPR panel) ---
    fig = plot_cc1pi_vs_pion_kinematics(
        metrics_cc1pi, data, baseline_cc1pi, uncertainties=True,
        fixed_fpr=[baseline_fpr_cc1pi],
        reco_baseline_tpr=reco_baseline_tpr_cc1pi,
        reco_baseline_label=reco_label_cc1pi,
    )
    figs_cc1pi.append(fig)
    fig.show()

    fig = plot_prc_curves(
        results, signal_classes=cc1pi_classes,
        title=r"PRC — $CC1\pi^\pm$ tagging", uncertainties=True,
        playlist=playlist,
    )
    figs_cc1pi.append(fig)
    fig.show()

    fig = plot_binned_by_inttype(
        results, data,
        signal_classes=cc1pi_classes,
        x_var="pion_E",
        xlabel="Pion energy [GeV]",
        title=r"$CC1\pi^\pm$ tagging vs. pion energy — by interaction type",
        log_x=True,
        uncertainties=True,
        fixed_fpr=[baseline_fpr_cc1pi],
        reco_baseline_pred=y_pred_cc1pi,
        reco_baseline_label=reco_label_cc1pi,
        playlist=playlist,
    )
    figs_cc1pi.append(fig)
    fig.show()

    fig = plot_binned_by_inttype(
        results, data,
        signal_classes=cc1pi_classes,
        x_var="pion_theta",
        xlabel="Pion angle [rad]",
        title=r"$CC1\pi^\pm$ tagging vs. pion angle — by interaction type",
        uncertainties=True,
        fixed_fpr=[baseline_fpr_cc1pi],
        reco_baseline_pred=y_pred_cc1pi,
        reco_baseline_label=reco_label_cc1pi,
        playlist=playlist,
    )
    figs_cc1pi.append(fig)
    fig.show()

