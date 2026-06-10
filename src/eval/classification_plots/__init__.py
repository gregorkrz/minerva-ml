"""Evaluation plotting utilities for charged-pion classification models.

Loads evaluation results from checkpoint directories and produces
AUPRC / AUROC / TPR vs kinematic bins for pion energy, pion angle, true *q₃*,
or true MC hadronic invariant mass *W* (GeV) from baselines.  TPR at a target
FPR can use either a **single score threshold** fit on the full masked sample
(``use_global_fpr=True``) or **per-bin** ROC cuts (``use_global_fpr=False``).
Supports multi-run uncertainty bands.

Example::

    from src.eval.classification_plots import (
        load_results,
        load_truth_and_baselines,
        compute_all_metrics,
        plot_cc1pi_vs_pion_kinematics,
    )
"""

from __future__ import annotations

from ._binning import data_with_signal_pion_bins, equal_frequency_bin_edges
from ._constants import (
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    DEFAULT_FIXED_FPR,
    DEFAULT_N_BINS,
    DEFAULT_Q3_BIN_EDGES,
    DEFAULT_W_BIN_EDGES_GEV,
    INT_TYPE_COLORS,
    MC_INT_TYPE,
    MC_INT_TYPE_MERGED,
    MUON_MASS_MEV,
    PROTON_MASS_MEV,
    W_METRICS_XLIM_GEV,
    merge_int_type_arr,
)
from ._hadronic_w import (
    add_hadronic_W_to_classification_data,
    hadronic_invariant_W_gev_from_baselines,
    mc_true_hadronic_W_gev_from_baselines,
)
from ._io import load_results, load_truth_and_baselines
from ._metrics_binned import (
    aggregate_metrics,
    bin_separation_metrics,
    compute_binned_metrics_q3,
    compute_binned_metrics_single,
    compute_binned_metrics_W,
    get_signal_probabilities,
    mc_value_in_bin,
)
from ._metrics_tasks import (
    compute_all_metrics,
    compute_all_metrics_q3,
    compute_all_metrics_W,
    compute_signal_baseline,
    compute_signal_baseline_W,
)
from ._plotting import (
    _plot_metric_line,
    _shared_light_legend,
    plot_binned_by_inttype,
    plot_cc1pi_vs_pion_kinematics,
    plot_composition_vs_kinematic,
    plot_confusion_matrices_at_threshold_W,
    plot_multi_classification_vs_W,
    plot_multi_pion_vs_q3,
    plot_prc_curves,
    plot_signal_composition_single_pion,
    save_figures_to_pdf,
)
from ._reco_baseline import (
    compute_reco_baseline_fpr_per_bin,
    compute_reco_baseline_recall_per_bin,
)

__all__ = [
    "CLASSIFICATION_PERFORMANCE_LEGEND_TITLE",
    "DEFAULT_FIXED_FPR",
    "DEFAULT_N_BINS",
    "DEFAULT_Q3_BIN_EDGES",
    "DEFAULT_W_BIN_EDGES_GEV",
    "INT_TYPE_COLORS",
    "MC_INT_TYPE",
    "MC_INT_TYPE_MERGED",
    "MUON_MASS_MEV",
    "PROTON_MASS_MEV",
    "W_METRICS_XLIM_GEV",
    "add_hadronic_W_to_classification_data",
    "aggregate_metrics",
    "bin_separation_metrics",
    "compute_all_metrics",
    "compute_all_metrics_q3",
    "compute_all_metrics_W",
    "compute_binned_metrics_q3",
    "compute_binned_metrics_single",
    "compute_binned_metrics_W",
    "compute_reco_baseline_fpr_per_bin",
    "compute_reco_baseline_recall_per_bin",
    "compute_signal_baseline",
    "compute_signal_baseline_W",
    "data_with_signal_pion_bins",
    "equal_frequency_bin_edges",
    "get_signal_probabilities",
    "hadronic_invariant_W_gev_from_baselines",
    "load_results",
    "load_truth_and_baselines",
    "mc_true_hadronic_W_gev_from_baselines",
    "mc_value_in_bin",
    "merge_int_type_arr",
    "plot_binned_by_inttype",
    "plot_cc1pi_vs_pion_kinematics",
    "plot_composition_vs_kinematic",
    "plot_confusion_matrices_at_threshold_W",
    "plot_multi_classification_vs_W",
    "plot_multi_pion_vs_q3",
    "plot_prc_curves",
    "plot_signal_composition_single_pion",
    "save_figures_to_pdf",
    "_plot_metric_line",
    "_shared_light_legend",
]
