"""Evaluation plotting utilities for E_available models.

Loads evaluation results from checkpoint directories and produces
RMS / IQR vs q3 summary plots (and optional intermediate diagnostics).

Example::

    from src.eval.e_available_plots import plot_rms_iqr

    training_names = {
        "LogMSE": {
            "Transformer": "E_avail_LogMSE_20260224_062703",
        },
    }

    fig = plot_rms_iqr(
        CKPT_DIR="/path/to/checkpoints",
        training_names=training_names,
    )
"""

from __future__ import annotations

from ._constants import DEFAULT_BASELINE_KEY, EAVAILABLE_SCALE
from ._grouped import flatten_grouped_training_names, load_eval_data_grouped
from ._load import load_eval_data
from ._plot_examples import plot_example_E_pred_true, plot_scaling_law
from ._plot_residuals import plot_residuals_by_energy, plot_residuals_by_q3
from ._plot_rms import plot_rms_iqr
from ._plot_uncertainty import plot_rms_iqr_with_uncertainty

__all__ = [
    "DEFAULT_BASELINE_KEY",
    "EAVAILABLE_SCALE",
    "flatten_grouped_training_names",
    "load_eval_data",
    "load_eval_data_grouped",
    "plot_example_E_pred_true",
    "plot_residuals_by_energy",
    "plot_residuals_by_q3",
    "plot_rms_iqr",
    "plot_rms_iqr_with_uncertainty",
    "plot_scaling_law",
]
