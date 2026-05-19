"""Shared defaults for eval scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Strip trailing data-cap suffix from grouped regression labels, e.g. ``BERT-tiny 6M`` → ``BERT-tiny``.
_GROUPED_LABEL_CAP_SUFFIX = re.compile(r"\s+\d+(?:\.\d+)?[kKmM]$")


def base_model_key_from_plot_label(label: str) -> str:
    """Base wandb model key from a flat or grouped training label."""
    return _GROUPED_LABEL_CAP_SUFFIX.sub("", label).strip()


def model_key_excluded_from_metric_eval_plots(model_key: str) -> bool:
    """Exclude Transformer2-DIS and BERT-tiny* (legend: BERT-small*) from standard eval figures."""
    if model_key == "Transformer2-DIS":
        return True
    if model_key.startswith("BERT-tiny"):
        return True
    return False


def grouped_label_excluded_from_metric_eval_plots(label: str) -> bool:
    """Same as :func:`model_key_excluded_from_metric_eval_plots` for ``ModelName 6M``-style keys."""
    return model_key_excluded_from_metric_eval_plots(base_model_key_from_plot_label(label))


def model_key_excluded_from_training_curve_flops(model_key: str) -> bool:
    """Omit DIS from log-FLOPs training curves; BERT-tiny* (legend BERT-small*) included."""
    return model_key == "Transformer2-DIS"


def model_key_excluded_from_training_curve_steps(model_key: str) -> bool:
    """Only Transformer2-DIS is omitted from log-steps vs validation loss plots."""
    return model_key == "Transformer2-DIS"


def model_key_excluded_from_small_paper_ratio_histogram(model_key: str) -> bool:
    """Small-paper *E* ratio histograms: no DIS, no BERT-tiny* (legend BERT-small*)."""
    if model_key == "Transformer2-DIS":
        return True
    if model_key.startswith("BERT-tiny"):
        return True
    return False


def model_key_excluded_from_small_paper_regression_with_bert(model_key: str) -> bool:
    """Small-paper curves that should list BERT-small*: drop DIS only."""
    return model_key == "Transformer2-DIS"


def filter_classification_results_for_standard_plots(results: dict[str, Any]) -> dict[str, Any]:
    """Drop models that should not appear on classification metric PDFs."""
    return {
        k: v
        for k, v in results.items()
        if not model_key_excluded_from_metric_eval_plots(k)
    }


def filter_regression_training_names(
    training_names: dict[str, dict[str, Any]],
    *,
    small_paper: bool = False,
    small_paper_include_bert: bool = False,
) -> dict[str, dict[str, Any]]:
    """Subset ``training_names`` for regression plots.

    * ``small_paper=False`` — standard eval bundle (no DIS, no BERT-tiny*).
    * ``small_paper=True``, ``small_paper_include_bert=False`` — ratio histograms
      (no DIS, no BERT-tiny*).
    * ``small_paper=True``, ``small_paper_include_bert=True`` — compact IQR/MPV
      style curves (no DIS; BERT-tiny* kept).
    """
    if small_paper:
        pred = (
            model_key_excluded_from_small_paper_regression_with_bert
            if small_paper_include_bert
            else model_key_excluded_from_small_paper_ratio_histogram
        )
    else:
        pred = model_key_excluded_from_metric_eval_plots
    return {
        loss: {k: v for k, v in models.items() if not pred(k)}
        for loss, models in training_names.items()
    }


DEFAULT_CKPT_DIR = Path("/global/cfs/cdirs/m3246/gregork/checkpoints")
DEFAULT_WANDB_TAG = "Run_2703"
DEFAULT_OUT_DIR = Path("out")
# Default directory for plot PDFs (under repo unless path is absolute).
DEFAULT_PLOTS_DIR = Path("plots")
CLASSIFICATION_PICKLE_STEM = "classification"
REGRESSION_PICKLE_STEM = "regression"

# FLOPs per training step (batch size 2048) — same tables as notebooks.
# OmniLearned-medium: scalar used for log-FLOPs plots (full-model training proxy). Frozen-backbone
# training uses fewer FLOPs per step; measure with ``src.scripts.train --calculate-flops``.
# BERT-tiny / tiny-rw: ``train.py --calculate-flops`` (bs=2048, max_particles=33): inference
# 55_128_883_200 FLOPs/batch; table uses training heuristic 3× inference fwd = 165_386_649_600 (same arch).
FLOPS_PER_STEP: dict[str, float] = {
    "BERT-tiny": 165_386_649_600.0,
    "BERT-tiny-rw": 165_386_649_600.0,
    "Transformer-xsmall": 358.5 * 1e9,
    "OmniLearned-small": 1769 * 1e9,
    "OmniLearned-small-int": 1769 * 1e9,
    "MLP": 2.6 * 1e9,
    "OmniLearned-medium": 8035 * 1e9,
    "OmniLearned-small-rw": 1769 * 1e9,
    "Transformer2": 6236 * 1e9,
    # Same architecture / step cost as Transformer2; training data filtered to DIS in submit_train_jobs.
    "Transformer2-DIS": 6236 * 1e9,
    "Transformer-small": 1263 * 1e9,
}

CLRS_CLASSIFICATION: dict[str, str] = {
    "BERT-tiny": "#0d9488",  # teal
    "BERT-tiny-rw": "#7c3aed",  # violet
    # Aliases (same as above): figures use :func:`plot_model_label` for display names.
    "BERT-small": "#0d9488",
    "BERT-small-rw": "#7c3aed",
    "Transformer": "#1f77b4",
    "Transformer-xsmall": "#1f77b4",
    "OmniLearned-small": "#ff7f0e",
    "OmniLearned-small-int": "#66c2a5",
    "Transformer-small": "#17becf",
    "MLP": "#2ca02c",
    "OmniLearned-medium": "#9467bd",
    "OmniLearned-small-rw": "#e377c2",
    "Transformer2": "#d62728",
    "Transformer2-DIS": "#f59e0b",  # amber — distinct from full-data Transformer2 (red)
}


def plot_model_label(name: str) -> str:
    """Human-readable model name for figure legends/titles only.

    Wandb keys and internal dict keys stay ``BERT-tiny`` / ``BERT-tiny-rw``; this
    maps them to ``BERT-small`` / ``BERT-small-rw`` for display. Suffixes such as
    `` 6M`` or ``§0`` are preserved.

    If a series never appears in a plot, the pickle is missing that model: runs must
    be discoverable for the tag (``get_*_runs_by_model_and_cap``) with ``data_cap=-1``,
    and for training-curve PDFs, wandb history must include non-empty ``eval_loss``.
    Log-FLOPs and log-steps combined figures include BERT-small*; DIS is omitted.
    """
    if name.startswith("BERT-tiny-rw"):
        return "BERT-small-rw" + name[len("BERT-tiny-rw") :]
    if name.startswith("BERT-tiny"):
        return "BERT-small" + name[len("BERT-tiny") :]
    return name


def repo_output_path(repo_root: Path, path: Path) -> Path:
    """Resolve *path* under *repo_root* unless *path* is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


CLRS_REGRESSION: dict[str, str] = {
    "BERT-tiny": "#0d9488",  # teal
    "BERT-tiny-rw": "#7c3aed",  # violet
    "BERT-small": "#0d9488",
    "BERT-small-rw": "#7c3aed",
    "Transformer-xsmall": "#1f77b4",
    "Transformer-small": "#17becf",
    "Transformer2": "#d62728",
    "Transformer2-DIS": "#f59e0b",  # amber
    "OmniLearned-small": "#ff7f0e",
    "OmniLearned-small-int": "#66c2a5",
    "MLP": "#2ca02c",
    "OmniLearned-medium": "#9467bd",
    "OmniLearned-small-rw": "#e377c2",
}
