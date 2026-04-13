"""Shared defaults for eval scripts."""

from __future__ import annotations

from pathlib import Path

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
FLOPS_PER_STEP: dict[str, float] = {
    "Transformer-xsmall": 358.5 * 1e9,
    "OmniLearned-small": 1769 * 1e9,
    "OmniLearned-small-int": 1769 * 1e9,
    "MLP": 2.6 * 1e9,
    "OmniLearned-medium": 8035 * 1e9,
    "OmniLearned-small-rw": 1769 * 1e9,
    "Transformer2": 6236 * 1e9,
    "Transformer-small": 1263 * 1e9,
}

CLRS_CLASSIFICATION: dict[str, str] = {
    "Transformer": "#1f77b4",
    "Transformer-xsmall": "#1f77b4",
    "OmniLearned-small": "#ff7f0e",
    "OmniLearned-small-int": "#66c2a5",
    "Transformer-small": "#17becf",
    "MLP": "#2ca02c",
    "OmniLearned-medium": "#9467bd",
    "OmniLearned-small-rw": "#e377c2",
    "Transformer2": "#d62728",
}


def repo_output_path(repo_root: Path, path: Path) -> Path:
    """Resolve *path* under *repo_root* unless *path* is already absolute."""
    path = Path(path)
    return path if path.is_absolute() else repo_root / path


CLRS_REGRESSION: dict[str, str] = {
    "Transformer-xsmall": "#1f77b4",
    "Transformer-small": "#17becf",
    "OmniLearned-small": "#ff7f0e",
    "OmniLearned-small-int": "#66c2a5",
    "MLP": "#2ca02c",
    "OmniLearned-medium": "#9467bd",
    "OmniLearned-small-rw": "#e377c2",
}
