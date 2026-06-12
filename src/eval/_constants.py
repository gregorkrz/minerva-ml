"""Shared defaults for eval scripts."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CKPT_DIR = Path("/global/cfs/cdirs/m3246/gregork/checkpoints")
DEFAULT_WANDB_TAG = "Run_2703"
DEFAULT_OUT_DIR = Path("out")
# Default directory for plot PDFs (under repo unless path is absolute).
CLASSIFICATION_PICKLE_STEM = "classification"
REGRESSION_PICKLE_STEM = "regression"
# Canonical cache paths used by --plots-only (no flag suffix, no wandb needed).
DEFAULT_CACHE_DIR = Path("plots/tmp_results")
DEFAULT_CFS_PLOTS_DIR = Path("/global/cfs/cdirs/m3246/gregork/Minerva/runs/plots")
DEFAULT_CFS_CACHE_DIR = DEFAULT_CFS_PLOTS_DIR / "tmp_results"
CANONICAL_CLASSIFICATION_PICKLE = "classification.pkl"
CANONICAL_REGRESSION_PICKLE = "regression.pkl"


def canonical_classification_pickle_paths(
    repo_root: Path,
    flag: str = DEFAULT_WANDB_TAG,
    data_root: Path | None = None,
) -> list[Path]:
    """Candidate paths for the classification pickle (largest existing file wins)."""
    runs_root = DEFAULT_CFS_PLOTS_DIR.parent
    paths = [
        runs_root / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl",
        DEFAULT_CFS_CACHE_DIR / CANONICAL_CLASSIFICATION_PICKLE,
        repo_output_path(repo_root, DEFAULT_CACHE_DIR) / CANONICAL_CLASSIFICATION_PICKLE,
    ]
    if data_root is not None:
        paths.insert(1, data_root / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl")
    return paths

# FLOPs per training step (batch size 2048) — same tables as notebooks.
# OmniLearned-medium: scalar used for log-FLOPs plots (full-model training proxy). Frozen-backbone
# training uses fewer FLOPs per step; measure with ``src.scripts.train --calculate-flops``.
# BERT-tiny / tiny-rw: ``train.py --calculate-flops`` (bs=2048, max_particles=33): inference
# 55_128_883_200 FLOPs/batch; table uses training heuristic 3× inference fwd = 165_386_649_600 (same arch).
# HyperScale-small / -rw (embedding, bs=2048): training heuristic 3× inference = 1_015_349_342_208.
# HyperScale-medium / -rw (embedding, bs=2048): training heuristic 3× inference = 6_766_708_033_536.
FLOPS_PER_STEP: dict[str, float] = {
    "BERT-tiny": 165_386_649_600.0,
    "BERT-tiny-rw": 165_386_649_600.0,
    "BERT-tiny-energy-order": 165_386_649_600.0,
    "Transformer-xsmall": 358.5 * 1e9,
    "Transformer-xsmall-Weigh1": 358.5 * 1e9,
    "Transformer-xsmall-Weigh2": 358.5 * 1e9,
    "OmniLearned-small": 1769 * 1e9,
    "OmniLearned-small-int": 1769 * 1e9,
    "MLP": 2.6 * 1e9,
    "OmniLearned-medium": 8035 * 1e9,
    "OmniLearned-small-rw": 1769 * 1e9,
    "Transformer2": 6236 * 1e9,
    # Same architecture / step cost as Transformer2; training data filtered to DIS in submit_train_jobs.
    "Transformer2-DIS": 6236 * 1e9,
    "Transformer-small": 1263 * 1e9,
    "HyperScale-small": 1_015_349_342_208.0,
    "HyperScale-small-rw": 1_015_349_342_208.0,
    "HyperScale-medium": 6_766_708_033_536.0,
    "HyperScale-medium-rw": 6_766_708_033_536.0,
}

CLRS_CLASSIFICATION: dict[str, str] = {
    "BERT-tiny": "#0d9488",  # teal
    "BERT-tiny-rw": "#7c3aed",  # violet
    "BERT-tiny-energy-order": "#059669",  # emerald
    # Aliases (same as above): figures use :func:`plot_model_label` for display names.
    "BERT-small": "#0d9488",
    "BERT-small-rw": "#7c3aed",
    "Transformer": "#1f77b4",
    "Transformer-xsmall": "#1f77b4",
    "Transformer-xsmall-Weigh1": "#2563eb",
    "Transformer-xsmall-Weigh2": "#1d4ed8",
    "OmniLearned-small": "#ff7f0e",
    "OmniLearned-small-int": "#66c2a5",
    "Transformer-small": "#17becf",
    "MLP": "#2ca02c",
    "OmniLearned-medium": "#9467bd",
    "OmniLearned-small-rw": "#e377c2",
    "Transformer2": "#d62728",
    "Transformer2-DIS": "#f59e0b",  # amber — distinct from full-data Transformer2 (red)
    "HyperScale-small": "#0284c7",  # sky-600
    "HyperScale-small-rw": "#38bdf8",  # sky-400
    "HyperScale-medium": "#c026d3",  # fuchsia-600
    "HyperScale-medium-rw": "#e879f9",  # fuchsia-400
}


def is_bert_model(name: str) -> bool:
    return name.startswith("BERT-")


def is_hyperscale_model(name: str) -> bool:
    return name.startswith("HyperScale-")


def is_base_steps_model(name: str) -> bool:
    """Default steps-plot filter: excludes BERT and HyperScale models."""
    return not is_bert_model(name) and not is_hyperscale_model(name)


# ``plot_steps`` small-architecture comparison (pretrained dark, rw light per family).
STEPS_SMALL_MODELS: frozenset[str] = frozenset({
    "HyperScale-small",
    "HyperScale-small-rw",
    "HyperScale-medium",
    "HyperScale-medium-rw",
    "OmniLearned-small",
    "OmniLearned-small-rw",
    "BERT-tiny",
    "BERT-tiny-rw",
})

STEPS_SMALL_MODEL_COLORS: dict[str, str] = {
    "HyperScale-small": "#0369a1",  # sky-700
    "HyperScale-small-rw": "#7dd3fc",  # sky-300
    "HyperScale-medium": "#86198f",  # fuchsia-800
    "HyperScale-medium-rw": "#e879f9",  # fuchsia-400
    "OmniLearned-small": "#c2410c",  # orange-700
    "OmniLearned-small-rw": "#fdba74",  # orange-300
    "BERT-tiny": "#0f766e",  # teal-700
    "BERT-tiny-rw": "#5eead4",  # teal-300
}


def plot_model_label(name: str) -> str:
    """Human-readable model name for figure legends/titles only.

    Wandb keys and internal dict keys stay ``BERT-tiny`` / ``BERT-tiny-rw``; this
    maps them to ``BERT-small`` / ``BERT-small-rw`` for display. Suffixes such as
    `` 6M`` or ``§0`` are preserved.

    If a series never appears in a plot, the pickle is missing that model: runs must
    be discoverable for the tag (``get_*_runs_by_model_and_cap``) with ``data_cap=-1``,
    and for training-curve PDFs, wandb history must include non-empty ``eval_loss``.
    """
    if name.startswith("BERT-tiny-rw"):
        return "BERT-small-rw" + name[len("BERT-tiny-rw") :]
    if name.startswith("BERT-tiny-energy-order"):
        return "BERT-small-energy-order" + name[len("BERT-tiny-energy-order") :]
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
    "BERT-tiny-energy-order": "#059669",  # emerald
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
    "HyperScale-small": "#0284c7",
    "HyperScale-small-rw": "#38bdf8",
    "HyperScale-medium": "#c026d3",
    "HyperScale-medium-rw": "#e879f9",
}
