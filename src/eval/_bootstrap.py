"""Repo path setup for scripts under ``src/eval/``."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """``minerva-data-processing/`` root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def ensure_sys_path() -> Path:
    """Insert repo root and ``notebooks/`` on ``sys.path``; return repo root."""
    root = repo_root()
    for p in (root, root / "notebooks"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


def silence_classification_empty_bin_warnings() -> None:
    """Hide noisy numpy warnings when all runs are NaN in a kinematic bin (expected)."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r".*Mean of empty slice.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=r".*Degrees of freedom <= 0 for slice.*",
    )
