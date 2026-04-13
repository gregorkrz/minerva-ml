"""Grouped training_names (seeds) and colour helpers."""

from __future__ import annotations

import re
from typing import Any
from pathlib import Path
import numpy as np

from ._load import load_eval_data

_SEED_SEP = "§"


def _extract_sample_count(label: str) -> float:
    """Extract numeric sample count from labels like 'Transformer 6M' or '500k'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*([kKmM])\b", label)
    if match:
        num = float(match.group(1))
        suffix = match.group(2).upper()
        return num * (1e6 if suffix == "M" else 1e3)
    match = re.search(r"(\d+)", label)
    return float(match.group(1)) if match else 0.0


def _resolve_color_map(
    config_labels: list[str],
    colors: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a label → colour mapping from a user-supplied *colors* dict.

    *colors* maps human-readable sample-count tokens (e.g. ``"6M"``,
    ``"500k"``) to any matplotlib-compatible colour.  Each config label
    is matched to the first token that appears as a substring.  Labels
    without a match get a default grey.
    """
    if not config_labels:
        return {}
    if colors is None:
        colors = {}
    out: dict[str, Any] = {}
    for label in config_labels:
        if label in colors:
            out[label] = colors[label]
        else:
            out[label] = "tab:gray"
    return out


def flatten_grouped_training_names(
    training_names_grouped: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, str]]:
    """Flatten ``{loss: {label: [run, …]}}`` → ``{loss: {label§i: run, …}}``."""
    flat: dict[str, dict[str, str]] = {}
    for loss, configs in training_names_grouped.items():
        flat[loss] = {}
        for label, runs in configs.items():
            for i, run in enumerate(runs):
                flat[loss][f"{label}{_SEED_SEP}{i}"] = run
    return flat


def load_eval_data_grouped(
    CKPT_DIR: str | Path,
    training_names_grouped: dict[str, dict[str, list[str]]],
    **kwargs: Any,
) -> dict[str, Any]:
    """Load evaluation data for grouped training names (seeds as lists)."""
    return load_eval_data(
        CKPT_DIR,
        flatten_grouped_training_names(training_names_grouped),
        **kwargs,
    )
