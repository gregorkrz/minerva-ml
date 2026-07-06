"""Wandb validation loss history (``eval_loss``) for training-curve plots."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import numpy as np

from ._constants import FLOPS_PER_STEP, is_bdt_model


def get_validation_loss_history(
    run_name: str,
    project: str = "minerva-models",
    with_steps: bool = False,
    *,
    model_name: str | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Return logged ``eval_loss`` for a run (from wandb history).

    Neural runs: points at every 1000 steps. BDT baselines: final ``eval_loss``
    (typically logged once at step 0).
    """
    import wandb

    api = wandb.Api(timeout=60)
    entity = os.environ.get("WANDB_ENTITY") or getattr(api, "default_entity", None)
    if not entity:
        raise RuntimeError(
            "WANDB_ENTITY not set and wandb default_entity unknown. "
            "Run wandb login or set WANDB_ENTITY."
        )
    path = f"{entity}/{project}"
    runs = api.runs(path, filters={"displayName": run_name})
    run = next(iter(runs), None)
    if run is None:
        raise ValueError(f"Wandb run not found: {run_name!r} in {path}")
    hist = run.history()
    if "eval_loss" not in hist.columns:
        return (np.array([]), np.array([])) if with_steps else np.array([])
    mask = hist["eval_loss"].notna()
    steps = np.asarray(hist["_step"], dtype=float)[mask]
    losses = np.asarray(hist["eval_loss"].dropna(), dtype=float)
    step_int = np.round(steps).astype(int)
    if model_name is not None and is_bdt_model(model_name):
        keep = np.ones(len(steps), dtype=bool)
    else:
        keep = (step_int >= 1000) & (step_int % 1000 == 0)
    steps, losses = steps[keep], losses[keep]
    order = np.argsort(steps)
    steps, losses = steps[order], losses[order]
    _, idx = np.unique(np.round(steps).astype(int), return_index=True)
    steps = steps[np.sort(idx)]
    losses = losses[np.sort(idx)]
    if with_steps:
        return steps, losses
    return losses


def classification_runs_per_model(
    runs_by_model_cap: dict[str, dict[int, list[str]]],
    training_names: dict[str, str],
) -> OrderedDict[str, list[str]]:
    """Map model name -> list of run display names (seeds) for cap=-1."""
    runs_per_model: OrderedDict[str, list[str]] = OrderedDict()
    for model in sorted(FLOPS_PER_STEP):
        if model not in runs_by_model_cap or -1 not in runs_by_model_cap[model]:
            run_list = training_names.get(model)
            if run_list is not None:
                runs_per_model[model] = (
                    run_list if isinstance(run_list, list) else [run_list]
                )
            continue
        runs_per_model[model] = runs_by_model_cap[model][-1]
    return runs_per_model


def regression_runs_per_model(
    runs_by_model_cap: dict[str, dict[int, list[str]]],
    training_names_full: dict[str, dict[str, Any]],
) -> OrderedDict[str, list[str]]:
    """Regression: cap=-1 run lists per model."""
    runs_per_model: OrderedDict[str, list[str]] = OrderedDict()
    for model in sorted(FLOPS_PER_STEP):
        if model not in runs_by_model_cap or -1 not in runs_by_model_cap[model]:
            run_name = training_names_full["Log1p"].get(model)
            if run_name:
                runs_per_model[model] = (
                    [run_name] if isinstance(run_name, str) else list(run_name)
                )
            continue
        runs_per_model[model] = runs_by_model_cap[model][-1]
    return runs_per_model


def collect_histories_for_runs(
    runs_per_model: OrderedDict[str, list[str]],
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """For each model, list of (steps, losses) arrays per seed run."""
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for model, run_names in runs_per_model.items():
        series: list[tuple[np.ndarray, np.ndarray]] = []
        for rn in run_names:
            st, lo = get_validation_loss_history(
                rn, with_steps=True, model_name=model,
            )
            if len(st) > 0 and len(lo) > 0:
                series.append((st, lo))
        if series:
            out[model] = series
    return out
