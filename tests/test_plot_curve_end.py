"""Training-curve truncation (stop before overfitting)."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from src.eval._plot_config import CurveEndConfig, ModelCurveCuts
from src.eval.plot_steps import (
    _align_mean_val_loss,
    _resolve_model_step_cutoffs,
    _step_at_min_val_loss,
)


def test_step_at_min_val_loss():
    steps = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    loss = np.array([1.2, 1.0, 1.05, 1.1])
    assert _step_at_min_val_loss(steps, loss) == pytest.approx(2000.0)


def test_resolve_cutoffs_min_val_loss_and_match():
    ref_steps = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    ref_loss = np.array([1.2, 1.0, 1.05, 1.1])
    rw_steps = np.array([1000.0, 2000.0, 3000.0, 5000.0, 6000.0])
    rw_loss = np.array([1.2, 1.0, 1.05, 0.95, 1.3])

    curve_runs = OrderedDict(
        {
            "HyperScale-small": [(ref_steps, ref_loss)],
            "HyperScale-small-rw": [(rw_steps, rw_loss)],
        }
    )
    cfg = CurveEndConfig(
        policy="min_val_loss",
        match={"HyperScale-small-rw": "HyperScale-small"},
    )
    cuts = ModelCurveCuts(legacy_global_min_val_loss=True, legacy_match=dict(cfg.match))
    cutoffs = _resolve_model_step_cutoffs(curve_runs, model_curve_cuts=cuts)
    assert cutoffs["HyperScale-small"] == pytest.approx(2000.0)
    assert cutoffs["HyperScale-small-rw"] == pytest.approx(2000.0)


def test_align_mean_val_loss_single_seed():
    steps = np.array([0.0, 1000.0])
    loss = np.array([1.5, 1.2])
    packed = _align_mean_val_loss([(steps, loss)])
    assert packed is not None
    grid, mean, sigma = packed
    assert grid.tolist() == [0.0, 1000.0]
    assert mean.tolist() == [1.5, 1.2]
    assert np.all(sigma == 0.0)
