"""Per-model step / FLOP curve cuts from plot configs."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from src.eval._plot_config import ModelCurveCuts, PlotConfig, TaskScopedCut
from src.eval.plot_steps import (
    _apply_flop_cut_mask,
    _apply_flop_cut_step_mask,
    _resolve_model_step_cutoffs,
)


def test_resolve_cutoffs_from_per_model_policy_and_match():
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
    cuts = ModelCurveCuts(
        step_cut_policy={"HyperScale-small": "min_val_loss"},
        step_cut_match={"HyperScale-small-rw": "HyperScale-small"},
    )
    cutoffs = _resolve_model_step_cutoffs(curve_runs, model_curve_cuts=cuts)
    assert cutoffs["HyperScale-small"] == pytest.approx(2000.0)
    assert cutoffs["HyperScale-small-rw"] == pytest.approx(2000.0)


def test_explicit_step_cut_caps_policy():
    steps = np.array([1000.0, 2000.0, 3000.0])
    loss = np.array([1.2, 1.0, 1.1])
    curve_runs = OrderedDict({"MLP": [(steps, loss)]})
    cuts = ModelCurveCuts(
        step_cut_policy={"MLP": "min_val_loss"},
        step_cut={"MLP": 1500.0},
    )
    cutoffs = _resolve_model_step_cutoffs(curve_runs, model_curve_cuts=cuts)
    assert cutoffs["MLP"] == pytest.approx(1500.0)


def test_apply_flop_cut_mask():
    x = np.array([15.0, 16.0, 17.0])
    y = np.array([1.2, 1.1, 1.0])
    x_out, y_out = _apply_flop_cut_mask(x, y, max_log10_flop=16.0)
    assert x_out.tolist() == [15.0, 16.0]
    assert y_out.tolist() == [1.2, 1.1]


def test_apply_flop_xmin_mask():
    from src.eval.plot_steps import _apply_flop_xmin_mask

    x = np.array([13.0, 14.0, 15.0])
    y = np.array([1.3, 1.2, 1.1])
    x_out, y_out = _apply_flop_xmin_mask(x, y, min_log10_flop=14.0)
    assert x_out.tolist() == [14.0, 15.0]
    assert y_out.tolist() == [1.2, 1.1]


def test_apply_flop_cut_step_mask_matches_flops_cap():
    flops = 1e9
    steps = np.array([1000.0, 50_000.0, 40_000_000.0, 60_000_000.0])
    loss = np.array([1.2, 1.1, 1.0, 0.95])
    steps_out, loss_out = _apply_flop_cut_step_mask(
        steps, loss, flops_per_step=flops, max_log10_flop=16.7,
    )
    log10_flops = np.log10(steps_out * flops + 1.0)
    assert np.all(log10_flops <= 16.7)
    assert steps_out.tolist() == [1000.0, 50_000.0, 40_000_000.0]
    assert loss_out.tolist() == [1.2, 1.1, 1.0]


def test_apply_log_step_cut_mask():
    from src.eval.plot_steps import _apply_log_step_cut_mask

    steps = np.array([1000.0, 10_000.0, 100_000.0, 1_000_000.0])
    loss = np.array([1.2, 1.1, 1.0, 0.95])
    steps_out, loss_out = _apply_log_step_cut_mask(steps, loss, max_log10_step=4.55)
    log10_steps = np.log10(steps_out + 1.0)
    assert np.all(log10_steps <= 4.55)
    assert steps_out.tolist() == [1000.0, 10_000.0]


def test_task_scoped_cut_scalar_applies_to_both_tasks():
    cut = TaskScopedCut.from_json(16.55)
    assert cut is not None
    assert cut.for_task("classification") == 16.55
    assert cut.for_task("regression") == 16.55


def test_plot_config_loads_per_model_cuts(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        """
        {
          "models": [
            {"name": "A", "color": "#000", "step_cut": 1000, "flop_cut": 16.5, "log_step_cut": 4.55},
            {"name": "B", "color": "#111", "step_cut_match": "A"}
          ]
        }
        """
    )
    cfg = PlotConfig.load(cfg_path)
    cuts = cfg.model_curve_cuts("classification")
    assert cuts.step_cut["A"] == 1000.0
    assert cuts.flop_cut["A"] == 16.5
    assert cuts.log_step_cut["A"] == 4.55
    assert cuts.step_cut_match["B"] == "A"


def test_plot_config_loads_task_scoped_flop_and_step_cuts(tmp_path):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        """
        {
          "models": [
            {
              "name": "HyperScale-small-rw",
              "color": "#60a5fa",
              "flop_cut": {"classification": 16.55, "regression": 16.8},
              "log_step_cut": {"classification": 4.55, "regression": 4.7}
            }
          ]
        }
        """
    )
    cfg = PlotConfig.load(cfg_path)
    clf = cfg.model_curve_cuts("classification")
    reg = cfg.model_curve_cuts("regression")
    assert clf.flop_cut["HyperScale-small-rw"] == 16.55
    assert reg.flop_cut["HyperScale-small-rw"] == 16.8
    assert clf.log_step_cut["HyperScale-small-rw"] == 4.55
    assert reg.log_step_cut["HyperScale-small-rw"] == 4.7
