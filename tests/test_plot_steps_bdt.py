"""BDT horizontal baseline on FLOPs vs validation-loss plots."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from src.eval._constants import is_bdt_model
from src.eval.plot_steps import (
    _draw_bdt_baseline_hlines,
    _loss_values_in_log_steps_window,
    _mean_final_loss_per_seed,
    _split_bdt_runs,
    _validation_loss_y_limits,
    _ylim_including_bdt,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("BDT", True),
        ("BDT-binnedW", True),
        ("BDT-BC-binnedW", True),
        ("MLP", False),
        ("MLP-binnedW", False),
    ],
)
def test_is_bdt_model(name, expected):
    assert is_bdt_model(name) is expected


def test_mean_final_loss_per_seed():
    series = [
        (np.array([0]), np.array([1.0, 0.8])),
        (np.array([0]), np.array([1.2])),
    ]
    mean, sigma = _mean_final_loss_per_seed(series)
    assert mean == pytest.approx(1.0)
    assert sigma == pytest.approx(0.2)


def test_ylim_including_bdt_expands_classification_window():
    bdt = OrderedDict(
        {
            "BDT": [
                (np.array([0.0]), np.array([1.264])),
                (np.array([0.0]), np.array([1.263])),
            ]
        }
    )
    expanded = _ylim_including_bdt((1.075, 1.25), bdt)
    assert expanded is not None
    assert expanded[0] == pytest.approx(1.075)
    assert expanded[1] > 1.25
    assert expanded[1] >= 1.264


def test_validation_loss_y_limits_auto_from_data():
    curve = OrderedDict(
        {
            "MLP": [
                (np.array([1000.0, 2000.0]), np.array([1.20, 1.10])),
            ]
        }
    )
    bdt = OrderedDict(
        {
            "BDT": [
                (np.array([0.0]), np.array([1.264])),
            ]
        }
    )
    ymin, ymax = _validation_loss_y_limits(curve, bdt, ylim=None)
    assert ymin < 1.10 - 0.12 * (1.264 - 1.10)  # bottom pad below data min
    assert ymax >= 1.264
    assert ymax > ymin


def test_validation_loss_y_limits_respects_global_flops_xmin():
    flops = 1e9
    steps = np.array([1_000.0, 100_000_000.0])
    loss = np.array([0.5, 0.03])
    curve = OrderedDict({"M": [(steps, loss)]})
    _, ymax_full = _validation_loss_y_limits(
        curve,
        OrderedDict(),
        ylim=None,
        flops_per_step={"M": flops},
    )
    _, ymax_win = _validation_loss_y_limits(
        curve,
        OrderedDict(),
        ylim=None,
        flops_per_step={"M": flops},
        global_flops_xmin=16.0,
    )
    assert ymax_win < ymax_full
    assert ymax_win < 0.1


def test_loss_values_in_log_steps_window():
    steps = np.array([1e3, 1.5e4, 2e4, 1e5])
    losses = np.array([0.05, 0.04, 0.035, 0.03])
    win = _loss_values_in_log_steps_window(
        steps,
        losses,
        log_steps_xmin=4.0,
        log_steps_xmax=5.0,
    )
    assert len(win) == 2
    assert win.tolist() == [0.04, 0.035]


def test_runs_per_model_excludes_reco_baseline_and_bdt():
    from src.eval.plot_steps import _runs_per_model

    lh = {
        "MLP": [(np.array([1000.0]), np.array([1.0]))],
        "BDT": [(np.array([0.0]), np.array([1.26]))],
        "Reco-baseline": [(np.array([0.0]), np.array([0.08]))],
    }
    flops = {"MLP": 1e9, "BDT": 2.6e9, "Reco-baseline": 0.0}
    runs = _runs_per_model(lh, flops)
    assert "MLP" in runs
    assert "BDT" not in runs
    assert "Reco-baseline" not in runs


def test_is_steps_plot_excluded_model_covers_bdt():
    from src.eval._constants import is_steps_plot_excluded_model

    assert is_steps_plot_excluded_model("BDT")
    assert is_steps_plot_excluded_model("BDT-binnedW")
    assert not is_steps_plot_excluded_model("MLP")


def test_split_mlp_horizontal_ref_classification_only():
    lh = {
        "MLP": [(np.array([1000.0, 2000.0]), np.array([1.20, 1.10]))],
        "BDT": [(np.array([0.0]), np.array([1.26]))],
        "HyperScale-small": [(np.array([1000.0]), np.array([1.15]))],
    }
    flops = {"MLP": 1.0, "BDT": 0.0, "HyperScale-small": 2.0}
    clf_refs = {"BDT", "MLP"}
    reg_refs = {"BDT"}

    curve_c, ref_c = _split_bdt_runs(lh, flops, horizontal_ref_models=clf_refs)
    assert "MLP" in ref_c
    assert "MLP" not in curve_c
    assert "BDT" in ref_c

    curve_r, ref_r = _split_bdt_runs(lh, flops, horizontal_ref_models=reg_refs)
    assert "MLP" in curve_r
    assert "MLP" not in ref_r
    assert "BDT" in ref_r


def test_draw_bdt_baseline_hlines_adds_line_and_errorbar():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import OrderedDict

    fig, ax = plt.subplots()
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 1.5)
    ref = OrderedDict(
        {
            "BDT": [
                (np.array([0.0]), np.array([1.264])),
                (np.array([0.0]), np.array([1.263])),
            ]
        }
    )
    _draw_bdt_baseline_hlines(ax, ref, {"BDT": "#b91c1c"}, lambda m: m)
    dashed = [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert len(dashed) == 1
    assert dashed[0].get_ydata()[0] == pytest.approx(1.2635)
    assert len(ax.containers) == 1  # errorbar container
    plt.close(fig)
