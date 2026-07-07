"""Reco-baseline validation CE for training-curve reference lines."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.eval._constants import (
    is_horizontal_reference_model,
    is_steps_plot_excluded_model,
    plot_model_label,
)
from src.eval.baseline_val_loss import (
    RECO_BASELINE_MODEL_KEY,
    inject_reco_baseline_loss_history,
    reco_baseline_loss_series,
)


def test_reco_baseline_is_not_horizontal_reference():
    assert not is_horizontal_reference_model(RECO_BASELINE_MODEL_KEY)
    assert is_steps_plot_excluded_model(RECO_BASELINE_MODEL_KEY)
    assert plot_model_label(RECO_BASELINE_MODEL_KEY) == "Cut baseline"


def test_reco_baseline_loss_series_shape():
    series = reco_baseline_loss_series(1.5)
    assert len(series) == 1
    steps, losses = series[0]
    assert steps.tolist() == [0.0]
    assert losses.tolist() == [1.5]


def test_merge_config_horizontal_refs_regression_side():
    from src.eval.plot_steps import _merge_config_horizontal_refs

    lh_r = {"MLP": [(np.array([1000.0]), np.array([0.03]))]}
    lh_full = {
        **lh_r,
        "BDT": [(np.array([0.0]), np.array([0.05]))],
        "Reco-baseline": [(np.array([0.0]), np.array([0.08]))],
    }
    flops = {"MLP": 1e9, "BDT": 2.6e9, "Reco-baseline": 0.0}
    colors = {"MLP": "#2ca02c", "BDT": "#b91c1c", "Reco-baseline": "#64748b"}
    merged_lh, merged_flops, merged_colors = _merge_config_horizontal_refs(
        lh_r,
        {"MLP": flops["MLP"]},
        {"MLP": colors["MLP"]},
        ref_names={"BDT", "Reco-baseline"},
        source_lh=lh_full,
        source_flops=flops,
        source_colors=colors,
    )
    assert set(merged_lh) == {"MLP", "BDT"}
    assert merged_lh["BDT"][0][1].tolist() == [0.05]
    assert "Reco-baseline" not in merged_lh


def test_regression_reco_baseline_huber_matches_eval_recipe():
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    preds = np.array([1.1, 1.9, 3.2], dtype=np.float32)
    expected = F.huber_loss(
        torch.from_numpy(preds),
        torch.from_numpy(targets),
    ).item()
    assert expected == pytest.approx(
        float(
            F.huber_loss(
                torch.tensor([1.1, 1.9, 3.2]),
                torch.tensor([1.0, 2.0, 3.0]),
            )
        )
    )
    loss_histories: dict = {}
    from src.eval.baseline_val_loss import inject_reco_baseline_regression_loss_history

    inject_reco_baseline_regression_loss_history(
        loss_histories,
        data_path="/nonexistent",
        playlists=(),
    )
    assert RECO_BASELINE_MODEL_KEY not in loss_histories


def test_weighted_ce_matches_eval_recipe():
    """Synthetic sanity check: stable logits, same weighted CE as train.evaluate."""
    mc = torch.tensor([0, 1, 2, 3, 4, 0])
    pred = torch.tensor([0, 1, 2, 3, 4, 3])  # one wrong
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    logits = torch.full((mc.shape[0], 5), -1.0)
    logits[torch.arange(mc.shape[0]), pred] = 0.0
    expected = F.cross_entropy(logits, mc, weight=weights).item()

    loss_histories: dict = {}
    inject_reco_baseline_loss_history(
        loss_histories,
        data_path="/nonexistent",
        playlists=(),
    )
    assert RECO_BASELINE_MODEL_KEY not in loss_histories

    assert expected == F.cross_entropy(logits, mc, weight=weights).item()
