"""Tests for kinematic-binned classification loss weights."""

import numpy as np
import pytest

from src.dataset.binned_loss_weights import (
    INVALID_BIN,
    assign_bin_indices,
    compute_binned_class_weights,
    per_event_loss_weights,
)
from src.eval.classification_plots._signal_definitions import resolve_signal_classes


def test_resolve_signal_classes_case_insensitive():
    assert resolve_signal_classes("CC1pipm") == [0]
    assert resolve_signal_classes("cc1pi0") == [2]
    assert resolve_signal_classes("CCN1pipm") == [0, 1]
    assert resolve_signal_classes("ccnpipm") == [1]


def test_resolve_signal_classes_unknown_raises():
    with pytest.raises(ValueError, match="Unknown signal tag"):
        resolve_signal_classes("not_a_signal")


def test_assign_bin_indices_last_bin_closed():
    edges = np.array([0.0, 1.0, 2.0])
    values = np.array([0.0, 0.5, 1.0, 2.0, np.nan])
    bins = assign_bin_indices(values, edges)
    assert bins.tolist() == [0, 0, 1, 1, INVALID_BIN]


def test_assign_bin_indices_out_of_range_is_invalid():
    edges = np.array([0.0, 1.0, 2.0])
    values = np.array([-0.1, 2.1])
    bins = assign_bin_indices(values, edges)
    assert bins.tolist() == [INVALID_BIN, INVALID_BIN]


def test_compute_binned_class_weights_normal_bin():
    labels = np.array([0, 0, 1, 2, 3, 4])
    bin_indices = np.array([0, 0, 0, 0, 0, 0])
    global_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    signal_classes = [0]

    table = compute_binned_class_weights(
        labels, bin_indices, 5, global_weights, signal_classes
    )
    assert table.shape == (1, 5)
    # class 0 appears twice out of 6 -> weight 6/2 = 3
    assert table[0, 0] == pytest.approx(3.0)
    assert table[0, 1] == pytest.approx(6.0)


def test_compute_binned_class_weights_signal_only_fallback():
    labels = np.array([3, 4, 3])
    bin_indices = np.array([0, 0, 0])
    global_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    signal_classes = [0]

    table = compute_binned_class_weights(
        labels, bin_indices, 5, global_weights, signal_classes
    )
    np.testing.assert_array_equal(table[0], global_weights)


def test_compute_binned_class_weights_background_only_fallback():
    labels = np.array([0, 0])
    bin_indices = np.array([1, 1])
    global_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    signal_classes = [0]

    table = compute_binned_class_weights(
        labels, bin_indices, 5, global_weights, signal_classes
    )
    np.testing.assert_array_equal(table[1], global_weights)


def test_per_event_loss_weights_invalid_bin_uses_global():
    labels = np.array([0, 1])
    bin_indices = np.array([INVALID_BIN, 0])
    weight_table = np.array([[2.0, 4.0, 6.0, 8.0, 10.0]])
    global_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    out = per_event_loss_weights(labels, bin_indices, weight_table, global_weights)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(4.0)
