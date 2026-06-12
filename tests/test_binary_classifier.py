"""Binary CCN1pipm classifier (Weigh2) training helpers."""

import numpy as np
import pytest

from src.dataset.binned_loss_weights import compute_binned_class_weights
from src.dataset.dataloader import (
    binary_class_counts_from_pid_column,
    pid_to_binary_label,
)
from src.eval.classification_plots._metrics_binned import get_signal_probabilities
from src.eval.classification_plots._signal_definitions import resolve_signal_classes
from src.utils.utils import parse_binned_classifier_model_cap


def test_pid_to_binary_ccn1pipm():
    signal = resolve_signal_classes("CCN1pipm")
    assert signal == [0, 1]
    assert pid_to_binary_label(0, signal) == 1
    assert pid_to_binary_label(1, signal) == 1
    assert pid_to_binary_label(2, signal) == 0
    assert pid_to_binary_label(3, signal) == 0
    assert pid_to_binary_label(4, signal) == 0


def test_binary_class_counts():
    pid = np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)
    idx = np.arange(len(pid))
    counts = binary_class_counts_from_pid_column(pid, idx, [0, 1])
    assert counts.tolist() == [3.0, 3.0]


def test_binned_weights_binary_signal_class_index():
    labels = np.array([0, 1, 0, 1, 0, 1])
    bin_indices = np.zeros_like(labels)
    global_weights = np.array([1.0, 2.0])
    table = compute_binned_class_weights(
        labels, bin_indices, 2, global_weights, signal_classes=[1]
    )
    assert table.shape == (1, 2)
    assert table[0, 1] == pytest.approx(3.0)


def test_get_signal_probabilities_two_class_output():
    result = {
        "1A": {
            "pid": np.array([0, 1, 3, 4]),
            "prediction": np.array(
                [
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.9, 0.1],
                    [0.6, 0.4],
                ]
            ),
        }
    }
    out = get_signal_probabilities(result, [0, 1], playlist="1A")
    np.testing.assert_array_equal(out["ytrue"], [1, 1, 0, 0])
    np.testing.assert_allclose(out["ypred"], [0.8, 0.7, 0.1, 0.4])


def test_parse_binned_classifier_weigh2():
    name = (
        "Run_1703_classifier_Transformer1_data_cap_-1_seed_55_"
        "binnedW_CCN1pipmBin_20260611_120000"
    )
    assert parse_binned_classifier_model_cap(name) == ("Transformer-xsmall-Weigh2", -1)
