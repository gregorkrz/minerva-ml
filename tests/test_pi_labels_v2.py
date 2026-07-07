"""Tests for MC-truth Pi_labels_v2 mapping."""

import numpy as np
import torch

from src.dataset.dataloader import get_Pi_labels_v2


def _truth_row(*, current=1, n_pi_plus=0, n_pi_minus=0, n_pi_zero=0) -> np.ndarray:
    row = np.zeros(15, dtype=np.float64)
    row[3] = current
    row[5] = n_pi_plus
    row[6] = n_pi_minus
    row[10] = n_pi_zero
    return row


def test_cc1pi_plus_is_class_0():
    labels = get_Pi_labels_v2(torch.from_numpy(_truth_row(n_pi_plus=1)[None]))
    assert labels.item() == 0


def test_cc1pi_minus_is_class_0():
    labels = get_Pi_labels_v2(torch.from_numpy(_truth_row(n_pi_minus=1)[None]))
    assert labels.item() == 0


def test_cc_multi_charged_is_class_1():
    labels = get_Pi_labels_v2(torch.from_numpy(_truth_row(n_pi_plus=2)[None]))
    assert labels.item() == 1


def test_cc1pi0_is_class_2():
    labels = get_Pi_labels_v2(torch.from_numpy(_truth_row(n_pi_zero=1)[None]))
    assert labels.item() == 2


def test_cc_other_zero_charged_is_class_3():
    labels = get_Pi_labels_v2(torch.from_numpy(_truth_row()[None]))
    assert labels.item() == 3


def test_nc_is_class_4():
    labels = get_Pi_labels_v2(
        torch.from_numpy(_truth_row(current=2, n_pi_plus=1)[None])
    )
    assert labels.item() == 4
