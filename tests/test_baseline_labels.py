"""Tests for cut-based baseline Pi_labels_v2 mapping."""

import numpy as np

from src.dataset.baseline_labels import PI0_MASS_MEV, get_Pi_labels_v2_from_baseline


def test_baseline_labels_cc1pi():
    labels = get_Pi_labels_v2_from_baseline(
        n_muons=np.array([1]),
        n_charged_prongs=np.array([1]),
        improved_nmichel=np.array([1]),
        is_pizero_signal=np.array([0]),
        two_gamma_inv_mass=np.array([-1.0]),
    )
    assert labels[0] == 0


def test_baseline_labels_ccnpi_gt1():
    labels = get_Pi_labels_v2_from_baseline(
        n_muons=np.array([1]),
        n_charged_prongs=np.array([2]),
        improved_nmichel=np.array([1]),
        is_pizero_signal=np.array([0]),
        two_gamma_inv_mass=np.array([-1.0]),
    )
    assert labels[0] == 1


def test_baseline_labels_cc1pi_multi_michel():
    # 1 charged prong with >=2 Michel electrons is still CC-1-charged-pion (class 0),
    # matching the CCNpi+- cut (n_charged_prongs>=1 & improved_nmichel>=1).
    labels = get_Pi_labels_v2_from_baseline(
        n_muons=np.array([1, 1]),
        n_charged_prongs=np.array([1, 1]),
        improved_nmichel=np.array([2, 3]),
        is_pizero_signal=np.array([0, 0]),
        two_gamma_inv_mass=np.array([-1.0, -1.0]),
    )
    assert labels[0] == 0
    assert labels[1] == 0


def test_baseline_labels_cc1pi0():
    labels = get_Pi_labels_v2_from_baseline(
        n_muons=np.array([1]),
        n_charged_prongs=np.array([0]),
        improved_nmichel=np.array([0]),
        is_pizero_signal=np.array([2]),
        two_gamma_inv_mass=np.array([PI0_MASS_MEV]),
    )
    assert labels[0] == 2


def test_baseline_labels_nc():
    labels = get_Pi_labels_v2_from_baseline(
        n_muons=np.array([0]),
        n_charged_prongs=np.array([0]),
        improved_nmichel=np.array([0]),
        is_pizero_signal=np.array([0]),
        two_gamma_inv_mass=np.array([-1.0]),
    )
    assert labels[0] == 4


def test_baseline_labels_cc_other():
    labels = get_Pi_labels_v2_from_baseline(
        n_muons=np.array([1]),
        n_charged_prongs=np.array([1]),
        improved_nmichel=np.array([0]),
        is_pizero_signal=np.array([0]),
        two_gamma_inv_mass=np.array([-1.0]),
    )
    assert labels[0] == 3
