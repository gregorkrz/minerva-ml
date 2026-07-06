"""Cut-based baseline labels for CC1orNPi (Pi_labels_v2) classifier training."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

PI0_MASS_MEV = 134.977


def get_Pi_labels_v2_from_baseline(
    n_muons: np.ndarray,
    n_charged_prongs: np.ndarray,
    improved_nmichel: np.ndarray,
    is_pizero_signal: np.ndarray,
    two_gamma_inv_mass: np.ndarray,
) -> np.ndarray:
    """Map reco cut-based baseline variables to Pi_labels_v2 classes (0–4).

    Uses the same class index scheme as :func:`get_Pi_labels_v2` in
    ``dataloader.py`` and the binary reco baselines in classification eval:

    * 0 — CC 1 charged pion (``n_muons==1``, ``n_charged_prongs==1``, michel≥1)
    * 1 — CC N>1 charged pion (``n_muons==1``, ``n_charged_prongs≥2``, michel≥1)
    * 2 — CC 1 π⁰, no charged pions (π⁰ mass cut, michel==0)
    * 3 — CC other (CC-like reco, none of the above)
    * 4 — NC (``n_muons != 1``)
    """
    n_muons = np.asarray(n_muons)
    n_charged_prongs = np.asarray(n_charged_prongs)
    improved_nmichel = np.asarray(improved_nmichel)
    is_pizero_signal = np.asarray(is_pizero_signal)
    two_gamma_inv_mass = np.asarray(two_gamma_inv_mass)

    labels = np.full(n_muons.shape[0], 4, dtype=np.int64)
    is_cc_reco = n_muons == 1

    cc1pi = is_cc_reco & (n_charged_prongs == 1) & (improved_nmichel >= 1)
    ccnpi_gt1 = is_cc_reco & (n_charged_prongs >= 2) & (improved_nmichel >= 1)
    cc1pi0 = (
        is_cc_reco
        & (is_pizero_signal == 2)
        & (np.abs(two_gamma_inv_mass - PI0_MASS_MEV) < PI0_MASS_MEV)
        & (improved_nmichel == 0)
    )
    cc_other = is_cc_reco & ~cc1pi & ~ccnpi_gt1 & ~cc1pi0

    labels[cc1pi] = 0
    labels[ccnpi_gt1] = 1
    labels[cc1pi0] = 2
    labels[cc_other] = 3
    return labels


def load_baseline_pi_labels_v2(
    data_path: str | Path,
    playlist: str,
    split: str,
) -> np.ndarray:
    """Load per-split baseline Pi_labels_v2 aligned with split-local event indices."""
    data_path = Path(data_path)
    result_path = data_path / "result.pkl"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Missing split index file: {result_path}. "
            "Run split_dataset.py on this dataset first."
        )
    with open(result_path, "rb") as f:
        split_idx = pickle.load(f)
    if playlist not in split_idx:
        raise KeyError(
            f"Playlist '{playlist}' not found in {result_path}. "
            f"Available: {sorted(split_idx)}"
        )
    split_key = f"{split}_idx"
    if split_key not in split_idx[playlist]:
        raise KeyError(
            f"Split '{split}' not found for playlist '{playlist}' in {result_path}."
        )
    local_to_global = np.asarray(split_idx[playlist][split_key], dtype=np.int64)

    baseline_file = data_path / "baselines" / f"{playlist}_enu_baselines.npz"
    if not baseline_file.exists():
        raise FileNotFoundError(
            f"Missing baseline file: {baseline_file}. "
            "Run src/scripts/extract_baselines.py to generate baselines."
        )
    baselines = dict(np.load(baseline_file))

    return get_Pi_labels_v2_from_baseline(
        baselines["n_muons"][local_to_global],
        baselines["n_charged_prongs"][local_to_global],
        baselines["improved_nmichel"][local_to_global],
        baselines["is_pizero_signal"][local_to_global],
        baselines["two_gamma_invariant_mass"][local_to_global],
    )
