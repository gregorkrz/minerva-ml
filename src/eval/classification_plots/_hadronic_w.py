"""Hadronic W augmentation for classification data."""

from __future__ import annotations

import numpy as np

from ._constants import PROTON_MASS_MEV, MUON_MASS_MEV, DEFAULT_W_BIN_EDGES_GEV


def mc_true_hadronic_W_gev_from_baselines(
    baselines: dict[str, np.ndarray],
    test_idx: np.ndarray,
) -> np.ndarray:
    """True MC hadronic *W* (GeV) per test event from baseline ``mc_true_hadronic_W_GeV``.

    This array is written by ``extract_baselines.py`` using
    :func:`extract_baselines.true_hadronic_invariant_W_gev_from_mc_part`
    (sum of non-lepton final-state four-momenta; see that script).  Sentinel
    invalid values are ``-1``; they become ``nan`` here for binning and metrics.

    Parameters
    ----------
    baselines
        Dictionary loaded from ``*_enu_baselines.npz`` (must include the key
        ``mc_true_hadronic_W_GeV``).
    test_idx
        Indices of the test split (same convention as :func:`load_truth_and_baselines`).

    Raises
    ------
    KeyError
        If ``mc_true_hadronic_W_GeV`` is missing — regenerate baselines with
        ``src/scripts/extract_baselines.py``.
    """
    if "mc_true_hadronic_W_GeV" not in baselines:
        raise KeyError(
            "Baselines must contain 'mc_true_hadronic_W_GeV' (true MC hadronic W in GeV). "
            "Re-run src/scripts/extract_baselines.py to regenerate *_enu_baselines.npz files."
        )
    w = np.asarray(baselines["mc_true_hadronic_W_GeV"][test_idx], dtype=np.float64)
    return np.where((w < 0.0) | ~np.isfinite(w), np.nan, w)


def hadronic_invariant_W_gev_from_baselines(
    baselines: dict[str, np.ndarray],
    test_idx: np.ndarray,
) -> np.ndarray:
    """Reco-derived hadronic *W* (GeV) per test event from baseline kinematics (**not** MC truth).

    For **classification vs. true MC** *W*, use :func:`mc_true_hadronic_W_gev_from_baselines`
    via :func:`add_hadronic_W_to_classification_data` (default).  This function remains
    available for comparisons that use the lab-frame expression below.

    Uses the lab-frame expression

        ``W² = M_p² + 2 M_p E_recoil - 2 (E_μ + E_recoil) (E_μ - |p_μ| cos θ_μ) + m_μ²``,

    with all energies in MeV.  The combination ``(E_μ - |p_μ| cos θ_μ)`` is
    reconstructed from the same *q0*, *q3*, and MC incoming neutrino energy
    *E_true* stored in the baseline file as

        ``E_μ - |p_μ| cos θ_μ = (Q² + m_μ²) / (2 E_ν)``,

    where ``Q² = q₃² - q₀²`` (MeV²) with *q₃* the magnitude returned by
    ``extract_baselines.get_q3`` and ``q₀ = E_ν - E_μ``.

    ``E_recoil`` is ``MasterAnaDev_hadron_recoil`` (``E_recoil_only`` in the
    npz); invalid recoil rows (``< 0``) yield NaN for *W*.

    Parameters
    ----------
    baselines
        Dictionary loaded from ``*_enu_baselines.npz``.
    test_idx
        Indices of the test split (same convention as :func:`load_truth_and_baselines`).
    """
    E_mu = np.asarray(baselines["E_muon"][test_idx], dtype=np.float64)
    E_rec = np.asarray(baselines["E_recoil_only"][test_idx], dtype=np.float64)
    q0 = np.asarray(baselines["q0"][test_idx], dtype=np.float64)
    q3 = np.asarray(baselines["q3"][test_idx], dtype=np.float64)
    E_nu = np.asarray(baselines["E_true"][test_idx], dtype=np.float64)

    Mp, mm = PROTON_MASS_MEV, MUON_MASS_MEV
    Q2 = q3 * q3 - q0 * q0
    with np.errstate(divide="ignore", invalid="ignore"):
        emu_minus_pl = (Q2 + mm * mm) / (2.0 * E_nu)

    valid = (E_mu > 0) & (E_nu > 0) & np.isfinite(emu_minus_pl) & (E_rec >= 0)
    W2 = Mp * Mp + 2.0 * Mp * E_rec - 2.0 * (E_mu + E_rec) * emu_minus_pl + mm * mm
    W_gev = np.sqrt(np.maximum(W2, 0.0)) / 1000.0
    W_gev[~valid] = np.nan
    return W_gev


def add_hadronic_W_to_classification_data(
    data: dict[str, Any],
    playlist: str,
    w_bin_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Shallow copy of *data* with ``W_GeV``, ``W_bin_edges``, and ``W_bin_mids``.

    ``W_GeV`` is **true MC hadronic invariant mass** (GeV) from the baselines
    field ``mc_true_hadronic_W_GeV`` produced by ``extract_baselines.py`` — not
    the reco-derived lab-frame *W* from :func:`hadronic_invariant_W_gev_from_baselines`.

    Required keys: ``baselines``, ``test_idx`` (as returned by
    :func:`load_truth_and_baselines`).
    """
    if w_bin_edges is None:
        w_bin_edges = DEFAULT_W_BIN_EDGES_GEV.copy()
    else:
        w_bin_edges = _as_strictly_increasing_bin_edges(w_bin_edges, "w_bin_edges")

    out = dict(data)
    test_idx = data["test_idx"][playlist]
    bl = data["baselines"][playlist]
    out["W_GeV"] = mc_true_hadronic_W_gev_from_baselines(bl, test_idx)
    out["W_bin_edges"] = w_bin_edges
    out["W_bin_mids"] = (w_bin_edges[:-1] + w_bin_edges[1:]) / 2
    return out
