"""Pion binning utilities."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._constants import DEFAULT_N_BINS
from ._hadronic_w import add_hadronic_W_to_classification_data

def equal_frequency_bin_edges(
    x: np.ndarray,
    mask: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    """Histogram edges so **true-signal** rows in *mask* are split ~evenly across bins.

    Sorts finite ``x[mask]`` and places cuts between consecutive blocks of
    ``~n // n_bins`` sorted signal values. Interior cut between block
    boundaries is the midpoint of the two adjacent values (ties broken with
    ``np.nextafter``). ``edges[0]`` and ``edges[-1]`` are the signal min and
    max so all signal land in some bin under :func:`mc_value_in_bin`.

    All events (signal + background) are then assigned with :func:`mc_value_in_bin`.
    Background **histogram** counts per bin still vary with kinematics; only
    **signal** counts are balanced by construction.
    """
    vals = np.asarray(x[mask], dtype=float)
    vals = vals[np.isfinite(vals)]
    vals.sort()
    n = len(vals)
    if n < n_bins:
        raise ValueError(
            f"equal_frequency_bin_edges: need at least n_bins={n_bins} "
            f"finite values under mask, got {n}"
        )
    splits = (np.arange(n_bins + 1, dtype=np.int64) * n / n_bins).astype(np.int64)
    splits[-1] = n
    for j in range(1, n_bins + 1):
        splits[j] = max(int(splits[j]), int(splits[j - 1]))
    splits[0] = 0

    edges = np.empty(n_bins + 1, dtype=np.float64)
    edges[0] = float(vals[0])
    edges[-1] = float(vals[-1])
    for j in range(1, n_bins):
        sj = int(splits[j])
        sj = min(max(sj, 1), n - 1)
        left_max = float(vals[sj - 1])
        right_min = float(vals[sj])
        if right_min > left_max:
            edges[j] = 0.5 * (left_max + right_min)
        else:
            edges[j] = float(np.nextafter(left_max, np.inf))

    for j in range(1, n_bins):
        if edges[j] <= edges[j - 1]:
            edges[j] = float(np.nextafter(edges[j - 1], np.inf))
    edges[-1] = float(vals[-1])
    return edges


def _as_strictly_increasing_bin_edges(edges: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Validate user-supplied histogram edges (1-D, finite, strictly increasing)."""
    arr = np.asarray(edges, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1-D sequence with at least 2 edges, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(arr) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def data_with_signal_pion_bins(
    data: dict[str, Any],
    pid: np.ndarray,
    signal_classes: list[int],
    n_bins: int | None = None,
    pion_quantile_require_has_pion: bool = True,
    pion_bin_edge_method: str = "equal_frequency",
    *,
    pion_E_bin_edges: np.ndarray | Sequence[float] | None = None,
    pion_theta_bin_edges: np.ndarray | Sequence[float] | None = None,
) -> dict[str, Any]:
    """Shallow copy of *data* with pion E/θ bin edges from **true signal** only.

    **Edge methods** (*pion_bin_edge_method*):

    * ``"equal_frequency"`` (default): edges split sorted signal :math:`E` / θ
      into nearly equal counts per bin (best-effort with ties).
    * ``"quantile"``: ``np.quantile`` on the same signal subset (can differ from
      exact equal counts when there are duplicate values).
    * ``"custom"``: use keyword arguments *pion_E_bin_edges* and *pion_theta_bin_edges*
      (each strictly increasing, length :math:`N+1` for :math:`N` bins). *n_bins* is
      ignored in this mode; :math:`E` and θ may use different bin counts.

    **Masks for which signal rows define edges** (*pion_quantile_require_has_pion*):

    * True (legacy): ``has_pion``, ``E > 0``, finite θ, restricted to signal.
    * False: ``signal & finite E`` for :math:`E`, ``signal & finite θ`` for θ,
      matching ``compute_all_metrics(..., pion_bins_require_has_pion=False)``.

    **ROC / AUPRC per bin** (:func:`bin_separation_metrics`): positives are
    signal in that kinematic bin; **negatives are the full background set**
    (every non-signal event). So **the same background events — and the same
    count of negatives — are used for every bin’s curve**; only the in-bin
    signal positives change.

    **Count histograms** (``N_\\mathrm{total}``, ``N_\\mathrm{signal}``): background
    per bin is “non-signal whose :math:`E` or θ falls in that bin”; those counts
    **cannot** all match each other while also using signal-balanced edges,
    unless signal and background share the same kinematic distribution.

    Use separate calls for CC1π± (e.g. ``[0]``) and CCπ⁰ (e.g. ``[2]``).
    """
    if n_bins is None:
        n_bins = len(data["pion_E_MC_bins"]) - 1

    pid_i = np.asarray(pid).astype(int, copy=False)
    sig = np.isin(pid_i, np.asarray(signal_classes, dtype=int))

    has_pion = data["has_pion"]
    pion_E = data["pion_E_MC"]
    pion_theta = data["pion_theta_MC"]

    if pion_quantile_require_has_pion:
        m_e = sig & has_pion & (pion_E > 0)
        m_th = sig & has_pion & np.isfinite(pion_theta)
    else:
        m_e = sig & np.isfinite(pion_E)
        m_th = sig & np.isfinite(pion_theta)

    if m_e.sum() < 2:
        raise ValueError(
            f"Too few events for E binning (n={int(m_e.sum())}); "
            f"signal_classes={signal_classes}"
        )
    if m_th.sum() < 2:
        raise ValueError(
            f"Too few events for θ binning (n={int(m_th.sum())}); "
            f"signal_classes={signal_classes}"
        )

    if pion_bin_edge_method != "custom" and (
        pion_E_bin_edges is not None or pion_theta_bin_edges is not None
    ):
        raise ValueError(
            "pion_E_bin_edges / pion_theta_bin_edges are only used when "
            "pion_bin_edge_method='custom'"
        )

    if pion_bin_edge_method == "quantile":
        pion_E_bins = np.quantile(pion_E[m_e], np.linspace(0, 1, n_bins + 1))
        pion_theta_bins = np.quantile(pion_theta[m_th], np.linspace(0, 1, n_bins + 1))
    elif pion_bin_edge_method == "equal_frequency":
        pion_E_bins = equal_frequency_bin_edges(pion_E, m_e, n_bins)
        pion_theta_bins = equal_frequency_bin_edges(pion_theta, m_th, n_bins)
    elif pion_bin_edge_method == "custom":
        if pion_E_bin_edges is None or pion_theta_bin_edges is None:
            raise ValueError(
                "pion_bin_edge_method='custom' requires keyword arguments "
                "pion_E_bin_edges and pion_theta_bin_edges"
            )
        pion_E_bins = _as_strictly_increasing_bin_edges(pion_E_bin_edges, "pion_E_bin_edges")
        pion_theta_bins = _as_strictly_increasing_bin_edges(
            pion_theta_bin_edges, "pion_theta_bin_edges"
        )
    else:
        raise ValueError(
            f"pion_bin_edge_method must be 'quantile', 'equal_frequency', or 'custom', "
            f"got {pion_bin_edge_method!r}"
        )

    out = dict(data)
    out["pion_E_MC_bins"] = pion_E_bins
    out["pion_E_MC_bins_mid"] = (pion_E_bins[:-1] + pion_E_bins[1:]) / 2
    out["pion_theta_MC_bins"] = pion_theta_bins
    out["pion_theta_MC_bins_mid"] = (pion_theta_bins[:-1] + pion_theta_bins[1:]) / 2
    return out

