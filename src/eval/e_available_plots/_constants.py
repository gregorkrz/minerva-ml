"""Constants for E_available evaluation plots."""

from __future__ import annotations

EAVAILABLE_SCALE = 1.17  # E_available scale factor for blob recoil energy
DEFAULT_BASELINE_KEY = "blob_recoil_E_scaled"

# Width (in) of the compact IQR/MPV small-paper column; used as the **height** of the
# two-panel :math:`E_\mathrm{reco}/E_\mathrm{true}` ratio figure so they match in print.
SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES: tuple[float, float] = (4.15, 4.32)
# Legend only for compact IQR/MPV small-paper figure (7 pt base + 7%).
SMALL_PAPER_COMPACT_IQR_MPV_LEGEND_FS = 7 * 1.07
