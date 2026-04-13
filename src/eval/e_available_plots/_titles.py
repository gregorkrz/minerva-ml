"""Axis title helpers for q3 / E_true bin labels."""

from __future__ import annotations


def _format_bin_edge_for_title(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:g}"


def _q3_bin_title(qlow: float, qhigh: float, upper_threshold: float = 50.0) -> str:
    lo = _format_bin_edge_for_title(qlow)
    hi = _format_bin_edge_for_title(qhigh)
    if qhigh >= upper_threshold:
        return rf"$q_3 \in [{lo}, \infty)\ \mathrm{{GeV}}$"
    return rf"$q_3 \in [{lo}, {hi}]\ \mathrm{{GeV}}$"


def _Etrue_bin_title(elow: float, ehigh: float, upper_threshold: float = 100.0) -> str:
    lo = _format_bin_edge_for_title(elow)
    hi = _format_bin_edge_for_title(ehigh)
    if ehigh >= upper_threshold:
        return rf"$E_{{\mathrm{{available}}}}^{{\mathrm{{true}}}} \geq {lo}\ \mathrm{{GeV}}$"
    return rf"$E_{{\mathrm{{available}}}}^{{\mathrm{{true}}}} \in [{lo}, {hi})\ \mathrm{{GeV}}$"
