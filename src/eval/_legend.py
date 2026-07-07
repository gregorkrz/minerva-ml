"""Shared figure legends (column stacks, outside placement)."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_LEGEND_FS = 10


def legend_spacer_label(slot: int) -> str:
    """Unique invisible label so matplotlib keeps empty legend grid cells."""
    return "\u200b" * (slot + 1)


def order_legend_handles_labels(
    handles: list,
    labels: list[str],
    label_order: list[str] | None,
) -> tuple[list, list[str]]:
    if not label_order:
        return handles, labels
    by_label = dict(zip(labels, handles))
    ordered_labels = [lab for lab in label_order if lab in by_label]
    ordered_labels.extend(sorted(set(by_label) - set(ordered_labels)))
    return [by_label[lab] for lab in ordered_labels], ordered_labels


def layout_legend_with_column_stacks(
    handles: list,
    labels: list[str],
    stacks: tuple[tuple[str, ...], ...],
) -> tuple[list, list[str], int]:
    """Lay out a multi-row legend: each stack shares one column; remaining
    labels (e.g. MLP-rw, Baseline) fill the rightmost column top-to-bottom."""
    by_label = dict(zip(labels, handles))
    resolved_stacks: list[tuple[str, ...]] = []
    stacked_labels: set[str] = set()
    for stack in stacks:
        s = tuple(lab for lab in stack if lab in by_label)
        if s:
            resolved_stacks.append(s)
            stacked_labels.update(s)

    if not resolved_stacks:
        n = len(labels)
        ncol = max(3, min(6, (n + 2) // 3)) if n > 2 else n
        return handles, labels, min(n, ncol + 1)

    non_stack = [lab for lab in labels if lab not in stacked_labels]
    tail_labels = [lab for lab in non_stack if lab != "Baseline"]
    if "Baseline" in non_stack:
        tail_labels.append("Baseline")

    n_rows = max(len(s) for s in resolved_stacks)
    if tail_labels:
        n_rows = max(n_rows, len(tail_labels))
    ncol = len(resolved_stacks) + (1 if tail_labels else 0)
    grid_h: list[list] = [[None] * ncol for _ in range(n_rows)]
    grid_l: list[list[str]] = [[None] * ncol for _ in range(n_rows)]

    for col, stack in enumerate(resolved_stacks):
        for row, lab in enumerate(stack):
            grid_h[row][col] = by_label[lab]
            grid_l[row][col] = lab

    if tail_labels:
        tail_col = len(resolved_stacks)
        for row, lab in enumerate(tail_labels):
            grid_h[row][tail_col] = by_label[lab]
            grid_l[row][tail_col] = lab

    for row in range(n_rows):
        for col in range(ncol):
            if grid_h[row][col] is None:
                slot = row * ncol + col
                grid_h[row][col] = Line2D([], [], linestyle="", marker="", alpha=0.0)
                grid_l[row][col] = legend_spacer_label(slot)

    out_h: list = []
    out_l: list[str] = []
    for col in range(ncol):
        for row in range(n_rows):
            out_h.append(grid_h[row][col])
            out_l.append(grid_l[row][col])
    return out_h, out_l, ncol


def shared_figure_legend(
    fig: plt.Figure,
    axes: tuple[plt.Axes, ...],
    *,
    label_order: list[str] | None = None,
    column_stack_labels: list[list[str]] | None = None,
    legend_fontsize: float | None = None,
) -> None:
    """One legend for all *axes*, de-duplicated by model name."""
    by_label: dict[str, plt.Artist] = {}
    for ax in axes:
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if li and li != "_nolegend_" and li not in by_label:
                by_label[li] = hi
    labels = sorted(by_label.keys())
    handles = [by_label[k] for k in labels]
    handles, labels = order_legend_handles_labels(handles, labels, label_order)
    if not handles:
        return
    if column_stack_labels:
        stacks = tuple(tuple(s) for s in column_stack_labels)
        handles, labels, ncol = layout_legend_with_column_stacks(
            handles, labels, stacks
        )
    else:
        n = len(labels)
        ncol = max(3, min(6, (n + 2) // 3)) if n > 2 else n
        ncol = min(n, ncol + 1)
    fs = float(legend_fontsize if legend_fontsize is not None else DEFAULT_LEGEND_FS)
    legend_kw: dict = dict(
        ncol=ncol,
        fontsize=fs,
        frameon=True,
        fancybox=True,
        facecolor="white",
        edgecolor="0.4",
        columnspacing=1.0,
        handletextpad=0.5,
    )
    try:
        fig.legend(handles, labels, loc="outside lower center", **legend_kw)
    except (TypeError, ValueError):
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            **legend_kw,
        )
