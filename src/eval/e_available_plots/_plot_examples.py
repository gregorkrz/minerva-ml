"""Example scatter and scaling-law plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np

from src.eval._constants import plot_model_label

from ._constants import DEFAULT_BASELINE_KEY, MIN_ETRUE_GEV
from ._load import _build_event_mask
from ._grouped import (
    _extract_sample_count,
    _resolve_color_map,
    flatten_grouped_training_names,
    load_eval_data_grouped,
    _SEED_SEP,
)


def plot_example_E_pred_true(
    CKPT_DIR: str | Path,
    training_names: dict[str, dict[str, str]] | dict[str, dict[str, list[str]]],
    playlists: list[str] | None = None,
    dataset_to_plot: str = "1A",
    baseline_run: str | None = None,
    baseline_ref: tuple[str, str] | None = None,
    baseline_key: str = DEFAULT_BASELINE_KEY,
    use_cc_selection: int = 2,
    n_examples: int = 10,
    seed: int = 42,
    data: dict | None = None,
    suppress_errors: bool = False,
    verbose: bool = True,
    colors: dict[str, Any] | None = None,
    transform=None,
    first_seed_only: bool = True,
) -> plt.Figure:
    """Scatter *E_true* vs *E_pred* for a random subset of selected events (debug).

    Uses the same event selection as :func:`plot_rms_iqr_with_uncertainty`
    (*use_cc_selection*, *baseline_key*).  By default samples from the first
    seed of each config only (*first_seed_only=True*).

    Also prints a small table to stdout for each method.
    """
    training_names_grouped: dict[str, dict[str, list[str]]] = {}
    for loss, models in training_names.items():
        training_names_grouped.setdefault(loss, {})
        for key, val in models.items():
            if isinstance(val, list):
                runs = val
                if not runs:
                    continue
                training_names_grouped[loss][key] = runs
            else:
                run_name = val
                if run_name is None:
                    continue
                config_label = str(key)
                training_names_grouped[loss].setdefault(config_label, [])
                training_names_grouped[loss][config_label].append(run_name)

    dp = dataset_to_plot
    if data is None:
        data = load_eval_data_grouped(
            CKPT_DIR,
            training_names_grouped,
            playlists=playlists,
            baseline_ref=baseline_ref,
            baseline_run=baseline_run,
            verbose=verbose,
            transform=transform,
            suppress_errors=suppress_errors,
        )

    E_pred_dict = data["E_pred_dict"]
    E_true_dict = data["E_true_dict"]
    Enu_baselines = data["Enu_baselines"]
    Enu_filters = data["Enu_filters"]

    if dp not in Enu_filters:
        warnings.warn(f"No filter data for dataset '{dp}'.")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, f"No Enu_filters for {dp}", ha="center", va="center")
        return fig

    mask_sel = _build_event_mask(
        dp, Enu_filters, Enu_baselines, baseline_key, use_cc_selection
    )

    entries: list[tuple[str, str, str, np.ndarray, np.ndarray, str]] = []
    all_labels: list[str] = []
    for loss in training_names_grouped:
        for config_label, runs in training_names_grouped[loss].items():
            if not runs:
                continue
            seed_indices = [0] if first_seed_only else list(range(len(runs)))
            for s in seed_indices:
                flat_key = f"{config_label}{_SEED_SEP}{s}"
                if flat_key not in E_pred_dict.get(dp, {}).get(loss, {}):
                    if verbose:
                        print(
                            f"  [skip] {flat_key} not in E_pred_dict['{dp}']['{loss}']"
                        )
                    continue
                true_vec = np.asarray(E_true_dict[dp][loss][flat_key]).flatten()
                pred_vec = np.asarray(E_pred_dict[dp][loss][flat_key]).flatten()
                valid = mask_sel & (true_vec >= MIN_ETRUE_GEV)
                idx = np.flatnonzero(valid)
                if idx.size == 0:
                    if verbose:
                        print(f"  [skip] {flat_key}: no events pass selection")
                    continue
                label_off = sum(ord(c) for c in config_label) % 10_007
                rng = np.random.default_rng(seed + s * 100_003 + label_off)
                k = min(n_examples, int(idx.size))
                pick = rng.choice(idx, size=k, replace=False)
                t = true_vec[pick]
                p = pred_vec[pick]
                base = plot_model_label(config_label)
                title = f"{base}" if first_seed_only else f"{base} §{s}"
                entries.append((title, loss, flat_key, t, p, config_label))
                if config_label not in all_labels:
                    all_labels.append(config_label)

    if not entries:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No E_pred / selection overlap", ha="center", va="center")
        return fig

    color_map = _resolve_color_map(all_labels, colors)
    n_plots = len(entries)
    ncols = int(np.ceil(np.sqrt(n_plots)))
    nrows = int(np.ceil(n_plots / ncols))
    fig_w = min(4.0 * ncols, 20)
    fig_h = min(3.8 * nrows, 24)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for i, (title, loss, flat_key, t, p, cfg_key) in enumerate(entries):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ax.set_visible(True)
        color = color_map.get(cfg_key, "tab:gray")
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        if lo >= hi:
            hi = lo + 1e-6
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, alpha=0.5)
        ax.scatter(t, p, c=color, s=36, zorder=2, edgecolors="white", linewidths=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\n({loss})", fontsize=9)
        ax.set_xlabel(r"$E_{\mathrm{available}}^{\mathrm{true}}$ [GeV]")
        ax.set_ylabel(r"$E_{\mathrm{available}}^{\mathrm{reco}}$ [GeV]")
        ax.grid(True, alpha=0.3)
        if verbose:
            print(f"\n--- {title} ({loss}) [{flat_key}] ---")
            print(f"{'i':>3}  {'E_true':>12}  {'E_pred':>12}  {'pred/true':>10}")
            for j in range(len(t)):
                ratio = p[j] / t[j] if t[j] != 0 else float("nan")
                print(
                    f"{j:3d}  {float(t[j]):12.6g}  {float(p[j]):12.6g}  {ratio:10.6g}"
                )

    fig.suptitle(
        f"{dp}: {n_examples} random selected events per method "
        f"(use_cc_selection={use_cc_selection})",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_scaling_law(
    values: dict[str, Any],
    metric: str = "iqr",
    q3_bin_index: int = -1,
    colors: dict[str, Any] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a scaling-law curve: IQR (or RMS) of the highest q3 bin vs training samples.

    Parameters
    ----------
    values : the *values* dict returned by
        ``plot_rms_iqr_with_uncertainty(..., return_values=True)``.
    metric : ``"iqr"`` or ``"rms"``.
    q3_bin_index : which q3 bin to use.  ``-1`` (default) picks the last
        plotted bin (highest q3).
    colors : optional ``{substring_token: colour}`` mapping applied to
        config labels (same convention as *plot_rms_iqr_with_uncertainty*).
    ax : optional axes to draw on; a new figure is created if *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    q3_mids = values["q3_bin_mids"]
    bin_idx = q3_bin_index if q3_bin_index >= 0 else len(q3_mids) + q3_bin_index

    all_labels: list[str] = []
    for loss in values:
        if loss in ("q3_bin_mids", "baseline"):
            continue
        for label in values[loss]:
            if label not in all_labels:
                all_labels.append(label)
    color_map = _resolve_color_map(all_labels, colors)

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    for loss in values:
        if loss in ("q3_bin_mids", "baseline"):
            continue
        for config_label, v in values[loss].items():
            n_samples = _extract_sample_count(config_label)
            if n_samples == 0:
                continue
            y_mean = v[mean_key][bin_idx]
            y_std = v[std_key][bin_idx]
            color = color_map.get(config_label, "tab:gray")
            ax.errorbar(
                n_samples,
                y_mean,
                yerr=y_std,
                fmt="o",
                color=color,
                capsize=4,
            )
            ax.annotate(
                plot_model_label(config_label),
                (n_samples, y_mean),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color=color,
            )

    if "baseline" in values:
        bl_val = values["baseline"][metric][bin_idx]
        ax.axhline(bl_val, color="black", ls="--", lw=1, label="Baseline")

    ax.set_xscale("log")
    ax.set_xlabel("Number of training samples")
    metric_label = "IQR" if metric == "iqr" else "RMS"
    q3_lo = q3_mids[bin_idx] - (q3_mids[1] - q3_mids[0]) / 2 if len(q3_mids) > 1 else 0
    ax.set_ylabel(f"{metric_label} [GeV]  (q$_3$ bin {bin_idx})")
    ax.set_title(f"Scaling law – {metric_label} vs training set size")
    ax.legend(fontsize=7)
    ax.grid(True)
    fig.tight_layout()
    return fig
