#!/usr/bin/env python3
"""Fast paper figures under ``<plots-dir>/small_paper/`` from offline caches.

Designed to run **first** in ``generate_comparison_plots.sh`` so poster/paper
figures are ready before the heavier classification/regression bundles.

Pass ``--config JSON`` to restrict models, colors, display names, ``step_cutoff``,
and ``flops_xmin`` (same schema as other plot scripts).

Outputs (per config):

* ``regression_q3_iqr_mpv_1A_compact.pdf`` — IQR/MPV vs MC-truth *q₃* (compact layout)
* ``classification_tpr_at_fixed_fpr_baseline_1A.pdf`` — three columns (CC1π±, CC1π⁰, CCNπ TPR);
  shared model legend below; per-panel Baseline (FPR) legend
* ``classification_tpr_at_perbin_baseline_fpr_1A.pdf`` — same tasks, models read at the reco
  cut's per-bin FPR (equal-FPR comparison)
* ``ccnpi_roc_with_cut_by_W_1A.pdf`` — CCNπ per-*W* ROC curves with the reco cut as a point
* ``ccnpi_roc_with_cut_by_W_core_1A.pdf`` — same, MLP + HyperScale-small (±rw) + OL-small (±rw)
* ``classification_val_loss_vs_log10_flops_and_log10_steps.pdf``

``plot_regression.py`` still writes ``regression_e_ratio_hist_q3_0_1_and_1_2_1A.pdf``
into the same ``small_paper/`` directory later in the pipeline.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._classification_light import (
    _SMALL_PAPER_TPR_TASKS,
    _figure_metrics_3tasks_tpr_baseline,
    _light_specs_by_filename,
    _save_single_fig,
    _small_paper_tpr_row_from_spec,
)
from src.eval._constants import (
    CANONICAL_CLASSIFICATION_PICKLE,
    DEFAULT_CACHE_DIR,
    DEFAULT_CFS_CACHE_DIR,
    repo_output_path,
)
from src.eval._plot_config import PlotConfig
from src.eval._plot_split_results import (
    load_overlay_results,
    merge_overlay_light_metrics,
    overlay_model_names,
)
from src.eval._classification_metrics_cache import (
    _compute_reco_pred,
    _ensure_data_test_idx,
    _normalize_data_arrays,
)
from src.eval.classification_plots import (
    TRUE_W_XLABEL,
    data_with_signal_pion_bins,
)
from src.eval.classification_plots._metrics_binned import (
    _pion_kinematic_bin_mask,
    _align_per_event_array,
    get_signal_probabilities,
    mc_value_in_bin,
)
from src.eval.e_available_plots import (
    SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES,
    plot_rms_iqr_with_uncertainty,
)
from src.eval.plot_steps import (
    _config_horizontal_ref_names,
    write_classification_val_loss_log_flops_steps_pdf,
)

_REG_CACHE_NAME = "regression.pkl"
_STEPS_CACHE_NAME = "steps.pkl"
_LIGHT_CACHE_NAME = "classification_light.pkl"


_SMALL_PAPER_SIGNAL_CLASSES: dict[str, list[int]] = {
    r"$CC1\pi^\pm$": [0],
    r"$CC1\pi^0$": [2],
    r"$CCN\pi^\pm$ ($N \geq 1$)": [0, 1],
}

# CCNπ (N≥1) per-W ROC diagnostic: zoom on the low-FPR region where the reco cut lives.
_CCNPI_ROC_SIGNAL_CLASSES = [0, 1]
_CCNPI_ROC_CUT_TAG = "ccnpi_ge1"
_CCNPI_ROC_W_BINS: tuple[tuple[float, float], ...] = (
    (1.0, 1.5),
    (1.5, 2.0),
    (2.0, 2.5),
)
_CCNPI_ROC_CORE_MODELS: tuple[str, ...] = (
    "MLP",
    "HyperScale-small",
    "HyperScale-small-rw",
    "OmniLearned-small",
    "OmniLearned-small-rw",
)


def _resolve_classification_pickle(
    explicit: Path | None, cache_root: Path
) -> Path | None:
    if explicit is not None:
        return explicit
    candidates = [
        DEFAULT_CFS_CACHE_DIR / CANONICAL_CLASSIFICATION_PICKLE,
        cache_root / CANONICAL_CLASSIFICATION_PICKLE,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_overlay_context(
    clf_pickle: Path | None,
    cfg: PlotConfig,
    playlist: str = "1A",
) -> tuple[dict, dict, dict]:
    if clf_pickle is None or not overlay_model_names(cfg):
        return {}, {}, {}
    print(f"Loading split overlays for small_paper from {clf_pickle} …")
    with open(clf_pickle, "rb") as f:
        clf = pickle.load(f)
    return load_overlay_results(clf, cfg, playlists=[playlist])


def _apply_cfg_regression(reg: dict, cfg: PlotConfig) -> tuple[dict, dict, dict]:
    clrs = {**reg["clrs_dict_full"], **cfg.colors()}
    training_names = cfg.filter_nested_ordered(reg["training_names_full_no_rw"])
    return training_names, clrs, reg.get("eval_data_no_rw")


def _apply_cfg_steps(steps: dict, cfg: PlotConfig) -> tuple[dict, dict, dict]:
    clrs = {**steps["colors_c"], **cfg.colors()}
    model_names = set(cfg.model_names())
    lh = {k: v for k, v in steps["lh_c"].items() if k in model_names}
    flops = {k: v for k, v in steps["flops_c"].items() if k in model_names}
    clrs = {k: v for k, v in clrs.items() if k in model_names}
    return lh, flops, clrs


def _save_regression_q3_compact(
    *,
    reg: dict,
    cfg: PlotConfig,
    out_pdf: Path,
) -> None:
    training_names, clrs, eval_data = _apply_cfg_regression(reg, cfg)
    if not training_names.get("Log1p"):
        print("Skip (no regression models):", out_pdf)
        return

    fig = plot_rms_iqr_with_uncertainty(
        CKPT_DIR=Path(reg["ckpt_dir"]),
        training_names=training_names,
        playlists=["1A"],
        dataset_to_plot="1A",
        baseline_run=reg["baseline_run"],
        q3_bins=[0, 0.6, 1.2, 1.8, 2.4, 3.0, 100],
        show_q3_histograms=True,
        return_hist_fig=False,
        suppress_errors=bool(reg["suppress_errors"]),
        use_cc_selection=2,
        colors=clrs,
        text="",
        data=eval_data,
        label_fn=cfg.label_for,
        compact_figsize=SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES,
        compact_style=True,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_pdf)


def _save_classification_tpr_baseline(
    *,
    light_specs: list[dict],
    cfg: PlotConfig,
    clrs: dict[str, str],
    out_pdf: Path,
    overlay_results: dict | None = None,
    data_by_split: dict | None = None,
    data_w_by_split: dict | None = None,
    playlist: str = "1A",
) -> None:
    """CC1π± | CC1π⁰ | CCNπ TPR columns; shared model legend, Baseline per panel."""
    overlay_results = overlay_results or {}
    data_by_split = data_by_split or {}
    data_w_by_split = data_w_by_split or {}
    by_name = _light_specs_by_filename(light_specs)
    task_rows: list[tuple[str, dict]] = []
    for task_label, panel, filename in _SMALL_PAPER_TPR_TASKS:
        spec = by_name.get(filename)
        if spec is None:
            print(f"Skip classification TPR (missing light spec): {filename}")
            continue
        row = _small_paper_tpr_row_from_spec(spec, panel)
        row["all_metrics"] = cfg.filter_dict_ordered(row["all_metrics"])
        if overlay_results:
            sig_classes = _SMALL_PAPER_SIGNAL_CLASSES[task_label]
            row["all_metrics"] = merge_overlay_light_metrics(
                row["all_metrics"],
                panel,
                sig_classes,
                row["fixed_fpr"],
                overlay_results,
                data_by_split,
                data_w_by_split,
                cfg,
                playlist=playlist,
            )
        if not row["all_metrics"]:
            print(f"Skip classification TPR row (no models in config): {task_label}")
            continue
        task_rows.append((task_label, row))
    if not task_rows:
        print("Skip (no classification task rows):", out_pdf)
        return

    fig = _figure_metrics_3tasks_tpr_baseline(
        task_rows,
        {**clrs, **cfg.colors()},
        label_fn=cfg.label_for,
        model_order=cfg.ordered_model_names(),
        legend_label_order=cfg.legend_labels(),
    )
    _save_single_fig(fig, out_pdf)


# --- Per-bin (equal-FPR) variant: read each model at the reco cut's own -----
# per-bin FPR, so the comparison against the cut is apples-to-apples in every
# kinematic bin (removes the operating-point artifact where the cut's FPR
# drifts with the binning variable). Cut curve stays at its own per-bin point.
_CUT_FPR_KEY = "cutfpr"

# (task title, signal classes, reco-cut tag, kinematic panel)
_SMALL_PAPER_PERBIN_TASKS: tuple[tuple[str, list[int], str, str], ...] = (
    (r"$CC1\pi^\pm$", [0], "cc1pi", "pion_E"),
    (r"$CC1\pi^0$", [2], "cc1pi0", "pion_E"),
    (r"$CCN\pi^\pm$ ($N \geq 1$)", [0, 1], "ccnpi_ge1", "w"),
)


def _perbin_masks(
    data: dict,
    data_w: dict,
    pid: np.ndarray,
    signal_classes: list[int],
    panel: str,
    playlist: str,
    n_pred: int,
) -> tuple[list[np.ndarray], np.ndarray, str, bool]:
    """Return (per-bin boolean masks, bin midpoints, xlabel, log_x)."""
    if panel == "w":
        W = _align_per_event_array(
            data_w["W_GeV"], data_w, playlist, n_pred, key="W_GeV"
        )
        edges = np.asarray(data_w["W_bin_edges"])
        mids = np.asarray(data_w["W_bin_mids"])
        masks = [mc_value_in_bin(W, edges, i) for i in range(len(edges) - 1)]
        return masks, mids, TRUE_W_XLABEL, False

    if panel == "pion_E":
        data_sp = data_with_signal_pion_bins(
            data,
            pid,
            signal_classes,
            pion_quantile_require_has_pion=False,
            pion_bin_edge_method="equal_frequency",
        )
        edges = np.asarray(data_sp["pion_E_MC_bins"])
        mids = np.asarray(data_sp["pion_E_MC_bins_mid"])
        masks = [
            _pion_kinematic_bin_mask(
                data_sp,
                kind="E",
                bin_index=i,
                edges=edges,
                require_has_pion=False,
                playlist=playlist,
                n_pred=n_pred,
            )
            for i in range(len(edges) - 1)
        ]
        return masks, mids, r"True $E_\pi$ [GeV]", True

    raise ValueError(f"Unknown per-bin panel: {panel!r}")


def _perbin_model_tpr_at_cut_fpr(
    y: np.ndarray, p: np.ndarray, reco: np.ndarray, masks: list[np.ndarray]
) -> np.ndarray:
    """Per-bin model TPR read from the in-bin ROC at the cut's in-bin FPR."""
    from sklearn.metrics import roc_curve

    out = np.full(len(masks), np.nan)
    for i, bm in enumerate(masks):
        yb, pb, rb = y[bm], p[bm], reco[bm]
        valid = ~(np.isnan(yb) | np.isnan(pb))
        yb, pb, rb = yb[valid], pb[valid], rb[valid]
        nsig, nbkg = int((yb == 1).sum()), int((yb == 0).sum())
        if nsig == 0 or nbkg == 0 or len(np.unique(yb)) < 2:
            continue
        cut_fpr_b = ((rb == 1) & (yb == 0)).sum() / nbkg
        fpr, tpr, _ = roc_curve(yb, pb)
        idx = int(np.searchsorted(fpr, cut_fpr_b, side="right") - 1)
        idx = max(0, min(idx, len(tpr) - 1))
        out[i] = float(tpr[idx])
    return out


def _perbin_cut_tpr(
    y_true: np.ndarray, reco: np.ndarray, masks: list[np.ndarray]
) -> np.ndarray:
    """Per-bin reco-cut recall (its own per-bin TPR)."""
    out = np.full(len(masks), np.nan)
    for i, bm in enumerate(masks):
        yb, rb = y_true[bm], reco[bm]
        nsig = int((yb == 1).sum())
        if nsig == 0:
            continue
        out[i] = ((rb == 1) & (yb == 1)).sum() / nsig
    return out


def _save_classification_tpr_baseline_perbin_cutfpr(
    *,
    clf_pickle: Path | None,
    cfg: PlotConfig,
    clrs: dict[str, str],
    out_pdf: Path,
    playlist: str = "1A",
) -> None:
    """Like ``_save_classification_tpr_baseline`` but every model is read at the
    reco cut's *per-bin* FPR (equal-FPR-per-bin). The baseline goes into the
    shared model legend (no per-panel ``(FPR …)`` box)."""
    if clf_pickle is None or not Path(clf_pickle).is_file():
        print(f"Skip per-bin-FPR TPR plot (no classification pickle): {clf_pickle}")
        return

    print(f"Building per-bin-FPR TPR plot from {clf_pickle} …")
    with open(clf_pickle, "rb") as f:
        clf = pickle.load(f)
    results = clf["results"]
    present = [m for m in cfg.model_names() if m in results]
    if not present:
        print("Skip per-bin-FPR TPR plot (no config models in pickle).")
        return

    data = _ensure_data_test_idx(clf["data_by_playlist"][playlist], playlist, clf)
    data_w = _ensure_data_test_idx(
        clf["data_w_by_playlist"][playlist], playlist, clf
    )
    pid = np.asarray(results[present[0]][0][playlist]["pid"])
    n_pred = len(pid)

    task_rows: list[tuple[str, dict[str, Any]]] = []
    for task_label, sig_classes, cut_tag, panel in _SMALL_PAPER_PERBIN_TASKS:
        masks, mids, xlabel, log_x = _perbin_masks(
            data, data_w, pid, sig_classes, panel, playlist, n_pred
        )
        reco = np.asarray(_compute_reco_pred(data, pid, playlist, cut_tag))
        y_true = np.isin(pid, sig_classes).astype(int)
        cut_tpr = _perbin_cut_tpr(y_true, reco, masks)

        all_metrics: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for model in present:
            runs = results[model]
            per_run = []
            for k in range(len(runs)):
                sig = get_signal_probabilities(runs[k], sig_classes, playlist)
                per_run.append(
                    _perbin_model_tpr_at_cut_fpr(
                        sig["ytrue"], sig["ypred"], reco, masks
                    )
                )
            stacked = np.vstack(per_run)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_tpr = np.nanmean(stacked, axis=0)
                std_tpr = (
                    np.nanstd(stacked, axis=0)
                    if stacked.shape[0] > 1
                    else np.zeros(stacked.shape[1])
                )
            all_metrics[model] = {
                f"tpr@{_CUT_FPR_KEY}": {"mean": mean_tpr, "std": std_tpr}
            }

        task_rows.append(
            (
                task_label,
                {
                    "all_metrics": all_metrics,
                    "x": mids,
                    "fixed_fpr": [_CUT_FPR_KEY],
                    "reco_baseline_tpr": cut_tpr,
                    "reco_label": "Baseline",
                    "reco_baseline_global_fpr": None,  # -> plain "Baseline" in shared legend
                    "xlabel": xlabel,
                    "log_x": log_x,
                    "kinematic": None,
                },
            )
        )

    if not task_rows:
        print("Skip (no per-bin-FPR task rows):", out_pdf)
        return

    fig = _figure_metrics_3tasks_tpr_baseline(
        task_rows,
        {**clrs, **cfg.colors()},
        label_fn=cfg.label_for,
        model_order=cfg.ordered_model_names(),
        legend_label_order=cfg.legend_labels(),
    )
    _save_single_fig(fig, out_pdf)


def _model_score_from_run(run: dict, playlist: str, signal_classes: list[int]) -> np.ndarray:
    """Binary signal score for one run (same convention as ``get_signal_probabilities``)."""
    return get_signal_probabilities(run, signal_classes, playlist)["ypred"]


def _tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def _figure_ccnpi_roc_with_cut_by_W(
    *,
    results: dict[str, list[dict]],
    model_names: list[str],
    y_true: np.ndarray,
    reco: np.ndarray,
    w_gev: np.ndarray,
    colors: dict[str, str],
    label_fn: Callable[[str], str],
    signal_classes: list[int],
    playlist: str,
    title_suffix: str = "",
) -> plt.Figure | None:
    """Per-*W* ROC curves; reco cut as ★, models as ○ at the cut's in-bin FPR."""
    present = [m for m in model_names if m in results]
    if not present:
        return None

    scores = {
        m: _model_score_from_run(results[m][0], playlist, signal_classes)
        for m in present
    }

    fig, axes = plt.subplots(
        1,
        len(_CCNPI_ROC_W_BINS),
        figsize=(5.2 * len(_CCNPI_ROC_W_BINS), 5.0),
        sharey=True,
    )
    if len(_CCNPI_ROC_W_BINS) == 1:
        axes = [axes]

    for ax, (lo, hi) in zip(axes, _CCNPI_ROC_W_BINS):
        bmask = (w_gev >= lo) & (w_gev < hi)
        yb = y_true[bmask]
        recb = reco[bmask]
        nsig = int((yb == 1).sum())
        nbkg = int((yb == 0).sum())
        if nsig == 0 or nbkg == 0:
            ax.set_visible(False)
            continue

        cut_tpr = ((recb == 1) & (yb == 1)).sum() / nsig
        cut_fpr = ((recb == 1) & (yb == 0)).sum() / nbkg

        for name in present:
            sb = scores[name][bmask]
            fpr, tpr, _ = roc_curve(yb, sb)
            auc_val = roc_auc_score(yb, sb)
            tpr_match = _tpr_at_fpr(yb, sb, cut_fpr)
            color = colors.get(name, "tab:gray")
            lbl = label_fn(name)
            ax.plot(
                fpr,
                tpr,
                color=color,
                lw=1.8,
                label=f"{lbl}: AUC={auc_val:.3f}, TPR@cut FPR={tpr_match:.3f}",
            )
            ax.plot(
                [cut_fpr],
                [tpr_match],
                "o",
                color=color,
                ms=6,
                mfc="white",
                mew=1.5,
                zorder=5,
            )

        ax.plot(
            [cut_fpr],
            [cut_tpr],
            "*",
            color="k",
            ms=20,
            zorder=6,
            label=f"Reco cut: FPR={cut_fpr:.3f}, TPR={cut_tpr:.3f}",
        )
        ax.axvline(cut_fpr, color="k", ls=":", lw=1, alpha=0.6)
        ax.annotate(
            "cut FPR",
            xy=(cut_fpr, 0.004),
            fontsize=7,
            rotation=90,
            va="bottom",
            ha="right",
            alpha=0.7,
        )

        ax.set_xlim(0, 0.05)
        ax.set_ylim(0, 0.30)
        ax.set_xlabel("False positive rate")
        ax.set_title(
            rf"$W \in [{lo:.1f}, {hi:.1f})$ GeV   (sig={nsig}, bkg={nbkg})",
            fontsize=10,
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.92)

    axes[0].set_ylabel("True positive rate (efficiency)")
    suptitle = (
        r"$CCN\pi^\pm$ ($N\geq1$) per-$W$ ROC vs. reco cut — MINERvA Open Data "
        f"{playlist}"
    )
    if title_suffix:
        suptitle = f"{suptitle} ({title_suffix})"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def _save_ccnpi_roc_with_cut_by_W(
    *,
    clf_pickle: Path | None,
    cfg: PlotConfig,
    clrs: dict[str, str],
    out_pdf: Path,
    model_names: list[str] | None = None,
    title_suffix: str = "",
    playlist: str = "1A",
) -> None:
    """CCNπ per-*W* ROC with reco-cut operating point (equal in-bin FPR)."""
    if clf_pickle is None or not Path(clf_pickle).is_file():
        print(f"Skip CCNπ ROC-with-cut plot (no classification pickle): {clf_pickle}")
        return

    names = model_names if model_names is not None else cfg.ordered_model_names()
    print(f"Building CCNπ ROC-with-cut plot ({names}) from {clf_pickle} …")
    with open(clf_pickle, "rb") as f:
        clf = pickle.load(f)
    results = clf["results"]
    present = [m for m in names if m in results]
    if not present:
        print("Skip CCNπ ROC-with-cut plot (no requested models in pickle).")
        return

    data = _ensure_data_test_idx(clf["data_by_playlist"][playlist], playlist, clf)
    data_w = _ensure_data_test_idx(
        clf["data_w_by_playlist"][playlist], playlist, clf
    )
    pid = np.asarray(results[present[0]][0][playlist]["pid"])
    n_pred = len(pid)
    y_true = np.isin(pid, _CCNPI_ROC_SIGNAL_CLASSES).astype(int)
    reco = np.asarray(
        _compute_reco_pred(data, pid, playlist, _CCNPI_ROC_CUT_TAG)
    )
    data_w = _normalize_data_arrays(data_w, playlist, n_pred)
    w_gev = np.asarray(data_w["W_GeV"])
    if len(w_gev) != n_pred or len(reco) != n_pred:
        raise ValueError(
            f"CCNπ ROC alignment failed: W={len(w_gev)}, reco={len(reco)}, "
            f"predictions={n_pred}"
        )

    fig = _figure_ccnpi_roc_with_cut_by_W(
        results=results,
        model_names=names,
        y_true=y_true,
        reco=reco,
        w_gev=w_gev,
        colors={**clrs, **cfg.colors()},
        label_fn=cfg.label_for,
        signal_classes=_CCNPI_ROC_SIGNAL_CLASSES,
        playlist=playlist,
        title_suffix=title_suffix,
    )
    if fig is None:
        print("Skip (empty CCNπ ROC figure):", out_pdf)
        return
    _save_single_fig(fig, out_pdf)


def _save_classification_val_loss_curves(
    *,
    steps: dict,
    cfg: PlotConfig,
    out_pdf: Path,
) -> None:
    legend_label_order = [
        cfg.label_for(n)
        for n in cfg.ordered_model_names(
            ("BDT", "Transformer-xsmall", "Transformer-small", "MLP")
        )
    ]
    legend_column_stacks = (
        [[cfg.label_for(n) for n in stack] for stack in cfg.legend_column_stacks]
        if cfg.legend_column_stacks
        else None
    )
    write_classification_val_loss_log_flops_steps_pdf(
        steps["lh_c"],
        steps["flops_c"],
        {**steps["colors_c"], **cfg.colors()},
        None,
        out_pdf,
        model_names=set(cfg.model_names()),
        label_fn=cfg.label_for,
        step_cutoff=cfg.step_cutoff,
        flops_xmin=cfg.flops_xmin,
        model_curve_cuts=cfg.model_curve_cuts("classification"),
        legend_label_order=legend_label_order,
        legend_column_stacks=legend_column_stacks,
        config_horizontal_refs=_config_horizontal_ref_names(cfg, "classification"),
        horizontal_ref_models=_config_horizontal_ref_names(cfg, "classification"),
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plots-dir",
        type=Path,
        required=True,
        help="Plot root for this config (e.g. plots/poster_NPML)",
    )
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="JSON",
        help="Plot config JSON (models, colors, cutoffs).",
    )
    ap.add_argument(
        "--regression-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help=f"Regression plots cache (default: …/tmp_results/{_REG_CACHE_NAME})",
    )
    ap.add_argument(
        "--steps-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help=f"Steps plots cache (default: …/tmp_results/{_STEPS_CACHE_NAME})",
    )
    ap.add_argument(
        "--light-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help=f"Classification light draw-spec cache (default: …/tmp_results/{_LIGHT_CACHE_NAME})",
    )
    ap.add_argument(
        "--classification-pickle",
        type=Path,
        default=None,
        metavar="PKL",
        help="Classification pickle for split overlays (e.g. train-set sanity checks).",
    )
    ap.add_argument(
        "--skip-classification",
        action="store_true",
        help="Skip classification small-paper figures.",
    )
    ap.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip regression small-paper figures.",
    )
    args = ap.parse_args(argv)

    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    reg_cache = args.regression_cache or (cache_root / _REG_CACHE_NAME)
    steps_cache = args.steps_cache or (cache_root / _STEPS_CACHE_NAME)
    light_cache = args.light_cache or (cache_root / _LIGHT_CACHE_NAME)

    cfg = PlotConfig.load(args.config)
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    small_dir = plots_root / "small_paper"
    small_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== small_paper ({cfg.model_names()}) -> {small_dir} ===")

    if not args.skip_regression:
        if reg_cache.is_file():
            with open(reg_cache, "rb") as f:
                reg = pickle.load(f)
            _save_regression_q3_compact(
                reg=reg,
                cfg=cfg,
                out_pdf=small_dir / "regression_q3_iqr_mpv_1A_compact.pdf",
            )
        else:
            print(f"Skip regression small-paper figures (cache missing): {reg_cache}")
    else:
        print("Skip regression small-paper figures (--skip-regression).")

    if not args.skip_classification:
        clf_pkl = _resolve_classification_pickle(args.classification_pickle, cache_root)
        light_clrs: dict[str, str] = {}

        if steps_cache.is_file():
            with open(steps_cache, "rb") as f:
                steps = pickle.load(f)
            _save_classification_val_loss_curves(
                steps=steps,
                cfg=cfg,
                out_pdf=small_dir
                / "classification_val_loss_vs_log10_flops_and_log10_steps.pdf",
            )
        else:
            print(f"Skip classification val-loss (steps cache missing): {steps_cache}")

        if light_cache.is_file():
            with open(light_cache, "rb") as f:
                light_cached = pickle.load(f)
            light_clrs = light_cached["clrs"]
            overlay_results, data_by_split, data_w_by_split = _load_overlay_context(
                clf_pkl, cfg, playlist="1A"
            )
            _save_classification_tpr_baseline(
                light_specs=light_cached["specs"],
                cfg=cfg,
                clrs=light_clrs,
                out_pdf=small_dir / "classification_tpr_at_fixed_fpr_baseline_1A.pdf",
                overlay_results=overlay_results,
                data_by_split=data_by_split,
                data_w_by_split=data_w_by_split,
            )
            _save_classification_tpr_baseline_perbin_cutfpr(
                clf_pickle=clf_pkl,
                cfg=cfg,
                clrs=light_clrs,
                out_pdf=small_dir
                / "classification_tpr_at_perbin_baseline_fpr_1A.pdf",
                playlist="1A",
            )
        else:
            print(f"Skip classification TPR (light cache missing): {light_cache}")

        roc_clrs = {**light_clrs, **cfg.colors()}
        _save_ccnpi_roc_with_cut_by_W(
            clf_pickle=clf_pkl,
            cfg=cfg,
            clrs=roc_clrs,
            out_pdf=small_dir / "ccnpi_roc_with_cut_by_W_1A.pdf",
            playlist="1A",
        )
        _save_ccnpi_roc_with_cut_by_W(
            clf_pickle=clf_pkl,
            cfg=cfg,
            clrs=roc_clrs,
            out_pdf=small_dir / "ccnpi_roc_with_cut_by_W_core_1A.pdf",
            model_names=list(_CCNPI_ROC_CORE_MODELS),
            title_suffix="MLP, HyperScale-small, OL-small",
            playlist="1A",
        )
    else:
        print("Skip classification small-paper figures (--skip-classification).")


if __name__ == "__main__":
    main()
