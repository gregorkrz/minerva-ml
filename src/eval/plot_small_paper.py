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
* ``classification_val_loss_vs_log10_flops_and_log10_steps.pdf``

``plot_regression.py`` still writes ``regression_e_ratio_hist_q3_0_1_and_1_2_1A.pdf``
into the same ``small_paper/`` directory later in the pipeline.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
from src.eval._constants import DEFAULT_CACHE_DIR, repo_output_path
from src.eval._plot_config import PlotConfig
from src.eval.e_available_plots import (
    SMALL_PAPER_COMPACT_IQR_MPV_FIGSIZE_INCHES,
    plot_rms_iqr_with_uncertainty,
)
from src.eval.plot_steps import write_classification_val_loss_log_flops_steps_pdf

_REG_CACHE_NAME = "regression.pkl"
_STEPS_CACHE_NAME = "steps.pkl"
_LIGHT_CACHE_NAME = "classification_light.pkl"


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
) -> None:
    """CC1π± | CC1π⁰ | CCNπ TPR columns; shared model legend, Baseline per panel."""
    by_name = _light_specs_by_filename(light_specs)
    task_rows: list[tuple[str, dict]] = []
    for task_label, panel, filename in _SMALL_PAPER_TPR_TASKS:
        spec = by_name.get(filename)
        if spec is None:
            print(f"Skip classification TPR (missing light spec): {filename}")
            continue
        row = _small_paper_tpr_row_from_spec(spec, panel)
        row["all_metrics"] = cfg.filter_dict_ordered(row["all_metrics"])
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


def _save_classification_val_loss_curves(
    *,
    steps: dict,
    cfg: PlotConfig,
    out_pdf: Path,
) -> None:
    lh, flops, clrs = _apply_cfg_steps(steps, cfg)
    write_classification_val_loss_log_flops_steps_pdf(
        lh,
        flops,
        clrs,
        (1.075, 1.25),
        out_pdf,
        model_names=set(cfg.model_names()),
        label_fn=cfg.label_for,
        step_cutoff=cfg.step_cutoff,
        flops_xmin=cfg.flops_xmin,
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

    with open(reg_cache, "rb") as f:
        reg = pickle.load(f)
    _save_regression_q3_compact(
        reg=reg,
        cfg=cfg,
        out_pdf=small_dir / "regression_q3_iqr_mpv_1A_compact.pdf",
    )

    with open(steps_cache, "rb") as f:
        steps = pickle.load(f)
    _save_classification_val_loss_curves(
        steps=steps,
        cfg=cfg,
        out_pdf=small_dir / "classification_val_loss_vs_log10_flops_and_log10_steps.pdf",
    )

    if light_cache.is_file():
        with open(light_cache, "rb") as f:
            light_cached = pickle.load(f)
        _save_classification_tpr_baseline(
            light_specs=light_cached["specs"],
            cfg=cfg,
            clrs=light_cached["clrs"],
            out_pdf=small_dir / "classification_tpr_at_fixed_fpr_baseline_1A.pdf",
        )
    else:
        print(f"Skip classification TPR (light cache missing): {light_cache}")


if __name__ == "__main__":
    main()
