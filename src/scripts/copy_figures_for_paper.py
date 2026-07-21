#!/usr/bin/env python3
"""
Copy selected evaluation PDFs into ``figures_latex/<config>/`` with stable names
for a LaTeX paper.

Pass ``--config`` (same names as ``generate_comparison_plots.sh``) so sources are
taken from ``plots/<config>/`` and outputs go to ``figures_latex/<config>/`` —
e.g. ``--config V1Paper`` reads ``plots/V1Paper/`` and writes
``figures_latex/V1Paper/``. Override with ``--plots-root`` / ``--figures-root``.

Joint classification/regression training curves from ``plot_steps`` go directly
under the figures root (not under ``regression/``). Other regression plots still
go to ``regression/``; all classification plots to ``classification/``.

Classification figures are taken from **single-page** PDFs under
``classification/light/`` (from ``python -m src.eval.plot_classification_light``),
not from the large multi-page ``pions/`` or ``q3/`` tagging bundles. Metrics vs
hadronic ``W`` are copied when those light PDFs exist (they require ``W``-binned
data in the classification pickle).

Source layout matches ``src.eval`` plotting scripts under the chosen plots root:
  - ``regression/`` — ``plot_regression.py``
  - ``steps_combined/`` — ``plot_steps.py`` (classification | regression, one legend)
  - ``classification/steps/`` — ``plot_steps.py`` (only with ``--separate-panels``)
  - ``regression/steps/`` — ``plot_steps.py`` (only with ``--separate-panels``)
  - ``classification/light/`` — ``plot_classification_light.py``
  - ``small_paper/`` — ``plot_small_paper.py`` (TPR kinematics + CCNπ ROC-by-W)

Regression copies are plain file copies. Classification light / small_paper copies are
plain copies (no PDF page extraction).
"""

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_DEFAULT_PLOTS = _PROJECT_ROOT / "plots"
_DEFAULT_FIGURES = _PROJECT_ROOT / "figures_latex"
_PLOT_CONFIGS_DIR = _PROJECT_ROOT / "plot_configs"


def resolve_plot_config(config: str) -> tuple[Path, str]:
    """Resolve ``--config`` to ``(json_path, slug)``.

    Accepts a basename (``V1Paper``), a ``.json`` filename, or a path to a JSON
    under ``plot_configs/`` (or elsewhere). Slug is the stem used for
    ``plots/<slug>/`` by ``generate_comparison_plots.sh``.
    """
    raw = Path(config)
    candidates: list[Path] = []
    if raw.is_file():
        candidates.append(raw.resolve())
    else:
        name = raw.name
        if not name.endswith(".json"):
            name = f"{name}.json"
        candidates.append((_PLOT_CONFIGS_DIR / name).resolve())
        if raw.suffix == ".json" or "/" in config or "\\" in config:
            candidates.append((_PROJECT_ROOT / raw).resolve())

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path, path.stem

    available = sorted(p.stem for p in _PLOT_CONFIGS_DIR.glob("*.json"))
    avail_msg = ", ".join(available) if available else "(none found)"
    raise FileNotFoundError(
        f"Plot config not found for --config {config!r}. "
        f"Tried: {', '.join(str(p) for p in seen)}. Available: {avail_msg}"
    )

# Combined clf | reg panels from ``plots/steps_combined/`` → ``figures_latex/<name>`` (root).
FIGURES_LATEX_ROOT_COPIES = [
    ("steps_combined/log_flops_vs_val_loss.pdf", "flops_val_loss_clf_reg.pdf"),
    ("steps_combined/log_steps_vs_val_loss.pdf", "steps_val_loss_clf_reg.pdf"),
]

REGRESSION_COPIES = [
    (
        "regression/residuals_by_q3_select_events_by_E_recoil_CCinc.pdf",
        "residuals_q3_muon_sel.pdf",
    ),
    ("regression/q3_vs_iqr_rms_full_1A.pdf", "iqr_E_resolution_plot_1A.pdf"),
    ("regression/q3_vs_iqr_rms_full_1A_1B.pdf", "iqr_E_resolution_plot_1A_1B.pdf"),
]

# From ``plot_classification_light`` (playlist 1A). *W* PDFs exist only if the pickle
# carried hadronic-W binned data; those are listed separately and skipped when absent.
CLASSIFICATION_COPIES = [
    (
        "classification/light/eval_classification_light_cc1pi_q3_1A.pdf",
        "CC1PiPMTagging.pdf",
    ),
    (
        "classification/light/eval_classification_light_cc1pi0_q3_1A.pdf",
        "CC1Pi0Tagging.pdf",
    ),
    (
        "classification/light/eval_classification_light_ccnpi_q3_1A.pdf",
        "CCNPiPMTagging.pdf",
    ),
    (
        "classification/light/eval_classification_light_cc1pi_pion_kinematics_1A.pdf",
        "CC1PiPM_pion_kinematics.pdf",
    ),
    (
        "classification/light/eval_classification_light_cc1pi0_pion_kinematics_1A.pdf",
        "CC1Pi0_pion_kinematics.pdf",
    ),
    (
        "classification/light/eval_classification_light_ccnpi_q3_1A.pdf",
        "CCNPiPM_q3.pdf",
    ),
]

CLASSIFICATION_COPIES_OPTIONAL_W = [
    ("classification/light/eval_classification_light_cc1pi_W_1A.pdf", "CC1PiPM_W.pdf"),
    ("classification/light/eval_classification_light_cc1pi0_W_1A.pdf", "CC1Pi0_W.pdf"),
    ("classification/light/eval_classification_light_ccnpi_W_1A.pdf", "CCNPiPM_W.pdf"),
]

# From ``plot_small_paper`` (optional — requires small_paper regen; skip if missing).
SMALL_PAPER_COPIES_OPTIONAL = [
    (
        "small_paper/classification_tpr_at_fixed_fpr_baseline_1A.pdf",
        "classification/TPR_fixed_fpr_baseline_1A.pdf",
    ),
    (
        "small_paper/classification_tpr_at_perbin_baseline_fpr_1A.pdf",
        "classification/TPR_perbin_baseline_fpr_1A.pdf",
    ),
    (
        "small_paper/ccnpi_roc_with_cut_by_W_1A.pdf",
        "classification/CCNPiPM_ROC_by_W_1A.pdf",
    ),
    (
        "small_paper/ccnpi_roc_with_cut_by_W_core_1A.pdf",
        "classification/CCNPiPM_ROC_by_W_core_1A.pdf",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Plot config name or JSON path (e.g. V1Paper or plot_configs/V1Paper.json). "
            "Sets source to plots/<name>/ and output to figures_latex/<name>/ "
            "unless --plots-root / --figures-root are given."
        ),
    )
    parser.add_argument(
        "--plots-root",
        "--out-root",
        type=Path,
        default=None,
        dest="plots_root",
        help=(
            "Root directory for plot PDFs (same layout as --plots-dir for src.eval). "
            "Default: plots/<config>/ when --config is set, else plots/."
        ),
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: figures_latex/<config>/ when --config is set, "
            f"else {_DEFAULT_FIGURES}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions only",
    )
    args = parser.parse_args()

    config_slug: str | None = None
    if args.config is not None:
        try:
            config_path, config_slug = resolve_plot_config(args.config)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Plot config: {config_path} (slug={config_slug})")

    if args.plots_root is not None:
        plots_root = args.plots_root.resolve()
    elif config_slug is not None:
        plots_root = (_DEFAULT_PLOTS / config_slug).resolve()
    else:
        plots_root = _DEFAULT_PLOTS.resolve()

    if args.figures_root is not None:
        figures_root = args.figures_root.resolve()
    elif config_slug is not None:
        figures_root = (_DEFAULT_FIGURES / config_slug).resolve()
    else:
        figures_root = _DEFAULT_FIGURES.resolve()

    def dest(subdir: str, name: str) -> Path:
        return figures_root / subdir / name

    print(f"Source plots root: {plots_root}")
    print(f"Output figures root: {figures_root}")
    if args.dry_run:
        print("(dry run)")

    for rel, name in FIGURES_LATEX_ROOT_COPIES:
        src = plots_root / rel
        d = figures_root / name
        if not src.is_file():
            print(f"ERROR: missing source file: {src}", file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print(f"  would copy {src} -> {d}")
            continue
        figures_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        print(f"  copied {src.name} -> {d}")

    for rel, name in REGRESSION_COPIES:
        src = plots_root / rel
        d = dest("regression", name)
        if not src.is_file():
            print(f"ERROR: missing source file: {src}", file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print(f"  would copy {src} -> {d}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        print(f"  copied {src.name} -> {d}")

    for rel, name in CLASSIFICATION_COPIES:
        src = plots_root / rel
        d = dest("classification", name)
        if not src.is_file():
            print(f"ERROR: missing source file: {src}", file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print(f"  would copy {src} -> {d}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        print(f"  copied {src.name} -> {d}")

    for rel, name in CLASSIFICATION_COPIES_OPTIONAL_W:
        src = plots_root / rel
        d = dest("classification", name)
        if not src.is_file():
            print(
                f"WARNING: skip (no source file — run light plots with W data?): {src}"
            )
            continue
        if args.dry_run:
            print(f"  would copy {src} -> {d}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        print(f"  copied {src.name} -> {d}")

    for rel, dest_rel in SMALL_PAPER_COPIES_OPTIONAL:
        src = plots_root / rel
        d = figures_root / dest_rel
        if not src.is_file():
            print(
                f"WARNING: skip (no source file — run plot_small_paper?): {src}"
            )
            continue
        if args.dry_run:
            print(f"  would copy {src} -> {d}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        print(f"  copied {src.name} -> {d}")

    print("Done.")


if __name__ == "__main__":
    main()
