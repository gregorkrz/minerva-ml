#!/usr/bin/env python3
"""
Copy selected evaluation PDFs into ``figures_latex/`` (root and subfolders) with
stable names for a LaTeX paper.

Joint classification/regression training curves from ``plot_steps`` go directly
under ``figures_latex/`` (not under ``regression/``). Other regression plots still
go to ``figures_latex/regression/``; all classification plots to ``classification/``.

Classification figures are taken from **single-page** PDFs under
``plots/classification/light/`` (from ``python -m src.eval.plot_classification_light``),
not from the large multi-page ``pions/`` or ``q3/`` tagging bundles. Metrics vs
hadronic ``W`` are copied when those light PDFs exist (they require ``W``-binned
data in the classification pickle).

Source layout matches ``src.eval`` plotting scripts (default under repo ``plots/``):
  - ``plots/regression/`` — ``plot_regression.py``
  - ``plots/steps_combined/`` — ``plot_steps.py`` (classification | regression, one legend)
  - ``plots/classification/steps/`` — ``plot_steps.py`` (only with ``--separate-panels``)
  - ``plots/regression/steps/`` — ``plot_steps.py`` (only with ``--separate-panels``)
  - ``plots/classification/light/`` — ``plot_classification_light.py``

Regression copies are plain file copies. Classification light copies are plain copies
(no PDF page extraction).
"""

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_DEFAULT_PLOTS = _PROJECT_ROOT / "plots"
_DEFAULT_FIGURES = _PROJECT_ROOT / "figures_latex"

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
    ("classification/light/eval_classification_light_cc1pi_q3_1A.pdf", "CC1PiPMTagging.pdf"),
    ("classification/light/eval_classification_light_cc1pi0_q3_1A.pdf", "CC1Pi0Tagging.pdf"),
    ("classification/light/eval_classification_light_ccnpi_q3_1A.pdf", "CCNPiPMTagging.pdf"),
    (
        "classification/light/eval_classification_light_cc1pi_pion_kinematics_1A.pdf",
        "CC1PiPM_pion_kinematics.pdf",
    ),
    (
        "classification/light/eval_classification_light_cc1pi0_pion_kinematics_1A.pdf",
        "CC1Pi0_pion_kinematics.pdf",
    ),
    ("classification/light/eval_classification_light_ccnpi_q3_1A.pdf", "CCNPiPM_q3.pdf"),
]

CLASSIFICATION_COPIES_OPTIONAL_W = [
    ("classification/light/eval_classification_light_cc1pi_W_1A.pdf", "CC1PiPM_W.pdf"),
    ("classification/light/eval_classification_light_cc1pi0_W_1A.pdf", "CC1Pi0_W.pdf"),
    ("classification/light/eval_classification_light_ccnpi_W_1A.pdf", "CCNPiPM_W.pdf"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots-root",
        "--out-root",
        type=Path,
        default=_DEFAULT_PLOTS,
        dest="plots_root",
        help=(
            "Root directory for plot PDFs (default: plots/ under project). "
            "Same layout as --plots-dir for src.eval scripts."
        ),
    )
    parser.add_argument(
        "--figures-root",
        type=Path,
        default=_DEFAULT_FIGURES,
        help=f"Output directory (default: {_DEFAULT_FIGURES})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions only",
    )
    args = parser.parse_args()
    plots_root = args.plots_root.resolve()
    figures_root = args.figures_root.resolve()

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
            print(f"WARNING: skip (no source file — run light plots with W data?): {src}")
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
