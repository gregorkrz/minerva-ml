#!/usr/bin/env python3
"""
Copy selected evaluation PDFs into figures_latex/{regression,classification}/ with
stable names for a LaTeX paper. Multi-page classification eval PDFs can be reduced
to a selected page.

Source layout matches ``src.eval`` plotting scripts (default under repo ``plots/``):
  - ``plots/regression/`` — ``plot_regression.py``
  - ``plots/steps_combined/`` — ``plot_steps.py`` (classification | regression, one legend)
  - ``plots/classification/steps/`` — ``plot_steps.py`` (only with ``--separate-panels``)
  - ``plots/regression/steps/`` — ``plot_steps.py`` (only with ``--separate-panels``)
  - ``plots/classification/pions/`` — ``plot_classification_Pions.py``
  - ``plots/classification/q3/`` — ``plot_classification_q3.py``

Requires either:
  pip install pypdf
or Ghostscript (gs) on PATH for first-page extraction.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
_DEFAULT_PLOTS = _PROJECT_ROOT / "plots"
_DEFAULT_FIGURES = _PROJECT_ROOT / "figures_latex"

REGRESSION_COPIES = [
    ("steps_combined/log_flops_vs_val_loss.pdf", "flops_val_loss_clf_reg.pdf"),
    ("steps_combined/log_steps_vs_val_loss.pdf", "steps_val_loss_clf_reg.pdf"),
    (
        "regression/residuals_by_q3_select_events_by_E_recoil_CCinc.pdf",
        "residuals_q3_muon_sel.pdf",
    ),
    ("regression/q3_vs_iqr_rms_full_1A.pdf", "iqr_E_resolution_plot_1A.pdf"),
    ("regression/q3_vs_iqr_rms_full_1A_1B.pdf", "iqr_E_resolution_plot_1A_1B.pdf"),
]

# Training-curve PDFs are combined clf|reg under steps_combined/ (see REGRESSION_COPIES).
CLASSIFICATION_COPIES: list[tuple[str, str]] = []

CLASSIFICATION_FIRST_PAGE = [
    ("classification/pions/eval_cc1pi_tagging_1A.pdf", "CC1PiPMTagging.pdf"),
    ("classification/pions/eval_cc1pi0_tagging_1A.pdf", "CC1Pi0Tagging.pdf"),
    ("classification/q3/eval_Npi_tagging_1A.pdf", "CCNPiPMTagging.pdf"),
]

CLASSIFICATION_DETAILED_PAGES = [
    # 1-based pages — order from plot_classification_Pions.py / plot_classification_q3.py
    ("classification/pions/eval_cc1pi_tagging_1A.pdf", "CC1PiPM_E.pdf", 4),
    ("classification/pions/eval_cc1pi_tagging_1A.pdf", "CC1PiPM_Theta.pdf", 5),
    ("classification/pions/eval_cc1pi0_tagging_1A.pdf", "CC1Pi0_E.pdf", 3),
    ("classification/pions/eval_cc1pi0_tagging_1A.pdf", "CC1Pi0_Theta.pdf", 4),
    ("classification/q3/eval_Npi_tagging_1A.pdf", "CCNPiPM_q3.pdf", 3),
]


def _extract_page_pypdf(src: Path, dst: Path, page_number: int) -> None:
    try:
        from pypdf import PdfReader, PdfWriter

        reader = PdfReader(str(src))
        if page_number < 1 or len(reader.pages) < page_number:
            raise RuntimeError(f"Requested page {page_number} missing in {src}")
        writer = PdfWriter()
        writer.add_page(reader.pages[page_number - 1])
    except ImportError:
        from PyPDF2 import PdfFileReader, PdfFileWriter

        reader = PdfFileReader(str(src))
        if page_number < 1 or reader.getNumPages() < page_number:
            raise RuntimeError(f"Requested page {page_number} missing in {src}")
        writer = PdfFileWriter()
        writer.addPage(reader.getPage(page_number - 1))
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        writer.write(f)


def _extract_page_gs(src: Path, dst: Path, page_number: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "gs",
            "-q",
            "-sDEVICE=pdfwrite",
            "-dNOPAUSE",
            "-dBATCH",
            "-dSAFER",
            f"-dFirstPage={page_number}",
            f"-dLastPage={page_number}",
            f"-sOutputFile={dst}",
            str(src),
        ],
        check=True,
    )


def extract_first_page(src: Path, dst: Path) -> None:
    extract_page(src, dst, 1)


def extract_page(src: Path, dst: Path, page_number: int) -> None:
    try:
        _extract_page_pypdf(src, dst, page_number)
    except ImportError:
        _extract_page_gs(src, dst, page_number)


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

    for rel, name in CLASSIFICATION_FIRST_PAGE:
        src = plots_root / rel
        d = dest("classification", name)
        if not src.is_file():
            print(f"ERROR: missing source file: {src}", file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print(f"  would extract page 1: {src} -> {d}")
            continue
        extract_first_page(src, d)
        print(f"  page 1 from {src.name} -> {d}")

    for rel, name, page in CLASSIFICATION_DETAILED_PAGES:
        src = plots_root / rel
        d = dest("classification_detailed", name)
        if not src.is_file():
            print(f"ERROR: missing source file: {src}", file=sys.stderr)
            sys.exit(1)
        if args.dry_run:
            print(f"  would extract page {page}: {src} -> {d}")
            continue
        extract_page(src, d, page)
        print(f"  page {page} from {src.name} -> {d}")

    print("Done.")


if __name__ == "__main__":
    main()
