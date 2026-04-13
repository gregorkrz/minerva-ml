#!/usr/bin/env python3
"""Classification “light” PDFs only (no main q₃ / pion figure bundles).

Reads the classification pickle and writes under
``<--plots-dir>/classification/light/`` — same outputs as the light appendix
steps in ``plot_classification_q3`` (CCNπ) and ``plot_classification_Pions``
(CC1π±, CCπ⁰), without re-running confusion matrices or full tagging PDFs.

Use ``--components`` to limit work:

* ``q3`` — CCNπ vs *q₃* / *W* light figures only.
* ``pion`` — CC1π± and CCπ⁰ light figures only.
* ``all`` (default) — both sets.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._bootstrap import silence_classification_empty_bin_warnings
from src.eval._classification_light import save_light_classification_pdfs
from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_OUT_DIR,
    DEFAULT_PLOTS_DIR,
    DEFAULT_WANDB_TAG,
    EVAL_DATA_SUBDIR,
    repo_output_path,
)
from src.eval.classification_plots import CLASSIFICATION_PERFORMANCE_LEGEND_TITLE


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / EVAL_DATA_SUBDIR / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl"


def main(argv: list[str] | None = None) -> None:
    silence_classification_empty_bin_warnings()
    _ = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Root containing eval_data/ pickles (default: out/ under repo)",
    )
    ap.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help="Root for PDF output (default: plots/ under repo)",
    )
    ap.add_argument(
        "--components",
        choices=("all", "q3", "pion"),
        default="all",
        help="Which light bundle to emit (default: all).",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    args = ap.parse_args(argv)

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    light_dir = plots_root / "classification" / "light"
    light_dir.mkdir(parents=True, exist_ok=True)

    pkl = args.classification_pickle or _pickle_path(data_root, args.flag)
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    results = clf["results"]
    data_by_playlist = clf["data_by_playlist"]
    data_w_by_playlist = clf.get("data_w_by_playlist")
    clrs = clf["clrs_dict_full"]
    playlists = clf["playlists"]

    if args.components == "all":
        components = ("pion", "q3")
    elif args.components == "q3":
        components = ("q3",)
    else:
        components = ("pion",)

    save_light_classification_pdfs(
        light_dir,
        results,
        data_by_playlist,
        clrs,
        playlists,
        components=components,
        data_w_by_playlist=data_w_by_playlist,
    )


if __name__ == "__main__":
    main()
