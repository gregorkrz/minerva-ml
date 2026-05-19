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
    filter_classification_results_for_standard_plots,
    repo_output_path,
)
from src.eval.classification_plots import (
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    DEFAULT_FIXED_FPR,
    get_signal_probabilities,
)


def _pickle_path(out_dir: Path, flag: str) -> Path:
    return out_dir / f"{CLASSIFICATION_PICKLE_STEM}_{flag}.pkl"


def main(argv: list[str] | None = None) -> None:
    silence_classification_empty_bin_warnings()
    _ = CLASSIFICATION_PERFORMANCE_LEGEND_TITLE

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory containing classification_<flag>.pkl (default: out/ under repo)",
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

    # Random / uninformative score (independent of label): ROC is the diagonal, so
    # TPR on signal equals FPR on background at the same threshold; fixing FPR=α
    # implies TPR=α in expectation (“perturb positives at random” → no extra skill).
    print(
        "Random baseline TPR/FPR: for scores independent of truth (same notion as "
        "the gray “Random baseline” in AUPRC), expected TPR equals FPR at every "
        "operating point. Fixed-FPR targets used in these figures: "
        f"{list(DEFAULT_FIXED_FPR)!r} → random TPR equals each target FPR."
    )

    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    light_dir = plots_root / "classification" / "light"
    light_dir.mkdir(parents=True, exist_ok=True)

    pkl = args.classification_pickle or _pickle_path(data_root, args.flag)
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    results = filter_classification_results_for_standard_plots(clf["results"])
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

    first_model = next(iter(results))
    run0 = results[first_model][0]
    for playlist in playlists:
        bits: list[str] = []
        if "pion" in components:
            for name, classes in (("CC1π±", [0]), ("CCπ⁰", [2])):
                y = get_signal_probabilities(run0, classes, playlist)["ytrue"]
                bits.append(f"{name} P(signal)={float(y.mean()):.6g} (n={len(y):,})")
        if "q3" in components:
            y = get_signal_probabilities(run0, [0, 1], playlist)["ytrue"]
            bits.append(f"CCNπ P(signal)={float(y.mean()):.6g} (n={len(y):,})")
        if bits:
            print(f"Playlist {playlist} — overall signal fraction: " + "; ".join(bits))

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
