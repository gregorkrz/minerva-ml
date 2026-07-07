#!/usr/bin/env python3
"""Build ``steps.pkl`` without loading both eval pickles at once.

Run after ``collect_eval_data`` so ``plot_steps --plots-only`` stays fast::

    python -m src.eval.build_steps_cache [--flag <tag>]

Writes ``plots/tmp_results/steps.pkl`` (typically tens of MB vs 30+ GB peak
if both source pickles were loaded together).

Use ``--additive`` to append loss histories for models missing from an existing
``steps.pkl``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    REGRESSION_PICKLE_STEM,
    repo_output_path,
)
from src.eval._steps_cache import (
    STEPS_CACHE_NAME,
    build_steps_cache_from_pickles,
    save_steps_cache,
    update_steps_cache_from_pickles,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument("--regression-pickle", type=Path, default=None)
    ap.add_argument(
        "--steps-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Override output path (default: plots/tmp_results/steps.pkl).",
    )
    ap.add_argument(
        "--force", action="store_true", help="Rebuild even if cache exists."
    )
    ap.add_argument(
        "--additive",
        action="store_true",
        help="Only append loss histories for models missing from an existing cache.",
    )
    ap.add_argument(
        "--skip-classification",
        action="store_true",
        help="With --additive, do not update the classification side.",
    )
    ap.add_argument(
        "--skip-regression",
        action="store_true",
        help="With --additive, do not update the regression side.",
    )
    args = ap.parse_args(argv)

    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    steps_cache_path = args.steps_cache or (cache_root / STEPS_CACHE_NAME)

    clf_p = args.classification_pickle or (
        data_root / f"{CLASSIFICATION_PICKLE_STEM}_{args.flag}.pkl"
    )
    reg_p = args.regression_pickle or (
        data_root / f"{REGRESSION_PICKLE_STEM}_{args.flag}.pkl"
    )

    if args.additive and not args.force:
        if not steps_cache_path.exists():
            print(f"No existing steps cache at {steps_cache_path}; running full build.")
        else:
            import pickle

            with open(steps_cache_path, "rb") as f:
                existing = pickle.load(f)
            payload = update_steps_cache_from_pickles(
                existing,
                clf_p,
                reg_p,
                update_classification=not args.skip_classification,
                update_regression=not args.skip_regression,
            )
            save_steps_cache(payload, steps_cache_path)
            print("Done (additive).")
            return

    if steps_cache_path.exists() and not args.force:
        size_mb = steps_cache_path.stat().st_size / 1e6
        print(f"Cache already exists ({size_mb:.0f} MB): {steps_cache_path}")
        print("Use --force to rebuild or --additive to append missing models.")
        return

    for p in (clf_p, reg_p):
        if not p.exists():
            sys.exit(f"Pickle not found: {p}")

    payload = build_steps_cache_from_pickles(clf_p, reg_p)
    save_steps_cache(payload, steps_cache_path)
    print("Done.")


if __name__ == "__main__":
    main()
