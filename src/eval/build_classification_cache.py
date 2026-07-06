#!/usr/bin/env python3
"""Build the classification metrics cache without generating any plots.

Run once after ``collect_eval_data`` to make ``--plots-only`` fast::

    python -m src.eval.build_classification_cache [--flag <tag>]

Writes ``plots/tmp_results/classification_metrics.pkl`` (~300–500 MB vs 16 GB
source pickle).  Subsequent ``--plots-only`` runs never load the source pickle.

Use ``--additive`` to merge metrics for models present in the source pickle but
missing from an existing cache (fast path when adding 1–2 new models).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.eval._cache_additive import (
    metrics_cache_model_keys,
    resolve_models_to_add,
)
from src.eval._classification_metrics_cache import (
    CLF_METRICS_CACHE_NAME,
    build_metrics_cache,
    load_metrics_cache,
    save_metrics_cache,
    update_metrics_cache,
)
from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    repo_output_path,
)


def _parse_models(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    models = [m.strip() for m in raw.split(",") if m.strip()]
    return models or None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flag", "-f", default=DEFAULT_WANDB_TAG)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument(
        "--metrics-cache", type=Path, default=None, metavar="PKL",
        help="Override output path (default: plots/tmp_results/classification_metrics.pkl).",
    )
    ap.add_argument("--force", action="store_true", help="Rebuild even if cache already exists.")
    ap.add_argument(
        "--additive",
        action="store_true",
        help="Only compute metrics for models missing from an existing cache.",
    )
    ap.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys to add (default: all missing from source pickle).",
    )
    args = ap.parse_args(argv)

    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
    metrics_cache_path = args.metrics_cache or (cache_root / CLF_METRICS_CACHE_NAME)
    models_filter = _parse_models(args.models)

    pkl = args.classification_pickle or (
        data_root / f"{CLASSIFICATION_PICKLE_STEM}_{args.flag}.pkl"
    )
    if not pkl.exists():
        sys.exit(f"Classification pickle not found: {pkl}")

    print(f"Loading {pkl} …")
    with open(pkl, "rb") as f:
        clf = pickle.load(f)

    if args.additive and not args.force:
        if not metrics_cache_path.exists():
            print(
                f"No existing metrics cache at {metrics_cache_path}; "
                "running full build."
            )
        else:
            existing = load_metrics_cache(metrics_cache_path)
            new_models = resolve_models_to_add(
                set(clf["results"]),
                metrics_cache_model_keys(existing),
                models_filter,
            )
            if not new_models:
                print("Metrics cache already contains all requested models.")
                return
            cache = update_metrics_cache(existing, clf, new_models)
            save_metrics_cache(cache, metrics_cache_path)
            print("Done (additive).")
            return

    if metrics_cache_path.exists() and not args.force:
        size_mb = metrics_cache_path.stat().st_size / 1e6
        print(f"Cache already exists ({size_mb:.0f} MB): {metrics_cache_path}")
        print("Use --force to rebuild or --additive to append missing models.")
        return

    print("Building metrics cache …")
    cache = build_metrics_cache(clf)
    save_metrics_cache(cache, metrics_cache_path)
    print("Done.")


if __name__ == "__main__":
    main()
