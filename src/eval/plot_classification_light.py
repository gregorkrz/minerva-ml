#!/usr/bin/env python3
"""Classification "light" PDFs only (no main q₃ / pion figure bundles).

Reads the classification pickle and writes under
``<--plots-dir>/classification/light/`` — same outputs as the light appendix
steps in ``plot_classification_q3`` (CCNπ) and ``plot_classification_Pions``
(CC1π±, CCπ⁰), without re-running confusion matrices or full tagging PDFs.

Use ``--components`` to limit work:

* ``q3`` — CCNπ vs *q₃* / *W* light figures only.
* ``pion`` — CC1π± and CCπ⁰ light figures only.
* ``all`` (default) — both sets.

Caching
-------
On first run the script computes all AUPRC/AUROC metrics and saves them to a
*plots cache* pickle (default ``plots/tmp_results/classification_light.pkl``).
Subsequent runs with ``--plots-only`` load that cache and skip the expensive
metric computation, going straight to figure drawing.

The cache stores metrics for *all* models; ``--config`` filtering is applied at
draw time, so one cache serves multiple configs.
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
from src.eval._cache_additive import light_cache_model_keys, resolve_models_to_add
from src.eval._classification_light import (
    compute_light_classification_data,
    draw_light_classification_from_cache,
    update_light_classification_cache,
)
from src.eval._constants import (
    CLASSIFICATION_PICKLE_STEM,
    DEFAULT_CACHE_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_WANDB_TAG,
    repo_output_path,
)
from src.eval._plot_config import PlotConfig
from src.eval.classification_plots import (
    CLASSIFICATION_PERFORMANCE_LEGEND_TITLE,
    DEFAULT_FIXED_FPR,
    get_signal_probabilities,
)

_LIGHT_CACHE_NAME = "classification_light.pkl"


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
        default=None,
        help="Root for PDF output (default: <out-dir>/plots)",
    )
    ap.add_argument(
        "--components",
        choices=("all", "q3", "pion"),
        default="all",
        help="Which light bundle to emit (default: all).",
    )
    ap.add_argument("--classification-pickle", type=Path, default=None)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="JSON",
        help="Plot config JSON (models, colors, optional display_name). "
        "When given, only listed models are drawn.",
    )
    ap.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip metric computation; read pre-computed draw specs from the plots "
        "cache (default: plots/tmp_results/classification_light.pkl). Requires a "
        "previous non-plots-only run to have written the cache.",
    )
    ap.add_argument(
        "--plots-cache",
        type=Path,
        default=None,
        metavar="PKL",
        help="Override the plots cache path used by --plots-only or written during "
        "a normal run (default: plots/tmp_results/classification_light.pkl).",
    )
    ap.add_argument(
        "--additive",
        action="store_true",
        help="Only compute light metrics for models missing from an existing cache.",
    )
    ap.add_argument(
        "--models",
        default=None,
        help="Comma-separated model keys to add (default: all missing from source pickle).",
    )
    args = ap.parse_args(argv)
    if args.plots_dir is None:
        args.plots_dir = Path(args.out_dir or DEFAULT_OUT_DIR) / "plots"

    print(
        "Random baseline TPR/FPR: for scores independent of truth (same notion as "
        'the gray "Random baseline" in AUPRC), expected TPR equals FPR at every '
        "operating point. Fixed-FPR targets used in these figures: "
        f"{list(DEFAULT_FIXED_FPR)!r} → random TPR equals each target FPR."
    )

    plots_root = repo_output_path(_REPO_ROOT, args.plots_dir)
    light_dir = plots_root / "classification" / "light"
    light_dir.mkdir(parents=True, exist_ok=True)

    cache_root = repo_output_path(_REPO_ROOT, DEFAULT_CACHE_DIR)
    plots_cache_path = args.plots_cache or (cache_root / _LIGHT_CACHE_NAME)

    cfg: PlotConfig | None = PlotConfig.load(args.config) if args.config else None

    def _parse_models(raw: str | None) -> list[str] | None:
        if raw is None:
            return None
        models = [m.strip() for m in raw.split(",") if m.strip()]
        return models or None

    models_filter = _parse_models(args.models)

    if args.plots_only:
        print(f"Loading plots cache from {plots_cache_path} …")
        with open(plots_cache_path, "rb") as f:
            cached = pickle.load(f)
        specs = cached["specs"]
        clrs = dict(cached["clrs"])
        if cfg is not None:
            clrs.update(cfg.colors())
        draw_light_classification_from_cache(specs, clrs, light_dir, cfg=cfg)
        return

    # --- Normal (compute + cache + draw) path ---
    data_root = repo_output_path(_REPO_ROOT, Path(args.out_dir or DEFAULT_OUT_DIR))
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

    if args.additive and plots_cache_path.exists():
        print(f"Loading existing light cache from {plots_cache_path} …")
        with open(plots_cache_path, "rb") as f:
            existing_cache = pickle.load(f)
        new_models = resolve_models_to_add(
            set(clf["results"]),
            light_cache_model_keys(existing_cache),
            models_filter,
        )
        if not new_models:
            print("Light cache already contains all requested models.")
            cached = existing_cache
        else:
            cached = update_light_classification_cache(
                existing_cache, clf, new_models, components
            )
            plots_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(plots_cache_path, "wb") as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved plots cache → {plots_cache_path}")
        specs = cached["specs"]
        clrs = dict(cached["clrs"])
        if cfg is not None:
            clrs.update(cfg.colors())
        draw_light_classification_from_cache(specs, clrs, light_dir, cfg=cfg)
        return

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

    # Compute metrics for ALL models (cache is config-independent).
    print("Computing light classification metrics (this may take a few minutes)…")
    specs = compute_light_classification_data(
        results,
        data_by_playlist,
        playlists,
        components,
        data_w_by_playlist=data_w_by_playlist,
    )

    # Save plots cache so --plots-only can skip recomputation next time.
    plots_cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plots_cache_path, "wb") as f:
        pickle.dump({"specs": specs, "clrs": clrs}, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved plots cache → {plots_cache_path}")

    # Draw (apply config filter at draw time).
    draw_clrs = {**clrs, **(cfg.colors() if cfg else {})}
    draw_light_classification_from_cache(specs, draw_clrs, light_dir, cfg=cfg)


if __name__ == "__main__":
    main()
