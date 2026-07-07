"""Build the small steps plots cache from eval pickles.

Loads classification and regression pickles **one at a time** so peak RAM is
~one source pickle instead of both (~17 GB + ~16 GB), which OOMs login nodes.
"""

from __future__ import annotations

import gc
import pickle
from pathlib import Path

from src.eval._constants import is_steps_plot_excluded_model

STEPS_CACHE_NAME = "steps.pkl"


def _strip_steps_plot_excluded_side(
    lh: dict,
    flops: dict,
    colors: dict,
) -> tuple[dict, dict, dict]:
    """Drop models that must never appear on steps plots."""
    excluded = {k for k in lh if is_steps_plot_excluded_model(k)}
    for key in excluded:
        lh.pop(key, None)
        flops.pop(key, None)
        colors.pop(key, None)
    return lh, flops, colors


def build_steps_cache_payload(
    lh_c: dict,
    flops_c: dict,
    colors_c: dict,
    lh_r: dict,
    flops_r: dict,
    colors_r: dict,
) -> dict:
    return {
        "lh_c": lh_c,
        "flops_c": flops_c,
        "colors_c": colors_c,
        "lh_r": lh_r,
        "flops_r": flops_r,
        "colors_r": colors_r,
    }


def save_steps_cache(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = path.stat().st_size / 1e6
    print(f"Saved steps cache ({size_mb:.0f} MB) → {path}")


def build_steps_cache_from_pickles(
    classification_pickle: Path,
    regression_pickle: Path,
) -> dict:
    """Extract loss histories from eval pickles with minimal peak memory."""
    print(f"Loading classification pickle (loss histories only) …")
    print(f"  {classification_pickle}")
    with open(classification_pickle, "rb") as f:
        clf = pickle.load(f)
    lh_c = clf["loss_histories"]
    flops_c = clf["flops_per_step"]
    colors_c = clf["clrs_dict_full"]
    del clf
    gc.collect()

    print(f"Loading regression pickle (loss histories only) …")
    print(f"  {regression_pickle}")
    with open(regression_pickle, "rb") as f:
        reg = pickle.load(f)
    lh_r = reg["loss_histories"]
    flops_r = reg["flops_per_step"]
    colors_r = reg["clrs_dict_full"]
    del reg
    gc.collect()

    lh_c, flops_c, colors_c = _strip_steps_plot_excluded_side(lh_c, flops_c, colors_c)
    lh_r, flops_r, colors_r = _strip_steps_plot_excluded_side(lh_r, flops_r, colors_r)

    print(
        "  classification loss_histories:",
        ", ".join(sorted(lh_c)) or "(none)",
    )
    print(
        "  regression loss_histories:",
        ", ".join(sorted(lh_r)) or "(none)",
    )

    return build_steps_cache_payload(
        lh_c,
        flops_c,
        colors_c,
        lh_r,
        flops_r,
        colors_r,
    )


def update_steps_cache_from_pickles(
    existing: dict,
    classification_pickle: Path,
    regression_pickle: Path,
    *,
    update_classification: bool = True,
    update_regression: bool = True,
) -> dict:
    """Merge new loss histories from eval pickles into *existing* steps cache."""
    from src.eval._cache_additive import merge_steps_side

    out = dict(existing)
    if update_classification and classification_pickle.exists():
        with open(classification_pickle, "rb") as f:
            clf = pickle.load(f)
        new_models = sorted(
            m
            for m in set(clf["loss_histories"]) - set(out.get("lh_c", {}))
            if not is_steps_plot_excluded_model(m)
        )
        if new_models:
            print(f"  Additive steps cache (classification): {', '.join(new_models)}")
            partial = {
                "lh_c": {m: clf["loss_histories"][m] for m in new_models},
                "flops_c": {
                    m: clf["flops_per_step"][m]
                    for m in new_models
                    if m in clf["flops_per_step"]
                },
                "colors_c": {
                    m: clf["clrs_dict_full"][m]
                    for m in new_models
                    if m in clf["clrs_dict_full"]
                },
            }
            merge_steps_side(
                out, partial, lh_key="lh_c", flops_key="flops_c", colors_key="colors_c"
            )
        else:
            print("  Steps cache (classification): no new models.")
        del clf
        gc.collect()

    if update_regression and regression_pickle.exists():
        with open(regression_pickle, "rb") as f:
            reg = pickle.load(f)
        new_models = sorted(
            m
            for m in set(reg["loss_histories"]) - set(out.get("lh_r", {}))
            if not is_steps_plot_excluded_model(m)
        )
        if new_models:
            print(f"  Additive steps cache (regression): {', '.join(new_models)}")
            partial = {
                "lh_r": {m: reg["loss_histories"][m] for m in new_models},
                "flops_r": {
                    m: reg["flops_per_step"][m]
                    for m in new_models
                    if m in reg["flops_per_step"]
                },
                "colors_r": {
                    m: reg["clrs_dict_full"][m]
                    for m in new_models
                    if m in reg["clrs_dict_full"]
                },
            }
            merge_steps_side(
                out, partial, lh_key="lh_r", flops_key="flops_r", colors_key="colors_r"
            )
        else:
            print("  Steps cache (regression): no new models.")
        del reg
        gc.collect()

    return out
