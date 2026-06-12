"""Build the small steps plots cache from eval pickles.

Loads classification and regression pickles **one at a time** so peak RAM is
~one source pickle instead of both (~17 GB + ~16 GB), which OOMs login nodes.
"""

from __future__ import annotations

import gc
import pickle
from pathlib import Path

STEPS_CACHE_NAME = "steps.pkl"


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

    print(
        "  classification loss_histories:",
        ", ".join(sorted(lh_c)) or "(none)",
    )
    print(
        "  regression loss_histories:",
        ", ".join(sorted(lh_r)) or "(none)",
    )

    return build_steps_cache_payload(
        lh_c, flops_c, colors_c, lh_r, flops_r, colors_r,
    )
