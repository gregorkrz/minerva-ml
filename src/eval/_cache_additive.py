"""Helpers for incremental (additive) plot-cache updates."""

from __future__ import annotations

from typing import Any


def metrics_cache_model_keys(cache: dict) -> set[str]:
    """Model keys present in a classification metrics cache."""
    return set(cache.get("confusion_matrices", {}))


def light_cache_model_keys(cache: dict) -> set[str]:
    """Model keys present in a classification light cache."""
    keys: set[str] = set()
    for spec in cache.get("specs", []):
        all_metrics = spec.get("all_metrics")
        if isinstance(all_metrics, dict):
            keys.update(all_metrics)
    return keys


def resolve_models_to_add(
    source_model_keys: set[str],
    cached_model_keys: set[str],
    models: list[str] | None = None,
) -> list[str]:
    """Return sorted model keys in *source* but not yet in *cache*.

    When *models* is set, only those names are considered (must exist in source).
    """
    if models is not None:
        unknown = sorted(set(models) - source_model_keys)
        if unknown:
            raise ValueError(
                f"Requested model(s) not in source pickle: {', '.join(unknown)}"
            )
        todo = set(models) - cached_model_keys
        return sorted(todo)
    return sorted(source_model_keys - cached_model_keys)


def merge_model_metrics_tree(existing: dict, partial: dict) -> None:
    """Merge ``tag -> playlist -> mask -> {model: agg}`` trees in place."""
    for tag, pl_dict in partial.items():
        if tag not in existing:
            existing[tag] = pl_dict
            continue
        for pl, mask_dict in pl_dict.items():
            if pl not in existing[tag]:
                existing[tag][pl] = mask_dict
                continue
            for mask, model_dict in mask_dict.items():
                if mask not in existing[tag][pl]:
                    existing[tag][pl][mask] = dict(model_dict)
                else:
                    existing[tag][pl][mask].update(model_dict)


def merge_prc_tree(existing: dict, partial: dict) -> None:
    """Merge ``tag -> playlist -> {model: prc_stats}`` trees in place."""
    for tag, pl_dict in partial.items():
        if tag not in existing:
            existing[tag] = pl_dict
            continue
        for pl, model_dict in pl_dict.items():
            if pl not in existing[tag]:
                existing[tag][pl] = dict(model_dict)
            else:
                existing[tag][pl].update(model_dict)


def merge_light_classification_specs(
    existing_specs: list[dict[str, Any]],
    new_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge draw specs by ``filename``, updating ``all_metrics`` for new models."""
    by_filename = {s["filename"]: s for s in existing_specs}
    for spec in new_specs:
        fn = spec["filename"]
        if fn not in by_filename:
            by_filename[fn] = spec
            continue
        cur = by_filename[fn]
        if cur.get("type") != spec.get("type"):
            raise ValueError(
                f"Light spec type mismatch for {fn!r}: "
                f"{cur.get('type')!r} vs {spec.get('type')!r}"
            )
        cur["all_metrics"].update(spec["all_metrics"])
    return list(by_filename.values())


def merge_steps_side(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    lh_key: str,
    flops_key: str,
    colors_key: str,
) -> None:
    """Merge one task side (classification or regression) of a steps cache."""
    existing.setdefault(lh_key, {}).update(incoming.get(lh_key, {}))
    existing.setdefault(flops_key, {}).update(incoming.get(flops_key, {}))
    existing.setdefault(colors_key, {}).update(incoming.get(colors_key, {}))
