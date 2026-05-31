"""Pretrained checkpoint loading for HyperScale ParticleViT models."""

import os

import torch


def _extract_state_dict(checkpoint):
    """Pull a flat ``state_dict`` out of common checkpoint container formats.

    Supports raw state dicts as well as containers like ``{"model_state_dict": ...}``
    (the format saved by ``src.scripts.train.save_checkpoint``) and
    ``{"model": ...}`` / ``{"state_dict": ...}`` (common upstream conventions).
    """
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unexpected HyperScale checkpoint type {type(checkpoint).__name__}; "
            "expected a dict-like object."
        )
    for key in ("model_state_dict", "state_dict", "model"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    # Heuristic: looks like a raw state_dict (all tensor values).
    if all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError(
        "Could not locate a state_dict in checkpoint; "
        f"top-level keys = {list(checkpoint.keys())}"
    )


def _strip_known_prefixes(state_dict):
    """Drop ``module.`` (DDP) or ``encoder.``/``model.`` wrappers from keys.

    Upstream HyperScale saves the bare ``ParticleVIT*`` state dict (no prefix),
    but DDP-trained checkpoints and a few minerva wrappers add a prefix; strip
    them so keys line up with the HyperScaleBaseline submodule names.
    """
    prefixes = ("module.", "model.", "encoder.")
    out = {}
    for k, v in state_dict.items():
        nk = k
        # Only strip if the leading segment is one of the wrappers AND the
        # remainder starts with a known HyperScale submodule name.
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
                break
        out[nk] = v
    return out


def _filter_compatible(checkpoint_state, model_state, skip_prefixes=("head.",), verbose=True):
    """Keep only ckpt entries that exist in the model with matching shape.

    ``skip_prefixes`` are dropped unconditionally (default: the task-specific
    output head, which has a different output dim per task).
    """
    filtered = {}
    skipped_shape = 0
    skipped_missing = 0
    skipped_head = 0
    for k, v in checkpoint_state.items():
        if any(k.startswith(p) for p in skip_prefixes):
            skipped_head += 1
            continue
        if k not in model_state:
            skipped_missing += 1
            if verbose:
                print(f"  Skip {k}: not in model")
            continue
        if model_state[k].shape != v.shape:
            skipped_shape += 1
            if verbose:
                print(
                    f"  Skip {k}: shape mismatch "
                    f"(ckpt {tuple(v.shape)}, model {tuple(model_state[k].shape)})"
                )
            continue
        filtered[k] = v
    if verbose:
        print(
            f"  Filter summary: kept {len(filtered)}, "
            f"skipped (head/init-from-scratch={skipped_head}, "
            f"missing={skipped_missing}, shape_mismatch={skipped_shape})"
        )
    return filtered


def load_pretrained_hyperscale(model, ckpt_path, verbose=True):
    """Load a HyperScale checkpoint into ``model`` (a ``HyperScaleBaseline``).

    Loads weights only — no optimizer/scheduler state. The task-specific output
    ``head`` and any minerva-wrapper-only modules (``global_proj``) are kept
    at their random init so the model can be fine-tuned on a new task without
    output-dim conflicts. Missing or shape-mismatched keys are skipped with a
    diagnostic print.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"HyperScale checkpoint not found: {ckpt_path}")

    print(f"Loading HyperScale pretrained checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = _strip_known_prefixes(_extract_state_dict(checkpoint))
    target = model.state_dict()
    filtered = _filter_compatible(state, target, verbose=verbose)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    n_loaded = len(filtered)
    n_total = len(target)
    print(
        f"HyperScale: loaded {n_loaded}/{n_total} model tensors "
        f"({len(missing)} left at init, {len(unexpected)} unexpected)"
    )
    return model
