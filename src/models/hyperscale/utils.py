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


# Upstream HyperScale ``train_config.yaml`` writes the model class under
# ``model_type``. Map each to our ``--use-hyperscale`` variant tag.
_HS_MODEL_TYPE_TO_VARIANT = {
    "ParticleVIT": "basic",
    "ParticleVIT_Embedding": "embedding",
    "ParticleVIT_Pool": "pool",
}


def _parse_hyperscale_train_config_yaml(yaml_path):
    """Parse upstream HyperScale's ``train_config.yaml`` into our args dict.

    The upstream file (gregorkrz/HyperScale ``train.py`` dump) uses
    ``model_params.{embed_dim, depth, num_heads, mlp_ratio}`` plus a
    ``model_type`` selector. Returns a dict keyed by the argparse attr names
    used in ``src.scripts.train`` so it can be consumed the same way as our
    own ``ckpt["args"]``.
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "Parsing upstream HyperScale train_config.yaml requires pyyaml "
            "(`pip install pyyaml`)."
        ) from e
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    model_type = cfg.get("model_type")
    variant = _HS_MODEL_TYPE_TO_VARIANT.get(model_type)
    if variant is None:
        raise ValueError(
            f"Unrecognized HyperScale model_type {model_type!r} in {yaml_path}; "
            f"expected one of {tuple(_HS_MODEL_TYPE_TO_VARIANT)}"
        )
    mp = cfg.get("model_params") or {}
    out = {"use_hyperscale": variant}
    if "embed_dim" in mp:
        out["d_model"] = int(mp["embed_dim"])
    if "depth" in mp:
        out["depth"] = int(mp["depth"])
    if "num_heads" in mp:
        out["n_heads"] = int(mp["num_heads"])
    if "mlp_ratio" in mp:
        out["hs_mlp_ratio"] = float(mp["mlp_ratio"])
    return out


def _resolve_hyperscale_ckpt_path(path):
    """Accept either a checkpoint file path or a directory containing one.

    Upstream HyperScale's convention is to ship a run directory containing
    ``pvit_final.pth`` plus a ``train_config.yaml`` describing the run; our
    own ``save_checkpoint`` writes ``best_model.pt``. If ``path`` is a
    directory, look for those names first, then fall back to a single
    ``*.pt`` / ``*.pth`` file in the directory.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"HyperScale checkpoint not found: {path}")
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ("pvit_final.pth", "best_model.pt", "final.pt", "last.pt"):
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                return cand
        ckpts = sorted(
            n for n in os.listdir(path)
            if (n.endswith(".pt") or n.endswith(".pth"))
            and os.path.isfile(os.path.join(path, n))
        )
        if len(ckpts) == 1:
            return os.path.join(path, ckpts[0])
        if len(ckpts) > 1:
            raise ValueError(
                f"Multiple .pt/.pth files in {path!r}; pass an explicit file "
                f"path. Found: {ckpts}"
            )
        raise FileNotFoundError(
            f"No .pt or .pth files found in directory: {path}"
        )
    raise FileNotFoundError(
        f"HyperScale checkpoint path is neither file nor dir: {path}"
    )


def peek_hyperscale_checkpoint_args(ckpt_path):
    """Return a dict of HyperScale arch knobs for ``ckpt_path``, or ``None``.

    Accepts either a ``.pt`` file or a directory containing one (see
    ``_resolve_hyperscale_ckpt_path``). Looks for arch metadata in two places,
    in order:

    1. The ``"args"`` field of the checkpoint itself
       (``src.scripts.train.save_checkpoint`` stores ``vars(args)`` here).
    2. A ``train_config.yaml`` in the same directory as the checkpoint
       (this is what upstream gregorkrz/HyperScale writes alongside each run).

    Returns ``None`` if neither is found, so the caller can fall back to CLI
    flags.
    """
    ckpt_path = _resolve_hyperscale_ckpt_path(ckpt_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        saved = checkpoint.get("args")
        if isinstance(saved, dict):
            return saved

    yaml_path = os.path.join(os.path.dirname(ckpt_path), "train_config.yaml")
    if os.path.exists(yaml_path):
        return _parse_hyperscale_train_config_yaml(yaml_path)

    return None


def load_pretrained_hyperscale(model, ckpt_path, verbose=True):
    """Load a HyperScale checkpoint into ``model`` (a ``HyperScaleBaseline``).

    Accepts either a ``.pt`` file or a directory containing one (see
    ``_resolve_hyperscale_ckpt_path``). Loads weights only — no optimizer or
    scheduler state. The task-specific output ``head`` and any
    minerva-wrapper-only modules (``global_proj``) are kept at their random
    init so the model can be fine-tuned on a new task without output-dim
    conflicts. Missing or shape-mismatched keys are skipped with a diagnostic
    print.
    """
    ckpt_path = _resolve_hyperscale_ckpt_path(ckpt_path)

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
