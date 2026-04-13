"""
Stripped-down utilities from OmniLearned, containing only what's needed
for model construction and pretrained checkpoint loading.
"""

import os
import torch
import requests


def get_model_parameters(model_size):
    model_dict = {}
    if model_size == "small":
        model_dict["num_transformers"] = 8
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 8
        model_dict["base_dim"] = 128
        model_dict["mlp_ratio"] = 2
    elif model_size == "medium":
        model_dict["num_transformers"] = 12
        model_dict["num_transformers_head"] = 2
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 16
        model_dict["base_dim"] = 512
        model_dict["mlp_ratio"] = 2
    elif model_size == "large":
        model_dict["num_transformers"] = 28
        model_dict["num_transformers_head"] = 4
        model_dict["num_tokens"] = 4
        model_dict["num_heads"] = 32
        model_dict["base_dim"] = 1024
        model_dict["mlp_ratio"] = 2
    else:
        raise ValueError(f"Invalid model size: {model_size}")
    return model_dict


def _filter_partial_state(checkpoint_state, model_state, verbose=True):
    """Load weights where shapes match, skip mismatched or missing keys."""
    filtered = {}
    for k, v in checkpoint_state.items():
        if "out." in k:
            if verbose:
                print(f"  Skipping {k}: output layer excluded from loading")
            continue
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            if verbose:
                model_shape = model_state[k].shape if k in model_state else "missing"
                print(
                    f"  Skipping {k}: shape mismatch (ckpt: {v.shape}, model: {model_shape})"
                )
    return filtered


PRETRAINED_URL = "https://portal.nersc.gov/cfs/m4567/checkpoints"


def load_pretrained_omnilearned(model, pretrain_tag, checkpoint_dir):
    """
    Load a pretrained OmniLearned checkpoint into model.
    Downloads from NERSC portal if not found locally.
    Uses shape-mismatch filtering for fine-tuning compatibility.
    """
    checkpoint_name = f"best_model_{pretrain_tag}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_dir, exist_ok=True)
        url = f"{PRETRAINED_URL}/{checkpoint_name}"
        print(f"Downloading pretrained checkpoint from {url} ...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(checkpoint_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Saved to {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "body" in checkpoint and model.body is not None:
        filtered = _filter_partial_state(checkpoint["body"], model.body.state_dict())
        model.body.load_state_dict(filtered, strict=False)
        print(f"Loaded body: {len(filtered)}/{len(model.body.state_dict())} params")

    if "classifier_head" in checkpoint and model.classifier is not None:
        filtered = _filter_partial_state(
            checkpoint["classifier_head"], model.classifier.state_dict()
        )
        model.classifier.load_state_dict(filtered, strict=False)
        print(
            f"Loaded classifier: {len(filtered)}/{len(model.classifier.state_dict())} params"
        )

    if "generator_head" in checkpoint and model.generator is not None:
        filtered = _filter_partial_state(
            checkpoint["generator_head"], model.generator.state_dict()
        )
        model.generator.load_state_dict(filtered, strict=False)
        print(
            f"Loaded generator: {len(filtered)}/{len(model.generator.state_dict())} params"
        )

    return model
