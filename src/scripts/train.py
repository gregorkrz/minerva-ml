"""
Training script for PointGlobalMixedViT, OmniLearned, BERT, or HyperScale on HEP data.

# Energy regression training:
## ViT-like transformer training:
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG --d_model 128 --depth 4 --n_heads 8  --max_steps 250000 [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]


## OmniLearned Small training:
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG  --max_steps 250000 --use-omnilearned small --use-pretrained pretrain_s [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]


## OmniLearned Small random weights training:
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG  --max_steps 250000 --use-omnilearned small  [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]



# Event Classification training:
## ViT-like transformer training:
python -m src.scripts.train -bs 10 --mode classifier -npi2 -name DEB g   UG --d_model 64 --depth 4 --n_heads 4  --max_steps 100000 [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

## OmniLearned Small training:
python -m src.scripts.train -bs 10 --mode classifier -npi2 -name DEBUG --max_steps 100000 --use-omnilearned small --use-pretrained pretrain_s [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

## OmniLearned Small random weights training:
python -m src.scripts.train -bs 10 --mode classifier -npi2 -name DEBUG --max_steps 100000 --use-omnilearned small  [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]


## BERT-style baseline (HF pretrained; same particle features as OmniLearned):
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG --max_steps 250000 --use-bert small [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

## BERT-tiny architecture with random encoder weights (hub config only; equivalent to `--use-bert tiny --bert-random-init`):
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG --max_steps 250000 --use-bert tiny-rw [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

## BERT with particles sorted by descending log(E) before the encoder:
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG --max_steps 250000 --use-bert tiny --bert-energy-order [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

## HyperScale ParticleViT (basic / embedding / pool variant; same particle features as OmniLearned):
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG --d_model 256 --depth 3 --n_heads 8 --max_steps 250000 --use-hyperscale basic [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]


# Restrict training + validation to specific GENIE interaction types (QE/RES/DIS/COH/MEC):
## Train only on DIS events:
python -m src.scripts.train ... --event-types DIS
## Train on DIS + RES (names and int codes are both accepted):
python -m src.scripts.train ... --event-types DIS RES
python -m src.scripts.train ... --event-types 3 2

"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime


def _load_project_env() -> None:
    """Load KEY=VALUE lines from the repository root `.env` into `os.environ`.

    Does not override variables already set in the environment (same default as python-dotenv).
    """
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_project_env()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pytorch_optimizer import Lion
import wandb
from tqdm import tqdm
import numpy as np

from src.dataset.dataloader import (
    load_data,
    Task,
    parse_event_types,
    INT_TYPE_NAME_TO_CODE,
)
from src.constants.dataset import GLOBAL_COND_BASE_DIM
from src.models.vit import PointGlobalMixedViT, PointGlobalMixedViTConfig
from src.models.omnilearned import (
    PET2,
    get_model_parameters,
    load_pretrained_omnilearned,
)
from src.models.hyperscale import (
    ParticleVIT as _HSParticleVIT,
    ParticleVIT_Embedding as _HSParticleVIT_Embedding,
    ParticleVIT_Pool as _HSParticleVIT_Pool,
    init_olmo_weights as _hs_init_olmo_weights,
    _zero_masked_tokens as _hs_zero_masked_tokens,
    load_pretrained_hyperscale,
    peek_hyperscale_checkpoint_args,
)

# Architecture knobs that --hs-pretrained will auto-fill from a saved checkpoint
# when the user hasn't passed --use-hyperscale on the CLI. Keys are the argparse
# attribute names on `args`.
_HYPERSCALE_AUTOFILL_KEYS = (
    "use_hyperscale",
    "d_model",
    "depth",
    "n_heads",
    "hs_mlp_ratio",
)


def _maybe_autofill_hyperscale_args(args):
    """If --hs-pretrained is given without --use-hyperscale, restore the
    architecture knobs (variant, d_model, depth, n_heads, hs_mlp_ratio) from
    the checkpoint's saved ``args`` dict. CLI takes precedence: if the user
    explicitly passed --use-hyperscale, we don't touch anything.

    Returns True if any knob was overridden, False otherwise.
    """
    if not getattr(args, "hs_pretrained", None):
        return False
    if getattr(args, "use_hyperscale", None):
        # User specified the variant on the CLI; trust their flags as-is.
        return False
    saved = peek_hyperscale_checkpoint_args(args.hs_pretrained)
    if saved is None:
        raise ValueError(
            "--hs-pretrained was given without --use-hyperscale, but the "
            f"checkpoint at {args.hs_pretrained!r} has no saved args to "
            "auto-fill arch from. Pass --use-hyperscale {basic,embedding,pool} "
            "(and --d_model/--depth/--n_heads/--hs-mlp-ratio to match) "
            "explicitly."
        )
    if not saved.get("use_hyperscale"):
        raise ValueError(
            f"Checkpoint at {args.hs_pretrained!r} was not produced by a "
            "HyperScale run (saved args have no 'use_hyperscale'). Either "
            "pass --use-hyperscale explicitly, or point --hs-pretrained at a "
            "HyperScale checkpoint."
        )
    overrides = {}
    for key in _HYPERSCALE_AUTOFILL_KEYS:
        if key in saved:
            overrides[key] = saved[key]
            setattr(args, key, saved[key])
    print(
        "Auto-filled HyperScale architecture from checkpoint "
        f"({args.hs_pretrained}): "
        + ", ".join(f"{k}={v}" for k, v in overrides.items())
    )
    return True

HYPERSCALE_VARIANTS = ("basic", "embedding", "pool")

# print CUDA_VISIBLE_DEVICES
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Hugging Face model ids for --use-bert (requires: pip install transformers)
BERT_PRESETS = {
    "small": "prajjwal1/bert-tiny",
    "tiny": "prajjwal1/bert-tiny",
    "tiny-rw": "prajjwal1/bert-tiny",  # same arch as tiny; random-init encoder (see create_bert_model)
    "distil": "distilbert-base-uncased",
}


def _load_hf_sequence_encoder(
    pretrained_model_name_or_path: str, random_init: bool = False
):
    """Load BERT or DistilBERT by class name — not AutoModel.

    `prajjwal1/bert-tiny` ships a minimal config.json without `model_type`, which breaks
    `AutoModel.from_pretrained` on recent `transformers` versions.

    If ``random_init``, architecture (and vocab size, etc.) is taken from the hub ``config.json``,
    but weights are freshly initialized (no checkpoint weights).
    """
    from transformers import (
        BertConfig,
        BertModel,
        DistilBertConfig,
        DistilBertModel,
    )
    name = pretrained_model_name_or_path.lower()
    if "distilbert" in name:
        if random_init:
            cfg = DistilBertConfig.from_pretrained(pretrained_model_name_or_path)
            return DistilBertModel(cfg)
        return DistilBertModel.from_pretrained(pretrained_model_name_or_path)
    if random_init:
        cfg = BertConfig.from_pretrained(pretrained_model_name_or_path)
        return BertModel(cfg)
    return BertModel.from_pretrained(pretrained_model_name_or_path)


class BertBaseline(nn.Module):
    """Pretrained BERT over per-particle embeddings (inputs_embeds), mean-pool or CLS, linear head."""

    def __init__(
        self,
        input_dim,
        output_dim,
        pretrained_model_name_or_path: str,
        use_cls_token: bool = False,
        bert_random_init: bool = False,
        global_cont_dim: int = 0,
    ):
        super().__init__()
        self.use_cls_token = use_cls_token
        self.bert = _load_hf_sequence_encoder(
            pretrained_model_name_or_path, random_init=bert_random_init
        )
        hidden = int(self.bert.config.hidden_size)
        self.proj = nn.Linear(input_dim, hidden)
        self.global_proj = (
            nn.Linear(global_cont_dim, hidden) if global_cont_dim > 0 else None
        )
        self.head = nn.Linear(hidden, output_dim)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)

    def forward(self, x, mask, global_cont=None):
        # x: [B, N, input_dim], mask: [B, N] (1 = valid, 0 = pad)
        # global_cont: [B, global_cont_dim] -> projected and prepended as token 0 when enabled
        x = self.proj(x)
        if self.global_proj is not None:
            if global_cont is None:
                raise ValueError("BERT global token enabled but global_cont is missing")
            gtok = (
                self.global_proj(global_cont)
                .unsqueeze(1)
                .to(device=x.device, dtype=x.dtype)
            )
            x = torch.cat([gtok, x], dim=1)
            ones = torch.ones(x.shape[0], 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([ones, mask], dim=1)
        if self.use_cls_token:
            B = x.shape[0]
            cls = self.cls_token.expand(B, -1, -1).to(device=x.device, dtype=x.dtype)
            x = torch.cat([cls, x], dim=1)
            ones = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            mask = torch.cat([ones, mask], dim=1)
        attn = mask.to(dtype=torch.long)
        out = self.bert(inputs_embeds=x, attention_mask=attn)
        hs = out.last_hidden_state
        if self.use_cls_token:
            pooled = hs[:, 0]
        else:
            w = mask.unsqueeze(-1).to(dtype=hs.dtype)
            pooled = (hs * w).sum(dim=1) / w.sum(dim=1).clamp(min=1e-6)
        return self.head(pooled)


class HyperScaleBaseline(nn.Module):
    """HyperScale ParticleViT (basic/embedding/pool variant) wrapped to optionally
    consume a projected global-feature token (mirrors BertBaseline)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        variant: str = "basic",
        mlp_ratio: float = 8 / 3,
        global_cont_dim: int = 0,
    ):
        super().__init__()
        if variant not in HYPERSCALE_VARIANTS:
            raise ValueError(
                f"Unknown HyperScale variant {variant!r}; expected one of {HYPERSCALE_VARIANTS}"
            )
        if variant == "basic":
            inner = _HSParticleVIT(
                input_dim, output_dim, embed_dim, depth, num_heads, mlp_ratio
            )
        elif variant == "embedding":
            inner = _HSParticleVIT_Embedding(
                input_dim, output_dim, embed_dim, depth, num_heads, mlp_ratio
            )
        else:  # pool
            inner = _HSParticleVIT_Pool(
                input_dim, output_dim, embed_dim, depth, num_heads, mlp_ratio
            )

        self.variant = variant
        self.embed_dim = embed_dim
        # Reuse Hyperscale-initialized submodules directly.
        self.token_embed = inner.token_embed
        self.blocks = inner.blocks
        self.norm = inner.norm
        self.head = inner.head
        self.cls_token = getattr(inner, "cls_token", None)
        self.pool = getattr(inner, "pool", None)

        if global_cont_dim > 0:
            self.global_proj = nn.Linear(global_cont_dim, embed_dim)
            _hs_init_olmo_weights(self.global_proj)
        else:
            self.global_proj = None

    def forward(self, X, mask, global_cont=None):
        # X: (B, N, input_dim); mask: (B, N) {0,1} float or bool; global_cont: (B, gcd) or None
        attn_mask = mask.to(dtype=torch.bool) if mask is not None else None

        if self.variant == "embedding":
            x = self.token_embed(X, attn_mask=attn_mask)
        else:
            x = self.token_embed(X)
        x = _hs_zero_masked_tokens(x, attn_mask)

        bs = x.shape[0]
        if self.global_proj is not None:
            if global_cont is None:
                raise ValueError(
                    "HyperScale global token enabled but global_cont is missing"
                )
            g = (
                self.global_proj(global_cont)
                .unsqueeze(1)
                .to(device=x.device, dtype=x.dtype)
            )
            x = torch.cat([g, x], dim=1)
            if attn_mask is not None:
                ones = torch.ones(
                    bs, 1, dtype=torch.bool, device=attn_mask.device
                )
                attn_mask = torch.cat([ones, attn_mask], dim=1)

        if self.cls_token is not None:
            cls = self.cls_token.expand(bs, -1, -1)
            x = torch.cat([cls, x], dim=1)
            if attn_mask is not None:
                ones = torch.ones(
                    bs, 1, dtype=torch.bool, device=attn_mask.device
                )
                attn_mask = torch.cat([ones, attn_mask], dim=1)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.norm(x)
        x = _hs_zero_masked_tokens(x, attn_mask)

        if self.pool is not None:
            pooled = self.pool(x, attn_mask=attn_mask)
        else:
            pooled = x[:, 0]
        return self.head(pooled)


class ResidualBlock(nn.Module):
    """Pre-norm residual block with SiLU (smooth, good for regression)."""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(self.norm(x))


class CondOnlyMLP(nn.Module):
    """MLP with residual blocks that operates only on global/conditional features.
    Uses SiLU activations and optional positive output for log(1+E) regression."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        n_layers=3,
        dropout=0.0,
        output_positive=False,
    ):
        super().__init__()
        self.output_positive = output_positive
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.head_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.head(self.head_norm(x))
        if self.output_positive:
            x = F.softplus(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PointGlobalMixedViT on HEP data"
    )
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="minerva_1A",
        help="Dataset name (e.g., minerva_1A)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset directory",
        default="/global/cfs/cdirs/m3246/gregork/Minerva/20260311",
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=2048, help="Batch size for training"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--max_particles",
        type=int,
        default=33,
        help="Maximum number of particles per event",
    )
    # Model arguments
    parser.add_argument(
        "--mode",
        type=str,
        default="regression",
        choices=["regression", "classifier"],
        help="Training mode: regression or classifier",
    )
    parser.add_argument(
        "--classification_event_type",
        "-ec",
        action="store_true",
        help="Classify event type (requires mode=classifier)",
    )
    parser.add_argument(
        "--classification_current",
        "-cc",
        action="store_true",
        help="Classify event current (requires mode=classifier)",
    )
    parser.add_argument(
        "--classification_cc_1pi",
        "-cc1pi",
        action="store_true",
        help="Classify CC 1pi (requires mode=classifier)",
    )
    parser.add_argument(
        "--classification_n_pions",
        "-npi",
        action="store_true",
        help="Classify number of pions (requires mode=classifier)",
    )
    parser.add_argument(
        "--classification_CC1orNPi",
        "-npi2",
        action="store_true",
        help="Classify CC 1pi or n pions, according to signal definition inEberly et al. 2015 (requires mode=classifier)",
    )
    parser.add_argument(
        "--regress-E-available",
        "-E-available",
        action="store_true",
        help="Regress available energy of the event (requires mode=regression)",
    )
    parser.add_argument(
        "--regress-E-available-no-muon",
        "-E-available-no-muon",
        action="store_true",
        help="Regress available energy of the event, without the muon energy(requires mode=regression)",
    )
    parser.add_argument(
        "--no_use_cond",
        action="store_true",
        help="Do NOT use global/conditional features",
    )
    parser.add_argument(
        "--cond_only",
        "--cond-only",
        action="store_true",
        help="Train a simple MLP using only global/conditional features (no transformer)",
    )
    parser.add_argument(
        "--mlp_layers",
        type=int,
        default=3,
        help="Number of residual blocks in CondOnlyMLP (independent of --depth)",
    )
    parser.add_argument(
        "--use_pid", type=bool, default=True, help="Use particle ID information"
    )
    parser.add_argument(
        "--pid_idx", type=int, default=4, help="Index of PID in features"
    )
    # Model architecture (defaults from vit.py example)
    parser.add_argument(
        "--point_cont_dim",
        type=int,
        default=9,
        help="Dimension of continuous point features",
    )
    parser.add_argument(
        "--coord_dim", type=int, default=2, help="Dimension of coordinates"
    )
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument(
        "--depth", type=int, default=4, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--mlp_ratio", type=float, default=4.0, help="MLP hidden dimension ratio"
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument(
        "--attn_dropout", type=float, default=0.0, help="Attention dropout rate"
    )
    parser.add_argument(
        "--weighted_regression_loss",
        "-wl",
        action="store_true",
        help="Use weighted regression loss",
    )
    parser.add_argument(
        "--binned-loss-var",
        choices=["W", "q3"],
        default=None,
        help="Kinematic variable for per-bin class loss weights (classifier + CC1orNPi)",
    )
    parser.add_argument(
        "--binned-loss-signal",
        default=None,
        help=(
            "Signal definition for binned loss fallback "
            "(CC1pipm, CC1pi0, CCN1pipm, CCNpipm; case-insensitive)"
        ),
    )
    parser.add_argument(
        "--binary-classifier",
        action="store_true",
        default=False,
        help=(
            "With -npi2: train a 2-class signal/background classifier. Pi_labels_v2 "
            "classes in --binary-signal (default CCN1pipm -> pid 0,1) map to label 1; "
            "all other pid classes map to label 0. Eval npz still stores the original "
            "pid class for plotting."
        ),
    )
    parser.add_argument(
        "--binary-signal",
        default="CCN1pipm",
        help=(
            "Pi_labels_v2 pid classes treated as signal when --binary-classifier is set "
            "(default CCN1pipm: CC with >=1 charged pion, pid 0 and 1)."
        ),
    )
    parser.add_argument(
        "--predict-baseline",
        action="store_true",
        default=False,
        help=(
            "With -npi2: train to predict cut-based baseline Pi_labels_v2 (reco topology "
            "cuts from baselines/*.npz) instead of MC truth. Multiclass uses the same "
            "5-class scheme; with --binary-classifier, baseline classes in --binary-signal "
            "map to label 1. Eval still compares against MC truth."
        ),
    )
    parser.add_argument(
        "--log_MSE_loss", "-log-mse", action="store_true", help="Use log MSE loss"
    )
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay", "-wd", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--event-cap",
        "-cap",
        type=int,
        default=-1,
        help="Maximum number of events to use in the dataset",
    )
    parser.add_argument(
        "--event-sampler-random-state",
        "-seed-event-sampler",
        type=int,
        default=42,
        help="Random seed for event sampler",
    )
    parser.add_argument(
        "--max_steps", type=int, default=100000, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--use_amp",
        "--fp16",
        action="store_true",
        default=False,
        help="Use automatic mixed precision (autocast; --fp16 is an alias)",
    )
    parser.add_argument(
        "--max_samples_per_epoch",
        type=int,
        default=None,
        help="Maximum number of samples to use per epoch",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps (virtual batch size = batch_size * grad_accum_steps)",
    )
    # Logging and evaluation
    parser.add_argument(
        "--log_interval", type=int, default=1000, help="Log training loss every N steps"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=1000, help="Run evaluation every N steps"
    )
    parser.add_argument(
        "--save_interval", type=int, default=1000, help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="minerva-models", help="Wandb project name"
    )
    parser.add_argument(
        "--run_name",
        "-name",
        type=str,
        default=None,
        help="Name for this training run (timestamp will be appended); not required when --calculate-flops",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/global/cfs/cdirs/m3246/gregork/checkpoints",
        help="Base output directory for checkpoints (run_name with timestamp will be appended)",
    )
    parser.add_argument("--log1p_loss", type=bool, default=True, help="Use log1p loss")
    # Other
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--resume-run-id",
        type=str,
        default=None,
        help="Wandb run id to resume (for logging). If not set when resuming, uses id saved in checkpoint.",
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "lion"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--include-E-sum",
        type=bool,
        default=True,
        help="Include per-PID energy sums (blob, prong types, aggregated) as extra global features",
    )
    parser.add_argument(
        "--zero-cond-feature",
        type=int,
        nargs="+",
        default=None,
        help="Zero out global/cond feature(s) at these indices (ablation). "
        "E.g. --zero-cond-feature 3 to ablate E_recoil_CCinc",
    )
    # OmniLearned (PET2) arguments
    parser.add_argument(
        "--use-omnilearned",
        type=str,
        default=None,
        choices=["small", "medium", "large"],
        help="Use OmniLearned PET2 model of given size instead of ViT",
    )
    parser.add_argument(
        "--use-bert",
        type=str,
        nargs="?",
        const="small",
        default=None,
        choices=["tiny", "tiny-rw", "small", "distil"],
        metavar="SIZE",
        help="Use BERT-style encoder on particle features (same layout as OmniLearned). "
        "Presets tiny/small load prajjwal1/bert-tiny weights; tiny-rw uses the same config but "
        "random encoder init (or use --use-bert tiny --bert-random-init). Requires `transformers`.",
    )
    parser.add_argument(
        "--use-cls-token",
        action="store_true",
        default=False,
        help="With --use-bert: prepend a learnable CLS embedding and use its output (position 0) "
        "instead of masked mean pooling over particles.",
    )
    parser.add_argument(
        "--bert-random-init",
        action="store_true",
        default=False,
        help="With --use-bert: use hub config for architecture but randomly initialize BERT weights "
        "(no pretrained checkpoint).",
    )
    parser.add_argument(
        "--bert-energy-order",
        action="store_true",
        default=False,
        help="With --use-bert: sort each event's particles by descending log(E) before the encoder "
        "(stable sort; padding stays at the end).",
    )
    # HyperScale ParticleViT arguments
    parser.add_argument(
        "--use-hyperscale",
        type=str,
        nargs="?",
        const="basic",
        default=None,
        choices=list(HYPERSCALE_VARIANTS),
        metavar="VARIANT",
        help="Use HyperScale ParticleViT (gregorkrz/HyperScale) instead of ViT. "
        "VARIANT selects the input head: basic (linear embed + CLS), "
        "embedding (split kin/PID/vertex embeds, requires 9 features), "
        "pool (linear embed + attention pool, no CLS). "
        "Architecture knobs come from --d_model, --depth, --n_heads, --hs-mlp-ratio.",
    )
    parser.add_argument(
        "--hs-mlp-ratio",
        type=float,
        default=8 / 3,
        help="MLP ratio for HyperScale SwiGLU FFN (default 8/3, matches upstream HyperScale).",
    )
    parser.add_argument(
        "--hs-pretrained",
        type=str,
        default=None,
        help="Path to a HyperScale checkpoint to initialize the encoder from. "
        "Loads weights only (no optimizer state); shape-mismatched tensors and "
        "the task output head are skipped so the model can be fine-tuned on a "
        "new task. Requires --use-hyperscale.",
    )
    parser.add_argument(
        "--use-pretrained",
        type=str,
        default=None,
        help="Load pretrained OmniLearned checkpoint (e.g. pretrain_s, pretrain_m)",
    )
    parser.add_argument(
        "--ol-num-feat",
        type=int,
        default=4,
        help="Number of kinematic input features for PET2 (excluding PID)",
    )
    parser.add_argument(
        "--ol-num-add",
        type=int,
        default=5,
        help="Number of additional features for PET2 add_info input",
    )
    parser.add_argument(
        "--ol-num-cond",
        type=int,
        default=10,
        help="Number of global conditioning features for PET2 (must match global_features width before energy-sum cols)",
    )
    parser.add_argument(
        "--ol-pid-dim",
        type=int,
        default=8,
        help="Number of unique PID classes for PET2 embedding",
    )
    parser.add_argument(
        "--ol-interaction",
        action="store_true",
        default=False,
        help="Enable interaction matrix in PET2",
    )
    parser.add_argument(
        "--ol-local-interaction",
        action="store_true",
        default=False,
        help="Enable local interaction matrix in PET2",
    )
    parser.add_argument(
        "--ol-interaction-type",
        type=str,
        default="lhc",
        choices=["lhc", "astro"],
        help="Interaction type for PET2",
    )
    parser.add_argument(
        "--calculate-flops",
        action="store_true",
        help="Only compute FLOPs per batch (inference and approx training) then exit",
    )
    parser.add_argument(
        "--event-types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="Restrict training/validation to events with the given GENIE interaction "
        "types (truth_labels[:, 1] = mc_intType). Accepts names (QE, RES, DIS, COH, MEC) "
        "and/or integer codes (1=QE, 2=RES, 3=DIS, 4=COH, 8=MEC). "
        "Example: --event-types DIS, or --event-types 3, or --event-types DIS RES. "
        "If omitted, train on all event types (current default).",
    )
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_schedule(optimizer, warmup_steps, max_steps):
    """Get learning rate schedule with linear warmup and cosine decay."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_task(args):
    if args.mode == "regression":
        class_label_idx = None
        class_idx = None
        if args.regress_E_available:
            class_label_idx = 8
        elif args.regress_E_available_no_muon:
            class_label_idx = 9
        return Task(
            type="regression",
            regress_E_available=args.regress_E_available,
            regress_E_available_no_muon=args.regress_E_available_no_muon,
            class_label_idx=class_label_idx,
            regress_log=False,
        )
    elif args.mode == "classifier":
        if "classification_n_pions" not in args.__dict__:
            args.classification_n_pions = False
        binary_signal_pid: list[int] = []
        if args.classification_event_type:
            class_label_idx = 1
            class_idx = [1, 2, 3, 4, 8]
            class_idx_map = {1: 0, 2: 1, 3: 2, 4: 3, 8: 4}
        elif args.classification_current:
            class_label_idx = 3
            class_idx = [1, 2]
            class_idx_map = {1: 0, 2: 1}
        elif args.classification_cc_1pi:
            class_label_idx = 4
            class_idx = [0, 1, 2]
            class_idx_map = {0: 0, 1: 1, 2: 2}
        elif args.classification_n_pions:
            class_label_idx = 7
            class_idx = [0, 1]
            class_idx_map = {0: 0, 1: 1}
        elif args.classification_CC1orNPi:
            class_label_idx = -1
            if getattr(args, "binary_classifier", False):
                from src.eval.classification_plots._signal_definitions import (
                    resolve_signal_classes,
                )

                binary_signal_pid = resolve_signal_classes(args.binary_signal)
                class_idx = [0, 1]
                class_idx_map = {0: 0, 1: 1}
            else:
                class_idx = [0, 1, 2, 3, 4]
                class_idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        return Task(
            type="classifier",
            classification_event_type=args.classification_event_type,
            classification_current=args.classification_current,
            classification_cc_1pi=args.classification_cc_1pi,
            classification_n_pions=args.classification_n_pions,
            class_idx=class_idx,
            class_idx_map=class_idx_map,
            class_label_idx=class_label_idx,
            classification_CC1orNPi=args.classification_CC1orNPi,
            classification_binary=(
                getattr(args, "binary_classifier", False)
                and args.classification_CC1orNPi
            ),
            binary_signal_pid_classes=binary_signal_pid,
            predict_baseline=getattr(args, "predict_baseline", False),
        )
    else:
        raise ValueError("Invalid mode")


def create_model(args, task: Task):
    """Create PointGlobalMixedViT model or CondOnlyMLP."""
    if task.type == "classifier":
        num_classes = len(task.class_idx)
    elif task.type == "regression":
        num_classes = 1
    else:
        raise ValueError("Invalid task type")

    e_sum_dim = 6 if args.include_E_sum else 0

    if args.cond_only:
        global_cont_dim = GLOBAL_COND_BASE_DIM + e_sum_dim
        model = CondOnlyMLP(
            input_dim=global_cont_dim,
            hidden_dim=args.d_model,
            output_dim=num_classes,
            n_layers=args.mlp_layers,
            dropout=args.dropout,
            output_positive=(task.type == "regression"),
        )
        return model

    # Point categorical features (PID if used)
    point_cat_num_classes = [8] if args.use_pid else []

    # Global categorical features (none by default)
    global_cat_num_classes = []

    # Global continuous dimension
    global_cont_dim = (GLOBAL_COND_BASE_DIM if args.use_cond else 0) + e_sum_dim
    use_event_token = args.use_cond or args.include_E_sum

    cfg = PointGlobalMixedViTConfig(
        point_cont_dim=args.point_cont_dim,
        point_cat_num_classes=point_cat_num_classes,
        global_cont_dim=global_cont_dim,
        global_cat_num_classes=global_cat_num_classes,
        coord_dim=args.coord_dim,
        d_model=args.d_model,
        depth=args.depth,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        use_cls_token=True,
        use_event_token=use_event_token,
        cat_emb_dim=16,
    )

    model = PointGlobalMixedViT(cfg)

    # Add output head
    if args.mode == "regression":
        model.head = nn.Linear(args.d_model, 1)
    else:  # classifier
        model.head = nn.Linear(args.d_model, num_classes)
    return model


def create_omnilearned_model(args, task):
    """Create an OmniLearned PET2 model."""
    if task.type == "classifier":
        num_classes = len(task.class_idx)
    elif task.type == "regression":
        num_classes = 1
    else:
        raise ValueError("Invalid task type")

    model_params = get_model_parameters(args.use_omnilearned)
    use_cond = not args.no_use_cond
    e_sum_dim = 6 if args.include_E_sum else 0
    cond_dim = args.ol_num_cond + e_sum_dim
    if e_sum_dim > 0:
        use_cond = True

    model = PET2(
        input_dim=args.ol_num_feat,
        use_int=args.ol_interaction,
        local_int=args.ol_local_interaction,
        int_type=args.ol_interaction_type,
        conditional=use_cond,
        cond_dim=cond_dim,
        pid=args.use_pid,
        pid_dim=args.ol_pid_dim,
        add_info=True,
        add_dim=args.ol_num_add,
        mode=args.mode,
        num_classes=num_classes,
        num_gen_classes=1,
        mlp_drop=args.dropout,
        attn_drop=args.attn_dropout,
        feature_drop=0.0,
        num_coord=args.coord_dim,
        K=10,
        **model_params,
    )
    return model


def _apply_omnilearned_medium_backbone_freeze(model, args):
    """Freeze PET2 backbone for OmniLearned medium (OLM) in classifier/regression."""
    if not getattr(args, "use_omnilearned", None):
        return
    if args.use_omnilearned != "medium":
        return
    if not isinstance(model, PET2):
        return
    for p in model.body.parameters():
        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.body.parameters())
    mode = getattr(args, "mode", "unknown")
    print(
        f"OmniLearned medium ({mode}): frozen backbone ({n_frozen:,} params), "
        f"training head only ({n_train:,} trainable params)."
    )


def _bert_input_dim(args):
    """Particle feature width after optional PID strip (matches OmniLearned PET2)."""
    return args.ol_num_feat if args.use_pid else (args.ol_num_feat + 1)


def create_bert_model(args, task):
    """Create BertBaseline with HF BERT/DistilBERT (pretrained or random-init per args)."""
    if task.type == "classifier":
        num_classes = len(task.class_idx)
    elif task.type == "regression":
        num_classes = 1
    else:
        raise ValueError("Invalid task type")
    pretrained_name = BERT_PRESETS[args.use_bert]
    ri = bool(getattr(args, "bert_random_init", False)) or args.use_bert == "tiny-rw"
    e_sum_dim = 6 if args.include_E_sum else 0
    global_cont_dim = (GLOBAL_COND_BASE_DIM if args.use_cond else 0) + e_sum_dim
    if ri:
        print(
            "BERT encoder: random weight init (config from preset id, no HF checkpoint weights)."
        )
    return BertBaseline(
        input_dim=_bert_input_dim(args),
        output_dim=num_classes,
        pretrained_model_name_or_path=pretrained_name,
        use_cls_token=bool(getattr(args, "use_cls_token", False)),
        bert_random_init=ri,
        global_cont_dim=global_cont_dim,
    )


def _hyperscale_input_dim(args):
    """Per-particle feature width seen by the HyperScale embedding.

    The minerva dataset stores X as ``[..., point_cont_dim + 1]`` when ``use_pid`` is True
    (PID lives at ``pid_idx``). HyperScale's basic/pool variants accept any width via a
    Linear; the embedding variant requires exactly 9 (4 kin + 1 PID + 4 vertex), so the
    wrapper truncates X to 9 features in ``prepare_batch_hyperscale``.
    """
    if getattr(args, "use_hyperscale", None) == "embedding":
        return 9
    return args.point_cont_dim + (1 if args.use_pid else 0)


def create_hyperscale_model(args, task):
    """Create a HyperScaleBaseline (ParticleVIT basic / embedding / pool variant)."""
    if task.type == "classifier":
        num_classes = len(task.class_idx)
    elif task.type == "regression":
        num_classes = 1
    else:
        raise ValueError("Invalid task type")
    e_sum_dim = 6 if args.include_E_sum else 0
    global_cont_dim = (GLOBAL_COND_BASE_DIM if args.use_cond else 0) + e_sum_dim
    return HyperScaleBaseline(
        input_dim=_hyperscale_input_dim(args),
        output_dim=num_classes,
        embed_dim=args.d_model,
        depth=args.depth,
        num_heads=args.n_heads,
        variant=args.use_hyperscale,
        mlp_ratio=args.hs_mlp_ratio,
        global_cont_dim=global_cont_dim,
    )


def prepare_batch_hyperscale(
    batch,
    device,
    use_pid=False,
    pid_idx=4,
    use_cond=False,
    include_E_sum=False,
    zero_cond_feature=None,
    variant="basic",
):
    """Prepare batch for HyperScaleBaseline.

    Keeps PID inline (the embedding variant indexes it from X[..., 4]); when use_pid=False
    the PID column is stripped to match the BERT/OmniLearned no-PID layout. The embedding
    variant requires exactly 9 features and X is truncated accordingly.
    """
    X = batch["X"].to(device, dtype=torch.float32)
    y = batch["y"].to(device)
    mask = batch["attention_mask"].to(device, dtype=torch.float32)

    if not use_pid and batch.get("pid") is not None:
        X = torch.cat([X[:, :, :pid_idx], X[:, :, pid_idx + 1 :]], dim=2)

    if variant == "embedding" and X.shape[-1] > 9:
        X = X[:, :, :9]

    global_cont = None
    if use_cond and batch.get("cond") is not None:
        global_cont = batch["cond"].to(device, dtype=torch.float32)

    if include_E_sum:
        if batch.get("energy_sums") is not None:
            e_sums = batch["energy_sums"].to(device, dtype=torch.float32)
            e_sums = torch.log(e_sums + 1e-3)
            if global_cont is not None:
                global_cont = torch.cat([global_cont, e_sums], dim=1)
            else:
                global_cont = e_sums
        elif global_cont is not None and global_cont.shape[1] in (10, 13, 16):
            pass

    if zero_cond_feature is not None and global_cont is not None:
        for idx in zero_cond_feature:
            global_cont[:, idx] = 0.0

    return {"X": X, "y": y, "attention_mask": mask, "global_cont": global_cont}


BERT_ENERGY_FEATURE_IDX = 3  # log(E) in [Δη, Δφ, log pT, log E]


def sort_particles_by_energy(X, mask, energy_idx=BERT_ENERGY_FEATURE_IDX):
    """Sort valid particles by descending energy; padded slots stay at the end."""
    energy = X[:, :, energy_idx].masked_fill(mask == 0, float("-inf"))
    order = torch.argsort(energy, dim=1, descending=True, stable=True)
    X = torch.gather(X, 1, order.unsqueeze(-1).expand_as(X))
    mask = torch.gather(mask, 1, order)
    return X, mask


def prepare_batch_bert(
    batch,
    device,
    use_pid=False,
    pid_idx=4,
    use_cond=False,
    include_E_sum=False,
    zero_cond_feature=None,
    energy_order=False,
):
    """Prepare batch for BertBaseline (particle tokens + optional global token + padding mask)."""
    X = batch["X"].to(device, dtype=torch.float32)
    y = batch["y"].to(device)
    mask = batch["attention_mask"].to(device, dtype=torch.float32)
    if use_pid and batch.get("pid") is not None:
        X = torch.cat([X[:, :, :pid_idx], X[:, :, pid_idx + 1 :]], dim=2)

    if energy_order:
        X, mask = sort_particles_by_energy(X, mask)

    global_cont = None
    if use_cond and batch.get("cond") is not None:
        global_cont = batch["cond"].to(device, dtype=torch.float32)

    if include_E_sum:
        if batch.get("energy_sums") is not None:
            e_sums = batch["energy_sums"].to(device, dtype=torch.float32)
            e_sums = torch.log(e_sums + 1e-3)
            if global_cont is not None:
                global_cont = torch.cat([global_cont, e_sums], dim=1)
            else:
                global_cont = e_sums
        elif global_cont is not None and global_cont.shape[1] in (10, 13, 16):
            # Cond already includes log energy sums; use as-is
            pass

    if zero_cond_feature is not None and global_cont is not None:
        for idx in zero_cond_feature:
            global_cont[:, idx] = 0.0

    return {"X": X, "y": y, "attention_mask": mask, "global_cont": global_cont}


def prepare_batch_omnilearned(
    batch, device, use_cond=False, use_pid=False, pid_idx=4, include_E_sum=False
):
    """Prepare batch for OmniLearned PET2 model input."""
    X = batch["X"].to(device, dtype=torch.float32)
    y = batch["y"].to(device)

    pid = None
    if use_pid and batch.get("pid") is not None:
        pid = batch["pid"].to(device)
        X = torch.cat([X[:, :, :pid_idx], X[:, :, pid_idx + 1 :]], dim=2)

    cond = None
    if use_cond and batch.get("cond") is not None:
        cond = batch["cond"].to(device, dtype=torch.float32)

    if include_E_sum:
        if batch.get("energy_sums") is not None:
            # Legacy: 4-col cond + raw energy_sums; concat log(e_sums+1e-3)
            e_sums = batch["energy_sums"].to(device, dtype=torch.float32)
            e_sums = torch.log(e_sums + 1e-3)
            if cond is not None:
                cond = torch.cat([cond, e_sums], dim=1)
            else:
                cond = e_sums
        elif cond is not None and cond.shape[1] in (10, 13, 16):
            # Cond already includes log energy sums (e.g. 4+6, 7+6, or 10+6); use as-is
            pass

    add_info = None
    if batch.get("add_info") is not None:
        add_info = batch["add_info"].to(device, dtype=torch.float32)

    return {"X": X, "y": y, "cond": cond, "pid": pid, "add_info": add_info}


class _FlopsWrapper(nn.Module):
    """Thin wrapper so calflops can run one forward with fixed inputs."""

    def __init__(self, model, args, inputs):
        super().__init__()
        self.model = model
        self.args = args
        self._inputs = inputs

    def forward(self, _dummy=None):
        return forward_model(self.model, self._inputs, self.args)


class _Pet2BodyFlopsWrapper(nn.Module):
    """PET2 ``body`` only (same inputs as full PET2 forward)."""

    def __init__(self, body, x, cond, pid, add_info):
        super().__init__()
        self.body = body
        self._x = x
        self._cond = cond
        self._pid = pid
        self._add_info = add_info

    def forward(self, _dummy=None):
        t = torch.zeros(self._x.shape[0], device=self._x.device, dtype=self._x.dtype)
        return self.body(self._x, self._cond, self._pid, self._add_info, t).sum()


class _Pet2ClassifierFlopsWrapper(nn.Module):
    """PET2 ``classifier`` only, on a fixed ``x_body`` tensor (shape from one body forward)."""

    def __init__(self, classifier, x_body: torch.Tensor):
        super().__init__()
        self.classifier = classifier
        self.register_buffer("_x_body", x_body.detach())

    def forward(self, _dummy=None):
        return self.classifier(self._x_body).sum()


def _make_dummy_batch(args, device):
    """Build a minimal batch dict with shape (batch_size, max_particles, ...) for FLOPs."""
    B = args.batch_size
    N = args.max_particles
    use_cond = not getattr(args, "no_use_cond", False)
    if getattr(args, "cond_only", False):
        use_cond = True
    e_sum_dim = 6 if getattr(args, "include_E_sum", True) else 0
    global_cont_dim = (GLOBAL_COND_BASE_DIM if use_cond else 0) + e_sum_dim
    point_cont_dim = getattr(args, "point_cont_dim", 9)
    coord_dim = getattr(args, "coord_dim", 2)
    pid_idx = getattr(args, "pid_idx", 4)
    use_pid = getattr(args, "use_pid", True)
    # OmniLearned / BERT expect X with last dim = ol_num_feat (4) after prepare_batch drops PID.
    # ViT expects point_cont with last dim = point_cont_dim (9) after prepare_batch drops PID.
    if getattr(args, "use_omnilearned", None) or getattr(args, "use_bert", None):
        ol_num_feat = getattr(args, "ol_num_feat", 4)
        total_feat_dim = ol_num_feat + (1 if use_pid else 0)
    else:
        # ViT, CondOnly, and HyperScale all see the full point_cont + optional PID layout.
        total_feat_dim = point_cont_dim + (1 if use_pid else 0)
    X = torch.zeros(B, N, total_feat_dim, device=device, dtype=torch.float32)
    if args.mode == "regression":
        y = torch.zeros(B, device=device, dtype=torch.float32)
    else:
        y = torch.zeros(B, device=device, dtype=torch.long)
    attention_mask = torch.ones(B, N, device=device, dtype=torch.float32)
    batch = {"X": X, "y": y, "attention_mask": attention_mask}
    if use_cond and global_cont_dim > 0:
        batch["cond"] = torch.zeros(
            B, GLOBAL_COND_BASE_DIM, device=device, dtype=torch.float32
        )
    if e_sum_dim > 0:
        batch["energy_sums"] = torch.ones(B, 6, device=device, dtype=torch.float32)
    if getattr(args, "use_omnilearned", None):
        batch["pid"] = torch.zeros(B, N, device=device, dtype=torch.long)
        batch["add_info"] = torch.zeros(
            B, N, getattr(args, "ol_num_add", 5), device=device, dtype=torch.float32
        )
        # PET_body uses mask = (X[:, :, 2:3] != 0). All-zero X marks every token as padding;
        # match real events by flagging all slots as present (third feature ≠ 0).
        X[:, :, 2:3] = 1.0
    elif getattr(args, "use_bert", None) and use_pid:
        batch["pid"] = torch.zeros(B, N, device=device, dtype=torch.long)
    elif getattr(args, "use_hyperscale", None) and use_pid:
        batch["pid"] = torch.zeros(B, N, device=device, dtype=torch.long)
    return batch


def run_calculate_flops(args):
    """Use calflops to compute inference FLOPs per batch and approx training FLOPs, then exit."""
    try:
        from calflops import calculate_flops
    except ImportError:
        raise SystemExit(
            "calflops is required for --calculate-flops. Install with: pip install calflops"
        )

    args.use_cond = not args.no_use_cond
    if args.cond_only:
        args.use_cond = True
    _maybe_autofill_hyperscale_args(args)
    device = torch.device("cpu")
    task = create_task(args)
    if args.use_omnilearned:
        model = create_omnilearned_model(args, task)
    elif getattr(args, "use_bert", None):
        model = create_bert_model(args, task)
    elif getattr(args, "use_hyperscale", None):
        model = create_hyperscale_model(args, task)
    else:
        model = create_model(args, task)
    model = model.to(device)
    model.eval()

    # Weights do not change FLOPs; skip checkpoint I/O here. Apply same freeze as training for medium.
    _apply_omnilearned_medium_backbone_freeze(model, args)

    dummy_batch = _make_dummy_batch(args, device)
    if args.use_omnilearned:
        inputs = prepare_batch_omnilearned(
            dummy_batch,
            device,
            args.use_cond,
            args.use_pid,
            args.pid_idx,
            include_E_sum=args.include_E_sum,
        )
    elif getattr(args, "use_bert", None):
        inputs = prepare_batch_bert(
            dummy_batch,
            device,
            args.use_pid,
            args.pid_idx,
            use_cond=args.use_cond,
            include_E_sum=args.include_E_sum,
            zero_cond_feature=args.zero_cond_feature,
            energy_order=args.bert_energy_order,
        )
    elif getattr(args, "use_hyperscale", None):
        inputs = prepare_batch_hyperscale(
            dummy_batch,
            device,
            args.use_pid,
            args.pid_idx,
            use_cond=args.use_cond,
            include_E_sum=args.include_E_sum,
            zero_cond_feature=args.zero_cond_feature,
            variant=args.use_hyperscale,
        )
    else:
        inputs = prepare_batch(
            dummy_batch,
            device,
            args.use_cond,
            args.use_pid,
            args.coord_dim,
            args.pid_idx,
            include_E_sum=args.include_E_sum,
            zero_cond_feature=args.zero_cond_feature,
        )

    wrapper = _FlopsWrapper(model, args, inputs)
    wrapper.eval()
    # One dummy input so calflops runs wrapper.forward(dummy); FLOPs come from the real model inside
    flops, macs, params = calculate_flops(
        model=wrapper,
        input_shape=(1,),
        output_as_string=False,
    )
    # calflops: "FLOPs" here counts multiply/add work in a theoretical graph (often FLOPs ≈ 2 × MACs).
    # nn.MultiheadAttention is usually counted like dense matmuls; real CUDA kernels (e.g. SDPA
    # backends) can do far less work, so these numbers are not hardware FLOPs and are not "exact".
    inference_flops = flops
    training_flops_approx = flops * 3
    Bsz = args.batch_size
    print(f"Batch size: {Bsz}, max_particles: {args.max_particles}")
    print(f"Inference FLOPs per batch (full model, calflops): {inference_flops:,}")
    print(f"  MACs per batch (calflops): {macs:,}  (~FLOPs/2 if counted as 2 FLOPs per MAC)")
    print(f"  Inference FLOPs per sample (÷ batch): {inference_flops / Bsz:,.0f}")
    print(f"Training FLOPs per batch (heuristic, full model ×3): {training_flops_approx:,}")
    print(f"Params (total): {params:,}")

    if args.use_omnilearned and isinstance(model, PET2) and model.classifier is not None:
        x = inputs["X"]
        t0 = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        body_wrap = _Pet2BodyFlopsWrapper(
            model.body, x, inputs.get("cond"), inputs.get("pid"), inputs.get("add_info")
        ).eval()
        body_flops, body_macs, body_params = calculate_flops(
            model=body_wrap, input_shape=(1,), output_as_string=False
        )
        with torch.no_grad():
            x_body = model.body(
                x, inputs.get("cond"), inputs.get("pid"), inputs.get("add_info"), t0
            )
        clf_wrap = _Pet2ClassifierFlopsWrapper(model.classifier, x_body).eval()
        clf_flops, clf_macs, clf_params = calculate_flops(
            model=clf_wrap, input_shape=(1,), output_as_string=False
        )
        n_body = sum(p.numel() for p in model.body.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_frozen = sum(
            p.numel() for p in model.body.parameters() if not p.requires_grad
        )
        split_sum = body_flops + clf_flops
        rel = abs(split_sum - inference_flops) / max(inference_flops, 1.0)
        print("")
        print("PET2 split (inference, per batch, calflops):")
        print(f"  body (backbone) FLOPs:     {body_flops:,}  MACs: {body_macs:,}")
        print(f"  classifier (head) FLOPs: {clf_flops:,}  MACs: {clf_macs:,}")
        print(f"  sum(body+head) FLOPs:      {split_sum:,}")
        print(
            f"  vs full-model forward above: rel. diff {100.0 * rel:.2f}% "
            "(large mismatch ⇒ counter quirks, not a guarantee body+head = traced full forward)"
        )
        print(f"  body params: {n_body:,}; trainable params: {n_train:,}; frozen body params: {n_frozen:,}")
        if args.use_omnilearned == "medium" and n_frozen > 0:
            # Heuristic: frozen body ⇒ no backward through backbone weights; backward cost is
            # dominated by trainable head (~2× head forward is a common rule of thumb, not exact).
            train_step_frozen_approx = body_flops + 3 * clf_flops
            print("")
            print(
                "OmniLearned-medium + frozen backbone — heuristic training work per batch "
                "(NOT measured; NOT equal to GPU FLOPs):"
            )
            print(f"  forward (body + head):     {split_sum:,}")
            print(
                f"  + backward (rough ~2× head only): 2 × {clf_flops:,} = {2 * clf_flops:,}"
            )
            print(
                f"  rule-of-thumb total:     {train_step_frozen_approx:,} (= body_fwd + 3× head_fwd)"
            )
            print(
                "  Expect this to stay large: the backbone still runs a full forward each step; "
                "only weight-update / backward through the body is skipped."
            )

    raise SystemExit(0)


def forward_model(model, inputs, args):
    """Run forward pass for ViT, PET2, BertBaseline, or HyperScaleBaseline; returns logits."""
    if getattr(args, "use_bert", None):
        return model(
            inputs["X"], inputs["attention_mask"], global_cont=inputs.get("global_cont")
        )
    if getattr(args, "use_hyperscale", None):
        return model(
            inputs["X"], inputs["attention_mask"], global_cont=inputs.get("global_cont")
        )
    if args.use_omnilearned:
        outputs = model(
            inputs["X"],
            inputs["y"],
            cond=inputs["cond"],
            pid=inputs["pid"],
            add_info=inputs["add_info"],
        )
        return outputs["y_pred"]
    elif args.cond_only:
        return model(inputs["global_cont"])
    else:
        features = model(
            point_cont=inputs["point_cont"],
            point_cats=inputs["point_cats"],
            pos=inputs["pos"],
            global_cont=inputs["global_cont"],
            global_cats=inputs["global_cats"],
            attn_mask=inputs["attn_mask"],
        )
        return model.head(features)


def prepare_batch(
    batch,
    device,
    use_cond=False,
    use_pid=False,
    coord_dim=2,
    pid_idx=4,
    include_E_sum=False,
    zero_cond_feature=None,
):
    """Prepare batch for model input."""
    X = batch["X"].to(device, dtype=torch.float32)
    y = batch["y"].to(device)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32)

    # Split features into coordinates and continuous features
    pos = X[:, :, :coord_dim]

    # Handle PID (categorical)
    point_cats = None
    if use_pid:
        point_cats = [X[:, :, pid_idx].long()]
        # concat up to pid idx and after pid idx
        # point_cont = X[:, :, :pid_idx]
        point_cont = torch.cat([X[:, :, :pid_idx], X[:, :, pid_idx + 1 :]], dim=2)
    else:
        point_cont = X  # use coord dim too as just normal features

    # Handle global features
    global_cont = None
    global_cats = None
    if use_cond and batch.get("cond") is not None:
        global_cont = batch["cond"].to(
            device, dtype=torch.float32
        )  # [B, global_cont_dim]

    if include_E_sum:
        if batch.get("energy_sums") is not None:
            # Legacy: 4-col cond + raw energy_sums; concat log(e_sums+1e-3)
            e_sums = batch["energy_sums"].to(device, dtype=torch.float32)
            e_sums = torch.log(e_sums + 1e-3)  # [B, 6]
            if global_cont is not None:
                global_cont = torch.cat([global_cont, e_sums], dim=1)
            else:
                global_cont = e_sums
        elif global_cont is not None and global_cont.shape[1] in (10, 13, 16):
            # Cond already includes log energy sums; use as-is
            pass

    if zero_cond_feature is not None and global_cont is not None:
        for idx in zero_cond_feature:
            global_cont[:, idx] = 0.0

    # Expand attention mask to account for special tokens
    # Model always adds CLS token, and adds EVT token if use_cond=True
    B = X.shape[0]
    use_event_token = use_cond or include_E_sum
    num_special_tokens = 1  # CLS token
    if use_event_token:
        num_special_tokens += 1  # EVT token

    # Prepend ones for special tokens (they are always "valid")
    special_token_mask = torch.ones(
        B, num_special_tokens, device=device, dtype=torch.float32
    )
    attention_mask = torch.cat(
        [special_token_mask, attention_mask], dim=1
    )  # [B, num_special + N]

    # Convert to boolean key padding mask for scaled_dot_product_attention.
    # For boolean SDPA masks, True means this key position is allowed to attend.
    # Our dataset mask uses 1=valid, 0=padding, so keep valid tokens as True.
    attention_mask = (
        (attention_mask > 0).unsqueeze(1).unsqueeze(2)
    )  # [B, 1, 1, seq_len]

    return {
        "point_cont": point_cont,
        "point_cats": point_cats,
        "pos": pos,
        "global_cont": global_cont,
        "global_cats": global_cats,
        "attn_mask": attention_mask,
        "y": y,
    }


def make_log1p_loss(criterion):
    # Transform target to log1p(target+1)
    def loss(pred, target, step=0):
        target = torch.log1p(target)
        return criterion(pred, target).mean()

    return loss


def compute_weighted_regression_loss(preds, targets, max_weight=10.0, min_weight=0.1):
    """Compute per-sample weighted Huber loss with 1/E weights."""
    targets = targets.to(dtype=preds.dtype)
    per_sample_loss = F.huber_loss(preds, targets, reduction="none")
    safe_targets = torch.where(targets > 0, targets, torch.ones_like(targets))
    weights = torch.where(
        targets > 0,
        1.0 / safe_targets,
        torch.full_like(targets, max_weight),
    )
    weights = torch.clamp(weights, max=max_weight, min=min_weight)
    return (per_sample_loss * weights).mean()


def compute_weighted_classification_loss(logits, targets, sample_weights):
    """Per-sample weighted cross-entropy (kinematic-binned class weights)."""
    per_sample_loss = F.cross_entropy(logits, targets, reduction="none")
    return (per_sample_loss * sample_weights.to(per_sample_loss.dtype)).mean()


def _attach_loss_weight(inputs, batch, device):
    """Copy per-sample loss weights from the collated batch into *inputs*."""
    lw = batch.get("loss_weight")
    if lw is not None:
        inputs["loss_weight"] = lw.to(device, dtype=torch.float32)
    return inputs


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    args,
    class_weights,
    use_amp=False,
    step=0,
    use_binned_loss=False,
):
    """Run evaluation on validation set."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    # Setup loss function
    if args.mode == "regression":
        if args.log_MSE_loss:
            criterion = make_log_loss(nn.MSELoss(reduction="none"))
        elif args.log1p_loss:
            criterion = make_log1p_loss(nn.HuberLoss(reduction="none"))
        else:
            criterion = nn.HuberLoss()
    elif use_binned_loss:
        criterion = None
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        if args.use_omnilearned:
            inputs = prepare_batch_omnilearned(
                batch,
                device,
                args.use_cond,
                args.use_pid,
                args.pid_idx,
                include_E_sum=args.include_E_sum,
            )
        elif getattr(args, "use_bert", None):
            inputs = prepare_batch_bert(
                batch,
                device,
                args.use_pid,
                args.pid_idx,
                use_cond=args.use_cond,
                include_E_sum=args.include_E_sum,
                zero_cond_feature=args.zero_cond_feature,
                energy_order=args.bert_energy_order,
            )
        elif getattr(args, "use_hyperscale", None):
            inputs = prepare_batch_hyperscale(
                batch,
                device,
                args.use_pid,
                args.pid_idx,
                use_cond=args.use_cond,
                include_E_sum=args.include_E_sum,
                zero_cond_feature=args.zero_cond_feature,
                variant=args.use_hyperscale,
            )
        else:
            inputs = prepare_batch(
                batch,
                device,
                args.use_cond,
                args.use_pid,
                args.coord_dim,
                args.pid_idx,
                include_E_sum=args.include_E_sum,
                zero_cond_feature=args.zero_cond_feature,
            )
        if use_binned_loss:
            _attach_loss_weight(inputs, batch, device)
        with autocast(enabled=use_amp):
            logits = forward_model(model, inputs, args)
            if args.mode == "regression":
                if args.weighted_regression_loss:
                    loss = compute_weighted_regression_loss(
                        logits.squeeze(-1), inputs["y"]
                    )
                else:
                    loss = criterion(logits.squeeze(-1), inputs["y"])
            elif use_binned_loss:
                loss = compute_weighted_classification_loss(
                    logits, inputs["y"], inputs["loss_weight"]
                )
            else:
                loss = criterion(logits, inputs["y"])

        total_loss += loss.item() * inputs["y"].size(0)
        total_samples += inputs["y"].size(0)

    avg_loss = total_loss / total_samples
    model.train()
    return {"eval_loss": avg_loss}


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    args,
    best_val_loss,
    filename="checkpoint.pt",
):
    """Save training checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "args": vars(args),
        "best_val_loss": best_val_loss,
        "wandb_run_id": wandb.run.id if wandb.run else None,
    }
    save_path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")
    return save_path


def load_checkpoint(
    checkpoint_path, model, optimizer=None, scheduler=None, scaler=None
):
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if (
        scaler is not None
        and "scaler_state_dict" in checkpoint
        and checkpoint["scaler_state_dict"] is not None
    ):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_step = checkpoint.get("step", 0)
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from step {start_step}")
    return start_step


# Make a log loss out of a criterion: the loss is criterion(log(pred+eps) - log(target+eps))


def make_log_loss(criterion):
    eps = 1e-6

    def loss(pred, target):
        pred_mask = (pred >= 0).float()
        return criterion(
            torch.log(pred_mask * pred + eps), torch.log(target + eps)
        ).mean()

    return loss


# region agent log
_AGENT_DEBUG_LOG_PATH = "/global/homes/g/gregork/.cursor/debug-781e93.log"


def _agent_debug_log(hypothesis_id, location, message, data, run_id="pre-fix"):
    try:
        payload = {
            "sessionId": "781e93",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def _body_l1_fp(model):
    if not hasattr(model, "body"):
        return None
    with torch.no_grad():
        return float(sum(p.float().abs().sum().item() for p in model.body.parameters()))


def _optimizer_trainable_alignment(optimizer, model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = len(trainable)
    st = optimizer.state_dict().get("state", {})
    n_state = len(st)
    mism = n_state != n_train
    return {
        "n_trainable_tensors": n_train,
        "n_optimizer_state_entries": n_state,
        "mismatch": mism,
    }


# endregion


def train(args):
    """Main training function."""
    # If resuming, load saved arguments from the checkpoint and override current ones,
    # so that training continues with the exact same configuration as the original run.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved_args = checkpoint.get("args", None)
        if saved_args is not None:
            # Preserve explicitly provided CLI controls like --resume, --no_wandb, --resume-run-id, --max_steps.
            preserve_keys = {"resume", "no_wandb", "resume_run_id", "max_steps"}
            for k, v in saved_args.items():
                if k in preserve_keys:
                    continue
                setattr(args, k, v)
        # Use wandb run id from checkpoint for resuming logging (unless --resume-run-id is set).
        args.wandb_run_id = checkpoint.get("wandb_run_id")
        print(f"Loaded training configuration from checkpoint: {args.resume}")

    # Determine run name and output directory
    if args.resume:
        # When resuming, derive run name from the directory containing the checkpoint.
        # This directory name is treated as the wandb run name and the output directory.
        checkpoint_dir = os.path.dirname(args.resume)
        run_name_with_timestamp = os.path.basename(checkpoint_dir)
        args.output_dir = checkpoint_dir
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # Create a new timestamped run name and corresponding output directory.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name_with_timestamp = f"{args.run_name}_{timestamp}"
        args.output_dir = os.path.join(args.output_dir, run_name_with_timestamp)
        os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Run name: {run_name_with_timestamp}")
    args.use_cond = not args.no_use_cond
    if args.cond_only:
        args.use_cond = True
    _maybe_autofill_hyperscale_args(args)
    if getattr(args, "use_bert", None):
        if args.use_omnilearned:
            raise ValueError("Cannot use --use-bert together with --use-omnilearned")
        if args.cond_only:
            raise ValueError("Cannot use --use-bert together with --cond-only")
        if getattr(args, "use_hyperscale", None):
            raise ValueError("Cannot use --use-bert together with --use-hyperscale")
    elif getattr(args, "use_hyperscale", None):
        if args.use_omnilearned:
            raise ValueError(
                "Cannot use --use-hyperscale together with --use-omnilearned"
            )
        if args.cond_only:
            raise ValueError("Cannot use --use-hyperscale together with --cond-only")
    elif getattr(args, "use_cls_token", False):
        raise ValueError("--use-cls-token requires --use-bert")
    elif getattr(args, "bert_random_init", False):
        raise ValueError("--bert-random-init requires --use-bert")
    elif getattr(args, "bert_energy_order", False):
        raise ValueError("--bert-energy-order requires --use-bert")
    if getattr(args, "binary_classifier", False):
        if args.mode != "classifier":
            raise ValueError("--binary-classifier requires --mode classifier")
        if not args.classification_CC1orNPi:
            raise ValueError("--binary-classifier requires -npi2 (--classification_CC1orNPi)")
    if getattr(args, "predict_baseline", False):
        if args.mode != "classifier":
            raise ValueError("--predict-baseline requires --mode classifier")
        if not args.classification_CC1orNPi:
            raise ValueError("--predict-baseline requires -npi2 (--classification_CC1orNPi)")
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    binned_var = getattr(args, "binned_loss_var", None)
    binned_signal = getattr(args, "binned_loss_signal", None)
    if (binned_var is None) ^ (binned_signal is None):
        raise ValueError(
            "Both --binned-loss-var and --binned-loss-signal must be set together, "
            "or neither."
        )
    if binned_var is not None and args.mode != "classifier":
        raise ValueError("Binned loss weighting requires --mode classifier.")
    if binned_signal is not None:
        from src.eval.classification_plots._signal_definitions import (
            resolve_signal_classes,
        )

        resolve_signal_classes(binned_signal)

    # Create dataloaders
    print("Creating dataloaders...")
    task = create_task(args)
    # HyperScale uses the ViT-style packed layout (point_cont_dim + PID per particle);
    # only OmniLearned/BERT split add_info out into a separate tensor.
    concat_additional_info = not (
        bool(args.use_omnilearned) or bool(getattr(args, "use_bert", None))
    )
    event_type_codes = parse_event_types(args.event_types) or None
    if event_type_codes:
        print(
            f"Restricting train+val to GENIE interaction types "
            f"{args.event_types} -> codes {event_type_codes}"
        )
    train_loader, class_weights, use_binned_loss = load_data(
        dataset_name=args.dataset_name,
        path=args.data_path,
        batch=args.batch_size,
        dataset_type="train",
        task=task,
        use_cond=args.use_cond,
        use_pid=args.use_pid,
        pid_idx=args.pid_idx,
        distributed=True,
        shuffle=True,
        max_particles=args.max_particles,
        num_workers=args.num_workers,
        rank=0,
        size=1,
        concat_additional_info=concat_additional_info,
        event_sampler_random_state=args.event_sampler_random_state,
        nevts=args.event_cap,
        use_energy_sums=args.include_E_sum,
        event_types=event_type_codes,
        binned_loss_var=binned_var,
        binned_loss_signal=binned_signal,
    )

    val_loader, _, _ = load_data(
        dataset_name=args.dataset_name,
        path=args.data_path,
        batch=args.batch_size,
        dataset_type="val",
        task=task,
        use_cond=args.use_cond,
        use_pid=args.use_pid,
        pid_idx=args.pid_idx,
        num_workers=args.num_workers,
        rank=0,
        size=1,
        distributed=True,
        shuffle=False,
        max_particles=args.max_particles,
        concat_additional_info=concat_additional_info,
        use_energy_sums=args.include_E_sum,
        event_types=event_type_codes,
        binned_loss_var=binned_var,
        binned_loss_signal=binned_signal,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("Creating model...")
    if args.use_omnilearned:
        model = create_omnilearned_model(args, task)
    elif getattr(args, "use_bert", None):
        model = create_bert_model(args, task)
    elif getattr(args, "use_hyperscale", None):
        model = create_hyperscale_model(args, task)
    else:
        model = create_model(args, task)
    model = model.to(device)

    # Load pretrained OmniLearned weights (before optimizer setup)
    if args.use_pretrained:
        if not args.use_omnilearned:
            raise ValueError("--use-pretrained requires --use-omnilearned")
        print(f"Loading pretrained weights: {args.use_pretrained}")
        load_pretrained_omnilearned(model, args.use_pretrained, args.output_dir)

    # Load pretrained HyperScale weights (before optimizer setup)
    if getattr(args, "hs_pretrained", None):
        if not getattr(args, "use_hyperscale", None):
            raise ValueError("--hs-pretrained requires --use-hyperscale")
        load_pretrained_hyperscale(model, args.hs_pretrained)

    _apply_omnilearned_medium_backbone_freeze(model, args)

    # region agent log
    _agent_debug_log(
        "H1",
        "train.py:post_freeze",
        "body fingerprint after pretrained+freeze (before optimizer/resume)",
        {
            "body_l1": _body_l1_fp(model),
            "use_pretrained": bool(args.use_pretrained),
            "resume": bool(args.resume),
            "omnilearned_size": getattr(args, "use_omnilearned", None),
        },
    )
    # endregion

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    task.class_weights = class_weights
    # Setup loss function
    if task.type == "regression":
        if args.log_MSE_loss:
            criterion = nn.MSELoss(reduction="none")
        elif args.log1p_loss:
            criterion = make_log1p_loss(nn.HuberLoss(reduction="none"))
        else:
            criterion = nn.HuberLoss()
    elif use_binned_loss:
        criterion = None
    else:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(task.class_weights, device=device, dtype=torch.float32)
        )

    steps_per_epoch = len(train_loader)

    # Setup optimizer and scheduler (trainable params only; e.g. frozen OmniLearned medium backbone)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "lion":
        optimizer = Lion(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.95, 0.98),
        )
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    scheduler = get_lr_schedule(optimizer, args.warmup_steps, max_steps=args.max_steps)

    # Setup AMP
    scaler = GradScaler() if args.use_amp else None

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)

    # region agent log
    opt_lr0 = optimizer.param_groups[0]["lr"] if optimizer.param_groups else None
    sched_last = getattr(scheduler, "last_epoch", None)
    sched_lr = scheduler.get_last_lr()[0] if scheduler is not None else None
    align = _optimizer_trainable_alignment(optimizer, model)
    scaler_scale = float(scaler.get_scale()) if scaler is not None else None
    _agent_debug_log(
        "H2",
        "train.py:post_resume_or_init",
        "scheduler vs optimizer lr; step alignment after resume",
        {
            "start_step": start_step,
            "scheduler_last_epoch": sched_last,
            "optimizer_param_group_lr0": opt_lr0,
            "scheduler_get_last_lr0": sched_lr,
            "lr_mismatch": (
                opt_lr0 is not None
                and sched_lr is not None
                and abs(float(opt_lr0) - float(sched_lr)) > 1e-12
            ),
            "warmup_steps": args.warmup_steps,
            "max_steps": args.max_steps,
        },
    )
    _agent_debug_log(
        "H3",
        "train.py:post_resume_or_init",
        "optimizer state vs trainable tensors",
        align,
    )
    _agent_debug_log(
        "H3",
        "train.py:post_resume_or_init",
        "body fingerprint after checkpoint load (if any)",
        {"body_l1": _body_l1_fp(model), "resume": bool(args.resume)},
    )
    _agent_debug_log(
        "H4",
        "train.py:post_resume_or_init",
        "AMP scaler after resume",
        {"scaler_scale": scaler_scale, "use_amp": bool(args.use_amp)},
    )
    # endregion

    # Initialize wandb
    if not args.no_wandb:
        wandb.login()
        init_kw = {
            "project": args.wandb_project,
            "name": run_name_with_timestamp,
            "config": vars(args),
        }
        if args.resume:
            wandb_id = args.resume_run_id or getattr(args, "wandb_run_id", None)
            if wandb_id:
                init_kw["id"] = wandb_id
                init_kw["resume"] = "allow"
            else:
                init_kw["resume"] = "allow"
        wandb.init(**init_kw)
        wandb.watch(model, log="all", log_freq=args.log_interval)

    # Training loop
    print("Starting training...")
    model.train()

    step = start_step  # counts optimizer steps (after grad accumulation)
    accum_counter = 0  # counts micro-batches since last optimizer step
    train_losses = []
    data_fetch_times = []
    backprop_times = []
    best_val_loss = float("inf")

    # --- Cond-only diagnostic: inspect first batch ---
    if args.cond_only:
        diag_batch = next(iter(train_loader))
        diag_inputs = prepare_batch(
            diag_batch,
            device,
            args.use_cond,
            args.use_pid,
            args.coord_dim,
            args.pid_idx,
            include_E_sum=args.include_E_sum,
            zero_cond_feature=args.zero_cond_feature,
        )
        gc = diag_inputs["global_cont"]
        y = diag_inputs["y"]
        print("=" * 60)
        print("COND-ONLY DIAGNOSTIC (first batch)")
        print("=" * 60)
        if gc is None:
            print("  *** WARNING: global_cont is None! Cond features not loaded. ***")
        else:
            print(f"  global_cont shape: {gc.shape}  dtype: {gc.dtype}")
            print(f"  global_cont stats per feature (min / mean / max / std):")
            for fi in range(gc.shape[1]):
                col = gc[:, fi]
                print(
                    f"    feat[{fi}]: {col.min().item():+.4f} / {col.mean().item():+.4f} / {col.max().item():+.4f} / {col.std().item():.4f}"
                )
            print(f"  target (y) shape: {y.shape}  dtype: {y.dtype}")
            if args.mode == "regression":
                print(
                    f"  target stats: min={y.min().item():.4f}  mean={y.mean().item():.4f}  max={y.max().item():.4f}  std={y.std().item():.4f}"
                )
                if args.log1p_loss:
                    y_log = torch.log1p(y)
                    print(
                        f"  log1p(target) stats: min={y_log.min().item():.4f}  mean={y_log.mean().item():.4f}  max={y_log.max().item():.4f}  std={y_log.std().item():.4f}"
                    )
            else:
                print(
                    f"  class labels (int): min={y.min().item()}  max={y.max().item()}"
                )
                for c in torch.sort(y.unique())[0]:
                    print(
                        f"    class {c.item()}: count {(y == c).sum().item()} (this batch)"
                    )
            with torch.no_grad():
                pred0 = model(gc)
                print(f"  model output (untrained) shape: {pred0.shape}")
                print(
                    f"  model output stats: min={pred0.min().item():.4f}  mean={pred0.mean().item():.4f}  max={pred0.max().item():.4f}  std={pred0.std().item():.4f}"
                )
            if args.mode == "regression":
                corr_matrix = torch.corrcoef(torch.cat([gc.T, y.unsqueeze(0)], dim=0))
                print(f"  Pearson correlation of each cond feature with target:")
                for fi in range(gc.shape[1]):
                    print(f"    feat[{fi}] <-> y: {corr_matrix[fi, -1].item():+.4f}")
            else:
                print(
                    "  (Pearson correlation skipped for classification — labels are discrete.)"
                )
        print("=" * 60)
        del diag_batch, diag_inputs, gc, y

    print(f"Steps per epoch: {steps_per_epoch}")
    print(
        f"Training for {args.max_steps} optimizer steps (grad_accum_steps={args.grad_accum_steps})"
    )
    epoch = 0
    done = False

    optimizer.zero_grad()
    while not done:
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
        iter_end_time = time.perf_counter()

        for batch in epoch_pbar:
            data_fetch_times.append(time.perf_counter() - iter_end_time)

            # Prepare inputs
            if args.use_omnilearned:
                inputs = prepare_batch_omnilearned(
                    batch,
                    device,
                    args.use_cond,
                    args.use_pid,
                    args.pid_idx,
                    include_E_sum=args.include_E_sum,
                )
            elif getattr(args, "use_bert", None):
                inputs = prepare_batch_bert(
                    batch,
                    device,
                    args.use_pid,
                    args.pid_idx,
                    use_cond=args.use_cond,
                    include_E_sum=args.include_E_sum,
                    zero_cond_feature=args.zero_cond_feature,
                    energy_order=args.bert_energy_order,
                )
            elif getattr(args, "use_hyperscale", None):
                inputs = prepare_batch_hyperscale(
                    batch,
                    device,
                    args.use_pid,
                    args.pid_idx,
                    use_cond=args.use_cond,
                    include_E_sum=args.include_E_sum,
                    zero_cond_feature=args.zero_cond_feature,
                    variant=args.use_hyperscale,
                )
            else:
                inputs = prepare_batch(
                    batch,
                    device,
                    args.use_cond,
                    args.use_pid,
                    args.coord_dim,
                    args.pid_idx,
                    include_E_sum=args.include_E_sum,
                    zero_cond_feature=args.zero_cond_feature,
                )

            if use_binned_loss:
                _attach_loss_weight(inputs, batch, device)

            # Forward pass
            with autocast(enabled=args.use_amp):
                logits = forward_model(model, inputs, args)

                if args.mode == "regression":
                    if args.weighted_regression_loss:
                        loss = compute_weighted_regression_loss(
                            logits.squeeze(-1), inputs["y"]
                        )
                    else:
                        loss = criterion(logits.squeeze(-1), inputs["y"])
                elif use_binned_loss:
                    loss = compute_weighted_classification_loss(
                        logits, inputs["y"], inputs["loss_weight"]
                    )
                else:
                    loss = criterion(logits, inputs["y"])

            # Normalize loss for gradient accumulation
            loss = loss / args.grad_accum_steps

            # Backward pass
            backprop_start_time = time.perf_counter()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_counter += 1

            # Only step optimizer after grad_accum_steps micro-batches
            performed_step = False
            if accum_counter % args.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                step += 1
                performed_step = True
                # region agent log
                if start_step > 0 and step == start_step + 1:
                    opt_lr_after = (
                        optimizer.param_groups[0]["lr"]
                        if optimizer.param_groups
                        else None
                    )
                    sched_lr_after = (
                        scheduler.get_last_lr()[0] if scheduler is not None else None
                    )
                    _agent_debug_log(
                        "H5",
                        "train.py:first_step_after_resume",
                        "first optimizer step after resume",
                        {
                            "step": step,
                            "train_loss_unscaled": float(
                                loss.item() * args.grad_accum_steps
                            ),
                            "lr_optimizer_pg0": opt_lr_after,
                            "lr_scheduler_get_last": sched_lr_after,
                            "scheduler_last_epoch_after_step": getattr(
                                scheduler, "last_epoch", None
                            ),
                        },
                    )
                # endregion
                # Print 5 example y_pred vs y_true for cond_only regression every 100 steps
                if args.cond_only and args.mode == "regression" and step % 100 == 0:
                    pred = logits.squeeze(-1).detach().cpu()
                    y_true = inputs["y"].detach().cpu()
                    n = min(5, pred.size(0))
                    print(
                        f"[step {step}] cond_only regression — 5 example y_pred vs y_true:"
                    )
                    for i in range(n):
                        print(
                            f"  [{i}] y_pred={pred[i].item():.6f}  y_true={y_true[i].item():.6f}"
                        )

            backprop_times.append(time.perf_counter() - backprop_start_time)

            # Track loss
            train_losses.append(loss.item() * args.grad_accum_steps)

            # Update progress bar with current metrics
            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": f"{step}/{args.max_steps}",
                }
            )

            # Log training loss (on optimizer steps)
            if performed_step and step % args.log_interval == 0:
                avg_train_loss = np.mean(train_losses)
                log_dict = {
                    "train_loss": avg_train_loss,
                    "data_fetch_time_s": float(np.mean(data_fetch_times)),
                    "backprop_time_s": float(np.mean(backprop_times)),
                    "learning_rate": current_lr,
                    "step": step,
                    "epoch": epoch,
                }

                if not args.no_wandb:
                    wandb.log(log_dict, step=step)

                train_losses = []
                data_fetch_times = []
                backprop_times = []
            iter_end_time = time.perf_counter()

            # Evaluation
            if performed_step and (step % args.eval_interval == 0 or step == 1):
                epoch_pbar.write(f"\nRunning evaluation at step {step}...")
                eval_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    args,
                    (
                        torch.tensor(
                            task.class_weights, device=device, dtype=torch.float32
                        )
                        if task.type == "classifier"
                        else None
                    ),
                    args.use_amp,
                    step,
                    use_binned_loss=use_binned_loss,
                )

                epoch_pbar.write(f"Eval loss: {eval_metrics['eval_loss']:.4f}")
                if eval_metrics["eval_loss"] < best_val_loss:
                    best_val_loss = eval_metrics["eval_loss"]
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        args,
                        best_val_loss,
                        filename="best_model.pt",
                    )
                    epoch_pbar.write(
                        f"New best model saved! Val loss: {best_val_loss:.4f}"
                    )

                if not args.no_wandb:
                    wandb.log(eval_metrics, step=step)
                    wandb.log({"best_val_loss": best_val_loss}, step=step)

            if step >= args.max_steps:
                done = True
                break

        epoch += 1
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_metrics = evaluate(
        model,
        val_loader,
        device,
        args,
        (
            torch.tensor(task.class_weights, device=device, dtype=torch.float32)
            if task.type == "classifier"
            else None
        ),
        args.use_amp,
        step,
        use_binned_loss=use_binned_loss,
    )
    print(f"Final eval loss: {eval_metrics['eval_loss']:.4f}")

    if not args.no_wandb:
        wandb.log(eval_metrics, step=step)
    print("Training complete!")

    if not args.no_wandb:
        wandb.finish()


def main():
    args = parse_args()
    if args.calculate_flops:
        run_calculate_flops(args)
        return
    if args.run_name is None:
        raise SystemExit(
            "--run_name / -name is required unless --calculate-flops is set"
        )
    train(args)


if __name__ == "__main__":
    main()
