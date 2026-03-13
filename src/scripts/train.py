"""
Training script for PointGlobalMixedViT or OmniLearned on HEP data.

# Energy regression training:
## ViT-like transformer training:
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG --d_model 128 --depth 4 --n_heads 8  --max_steps 250000 [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]


## OmniLearned Small training:
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG  --max_steps 250000 --use-omnilearned small --use-pretrained pretrain_s [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]


## OmniLearned Small random weights training:
python -m src.scripts.train -bs 10 --mode regression -E-available-no-muon -name DEBUG  --max_steps 250000 --use-omnilearned small  [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]



# Event Classification training:
## ViT-like transformer training:
python -m src.scripts.train -bs 10 --mode classifier -npi2 -name DEB    UG --d_model 64 --depth 4 --n_heads 4  --max_steps 100000 [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

## OmniLearned Small training:
python -m src.scripts.train -bs 10 --mode classifier -npi2 -name DEBUG --max_steps 100000 --use-omnilearned small --use-pretrained pretrain_s [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

## OmniLearned Small random weights training:
python -m src.scripts.train -bs 10 --mode classifier -npi2 -name DEBUG --max_steps 100000 --use-omnilearned small  [-cap {data_cap} -seed-event-sampler {seed} --seed {seed}]

"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pytorch_optimizer import Lion
import wandb
from tqdm import tqdm
import numpy as np

from src.dataset.dataloader import load_data, Task
from src.models.vit import PointGlobalMixedViT, PointGlobalMixedViTConfig
from src.models.omnilearned import PET2, get_model_parameters, load_pretrained_omnilearned

# print CUDA_VISIBLE_DEVICES
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class CondOnlyMLP(nn.Module):
    """MLP with residual blocks that operates only on global/conditional features."""
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.head(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointGlobalMixedViT on HEP data")
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="minerva_1A", 
                        help="Dataset name (e.g., minerva_1A)")
    parser.add_argument("--data_path", type=str,
                        help="Path to dataset directory",
                        default="/global/cfs/cdirs/m3246/gregork/Minerva/20260311")
    parser.add_argument("--batch_size", "-bs", type=int, default=2048,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--max_particles", type=int, default=33,
                        help="Maximum number of particles per event")
    # Model arguments
    parser.add_argument("--mode", type=str, default="regression", 
                        choices=["regression", "classifier"],
                        help="Training mode: regression or classifier")
    parser.add_argument("--classification_event_type", "-ec", action="store_true",
                        help="Classify event type (requires mode=classifier)")
    parser.add_argument("--classification_current", "-cc", action="store_true",
                        help="Classify event current (requires mode=classifier)")
    parser.add_argument("--classification_cc_1pi", "-cc1pi", action="store_true",
                        help="Classify CC 1pi (requires mode=classifier)")
    parser.add_argument("--classification_n_pions", "-npi", action="store_true",
                        help="Classify number of pions (requires mode=classifier)")
    parser.add_argument("--classification_CC1orNPi", "-npi2", action="store_true",
                        help="Classify CC 1pi or n pions, according to signal definition inEberly et al. 2015 (requires mode=classifier)")
    parser.add_argument("--regress-E-available", "-E-available", action="store_true",
                        help="Regress available energy of the event (requires mode=regression)")
    parser.add_argument("--regress-E-available-no-muon", "-E-available-no-muon", action="store_true",
                        help="Regress available energy of the event, without the muon energy(requires mode=regression)")
    parser.add_argument("--no_use_cond", action="store_true",
                        help="Do NOT use global/conditional features")
    parser.add_argument("--cond_only", "--cond-only", action="store_true",
                        help="Train a simple MLP using only global/conditional features (no transformer)")
    parser.add_argument("--mlp_layers", type=int, default=3,
                        help="Number of residual blocks in CondOnlyMLP (independent of --depth)")
    parser.add_argument("--use_pid", type=bool, default=True,
                        help="Use particle ID information")
    parser.add_argument("--pid_idx", type=int, default=4, help="Index of PID in features")
    # Model architecture (defaults from vit.py example)
    parser.add_argument("--point_cont_dim", type=int, default=9,
                        help="Dimension of continuous point features")
    parser.add_argument("--coord_dim", type=int, default=2,
                        help="Dimension of coordinates")
    parser.add_argument("--d_model", type=int, default=128,
                        help="Model dimension")
    parser.add_argument("--depth", type=int, default=4,
                        help="Number of transformer blocks")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0,
                        help="MLP hidden dimension ratio")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate")
    parser.add_argument("--attn_dropout", type=float, default=0.0,
                        help="Attention dropout rate")
    parser.add_argument("--weighted_regression_loss", "-wl", action="store_true",
                        help="Use weighted regression loss")
    parser.add_argument("--log_MSE_loss", "-log-mse", action="store_true",
                        help="Use log MSE loss")
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", "-wd", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--event-cap", "-cap", type=int, default=-1,
                        help="Maximum number of events to use in the dataset")
    parser.add_argument("--event-sampler-random-state", "-seed-event-sampler", type=int, default=42,
                        help="Random seed for event sampler")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Use automatic mixed precision")
    parser.add_argument("--max_samples_per_epoch", type=int, default=None,
                        help="Maximum number of samples to use per epoch")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps (virtual batch size = batch_size * grad_accum_steps)")
    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=1000,
                        help="Log training loss every N steps")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Run evaluation every N steps")
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--wandb_project", type=str, default="minerva-models",
                        help="Wandb project name")
    parser.add_argument("--run_name", "-name", type=str, default=None,
                        help="Name for this training run (timestamp will be appended); not required when --calculate-flops")
    parser.add_argument("--output_dir", type=str, default="/global/cfs/cdirs/m3246/gregork/checkpoints",
                        help="Base output directory for checkpoints (run_name with timestamp will be appended)")
    parser.add_argument("--log1p_loss", type=bool, default=True, help="Use log1p loss")
    # Other
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "lion"],
                        help="Optimizer to use")
    parser.add_argument("--include-E-sum", type=bool, default=True, help="Include per-PID energy sums (blob, prong types, aggregated) as extra global features")
    parser.add_argument("--zero-cond-feature", type=int, nargs="+", default=None,
                        help="Zero out global/cond feature(s) at these indices (ablation). "
                             "E.g. --zero-cond-feature 3 to ablate E_recoil_CCinc")
    # OmniLearned (PET2) arguments
    parser.add_argument("--use-omnilearned", type=str, default=None,
                        choices=["small", "medium", "large"],
                        help="Use OmniLearned PET2 model of given size instead of ViT")
    parser.add_argument("--use-pretrained", type=str, default=None,
                        help="Load pretrained OmniLearned checkpoint (e.g. pretrain_s, pretrain_m)")
    parser.add_argument("--ol-num-feat", type=int, default=4,
                        help="Number of kinematic input features for PET2 (excluding PID)")
    parser.add_argument("--ol-num-add", type=int, default=5,
                        help="Number of additional features for PET2 add_info input")
    parser.add_argument("--ol-num-cond", type=int, default=4,
                        help="Number of global conditioning features for PET2")
    parser.add_argument("--ol-pid-dim", type=int, default=8,
                        help="Number of unique PID classes for PET2 embedding")
    parser.add_argument("--ol-interaction", action="store_true", default=False,
                        help="Enable interaction matrix in PET2")
    parser.add_argument("--ol-local-interaction", action="store_true", default=False,
                        help="Enable local interaction matrix in PET2")
    parser.add_argument("--ol-interaction-type", type=str, default="lhc",
                        choices=["lhc", "astro"],
                        help="Interaction type for PET2")
    parser.add_argument("--calculate-flops", action="store_true",
                        help="Only compute FLOPs per batch (inference and approx training) then exit")
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
        return Task(type="regression", regress_E_available=args.regress_E_available,
                    regress_E_available_no_muon=args.regress_E_available_no_muon,
                    class_label_idx=class_label_idx, regress_log=False)
    elif args.mode == "classifier":
        if "classification_n_pions" not in args.__dict__:
            args.classification_n_pions = False
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
            class_idx = [0, 1, 2, 3, 4]
            class_idx_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        return Task(type="classifier", classification_event_type=args.classification_event_type, 
            classification_current=args.classification_current, classification_cc_1pi=args.classification_cc_1pi,
            classification_n_pions=args.classification_n_pions, class_idx=class_idx, class_idx_map=class_idx_map,
            class_label_idx=class_label_idx, classification_CC1orNPi=args.classification_CC1orNPi)
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
        global_cont_dim = 4 + e_sum_dim
        model = CondOnlyMLP(
            input_dim=global_cont_dim,
            hidden_dim=args.d_model,
            output_dim=num_classes,
            n_layers=args.mlp_layers,
            dropout=args.dropout,
        )
        return model

    # Point categorical features (PID if used)
    point_cat_num_classes = [8] if args.use_pid else []
    
    # Global categorical features (none by default)
    global_cat_num_classes = []
    
    # Global continuous dimension
    global_cont_dim = (4 if args.use_cond else 0) + e_sum_dim
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


def prepare_batch_omnilearned(batch, device, use_cond=False, use_pid=False, pid_idx=4,
                              include_E_sum=False):
    """Prepare batch for OmniLearned PET2 model input."""
    X = batch["X"].to(device, dtype=torch.float32)
    y = batch["y"].to(device)

    pid = None
    if use_pid and batch.get("pid") is not None:
        pid = batch["pid"].to(device)
        X = torch.cat([X[:, :, :pid_idx], X[:, :, pid_idx + 1:]], dim=2)

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
        elif cond is not None and cond.shape[1] == 10:
            # New: cond already has 10 cols (4 base + 6 log energy sums); use as-is
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


def _make_dummy_batch(args, device):
    """Build a minimal batch dict with shape (batch_size, max_particles, ...) for FLOPs."""
    B = args.batch_size
    N = args.max_particles
    use_cond = not getattr(args, "no_use_cond", False)
    if getattr(args, "cond_only", False):
        use_cond = True
    e_sum_dim = 6 if getattr(args, "include_E_sum", True) else 0
    global_cont_dim = (4 if use_cond else 0) + e_sum_dim
    point_cont_dim = getattr(args, "point_cont_dim", 9)
    coord_dim = getattr(args, "coord_dim", 2)
    pid_idx = getattr(args, "pid_idx", 4)
    use_pid = getattr(args, "use_pid", True)
    # OmniLearned expects X with last dim = ol_num_feat (4) after prepare_batch_omnilearned drops PID.
    # ViT expects point_cont with last dim = point_cont_dim (9) after prepare_batch drops PID.
    if getattr(args, "use_omnilearned", None):
        ol_num_feat = getattr(args, "ol_num_feat", 4)
        total_feat_dim = ol_num_feat + (1 if use_pid else 0)
    else:
        total_feat_dim = point_cont_dim + (1 if use_pid else 0)
    X = torch.zeros(B, N, total_feat_dim, device=device, dtype=torch.float32)
    if args.mode == "regression":
        y = torch.zeros(B, device=device, dtype=torch.float32)
    else:
        y = torch.zeros(B, device=device, dtype=torch.long)
    attention_mask = torch.ones(B, N, device=device, dtype=torch.float32)
    batch = {"X": X, "y": y, "attention_mask": attention_mask}
    if use_cond and global_cont_dim > 0:
        batch["cond"] = torch.zeros(B, 4, device=device, dtype=torch.float32)
    if e_sum_dim > 0:
        batch["energy_sums"] = torch.ones(B, 6, device=device, dtype=torch.float32)
    if getattr(args, "use_omnilearned", None):
        batch["pid"] = torch.zeros(B, N, device=device, dtype=torch.long)
        batch["add_info"] = torch.zeros(B, N, getattr(args, "ol_num_add", 5), device=device, dtype=torch.float32)
    return batch


def run_calculate_flops(args):
    """Use calflops to compute inference FLOPs per batch and approx training FLOPs, then exit."""
    try:
        from calflops import calculate_flops
    except ImportError:
        raise SystemExit("calflops is required for --calculate-flops. Install with: pip install calflops")

    args.use_cond = not args.no_use_cond
    if args.cond_only:
        args.use_cond = True
    device = torch.device("cpu")
    task = create_task(args)
    if args.use_omnilearned:
        model = create_omnilearned_model(args, task)
    else:
        model = create_model(args, task)
    model = model.to(device)
    model.eval()

    dummy_batch = _make_dummy_batch(args, device)
    if args.use_omnilearned:
        inputs = prepare_batch_omnilearned(
            dummy_batch, device, args.use_cond, args.use_pid, args.pid_idx,
            include_E_sum=args.include_E_sum,
        )
    else:
        inputs = prepare_batch(
            dummy_batch, device, args.use_cond, args.use_pid, args.coord_dim, args.pid_idx,
            include_E_sum=args.include_E_sum, zero_cond_feature=args.zero_cond_feature,
        )

    wrapper = _FlopsWrapper(model, args, inputs)
    wrapper.eval()
    # One dummy input so calflops runs wrapper.forward(dummy); FLOPs come from the real model inside
    flops, macs, params = calculate_flops(
        model=wrapper,
        input_shape=(1,),
        output_as_string=False,
    )
    # flops is total for one forward (inference) with the given batch size
    inference_flops = flops
    training_flops_approx = flops * 3
    print(f"Batch size: {args.batch_size}, max_particles: {args.max_particles}")
    print(f"Inference FLOPs per batch: {inference_flops:,}")
    print(f"Training FLOPs per batch (approx, ×3): {training_flops_approx:,}")
    print(f"Params: {params:,}")
    raise SystemExit(0)


def forward_model(model, inputs, args):
    """Run forward pass for either ViT or PET2, returns logits."""
    if args.use_omnilearned:
        outputs = model(
            inputs["X"], inputs["y"],
            cond=inputs["cond"], pid=inputs["pid"], add_info=inputs["add_info"],
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


def prepare_batch(batch, device, use_cond=False, use_pid=False, coord_dim=2, pid_idx=4,
                   include_E_sum=False, zero_cond_feature=None):
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
        #point_cont = X[:, :, :pid_idx]
        point_cont = torch.cat([X[:, :, :pid_idx], X[:, :, pid_idx+1:]], dim=2)
    else:
        point_cont = X # use coord dim too as just normal features

    # Handle global features
    global_cont = None
    global_cats = None
    if use_cond and batch.get("cond") is not None:
        global_cont = batch["cond"].to(device, dtype=torch.float32)  # [B, global_cont_dim]

    if include_E_sum:
        if batch.get("energy_sums") is not None:
            # Legacy: 4-col cond + raw energy_sums; concat log(e_sums+1e-3)
            e_sums = batch["energy_sums"].to(device, dtype=torch.float32)
            e_sums = torch.log(e_sums + 1e-3)  # [B, 6]
            if global_cont is not None:
                global_cont = torch.cat([global_cont, e_sums], dim=1)
            else:
                global_cont = e_sums
        elif global_cont is not None and global_cont.shape[1] == 10:
            # New: cond already has 10 cols (4 base + 6 log energy sums); use as-is
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
    special_token_mask = torch.ones(B, num_special_tokens, device=device, dtype=torch.float32)
    attention_mask = torch.cat([special_token_mask, attention_mask], dim=1)  # [B, num_special + N]
    
    # Convert to boolean key padding mask for scaled_dot_product_attention.
    # For boolean SDPA masks, True means this key position is allowed to attend.
    # Our dataset mask uses 1=valid, 0=padding, so keep valid tokens as True.
    attention_mask = (attention_mask > 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
    
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

@torch.no_grad()
def evaluate(model, dataloader, device, args, class_weights, use_amp=False, step=0):
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
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        if args.use_omnilearned:
            inputs = prepare_batch_omnilearned(batch, device, args.use_cond, args.use_pid, args.pid_idx,
                                              include_E_sum=args.include_E_sum)
        else:
            inputs = prepare_batch(batch, device, args.use_cond, args.use_pid, args.coord_dim, args.pid_idx, include_E_sum=args.include_E_sum, zero_cond_feature=args.zero_cond_feature)
        with autocast(enabled=use_amp):
            logits = forward_model(model, inputs, args)
            if args.mode == "regression":
                if args.weighted_regression_loss:
                    loss = compute_weighted_regression_loss(logits.squeeze(-1), inputs["y"])
                else:
                    loss = criterion(logits.squeeze(-1), inputs["y"])
            else:
                loss = criterion(logits, inputs["y"])

        total_loss += loss.item() * inputs["y"].size(0)
        total_samples += inputs["y"].size(0)
    
    avg_loss = total_loss / total_samples
    model.train()
    return {"eval_loss": avg_loss}


def save_checkpoint(model, optimizer, scheduler, scaler, step, args, best_val_loss, filename="checkpoint.pt"):
    """Save training checkpoint."""
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "args": vars(args),
        "best_val_loss": best_val_loss
    }
    save_path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")
    return save_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None):
    """Load checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    
    start_step = checkpoint.get("step", 0)
    print(f"Loaded checkpoint from {checkpoint_path}, resuming from step {start_step}")
    return start_step

# Make a log loss out of a criterion: the loss is criterion(log(pred+eps) - log(target+eps))

def make_log_loss(criterion):
    eps = 1e-6
    def loss(pred, target):
        pred_mask = (pred >= 0).float()
        return criterion(torch.log(pred_mask*pred+eps), torch.log(target+eps)).mean()
    return loss


def train(args):
    """Main training function."""
    # If resuming, load saved arguments from the checkpoint and override current ones,
    # so that training continues with the exact same configuration as the original run.
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        saved_args = checkpoint.get("args", None)
        if saved_args is not None:
            # Preserve explicitly provided CLI controls like --resume and --no_wandb.
            preserve_keys = {"resume", "no_wandb"}
            for k, v in saved_args.items():
                if k in preserve_keys:
                    continue
                setattr(args, k, v)
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
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    task = create_task(args)
    concat_additional_info = not bool(args.use_omnilearned)
    train_loader, class_weights = load_data(
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
    )
    
    val_loader, _ = load_data(
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
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    if args.use_omnilearned:
        model = create_omnilearned_model(args, task)
    else:
        model = create_model(args, task)
    model = model.to(device)

    # Load pretrained OmniLearned weights (before optimizer setup)
    if args.use_pretrained:
        if not args.use_omnilearned:
            raise ValueError("--use-pretrained requires --use-omnilearned")
        print(f"Loading pretrained weights: {args.use_pretrained}")
        load_pretrained_omnilearned(model, args.use_pretrained, args.output_dir)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    task.class_weights = class_weights
    # Setup loss function
    if task.type == "regression":
        if args.log_MSE_loss:
            criterion = (nn.MSELoss(reduction="none"))
        elif args.log1p_loss:
            criterion = make_log1p_loss(nn.HuberLoss(reduction="none"))
        else:
            criterion = nn.HuberLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(task.class_weights, device=device, dtype=torch.float32))
    
    steps_per_epoch = len(train_loader)

    # Setup optimizer and scheduler
    if args.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "lion":
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.98))
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")
    
    scheduler = get_lr_schedule(optimizer, args.warmup_steps, max_steps=args.max_steps)
    
    # Setup AMP
    scaler = GradScaler() if args.use_amp else None
    
    # Resume from checkpoint
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.login()
        wandb.init(
            project=args.wandb_project,
            name=run_name_with_timestamp,
            config=vars(args),
            # Always resume the wandb run when a checkpoint path is provided.
            resume="allow" if args.resume else None,
        )
        wandb.watch(model, log="all", log_freq=args.log_interval)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    step = start_step  # counts optimizer steps (after grad accumulation)
    accum_counter = 0  # counts micro-batches since last optimizer step
    train_losses = []
    data_fetch_times = []
    backprop_times = []
    best_val_loss = float('inf')
    
    # --- Cond-only diagnostic: inspect first batch ---
    if args.cond_only:
        diag_batch = next(iter(train_loader))
        diag_inputs = prepare_batch(diag_batch, device, args.use_cond, args.use_pid, args.coord_dim, args.pid_idx, include_E_sum=args.include_E_sum, zero_cond_feature=args.zero_cond_feature)
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
                print(f"    feat[{fi}]: {col.min().item():+.4f} / {col.mean().item():+.4f} / {col.max().item():+.4f} / {col.std().item():.4f}")
            print(f"  target (y) shape: {y.shape}  dtype: {y.dtype}")
            print(f"  target stats: min={y.min().item():.4f}  mean={y.mean().item():.4f}  max={y.max().item():.4f}  std={y.std().item():.4f}")
            if args.log1p_loss:
                y_log = torch.log1p(y)
                print(f"  log1p(target) stats: min={y_log.min().item():.4f}  mean={y_log.mean().item():.4f}  max={y_log.max().item():.4f}  std={y_log.std().item():.4f}")
            with torch.no_grad():
                pred0 = model(gc)
                print(f"  model output (untrained) shape: {pred0.shape}")
                print(f"  model output stats: min={pred0.min().item():.4f}  mean={pred0.mean().item():.4f}  max={pred0.max().item():.4f}  std={pred0.std().item():.4f}")
            corr_matrix = torch.corrcoef(torch.cat([gc.T, y.unsqueeze(0)], dim=0))
            print(f"  Pearson correlation of each cond feature with target:")
            for fi in range(gc.shape[1]):
                print(f"    feat[{fi}] <-> y: {corr_matrix[fi, -1].item():+.4f}")
        print("=" * 60)
        del diag_batch, diag_inputs, gc, y

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Training for {args.max_steps} optimizer steps (grad_accum_steps={args.grad_accum_steps})")
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
                inputs = prepare_batch_omnilearned(batch, device, args.use_cond, args.use_pid, args.pid_idx,
                                                  include_E_sum=args.include_E_sum)
            else:
                inputs = prepare_batch(batch, device, args.use_cond, args.use_pid, args.coord_dim, args.pid_idx, include_E_sum=args.include_E_sum, zero_cond_feature=args.zero_cond_feature)

            # Forward pass
            with autocast(enabled=args.use_amp):
                logits = forward_model(model, inputs, args)

                if args.mode == "regression":
                    if args.weighted_regression_loss:
                        loss = compute_weighted_regression_loss(logits.squeeze(-1), inputs["y"])
                    else:
                        loss = criterion(logits.squeeze(-1), inputs["y"])
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

            backprop_times.append(time.perf_counter() - backprop_start_time)

            # Track loss
            train_losses.append(loss.item() * args.grad_accum_steps)

            # Update progress bar with current metrics
            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
                "step": f"{step}/{args.max_steps}",
            })

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
                eval_metrics = evaluate(model, val_loader, device, args, torch.tensor(task.class_weights, device=device, dtype=torch.float32) if task.type == "classifier" else None, args.use_amp, step)

                epoch_pbar.write(f"Eval loss: {eval_metrics['eval_loss']:.4f}")
                if eval_metrics['eval_loss'] < best_val_loss:
                    best_val_loss = eval_metrics['eval_loss']
                    save_checkpoint(model, optimizer, scheduler, scaler, step, args, best_val_loss,
                                    filename="best_model.pt")
                    epoch_pbar.write(f"New best model saved! Val loss: {best_val_loss:.4f}")

                if not args.no_wandb:
                    wandb.log(eval_metrics, step=step)
                    wandb.log({"best_val_loss": best_val_loss}, step=step)

            if step >= args.max_steps:
                done = True
                break

        epoch += 1
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_metrics = evaluate(model, val_loader, device, args, torch.tensor(task.class_weights, device=device, dtype=torch.float32) if task.type == "classifier" else None, args.use_amp, step)
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
        raise SystemExit("--run_name / -name is required unless --calculate-flops is set")
    train(args)


if __name__ == "__main__":
    main()
