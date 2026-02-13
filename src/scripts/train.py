"""
Training script for PointGlobalMixedViT on HEP data.

This script trains a ViT model on particle physics data using the HEPTorchDataset.
It supports both regression and classification tasks with wandb logging and checkpointing.

python -m src.scripts.train -bs 1024 --mode regression -name Train_Regression
python -m src.scripts.train -bs 1024 --mode classifier -cc -name Train_CC 

python -m src.scripts.train -bs 1024 --mode regression -name Train_Regression_SmallModel --d_model 64 --depth 3 --n_heads 4 --dropout 0.0 --attn_dropout 0.0
python -m src.scripts.train -bs 1024 --mode classifier -cc -name Train_CC_SmallModel --d_model 32 --depth 3 --n_heads 4 --dropout 0.0 --attn_dropout 0.0


python -m src.scripts.train -bs 1024 --mode classifier --classification_cc_1pi -name Train_CC1pi --d_model 128 --depth 4 --n_heads 4 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260210_CCpi_labels_split


python -m src.scripts.train -bs 1024 --mode classifier --classification_cc_1pi -name Train_CC1pi --d_model 128 --depth 4 --n_heads 8 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260210_CCpi_labels_split


# Binary classification: has event > 1 charged pion produced?
python -m src.scripts.train -bs 2048 --mode classifier -npi -name Train_MultiPi --d_model 128 --depth 4 --n_heads 8 --dropout 0.01 --attn_dropout 0.01 --data_path /global/cfs/cdirs/m3246/gregork/Minerva/20260213_split --num_workers 2

"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import numpy as np

from src.dataset.dataloader import load_data, Task
from src.models.vit import PointGlobalMixedViT, PointGlobalMixedViTConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train PointGlobalMixedViT on HEP data")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="minerva_1A", 
                        help="Dataset name (e.g., minerva_1A)")
    parser.add_argument("--data_path", type=str,
                        help="Path to dataset directory",
                        default="/global/cfs/cdirs/m3246/gregork/Minerva/20260201_all_max_blobs_and_prongs_split_fix_anomalies")
    parser.add_argument("--batch_size", "-bs", type=int, default=1024,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
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
    parser.add_argument("--use_cond", default=True, type=bool,
                        help="Use global/conditional features")
    parser.add_argument("--use_pid", type=bool, default=True,
                        help="Use particle ID information")
    parser.add_argument("--pid_idx", type=int, default=4,
                        help="Index of PID in features")
    
    # Model architecture (defaults from vit.py example)
    parser.add_argument("--point_cont_dim", type=int, default=4,
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
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    parser.add_argument("--attn_dropout", type=float, default=0.0,
                        help="Attention dropout rate")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--max_samples_per_epoch", type=int, default=None,
                        help="Maximum number of samples to use per epoch")
    parser.add_argument("--single_batch_overfit", action="store_true",
                        help="Debug mode: repeatedly train on one fixed batch to verify model can overfit")
    parser.add_argument("--single_batch_steps", type=int, default=500,
                        help="Number of optimizer steps in single-batch overfit mode")
    parser.add_argument("--single_batch_eval_interval", type=int, default=100,
                        help="How often to log overfit-batch eval metrics in single-batch mode")
    
    # Logging and evaluation
    parser.add_argument("--log_interval", type=int, default=1000,
                        help="Log training loss every N steps")
    parser.add_argument("--eval_interval", type=int, default=10000,
                        help="Run evaluation every N steps")
    parser.add_argument("--save_interval", type=int, default=10000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--debug_regression_collapse", action="store_true",
                        help="Log per-batch regression variance and constant-baseline diagnostics")
    parser.add_argument("--wandb_project", type=str, default="minerva-models",
                        help="Wandb project name")
    parser.add_argument("--run_name", "-name", type=str, required=True,
                        help="Name for this training run (timestamp will be appended)")
    parser.add_argument("--output_dir", type=str, default="/global/cfs/cdirs/m3246/gregork/checkpoints",
                        help="Base output directory for checkpoints (run_name with timestamp will be appended)")
    
    # Other
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    
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
        return Task(type="regression")
    elif args.mode == "classifier":
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
        return Task(type="classifier", classification_event_type=args.classification_event_type, 
            classification_current=args.classification_current, classification_cc_1pi=args.classification_cc_1pi,
            classification_n_pions=args.classification_n_pions, class_idx=class_idx, class_idx_map=class_idx_map, class_label_idx=class_label_idx)
    else:
        raise ValueError("Invalid mode")

def create_model(args, task: Task):
    """Create PointGlobalMixedViT model."""
    if task.type == "classifier":
        num_classes = len(task.class_idx)
    elif task.type == "regression":
        num_classes = 1
    else:
        raise ValueError("Invalid task type")
    
    # Point categorical features (PID if used)
    point_cat_num_classes = [8] if args.use_pid else []
    
    # Global categorical features (none by default)
    global_cat_num_classes = []
    
    # Global continuous dimension
    global_cont_dim = 4 if args.use_cond else 0
    
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
        use_event_token=args.use_cond,
        cat_emb_dim=16,
    )
    
    model = PointGlobalMixedViT(cfg)
    
    # Add output head
    if args.mode == "regression":
        model.head = nn.Linear(args.d_model, 1)
    else:  # classifier
        model.head = nn.Linear(args.d_model, num_classes)
    return model


def prepare_batch(batch, device, use_cond=False, use_pid=False, coord_dim=2, pid_idx=4):
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
        point_cont = X[:, :, :pid_idx]
    else:
        point_cont = X # use coord dim too as just normal features

    # Handle global features
    global_cont = None
    global_cats = None
    if use_cond and batch.get("cond") is not None:
        global_cont = batch["cond"].to(device, dtype=torch.float32)  # [B, global_cont_dim]
    
    # Expand attention mask to account for special tokens
    # Model always adds CLS token, and adds EVT token if use_cond=True
    B = X.shape[0]
    num_special_tokens = 1  # CLS token
    if use_cond:
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

@torch.no_grad()
def evaluate(model, dataloader, device, args, class_weights, use_amp=False):
    """Run evaluation on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    # Setup loss function
    if args.mode == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = prepare_batch(batch, device, args.use_cond, args.use_pid, args.coord_dim, args.pid_idx)
        
        with autocast(enabled=use_amp):
            # Forward pass
            features = model(
                point_cont=inputs["point_cont"],
                point_cats=inputs["point_cats"],
                pos=inputs["pos"],
                global_cont=inputs["global_cont"],
                global_cats=inputs["global_cats"],
                attn_mask=inputs["attn_mask"],
            )

            logits = model.head(features)
            
            # Compute loss
            if args.mode == "regression":
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
        "best_val_loss": best_val_loss,
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


def train(args):
    """Main training function."""
    # Create timestamped run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name_with_timestamp = f"{args.run_name}_{timestamp}"
    
    # Update output directory with timestamped run name
    args.output_dir = os.path.join(args.output_dir, run_name_with_timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Run name: {run_name_with_timestamp}")
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    task = create_task(args)
    train_loader, class_weights = load_data(
        dataset_name=args.dataset_name,
        path=args.data_path,
        batch=args.batch_size,
        dataset_type="train",
        task=task,
        use_cond=args.use_cond,
        use_pid=args.use_pid,
        pid_idx=args.pid_idx,
        distributed=False,
        shuffle=True,
        max_particles=args.max_particles,
        num_workers=args.num_workers,
        rank=0,
        size=1,
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
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(args, task)
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    task.class_weights = class_weights
    # Setup loss function
    if task.type == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(task.class_weights, device=device, dtype=torch.float32))
    # Calculate number of epochs needed to reach max_steps
    
    steps_per_epoch = len(train_loader)
    max_steps = args.epochs * steps_per_epoch

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_schedule(optimizer, args.warmup_steps, max_steps=max_steps)
    
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
            resume="allow" if args.resume else None,
        )
        wandb.watch(model, log="all", log_freq=args.log_interval)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    step = start_step
    train_losses = []
    debug_pred_stds = []
    debug_target_stds = []
    debug_model_mses = []
    debug_baseline_mses = []
    best_val_loss = float('inf')
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Training for {args.epochs} epochs")
    
    if args.single_batch_overfit:
        print("Running single-batch overfit debug mode.")
        fixed_batch = next(iter(train_loader))
        fixed_inputs = prepare_batch(fixed_batch, device, args.use_cond, args.use_pid, args.coord_dim, args.pid_idx)
        overfit_pbar = tqdm(range(args.single_batch_steps), desc="Single-batch overfit", leave=True)

        for _ in overfit_pbar:
            optimizer.zero_grad()

            with autocast(enabled=args.use_amp):
                features = model(
                    point_cont=fixed_inputs["point_cont"],
                    point_cats=fixed_inputs["point_cats"],
                    pos=fixed_inputs["pos"],
                    global_cont=fixed_inputs["global_cont"],
                    global_cats=fixed_inputs["global_cats"],
                    attn_mask=fixed_inputs["attn_mask"],
                )
                logits = model.head(features)
                if args.mode == "regression":
                    loss = criterion(logits.squeeze(-1), fixed_inputs["y"])
                else:
                    loss = criterion(logits, fixed_inputs["y"])

            if args.mode == "regression" and args.debug_regression_collapse:
                with torch.no_grad():
                    preds = logits.squeeze(-1).detach()
                    targets = fixed_inputs["y"].detach()
                    debug_pred_stds.append(preds.std(unbiased=False).item())
                    debug_target_stds.append(targets.std(unbiased=False).item())
                    debug_model_mses.append(torch.mean((preds - targets) ** 2).item())
                    debug_baseline_mses.append(torch.mean((targets - targets.mean()) ** 2).item())

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            step += 1
            train_losses.append(loss.item())
            current_lr = scheduler.get_last_lr()[0]

            overfit_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
                "step": step
            })

            if step % args.log_interval == 0:
                log_dict = {
                    "train_loss": float(np.mean(train_losses)),
                    "learning_rate": current_lr,
                    "step": step,
                    "epoch": 0,
                    "debug/single_batch_mode": 1,
                }
                if args.mode == "regression" and args.debug_regression_collapse and len(debug_pred_stds) > 0:
                    log_dict.update({
                        "debug/pred_std": float(np.mean(debug_pred_stds)),
                        "debug/target_std": float(np.mean(debug_target_stds)),
                        "debug/model_mse": float(np.mean(debug_model_mses)),
                        "debug/baseline_mse": float(np.mean(debug_baseline_mses)),
                    })
                if not args.no_wandb:
                    wandb.log(log_dict, step=step)
                train_losses = []
                debug_pred_stds = []
                debug_target_stds = []
                debug_model_mses = []
                debug_baseline_mses = []

            if step % args.single_batch_eval_interval == 0:
                with torch.no_grad():
                    model.eval()
                    with autocast(enabled=args.use_amp):
                        eval_features = model(
                            point_cont=fixed_inputs["point_cont"],
                            point_cats=fixed_inputs["point_cats"],
                            pos=fixed_inputs["pos"],
                            global_cont=fixed_inputs["global_cont"],
                            global_cats=fixed_inputs["global_cats"],
                            attn_mask=fixed_inputs["attn_mask"],
                        )
                        eval_logits = model.head(eval_features)
                        if args.mode == "regression":
                            eval_loss = criterion(eval_logits.squeeze(-1), fixed_inputs["y"]).item()
                        else:
                            eval_loss = criterion(eval_logits, fixed_inputs["y"]).item()
                    model.train()

                overfit_log = {"debug/single_batch_eval_loss": eval_loss}
                if args.mode == "classifier":
                    with torch.no_grad():
                        pred_labels = torch.argmax(eval_logits, dim=1)
                        acc = (pred_labels == fixed_inputs["y"]).float().mean().item()
                    overfit_log["debug/single_batch_accuracy"] = acc
                if not args.no_wandb:
                    wandb.log(overfit_log, step=step)
                overfit_pbar.write(f"Single-batch eval @ step {step}: loss={eval_loss:.6f}")
    else:
        for epoch in range(args.epochs):
            # Epoch progress bar
            epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=True)
            
            for batch in epoch_pbar:
                # Prepare inputs
                inputs = prepare_batch(batch, device, args.use_cond, args.use_pid, args.coord_dim, args.pid_idx)
                
                # Forward pass
                optimizer.zero_grad()
                
                with autocast(enabled=args.use_amp):
                    features = model(
                        point_cont=inputs["point_cont"],
                        point_cats=inputs["point_cats"],
                        pos=inputs["pos"],
                        global_cont=inputs["global_cont"],
                        global_cats=inputs["global_cats"],
                        attn_mask=inputs["attn_mask"],
                    )
                    
                    logits = model.head(features)

                    # Compute loss
                    if args.mode == "regression":
                        loss = criterion(logits.squeeze(-1), inputs["y"])
                    else:
                        loss = criterion(logits, inputs["y"])

                # Optional diagnostics for regression collapse (predicting close to batch mean).
                if args.mode == "regression" and args.debug_regression_collapse:
                    with torch.no_grad():
                        preds = logits.squeeze(-1).detach()
                        targets = inputs["y"].detach()
                        debug_pred_stds.append(preds.std(unbiased=False).item())
                        debug_target_stds.append(targets.std(unbiased=False).item())
                        debug_model_mses.append(torch.mean((preds - targets) ** 2).item())
                        debug_baseline_mses.append(torch.mean((targets - targets.mean()) ** 2).item())
                
                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                
                scheduler.step()
                
                # Track loss
                train_losses.append(loss.item())
                step += 1
                
                # Update progress bar with current metrics
                current_lr = scheduler.get_last_lr()[0]
                epoch_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": step
                })
                
                # Log training loss
                if step % args.log_interval == 0:
                    avg_train_loss = np.mean(train_losses)
                    
                    log_dict = {
                        "train_loss": avg_train_loss,
                        "learning_rate": current_lr,
                        "step": step,
                        "epoch": epoch,
                    }

                    if args.mode == "regression" and args.debug_regression_collapse and len(debug_pred_stds) > 0:
                        log_dict.update({
                            "debug/pred_std": float(np.mean(debug_pred_stds)),
                            "debug/target_std": float(np.mean(debug_target_stds)),
                            "debug/model_mse": float(np.mean(debug_model_mses)),
                            "debug/baseline_mse": float(np.mean(debug_baseline_mses)),
                        })
                    
                    if not args.no_wandb:
                        wandb.log(log_dict, step=step)
                    
                    train_losses = []
                    debug_pred_stds = []
                    debug_target_stds = []
                    debug_model_mses = []
                    debug_baseline_mses = []
                
                # Evaluation
                if step % args.eval_interval == 0:
                    epoch_pbar.write(f"\nRunning evaluation at step {step}...")
                    eval_metrics = evaluate(model, val_loader, device, args, torch.tensor(task.class_weights, device=device, dtype=torch.float32), args.use_amp)
                    
                    epoch_pbar.write(f"Eval loss: {eval_metrics['eval_loss']:.4f}")
                    # save as best_model.pt if val loss is lower
                    if eval_metrics['eval_loss'] < best_val_loss:
                        best_val_loss = eval_metrics['eval_loss']
                        save_checkpoint(model, optimizer, scheduler, scaler, step, args, best_val_loss,
                                        filename="best_model.pt")
                        epoch_pbar.write(f"New best model saved! Val loss: {best_val_loss:.4f}")
                    
                    if not args.no_wandb:
                        wandb.log(eval_metrics, step=step)
                        wandb.log({"best_val_loss": best_val_loss}, step=step)
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_metrics = evaluate(model, val_loader, device, args, torch.tensor(task.class_weights, device=device, dtype=torch.float32), args.use_amp)
    print(f"Final eval loss: {eval_metrics['eval_loss']:.4f}")
    
    if not args.no_wandb:
        wandb.log(eval_metrics, step=step)
    
    # Save final checkpoint
    #save_checkpoint(model, optimizer, scheduler, scaler, step, args, best_val_loss,
    #                filename="checkpoint_final.pt")
    
    print("Training complete!")
    
    if not args.no_wandb:
        wandb.finish()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
