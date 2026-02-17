"""
Evaluation script for PointGlobalMixedViT on HEP data.

This script evaluates a trained ViT model on particle physics data using the HEPTorchDataset.
It supports both regression and classification tasks with wandb logging and metric computation.

Example usage:
python -m src.scripts.eval --checkpoint Train_Regression_SmallModel_20260210_003728 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_Regression_20260210_003302 --batch_size 1024 --dataset_name minerva_1A
Train_CC_SmallModel_20260210_004255
Train_CC_20260210_003828

python -m src.scripts.eval --checkpoint Train_CC_SmallModel_20260210_004255 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_CC_20260210_003828 --batch_size 1024 --dataset_name minerva_1A

python -m src.scripts.eval --checkpoint Train_MultiPi_20260213s_122832 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_MultiPi_20260213_122831 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_MultiPi_20260213_122827 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_MultiPi_20260213_122825 --batch_size 1024 --dataset_name minerva_1A


python -m src.scripts.eval --checkpoint Train_CC1pi_20260210_190400 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_CC1pi_20260210_190406 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_CC1pi_20260210_190413 --batch_size 1024 --dataset_name minerva_1A
python -m src.scripts.eval --checkpoint Train_CC1pi_20260210_190428 --batch_size 1024 --dataset_name minerva_1A


# eval /global/cfs/cdirs/m3246/gregork/checkpoints/Train_Regress_E_available2_20260214_232056/
python -m src.scripts.eval --checkpoint Train_Regress_E_available2_20260214_232056 --batch_size 1024 --dataset_name minerva_1A
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import wandb
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, r2_score, mean_absolute_error

from src.dataset.dataloader import load_data
from src.models.vit import PointGlobalMixedViT, PointGlobalMixedViTConfig
from src.scripts.train import set_seed, prepare_batch, create_task
from types import SimpleNamespace



def create_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    # checkpoint in checkpoint_path is a dictionary with the keys "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "step", "args", "best_val_loss"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get args from checkpoint
    if "args" not in checkpoint:
        raise ValueError("Checkpoint does not contain training arguments. Cannot reconstruct model.")
    
    args_dict = checkpoint["args"]

    args_attrs = SimpleNamespace(**args_dict)
    
    # Determine number of output classes for classification
    num_classes = None
    task = create_task(args_attrs)
    if task.type == "classifier":
        num_classes = len(task.class_idx)
    elif task.type == "regression":
        num_classes = 1
    else:
        raise ValueError("Invalid task type")
    
    # Reconstruct model config
    point_cat_num_classes = [8] if args_dict.get("use_pid", True) else []
    global_cat_num_classes = []
    global_cont_dim = 4 if args_dict.get("use_cond", False) else 0
    
    cfg = PointGlobalMixedViTConfig(
        point_cont_dim=args_dict.get("point_cont_dim", 2),
        point_cat_num_classes=point_cat_num_classes,
        global_cont_dim=global_cont_dim,
        global_cat_num_classes=global_cat_num_classes,
        coord_dim=args_dict.get("coord_dim", 2),
        d_model=args_dict.get("d_model", 128),
        depth=args_dict.get("depth", 4),
        n_heads=args_dict.get("n_heads", 4),
        mlp_ratio=args_dict.get("mlp_ratio", 4.0),
        dropout=args_dict.get("dropout", 0.1),
        attn_dropout=args_dict.get("attn_dropout", 0.0),
        use_cls_token=True,
        use_event_token=args_dict.get("use_cond", False),
        cat_emb_dim=16,
    )
    
    model = PointGlobalMixedViT(cfg)
    
    # Add output head
    if task.type == "regression":
        model.head = nn.Linear(args_dict.get("d_model", 128), 1)
    else:
        model.head = nn.Linear(args_dict.get("d_model", 128), num_classes)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Checkpoint was at step: {checkpoint.get('step', 'unknown')}")
    if "best_val_loss" in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, args_dict, task




@torch.no_grad()
def evaluate(model, dataloader, device, args_dict, use_amp=False):
    """Run evaluation and compute metrics."""
    model.eval()
    
    mode = args_dict.get("mode", "regression")
    use_cond = args_dict.get("use_cond", False)
    use_pid = args_dict.get("use_pid", True)
    coord_dim = args_dict.get("coord_dim", 2)
    pid_idx = args_dict.get("pid_idx", 4)
    
    # Setup loss function
    if mode == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    all_preds = []
    all_targets = []
    all_losses = []
    all_cond = []

    for batch in tqdm(dataloader, desc=f"Evaluating", leave=True):
        inputs = prepare_batch(batch, device, use_cond, use_pid, coord_dim, pid_idx)
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
            if mode == "regression":
                loss = criterion(logits.squeeze(-1), inputs["y"])
            else:
                loss = criterion(logits, inputs["y"])
        
        all_losses.append(loss.item())
        
        # Store predictions and targets
        if mode == "regression":
            all_preds.append(logits.squeeze(-1).cpu().numpy())
        else:
            #all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())# append the raw logits!
            all_preds.append(logits.cpu().numpy())
        
        all_targets.append(inputs["y"].cpu().numpy())
        all_cond.append(batch["cond"].cpu().numpy())
        # Each 10th step, print 1st 10 predictions and targets
        
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_cond = np.concatenate(all_cond)
    avg_loss = np.mean(all_losses)


    # Compute metrics
    metrics = {"loss": avg_loss}
    
    if mode == "regression":
        # Regression metrics
        metrics["r2_score"] = r2_score(all_targets, all_preds)
        metrics["mae"] = mean_absolute_error(all_targets, all_preds)
        metrics["rmse"] = np.sqrt(avg_loss)  # Since we used MSELoss
        
        # Compute residuals
        residuals = all_targets - all_preds
        metrics["mean_residual"] = np.mean(residuals)
        metrics["std_residual"] = np.std(residuals)
        
        print(f"\n{'='*60}")
        print(f"Regression Metrics:")
        print(f"{'='*60}")
        print(f"Loss (MSE):        {metrics['loss']:.6f}")
        print(f"RMSE:              {metrics['rmse']:.6f}")
        print(f"MAE:               {metrics['mae']:.6f}")
        print(f"R² Score:          {metrics['r2_score']:.6f}")
        print(f"Mean Residual:     {metrics['mean_residual']:.6f}")
        print(f"Std Residual:      {metrics['std_residual']:.6f}")
        print(f"{'='*60}\n")
        
    else:
        # Classification metrics are computed from class predictions (argmax of logits),
        # while preserving raw logits in all_preds for downstream analysis/saving.
        all_pred_labels = np.argmax(all_preds, axis=1)
        metrics["accuracy"] = accuracy_score(all_targets, all_pred_labels)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_pred_labels, average='weighted', zero_division=0
        )
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1_score"] = f1
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            all_targets, all_pred_labels, average=None, zero_division=0
        )
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_pred_labels)
        print(f"\n{'='*60}")
        print(f"Classification Metrics:")
        print(f"{'='*60}")
        print(f"Loss:              {metrics['loss']:.6f}")
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1 Score:          {metrics['f1_score']:.4f}")
        print(f"\nPer-class metrics:")
        for i in range(len(support_per_class)):
            print(f"  Class {i}: P={precision_per_class[i]:.4f}, R={recall_per_class[i]:.4f}, "
                  f"F1={f1_per_class[i]:.4f}, Support={support_per_class[i]}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"{'='*60}\n")
        
        # Store per-class metrics
        for i in range(len(support_per_class)):
            metrics[f"precision_class_{i}"] = precision_per_class[i]
            metrics[f"recall_class_{i}"] = recall_per_class[i]
            metrics[f"f1_class_{i}"] = f1_per_class[i]
            metrics[f"support_class_{i}"] = int(support_per_class[i])
    
    results = {
        "metrics": metrics,
        "predictions": all_preds,
        "targets": all_targets,
        "cond": all_cond
    }
    
    return results
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PointGlobalMixedViT on HEP data")    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--base_dir", type=str, default="/global/cfs/cdirs/m3246/gregork/checkpoints")
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="minerva_1A",
                        help="Dataset name (e.g., minerva_1A). If None, will use from checkpoint.")
    #parser.add_argument("--data_path", type=str, default="/global/cfs/cdirs/m3246/gregork/Minerva/20260201_all_max_blobs_and_prongs_split_fix_anomalies")
    parser.add_argument("--dataset_type", type=str, default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of dataloader workers")
    parser.add_argument("--max_particles", type=int, default=None,
                        help="Maximum number of particles per event. If None, will use from checkpoint.")    
    # Evaluation settings
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Use automatic mixed precision")

    args = parser.parse_args()
    args.checkpoint = os.path.join(args.base_dir, args.checkpoint, "best_model.pt")
    return args


def main():
    args = parse_args()
    
    # Set seed
    set_seed(42)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model from checkpoint

    model, args_dict, task = create_model_from_checkpoint(args.checkpoint, device)
    
    # Use checkpoint args if not provided
    dataset_name = args.dataset_name or args_dict.get("dataset_name", "minerva_1A")
    data_path = args_dict.get("data_path")
    max_particles = args.max_particles or args_dict.get("max_particles", 33)
    
    if data_path is None:
        raise ValueError("data_path must be provided either as argument or in checkpoint")
    
    print(f"\nLoading {args.dataset_type} dataset...")
    print(f"Dataset: {dataset_name}")
    print(f"Data path: {data_path}")
    
    # Create dataloader
    dataloader, _ = load_data(
        dataset_name=dataset_name,
        path=data_path,
        batch=args.batch_size,
        dataset_type=args.dataset_type,
        use_cond=args_dict.get("use_cond", False),
        use_pid=args_dict.get("use_pid", True),
        pid_idx=args_dict.get("pid_idx", 4),
        num_workers=args.num_workers,
        distributed=False,
        shuffle=False,
        max_particles=max_particles,
        task=task,
    )
    
    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).stem
    output_dir = os.path.join(os.path.dirname(args.checkpoint), "test_results")
    output_file = os.path.join(output_dir, f"outputs_{checkpoint_name}_{dataset_name}_0.npz")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    # Run evaluation
    print(f"\nStarting evaluation...")
    results = evaluate(
        model,
        dataloader,
        device,
        args_dict,
        use_amp=args.use_amp,
    )
    
    # Save results
    #results_file = os.path.join(output_dir, "metrics.json")
    #with open(results_file, "w") as f:
    #    json.dump(results["metrics"], f, indent=2)
    #print(f"Metrics saved to: {results_file}")
    
    np.savez(
        output_file,
        prediction=results["predictions"],
        pid=results["targets"],
        cond=results["cond"]
    )
    print(f"Predictions saved to: {output_file}")
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()