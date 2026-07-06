"""
Evaluation script for trained models on HEP data (PointGlobalMixedViT, OmniLearned PET2, BERT baseline).

This script evaluates a checkpoint on particle physics data using the HEPTorchDataset.
It supports both regression and classification tasks with wandb logging and metric computation.

Example usage:
python -m src.scripts.eval --checkpoint Training_Name --batch_size 1024 --dataset_name minerva_1A

"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Subset
import wandb
from tqdm import tqdm
import numpy as np

from src.dataset.dataloader import load_data
from src.models.vit import PointGlobalMixedViT, PointGlobalMixedViTConfig
from src.models.omnilearned import PET2, get_model_parameters
from src.scripts.train import (
    set_seed,
    prepare_batch,
    prepare_batch_omnilearned,
    prepare_batch_bert,
    prepare_batch_hyperscale,
    create_task,
    create_bert_model,
    create_hyperscale_model,
    forward_model,
    CondOnlyMLP,
    make_log_loss,
    _bdt_eval_batch,
)
from src.constants.dataset import GLOBAL_COND_BASE_DIM
from types import SimpleNamespace


def create_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    # checkpoint in checkpoint_path is a dictionary with the keys "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict", "step", "args", "best_val_loss"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get args from checkpoint
    if "args" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain training arguments. Cannot reconstruct model."
        )

    args_dict = checkpoint["args"]

    args_attrs = SimpleNamespace(**args_dict)

    # Determine number of output classes for classification
    num_classes = None
    task = create_task(args_attrs)
    # Evaluation always uses MC truth labels, even for models trained with --predict-baseline.
    task.predict_baseline = False
    if task.type == "classifier":
        num_classes = len(task.class_idx)
    elif task.type == "regression":
        num_classes = 1
    else:
        raise ValueError("Invalid task type")

    # BDT (sklearn HistGradientBoosting) is not a torch module; load the pickled
    # estimator saved alongside the checkpoint and return it directly.
    if args_dict.get("use_bdt", False):
        import joblib

        bdt_path = os.path.join(os.path.dirname(checkpoint_path), "bdt_model.joblib")
        if not os.path.exists(bdt_path):
            raise FileNotFoundError(f"BDT model file not found: {bdt_path}")
        print(f"Loading BDT model from {bdt_path}...")
        model = joblib.load(bdt_path)
        kind = "regressor" if args_dict.get("mode") == "regression" else "classifier"
        print(f"BDT loaded ({kind}, {type(model).__name__})")
        return model, args_dict, task

    # Reconstruct model
    ol_size = args_dict.get("use_omnilearned", None)
    if ol_size:
        model_params = get_model_parameters(ol_size)
        use_cond = args_dict.get("use_cond", True)
        e_sum_dim = 6 if args_dict.get("include_E_sum", False) else 0
        cond_dim = args_dict.get("ol_num_cond", 4) + e_sum_dim
        if e_sum_dim > 0:
            use_cond = True
        model = PET2(
            input_dim=args_dict.get("ol_num_feat", 4),
            use_int=args_dict.get("ol_interaction", False),
            local_int=args_dict.get("ol_local_interaction", False),
            int_type=args_dict.get("ol_interaction_type", "lhc"),
            conditional=use_cond,
            cond_dim=cond_dim,
            pid=args_dict.get("use_pid", True),
            pid_dim=args_dict.get("ol_pid_dim", 8),
            add_info=True,
            add_dim=args_dict.get("ol_num_add", 5),
            mode=args_dict.get("mode", "classifier"),
            num_classes=num_classes,
            num_gen_classes=1,
            mlp_drop=args_dict.get("dropout", 0.0),
            attn_drop=args_dict.get("attn_dropout", 0.0),
            feature_drop=0.0,
            num_coord=args_dict.get("coord_dim", 2),
            K=10,
            **model_params,
        )
    elif args_dict.get("use_bert", None):
        model = create_bert_model(args_attrs, task)
    elif args_dict.get("use_hyperscale", None):
        model = create_hyperscale_model(args_attrs, task)
    elif args_dict.get("cond_only", False):
        e_sum_dim = 6 if args_dict.get("include_E_sum", False) else 0
        model = CondOnlyMLP(
            input_dim=GLOBAL_COND_BASE_DIM + e_sum_dim,
            hidden_dim=args_dict.get("d_model", 128),
            output_dim=num_classes,
            n_layers=args_dict.get("mlp_layers", 3),
            dropout=args_dict.get("dropout", 0.1),
            output_positive=(task.type == "regression"),
        )
    else:
        e_sum_dim = 6 if args_dict.get("include_E_sum", False) else 0
        point_cat_num_classes = [8] if args_dict.get("use_pid", True) else []
        global_cat_num_classes = []
        global_cont_dim = (
            GLOBAL_COND_BASE_DIM if args_dict.get("use_cond", False) else 0
        ) + e_sum_dim
        use_event_token = args_dict.get("use_cond", False) or args_dict.get(
            "include_E_sum", False
        )

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
            use_event_token=use_event_token,
            cat_emb_dim=16,
        )

        model = PointGlobalMixedViT(cfg)

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


def make_log1p_loss(criterion):
    # Transform target to log1p(target+1)
    def loss(pred, target, step=0):
        target = torch.log1p(target)
        return criterion(pred, target).mean()

    return loss


@torch.no_grad()
def evaluate_bdt(model, dataloader, device, args_dict, task):
    """Evaluate a HistGradientBoosting model on cond features.

    Classifiers: ``predict_proba`` → log-probabilities for the classification pipeline.
    Regressors: ``predict`` → scalar targets for the regression pipeline.

    Validation loss uses the same batched aggregation as :func:`src.scripts.train.evaluate`.
    """
    args = SimpleNamespace(**args_dict)
    use_binned_loss = bool(
        args_dict.get("binned_loss_var") or args_dict.get("binned_loss_signal")
    )

    total_loss = 0.0
    total_samples = 0
    preds_parts, target_parts, cond_parts, pid_parts = [], [], [], []

    for batch in tqdm(dataloader, desc="Evaluating (BDT)", leave=True):
        loss, n, extras = _bdt_eval_batch(
            model, batch, args, task, use_binned_loss=use_binned_loss,
        )
        total_loss += loss * n
        total_samples += n
        preds_parts.append(extras["predictions"])
        target_parts.append(extras["targets"])
        cond_parts.append(extras["cond"])
        if batch.get("pid_label") is not None:
            pid_parts.append(batch["pid_label"].numpy())

    avg_loss = total_loss / total_samples if total_samples else float("nan")
    results = {
        "metrics": {"loss": float(avg_loss)},
        "predictions": np.concatenate(preds_parts),
        "targets": np.concatenate(target_parts),
        "cond": np.concatenate(cond_parts),
    }
    if pid_parts:
        results["pid_labels"] = np.concatenate(pid_parts)
    return results


@torch.no_grad()
def evaluate(model, dataloader, device, args_dict, use_amp=False):
    """Run evaluation and compute metrics."""
    model.eval()

    mode = args_dict.get("mode", "regression")
    use_cond = args_dict.get("use_cond", False)
    use_pid = args_dict.get("use_pid", True)
    coord_dim = args_dict.get("coord_dim", 2)
    pid_idx = args_dict.get("pid_idx", 4)
    cond_only = args_dict.get("cond_only", False)
    include_E_sum = args_dict.get("include_E_sum", False)
    zero_cond_feature = args_dict.get("zero_cond_feature", None)
    use_omnilearned = args_dict.get("use_omnilearned", None)
    use_bert = args_dict.get("use_bert", None)
    use_hyperscale = args_dict.get("use_hyperscale", None)
    args_ns = SimpleNamespace(**args_dict)

    # Setup loss function
    if mode == "regression":
        if args_dict.get("log1p_loss", False):
            print("Using log1p loss for eval!")
            criterion = make_log1p_loss(nn.HuberLoss(reduction="none"))
        else:
            criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    all_pid_labels = []
    all_losses = []
    all_cond = []

    for batch in tqdm(dataloader, desc=f"Evaluating", leave=True):
        if use_omnilearned:
            inputs = prepare_batch_omnilearned(
                batch, device, use_cond, use_pid, pid_idx, include_E_sum=include_E_sum
            )
        elif use_bert:
            inputs = prepare_batch_bert(
                batch,
                device,
                use_pid=use_pid,
                pid_idx=pid_idx,
                use_cond=use_cond,
                include_E_sum=include_E_sum,
                zero_cond_feature=zero_cond_feature,
                energy_order=args_dict.get("bert_energy_order", False),
            )
        elif use_hyperscale:
            inputs = prepare_batch_hyperscale(
                batch,
                device,
                use_pid=use_pid,
                pid_idx=pid_idx,
                use_cond=use_cond,
                include_E_sum=include_E_sum,
                zero_cond_feature=zero_cond_feature,
                variant=use_hyperscale,
            )
        else:
            inputs = prepare_batch(
                batch,
                device,
                use_cond,
                use_pid,
                coord_dim,
                pid_idx,
                include_E_sum=include_E_sum,
                zero_cond_feature=zero_cond_feature,
            )
        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            logits = forward_model(model, inputs, args_ns)

            if mode == "regression":
                loss = criterion(logits.squeeze(-1), inputs["y"])
            else:
                loss = criterion(logits, inputs["y"])

        all_losses.append(loss.item())

        # Store predictions and targets
        if mode == "regression":
            all_preds.append(logits.squeeze(-1).cpu().numpy())
        else:
            # all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())# append the raw logits!
            all_preds.append(logits.cpu().numpy())

        all_targets.append(inputs["y"].cpu().numpy())
        if mode == "classifier" and batch.get("pid_label") is not None:
            all_pid_labels.append(batch["pid_label"].cpu().numpy())
        all_cond.append(batch["cond"].cpu().numpy())
        # Each 10th step, print 1st 10 predictions and targets

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_cond = np.concatenate(all_cond)
    avg_loss = np.mean(all_losses)
    metrics = {"loss": avg_loss}

    results = {
        "metrics": metrics,
        "predictions": all_preds,
        "targets": all_targets,
        "cond": all_cond,
    }
    if all_pid_labels:
        results["pid_labels"] = np.concatenate(all_pid_labels)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate models on Minerva data")
    # Model checkpoint
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--base_dir", type=str, default="/global/cfs/cdirs/m3246/gregork/checkpoints"
    )
    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="minerva_1A",
        help="Dataset name (e.g., minerva_1A). If None, will use from checkpoint.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--nevts",
        type=int,
        default=-1,
        help="Maximum number of events to evaluate (-1 = all)",
    )
    parser.add_argument(
        "--batch_size", "-bs", type=int, default=2048, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--max_particles",
        type=int,
        default=None,
        help="Maximum number of particles per event. If None, will use from checkpoint.",
    )
    # Evaluation settings
    parser.add_argument(
        "--use_amp",
        dest="use_amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
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
        raise ValueError(
            "data_path must be provided either as argument or in checkpoint"
        )

    print(f"\nLoading {args.dataset_type} dataset...")
    print(f"Dataset: {dataset_name}")
    print(f"Data path: {data_path}")

    # Create dataloader
    use_omnilearned = args_dict.get("use_omnilearned", None)
    use_bert = args_dict.get("use_bert", None)
    concat_additional_info = not (bool(use_omnilearned) or bool(use_bert))
    tabular_only = args_dict.get("cond_only", False) or args_dict.get("use_bdt", False)
    dataloader, _, _ = load_data(
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
        nevts=args.nevts,
        use_energy_sums=args_dict.get("include_E_sum", False),
        concat_additional_info=concat_additional_info,
        tabular_only=tabular_only,
    )

    print(f"Number of samples: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    ds = dataloader.dataset
    if isinstance(ds, Subset):
        eval_indices = np.asarray(ds.indices, dtype=np.int64)
    else:
        eval_indices = np.arange(len(ds), dtype=np.int64)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(args.checkpoint).stem
    output_dir = os.path.join(
        os.path.dirname(args.checkpoint), args.dataset_type + "_results"
    )
    output_file = os.path.join(
        output_dir, f"outputs_{checkpoint_name}_{dataset_name}_0.npz"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")
    # Run evaluation
    print(f"\nStarting evaluation...")
    if args_dict.get("use_bdt", False):
        results = evaluate_bdt(model, dataloader, device, args_dict, task)
    else:
        results = evaluate(
            model,
            dataloader,
            device,
            args_dict,
            use_amp=args.use_amp,
        )

    # Save results
    # results_file = os.path.join(output_dir, "metrics.json")
    # with open(results_file, "w") as f:
    #    json.dump(results["metrics"], f, indent=2)
    # print(f"Metrics saved to: {results_file}")
    # log1p target transform is a regression-only feature; never apply it to
    # classifier logits/probabilities (otherwise it floors them to ~0).
    if args_dict.get("log1p_loss", False) and args_dict.get("mode") == "regression":
        results["predictions"] = np.exp(results["predictions"]) - 1
        # set neg predictions to 0
        results["predictions"] = np.maximum(results["predictions"], 0)

    np.savez(
        output_file,
        prediction=results["predictions"],
        pid=(
            results["pid_labels"]
            if "pid_labels" in results
            else results["targets"]
        ),
        cond=results["cond"],
        eval_indices=eval_indices,
    )
    print(f"Predictions saved to: {output_file}")

    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
