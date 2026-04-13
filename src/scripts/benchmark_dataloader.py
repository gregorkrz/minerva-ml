#!/usr/bin/env python3
"""
Benchmark HEP torch dataset loading and DataLoader batching.

Usage:
  python -m src.scripts.benchmark_dataloader --dataset-dir /path/to/1A/train --batch-size 2048
  python -m src.scripts.benchmark_dataloader -d /path/to/1A/train -bs 2048
"""

import argparse
import os
import time
from pathlib import Path

import torch

from src.dataset.dataloader import (
    HEPTorchDataset,
    Task,
    collate_point_cloud,
)
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark HEP torch dataset loading and batching."
    )
    p.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        required=True,
        help="Path to dataset folder containing .pb files (e.g. .../1A/train).",
    )
    p.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=2048,
        help="Batch size for the DataLoader (default: 2048).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0 = main process only).",
    )
    p.add_argument(
        "--max-particles",
        type=int,
        default=33,
        help="max_particles for collate (default: 33).",
    )
    p.add_argument(
        "--num-batches",
        "-n",
        type=int,
        default=None,
        help="Number of batches to iterate (default: all batches in one epoch).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise SystemExit(f"Dataset dir not found: {args.dataset_dir}")
    pb_files = list(Path(args.dataset_dir).glob("*.pb"))
    if not pb_files:
        raise SystemExit(f"No .pb files in {args.dataset_dir}")

    task = Task(type="regression", regress_E_available_no_muon=True, class_label_idx=0)

    # --- 1) Benchmark dataset loading (HEPTorchDataset init) ---
    print("=" * 60)
    print("1) Benchmark: loading HEP torch dataset (files into memory)")
    print("=" * 60)
    t0 = time.perf_counter()
    dataset = HEPTorchDataset(
        folder=args.dataset_dir,
        task=task,
        nevts=-1,
        max_particles=args.max_particles,
        concat_additional_info=True,
    )
    load_time = time.perf_counter() - t0
    n_samples = len(dataset)
    print(f"  Samples: {n_samples}")
    print(f"  Load time: {load_time:.2f} s")
    print(f"  Throughput: {n_samples / load_time:.0f} samples/s (load)")
    print()

    # --- 2) Benchmark batching (DataLoader iteration) ---
    print("=" * 60)
    print("2) Benchmark: batching (DataLoader iteration)")
    print("=" * 60)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: collate_point_cloud(x, max_particles=args.max_particles),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    n_batches_per_epoch = len(loader)
    n_batches = (
        args.num_batches if args.num_batches is not None else n_batches_per_epoch
    )
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {n_batches_per_epoch}")
    print(f"  Batches to run: {n_batches}")
    print(f"  Workers: {args.num_workers}")
    print()

    t0 = time.perf_counter()
    total_samples = 0
    batch_count = 0
    for batch in loader:
        total_samples += batch["X"].shape[0]
        batch_count += 1
        if batch_count >= n_batches:
            break
    batch_time = time.perf_counter() - t0

    print(f"  Batches iterated: {batch_count}")
    print(f"  Total samples iterated: {total_samples}")
    print(f"  Total time: {batch_time:.2f} s")
    print(f"  Throughput: {total_samples / batch_time:.0f} samples/s")
    print(f"  Time per batch: {batch_time / batch_count * 1000:.1f} ms")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
