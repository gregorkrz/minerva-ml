#!/usr/bin/env python3
"""Zip all checkpoint folders tagged with a given wandb tag.

Usage:
    python zip_wandb_runs.py                          # defaults: tag=Run_2703, output=Run_2703_checkpoints.tar.gz
    python zip_wandb_runs.py --tag Run_2703 -o runs.tar.gz
    python zip_wandb_runs.py --dry-run                # just list matching folders
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils.utils import fetch_runs_from_wandb

CKPT_DIR = Path("/global/cfs/cdirs/m3246/gregork/checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Zip wandb-tagged checkpoint folders.")
    parser.add_argument("--tag", default="Run_1203", help="Wandb tag to filter runs (default: Run_2703)")
    parser.add_argument("--project", default="minerva-models", help="Wandb project name")
    parser.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR, help="Checkpoint root directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output archive path (default: <tag>_checkpoints.tar.gz)")
    parser.add_argument("--dry-run", action="store_true", help="List matching folders without creating archive")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.tag}_checkpoints.tar.gz"

    run_names = fetch_runs_from_wandb(args.tag, args.project)

    found, missing = [], []
    for name in sorted(run_names):
        folder = args.ckpt_dir / name
        if folder.is_dir():
            found.append(name)
        else:
            missing.append(name)

    print(f"\n{len(found)} folder(s) found in {args.ckpt_dir}")
    if missing:
        print(f"{len(missing)} run(s) have no matching folder:")
        for m in missing:
            print(f"  [missing] {m}")

    if not found:
        print("Nothing to archive.")
        return

    if args.dry_run:
        print("\nFolders that would be archived:")
        for f in found:
            print(f"  {f}")
        return

    output_path = Path(args.output).resolve()
    print(f"\nCreating archive: {output_path}")
    print(f"This may take a while for {len(found)} folder(s)...")

    cmd = [
        "tar", "-czf", str(output_path),
        "-C", str(args.ckpt_dir),
        *found,
    ]
    subprocess.run(cmd, check=True)
    size_gb = output_path.stat().st_size / (1024 ** 3)
    print(f"Done. Archive size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
