#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 16:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --job-name=minerva_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "======================================"
echo "JobID: $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "Time:  $(date)"
echo "CWD:   $(pwd)"
echo "======================================"

# ---- Environment ----
module load conda
conda activate omni

# Verify GPU is available
nvidia-smi

# ---- Paths (consistent with preprocess_1A.sh) ----
export DATA_DIR=/global/cfs/cdirs/m3246/gregork
DATASET_PATH=$DATA_DIR/Minerva/20260129_split
CHECKPOINT_DIR=$DATA_DIR/checkpoints/20260129_small_training_scratch
PLAYLIST=1A

echo "Dataset path: $DATASET_PATH"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "======================================"

# ---- Training command ----
srun omnilearned train \
  --output_dir $CHECKPOINT_DIR \
  --save-tag M${PLAYLIST}_L1_regression \
  --dataset minerva_${PLAYLIST} \
  --path $DATASET_PATH \
  --mode regression \
  --regression-loss l1 \
  --batch 128 \
  --epoch 20 \
  --lr 5e-5 \
  --size small \
  --wd 0.1 \
  --num-workers 16 \
  --use-pid \
  --wandb 

