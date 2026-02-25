SLURM_TEMPLATE_GPU = """#!/bin/bash
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q {queue_name}
#SBATCH -t {time}
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}
#SBATCH --error={error_dir}

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

{env_commands}

{commands}

"""
