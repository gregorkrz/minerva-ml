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

# ---- Container configuration ----
IMAGE="{container_image}"
NAME="minerva_ml"
WORKSPACE="$HOME"


CONTAINER_CMD=$(cat << 'EOF'
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
cd /workspace/minerva-ml-hyperscale

# Verify GPU is available inside the container
nvidia-smi
# Environment variables
{env_vars}

# User commands
pip install transformers
export LD_LIBRARY_PATH="/opt/conda/lib:${{LD_LIBRARY_PATH:-}}"
{commands}
EOF
)

if podman-hpc container exists "$NAME"; then
    echo "🔁 Container $NAME already exists."
    if [ "$(podman-hpc inspect -f '{{.State.Running}}' "$NAME")" != "true" ]; then
        echo "▶️  Starting container..."
        podman-hpc start "$NAME"
    fi
else
    echo "🚀 Creating new container $NAME ..."
    podman-hpc run -d --name "$NAME" --shm-size=16g --gpu \
        -e CUDA_VISIBLE_DEVICES \
        -v "${{WORKSPACE}}:/workspace" \
        -v /global/cfs/cdirs/m3246/gregork:/global/cfs/cdirs/m3246/gregork \
        -v /pscratch/sd/g/gregork:/pscratch/sd/g/gregork \
        "$IMAGE" tail -f /dev/null
fi

echo "▶️ Executing job commands in container ..."
podman-hpc exec -e CUDA_VISIBLE_DEVICES "$NAME" bash -c "$CONTAINER_CMD"

"""
