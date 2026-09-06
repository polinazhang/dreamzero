#!/bin/bash
#SBATCH --job-name=DreamZero-static
#SBATCH --partition=gpu-h100
#SBATCH --account=gts-zk15
#SBATCH --qos=inferno
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --time=1-00:00:00
set -euo pipefail
module load python/3.11.9 cuda/12.9.1 anaconda3/2023.03
DREAMZERO_ROOT="/storage/scratch1/5/xzhang3205/vla-adaptation/models/dreamzero"
export HF_HOME="${DREAMZERO_ROOT}/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME="${DREAMZERO_ROOT}/.cache"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/dreamzero-static-triton-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/dreamzero-static-inductor-${SLURM_JOB_ID}"
export NO_ALBUMENTATIONS_UPDATE=1
export ATTENTION_BACKEND=TE
export NUM_DIT_STEPS=16
export ENABLE_DIT_CACHE=false
export DYNAMIC_CACHE_SCHEDULE=false
export PYTHONPATH="${DREAMZERO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${DREAMZERO_ROOT}"
"${DREAMZERO_ROOT}/.venv/bin/python" -m torch.distributed.run --standalone --nproc_per_node=1 \
    --module static_inference.run_robocasa "$@"
