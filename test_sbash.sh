#!/bin/bash
#SBATCH --job-name=diffusion_denoise_unet_simple
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00


set -eo pipefail


echo "[START] $(date)"
echo "[JOB] ${SLURM_JOB_NAME:-local} ${SLURM_JOB_ID:-no_job_id}"
echo "[NODE] ${SLURM_NODELIST:-local}"

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

cd "$REPO_DIR"

echo "[REPO] $REPO_DIR"

export BASHRCSOURCED="${BASHRCSOURCED:-0}"
source ~/.bashrc

module load scicomp-python-env

set -u

python main.py

echo "[END] $(date)"