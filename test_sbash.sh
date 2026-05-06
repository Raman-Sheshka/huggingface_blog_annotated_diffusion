# create my variables here

#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --partition=gpu-b300-288g-ellis
#SBATCH --gres=gpu:__NUM_GPUS__
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=__TIME_LIMIT__
#SBATCH --output=__LOG_FILE__
#SBATCH --error=__LOG_FILE__

echo "[START] $(date) - Running: __SCRIPT_TO_RUN__"

# ========== load environment and modules ==========
source ~/.bashrc

# load necessary modules (adjust as needed for your environment)

#module load triton/2025.1-gcc
#module load gcc/13.3.0
#module load cuda/12.6.2

source activate pytorch-env

# set environment variables for CUDA and compilers
#export CC=$(which gcc)
#export CXX=$(which g++)
#export CUDAHOSTCXX=$(which g++)
#export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
#export PATH="$CUDA_HOME/bin:$PATH"
#export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"