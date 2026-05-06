#!/bin/bash

# inspired by Wenwen Hou sbatch script
REPO_DIR_NAME="retaining-by-doing-RLand-SFT"  # the name of the repository directory, e.g., "retaining-by-doing-RLand-SFT"

# ========== environment variables ==========
USER="$(whoami)"                            # current user
WORK_DIR="$WRKDIR"                          # scratch working directory

# ========== command-line arguments ==========
SCRIPT_TO_RUN="$1"                          # takes the first argument, the script to run, e.g., "bash scripts/rl.sh"
JOB_NAME="${2:-myjob}"                      # SLURM job name, takes the second argument or defaults to "myjob"
NUM_GPUS="${3:-2}"                          # number of GPUs, takes the third argument or defaults to 2
TIME_LIMIT="${4:-5-00:00:00}"               # time limit, takes the fourth argument or defaults to 5 days (format: D-HH:MM:SS)

# if bash sbatch.sh "bash scripts/rl.sh" myjob 4 2-00:00:00
# then:
# - SCRIPT_TO_RUN="bash scripts/rl.sh"
# - JOB_NAME="myjob"
# - NUM_GPUS=4
# - TIME_LIMIT="2-00:00:00"


TIME_TAG=$(date +%Y%m%d_%H%M%S).            # create a timestamp in the format YYYYMMDD_HHMMSS

# ========== log file path ==========
LOG_FILE="$WORK_DIR/code/$REPO_DIR_NAME/logs/${JOB_NAME}_${TIME_TAG}.log"

# ========== SLURM script path ==========
SLURM_SCRIPT="/tmp/slurm_${JOB_NAME}_${TIME_TAG}.sh"

# GPU options
# #SBATCH --partition=gpu-h200-141g-ellis

cat > "$SLURM_SCRIPT" << 'SLURM_EOF'
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
module load triton/2025.1-gcc
module load gcc/13.3.0
module load cuda/12.6.2

# set environment variables for CUDA and compilers
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export WANDB_API_KEY="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"


# ========== go to the repository directory ==========
cd $WORK_DIR/code/$REPO_DIR_NAME/



__SCRIPT_TO_RUN__
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "[SUCCESS] Job finished successfully ✅"
else
    echo "[ERROR] Job failed with exit code: $EXIT_CODE ❌"
fi
echo "[END] $(date)"
SLURM_EOF

# ========= replace placeholders in the SLURM script ==========
sed -i "s|__JOB_NAME__|$JOB_NAME|g" "$SLURM_SCRIPT"
sed -i "s|__NUM_GPUS__|$NUM_GPUS|g" "$SLURM_SCRIPT"
sed -i "s|__TIME_LIMIT__|$TIME_LIMIT|g" "$SLURM_SCRIPT"
sed -i "s|__LOG_FILE__|$LOG_FILE|g" "$SLURM_SCRIPT"
sed -i "s|__SCRIPT_TO_RUN__|$SCRIPT_TO_RUN|g" "$SLURM_SCRIPT"

# ========== submit SLURM job ==========
echo "[INFO] Submitting SLURM job"
echo "[INFO] Job name: $JOB_NAME"
echo "[INFO] Command to run: $SCRIPT_TO_RUN"
echo "[INFO] Number of GPUs: $NUM_GPUS"
echo "[INFO] Time limit: $TIME_LIMIT"
echo "[INFO] Log file: $LOG_FILE"

JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $NF}')

# ========== print instructions for the user ==========
echo ""
echo "✅ SLURM job submitted successfully!"
echo "   Job ID: $JOB_ID"
echo ""
echo "📋 Check job status:"
echo "   squeue -j $JOB_ID"
echo ""
echo "📝 Check log file:"
echo "   tail -f $LOG_FILE"
echo ""
echo "❌ Cancel job:"
echo "   scancel $JOB_ID"