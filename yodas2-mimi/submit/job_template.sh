#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sc-loprio            # from: -q jag -p standard
#SBATCH --job-name=shard_job             # default name, override with --job-name when submitting
#SBATCH --time=14-00:00:00               # 14 days
#SBATCH --nodes=1
#SBATCH --ntasks=1                       # torchrun will spawn local workers
#SBATCH --cpus-per-task=8                # from: -c 8
#SBATCH --mem=20G                        # from: -r 20G
#SBATCH --gres=gpu:1                     # from: -g 1
#SBATCH --open-mode=append
#SBATCH --output=/nlp/scr/potsawee/workspace/tokenize-audio/yodas2-mimi/logs/default.log
#SBATCH --error=/nlp/scr/potsawee/workspace/tokenize-audio/yodas2-mimi/logs/default.log
#SBATCH --constraint=[40G|48G|80G]

set -euo pipefail

# Get shard ID from command line argument
SHARD_ID="${1:-}"

if [ -z "$SHARD_ID" ]; then
    echo "Error: Shard ID not provided"
    echo "Usage: sbatch submit/job_template.sh <shard_id>"
    echo "Example: sbatch submit/job_template.sh en006"
    exit 1
fi

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/nlp/scr/potsawee/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nlp/scr/potsawee/etc/profile.d/conda.sh" ]; then
        . "/nlp/scr/username/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nlp/scr/potsawee/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# helper env for nlprun to set environment
if [ "${ANACONDA_ENV:-}" != "" ]; then
    conda activate $ANACONDA_ENV
fi

umask 002

export PATH=$PATH:/u/nlp/bin:/usr/local/cuda
export PYTHONPATH='.'

# — HuggingFace shared cache —
export HF_HOME=/nlp/scr-sync/nlp/huggingface_hub_llms
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=<YOUR_HF_TOKEN>

# Put cache in my own directory (my job previous crashed due to /tmp/ being full)
export TMPDIR="/nlp/scr/potsawee/workspace/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export WANDB_DIR="/nlp/scr/potsawee/workspace/tmp/wandb_cache"
export WANDB_CACHE_DIR="$WANDB_DIR"
export WANDB_DATA_DIR="$WANDB_DIR"

# --- env setup (edit to match your system) ---
# If you use modules, load CUDA, etc.:
# module load cuda/12.1
# Activate your conda env
source ~/.bashrc
conda activate env04

# --- working directory (edit if needed) ---
WORKDIR=/nlp/scr/potsawee/workspace/tokenize-audio/yodas2-mimi/
cd "$WORKDIR"

# Log some basics
echo "Job ${SLURM_JOB_ID} on nodes: ${SLURM_JOB_NODELIST}"
echo "GPUs visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Processing shard: ${SHARD_ID}"

# --- launch ---
# Use srun so Slurm binds resources correctly; torchrun spawns 4 local processes.
srun --unbuffered bash -lc \
"export HF_TOKEN=<YOUR_HF_TOKEN> && python process_shard.py --shard-id ${SHARD_ID} --work-dir ./work --output-dir ./output --progress-dir ./progress --device cuda --num-workers 1 --batch-size 16 --save-every 48 --hf-repo-id potsawee/yodas2-mm"
