#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sc-loprio            # from: -q sc-loprio, jag-standard
#SBATCH --job-name=pretrain_prep         # default name, override with --job-name when submitting
#SBATCH --time=14-00:00:00               # 14 days
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                # from: -c 4
#SBATCH --mem=30G                        # from: -r 30G (might need more if accumulating large batches)
#SBATCH --open-mode=append
#SBATCH --output=/sphinx/u/salt-checkpoints/yodas2-mm-pretrain/logs/default.log
#SBATCH --error=/sphinx/u/salt-checkpoints/yodas2-mm-pretrain/logs/default.log
# Note: These defaults are overridden by submit_shard.sh with shard-specific log files
# No GPU needed for this preprocessing job
#SBATCH --exclude=tiger-hgx-1,cocoflops[1,2],cocoflops-hgx-1,pasteur-hgx-1,jagupard[19,20,26,27,30,31],iliad[1,2,3,4,5],tiger[1,2,3,4],iliad-hgx-1,pasteur[1,2,3,4,5,6],sphinx[1,2,3,4,5,6]

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

# — HuggingFace cache (using local directory with more space) —
export HF_HOME=/sphinx/u/salt-checkpoints/yodas2-mm-pretrain/huggingface_hub_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN=<YOUR_HF_TOKEN>

# Create cache directory if it doesn't exist
mkdir -p $HF_HOME

# Put cache in my own directory (my job previous crashed due to /tmp/ being full)
export TMPDIR="/nlp/scr/potsawee/workspace/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export WANDB_DIR="/nlp/scr/potsawee/workspace/tmp/wandb_cache"
export WANDB_CACHE_DIR="$WANDB_DIR"
export WANDB_DATA_DIR="$WANDB_DIR"

# ---------------------------------
# quick fix for tmp directory being full ---
export TMPROOT=/tmp/pm574-cache   # or /tmp/$USER if no scr1
mkdir -p "$TMPROOT"/{work,output,hf-cache,logs}

export TMPDIR="$TMPROOT"
export HF_HOME="$TMPROOT/hf-cache"
export HF_HUB_CACHE="$TMPROOT/hf-cache"
export TRANSFORMERS_CACHE="$TMPROOT/hf-cache"
# ---------------------------------

# --- env setup (edit to match your system) ---
# If you use modules, load CUDA, etc.:
# module load cuda/12.1
# Activate your conda env
source ~/.bashrc
conda activate env04

# --- working directory (edit if needed) ---
WORKDIR=/nlp/scr/potsawee/workspace/tokenize-audio/pretraining-data/
cd "$WORKDIR"

# Log some basics
echo "Job ${SLURM_JOB_ID} on nodes: ${SLURM_JOB_NODELIST}"
echo "Processing shard: ${SHARD_ID}"

# --- launch ---
# No GPU needed for this preprocessing job, just CPU
srun --unbuffered --cpu-bind=none bash -lc \
"export HF_TOKEN=<YOUR_HF_TOKEN> && python prepare_pretraining_data.py \
--shard-id ${SHARD_ID} \
--source-repo-id potsawee/yodas2-mm \
--dest-repo-id potsawee/yodas2-mm-pretrain \
--work-dir /sphinx/u/salt-checkpoints/yodas2-mm-pretrain/work \
--output-dir /sphinx/u/salt-checkpoints/yodas2-mm-pretrain/output \
--progress-dir /sphinx/u/salt-checkpoints/yodas2-mm-pretrain/progress \
--num-codebooks 8 \
--codebook-size 2048 \
--unicode-offset 0xe000 \
--parquet-batch-size 10000 \
--upload-batch-size 1 \
--checkpoint-interval 5 \
--max-consecutive-missing 5"

