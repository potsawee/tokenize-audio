#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sc-loprio
#SBATCH --job-name=conv
#SBATCH --time=14-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --output=/sphinx/u/salt-checkpoints/emilia-mm-conversational/logs-yodas/default.log
#SBATCH --error=/sphinx/u/salt-checkpoints/emilia-mm-conversational/logs-yodas/default.log
#SBATCH --constraint=[24G|40G|48G|80G]
#SBATCH --exclude=tiger-hgx-1,cocoflops2,jagupard27,jagupard30,jagupard31,tiger5,tiger7,tiger8,cocoflops-hgx-1,jagupard26,jagupard27,jagupard28,jagupard29

set -euo pipefail

SHARD_ID="${1:-}"

if [ -z "$SHARD_ID" ]; then
    echo "Error: Shard ID not provided"
    echo "Usage: sbatch submit/job_template_conversational.sh <shard_id>"
    echo "Example: sbatch submit/job_template_conversational.sh EN-B001361"
    exit 1
fi

# >>> conda initialize >>>
__conda_setup="$('/nlp/scr/potsawee/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nlp/scr/potsawee/etc/profile.d/conda.sh" ]; then
        . "/nlp/scr/potsawee/etc/profile.d/conda.sh"
    else
        export PATH="/nlp/scr/potsawee/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

if [ "${ANACONDA_ENV:-}" != "" ]; then
    conda activate $ANACONDA_ENV
fi

umask 002

export PATH=$PATH:/u/nlp/bin:/usr/local/cuda
export PYTHONPATH='.'

# HuggingFace cache - use work directory for this job to avoid quota issues
WORK_DIR="/sphinx/u/salt-checkpoints/emilia-mm-conversational/work-yodas"
export HF_HOME="${WORK_DIR}/hf_cache"
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE

# Put cache in my own directory
export TMPDIR="/nlp/scr/potsawee/workspace/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"

# Activate conda env
source ~/.bashrc
conda activate env05

# Working directory
WORKDIR=/nlp/scr/potsawee/workspace/tokenize-audio/emilia-mimi/
cd "$WORKDIR"

# Log some basics
echo "Job ${SLURM_JOB_ID} on nodes: ${SLURM_JOB_NODELIST}"
echo "GPUs visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Processing shard: ${SHARD_ID}"

# Launch the processing script
srun --unbuffered bash -lc \
"python -u process_shard_conversational.py \
    --split Emilia-YODAS \
    --shard-id ${SHARD_ID} \
    --work-dir /sphinx/u/salt-checkpoints/emilia-mm-conversational/work-yodas \
    --output-dir /sphinx/u/salt-checkpoints/emilia-mm-conversational/output-yodas \
    --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-conversational/progress-yodas \
    --hf-repo-id potsawee/emilia-mm-conversational \
    --device cuda \
    --batch-size 16 \
    --cache-interval 1024"



