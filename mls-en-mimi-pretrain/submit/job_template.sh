#!/bin/bash
#SBATCH --account=nlp
#SBATCH --partition=sc-loprio
#SBATCH --job-name=shard_job
#SBATCH --time=14-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --output=/sphinx/u/salt-checkpoints/mls-mm-pretrain/logs/default.log
#SBATCH --error=/sphinx/u/salt-checkpoints/mls-mm-pretrain/logs/default.log
#SBATCH --constraint=[24G|40G|48G|80G]
#SBATCH --exclude=tiger-hgx-1

set -euo pipefail

SHARD_ID="${1:-}"

if [ -z "$SHARD_ID" ]; then
    echo "Error: Shard ID not provided"
    echo "Usage: sbatch submit/job_template.sh <shard_id>"
    echo "Example: sbatch submit/job_template.sh train-00000-of-01416"
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
WORK_DIR="/sphinx/u/salt-checkpoints/mls-mm-pretrain/work"
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
WORKDIR=/nlp/scr/potsawee/workspace/tokenize-audio/mls-en-mimi-pretrain/
cd "$WORKDIR"

# Log some basics
echo "Job ${SLURM_JOB_ID} on nodes: ${SLURM_JOB_NODELIST}"
echo "GPUs visible: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Processing shard: ${SHARD_ID}"

# Launch the processing script
srun --unbuffered bash -lc \
"python process_shard.py \
    --shard-id ${SHARD_ID} \
    --work-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/work \
    --output-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train \
    --progress-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/progress \
    --device cuda"

