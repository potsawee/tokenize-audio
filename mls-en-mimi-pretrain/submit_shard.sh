#!/bin/bash

# Wrapper script to submit a shard processing job with dynamic shard ID
# Usage: ./submit_shard.sh <shard_id>
# Example: ./submit_shard.sh train-00000-of-01416

set -euo pipefail

SHARD_ID="${1:-}"

if [ -z "$SHARD_ID" ]; then
    echo "Error: Shard ID not provided"
    echo "Usage: ./submit_shard.sh <shard_id>"
    echo "Example: ./submit_shard.sh train-00000-of-01416"
    exit 1
fi

# Extract numeric part for shorter job name (train-00000-of-01416 -> mls00000)
SHARD_NUM=$(echo "$SHARD_ID" | grep -oP 'train-\K\d+' || echo "$SHARD_ID")
JOB_NAME="mls${SHARD_NUM}"

# Set log directory
LOG_DIR="/sphinx/u/salt-checkpoints/mls-mm-pretrain/logs"
LOG_FILE="${LOG_DIR}/${SHARD_ID}.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Submit job with dynamic parameters
sbatch \
    --job-name="${JOB_NAME}" \
    --output="${LOG_FILE}" \
    --error="${LOG_FILE}" \
    submit/job_template.sh "${SHARD_ID}"
SBATCH_EXIT=$?

if [ $SBATCH_EXIT -eq 0 ]; then
    echo "Submitted job: ${JOB_NAME} for shard: ${SHARD_ID}"
    echo "Log file: ${LOG_FILE}"
else
    echo "Error: Failed to submit job ${JOB_NAME} (exit code: $SBATCH_EXIT)" >&2
    exit 1
fi

