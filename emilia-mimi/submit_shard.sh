#!/bin/bash

# Wrapper script to submit a shard processing job with dynamic shard ID
# Usage: ./submit_shard.sh <shard_id>
# Example: ./submit_shard.sh EN-B000000

set -euo pipefail

SHARD_ID="${1:-}"

if [ -z "$SHARD_ID" ]; then
    echo "Error: Shard ID not provided"
    echo "Usage: ./submit_shard.sh <shard_id>"
    echo "Example: ./submit_shard.sh EN-B000000"
    exit 1
fi

# Create job name that fits in 8 chars (EN-B000000 -> en_0 or en_1234)
LANG_CODE=$(echo "$SHARD_ID" | cut -c1-2 | tr '[:upper:]' '[:lower:]')
NUMERIC_ID=$(echo "$SHARD_ID" | grep -oP '\d+' | head -1 | sed 's/^0*//')
JOB_NAME="${LANG_CODE}_${NUMERIC_ID}"

# Set log directory
LOG_DIR="/sphinx/u/salt-checkpoints/emilia-mm-pretrain/logs"
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


