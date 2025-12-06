#!/bin/bash

# Wrapper script to submit a YODAS shard processing job with dynamic shard ID
# Usage: ./submit_shard_yodas.sh <shard_id>
# Example: ./submit_shard_yodas.sh EN-B001361

set -euo pipefail

SHARD_ID="${1:-}"

if [ -z "$SHARD_ID" ]; then
    echo "Error: Shard ID not provided"
    echo "Usage: ./submit_shard_yodas.sh <shard_id>"
    echo "Example: ./submit_shard_yodas.sh EN-B001361"
    exit 1
fi

# Create job name that fits in 8 chars (EN-B001361 -> yn_1361)
# Use "y" prefix for YODAS to distinguish from regular Emilia
LANG_CODE=$(echo "$SHARD_ID" | cut -c1-2 | tr '[:upper:]' '[:lower:]')
NUMERIC_ID=$(echo "$SHARD_ID" | grep -oP '\d+' | head -1 | sed 's/^0*//')
JOB_NAME="y${LANG_CODE:0:1}_${NUMERIC_ID}"

# Set log directory
LOG_DIR="/sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/logs-yodas"
LOG_FILE="${LOG_DIR}/${SHARD_ID}.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Submit job with dynamic parameters
sbatch \
    --job-name="${JOB_NAME}" \
    --output="${LOG_FILE}" \
    --error="${LOG_FILE}" \
    submit/job_template_yodas_fix.sh "${SHARD_ID}"
SBATCH_EXIT=$?

if [ $SBATCH_EXIT -eq 0 ]; then
    echo "Submitted job: ${JOB_NAME} for shard: ${SHARD_ID}"
    echo "Log file: ${LOG_FILE}"
else
    echo "Error: Failed to submit job ${JOB_NAME} (exit code: $SBATCH_EXIT)" >&2
    exit 1
fi



