#!/bin/bash

# Wrapper script to submit a shard processing job with dynamic shard ID
# Usage: ./submit_shard.sh <shard_id>
# Example: ./submit_shard.sh en006
#          ./submit_shard.sh th007

set -euo pipefail

SHARD_ID="${1:-}"

if [ -z "$SHARD_ID" ]; then
    echo "Error: Shard ID not provided"
    echo "Usage: ./submit_shard.sh <shard_id>"
    echo "Example: ./submit_shard.sh en006"
    echo "         ./submit_shard.sh th007"
    exit 1
fi

# Set log directory
LOG_DIR="/sphinx/u/salt-checkpoints/yodas2-mm/logs"
LOG_FILE="${LOG_DIR}/${SHARD_ID}.log"

# Submit job with dynamic parameters
sbatch \
    --job-name="${SHARD_ID}" \
    --output="${LOG_FILE}" \
    --error="${LOG_FILE}" \
    submit/job_template.sh "${SHARD_ID}"

echo "Submitted job for shard: ${SHARD_ID}"
echo "Log file: ${LOG_FILE}"

