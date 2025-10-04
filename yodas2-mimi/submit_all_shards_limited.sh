#!/bin/bash

# Submit multiple shard processing jobs with concurrency limit
# Usage: ./submit_all_shards_limited.sh <shard_ids_file> [max_concurrent_jobs]
# Example: ./submit_all_shards_limited.sh shard_ids.txt 100
#
# The file should contain one shard ID per line:
# en001
# en002
# en003
# ...

set -euo pipefail

SHARD_IDS_FILE="${1:-}"
MAX_CONCURRENT="${2:-100}"  # Default to 100 concurrent jobs

if [ -z "$SHARD_IDS_FILE" ]; then
    echo "Error: Shard IDs file not provided"
    echo "Usage: ./submit_all_shards_limited.sh <shard_ids_file> [max_concurrent_jobs]"
    echo "Example: ./submit_all_shards_limited.sh shard_ids.txt 100"
    exit 1
fi

if [ ! -f "$SHARD_IDS_FILE" ]; then
    echo "Error: File not found: $SHARD_IDS_FILE"
    exit 1
fi

# Function to count running/pending jobs in sc-loprio partition
count_jobs() {
    squeue -u $USER -p sc-loprio -h -t RUNNING,PENDING | wc -l
}

# Read all shard IDs into an array
SHARD_IDS=()
while IFS= read -r SHARD_ID || [ -n "$SHARD_ID" ]; do
    # Skip empty lines and comments
    if [[ -z "$SHARD_ID" || "$SHARD_ID" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    # Trim whitespace
    SHARD_ID=$(echo "$SHARD_ID" | xargs)
    SHARD_IDS+=("$SHARD_ID")
done < "$SHARD_IDS_FILE"

TOTAL_SHARDS=${#SHARD_IDS[@]}
echo "Found $TOTAL_SHARDS shard(s) to submit"
echo "Maximum concurrent jobs: $MAX_CONCURRENT"
echo "============================================"

# Submit jobs with concurrency control
SUBMITTED=0
for SHARD_ID in "${SHARD_IDS[@]}"; do
    # Wait until we have space for another job
    while true; do
        CURRENT_JOBS=$(count_jobs)
        if [ "$CURRENT_JOBS" -lt "$MAX_CONCURRENT" ]; then
            break
        fi
        echo "[$SUBMITTED/$TOTAL_SHARDS] Currently $CURRENT_JOBS jobs running/pending. Waiting for slot... (checking every 10s)"
        sleep 10
    done
    
    SUBMITTED=$((SUBMITTED + 1))
    CURRENT_JOBS=$(count_jobs)
    echo "[$SUBMITTED/$TOTAL_SHARDS] Submitting shard: $SHARD_ID (current jobs: $CURRENT_JOBS)"
    ./submit_shard.sh "$SHARD_ID"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "============================================"
echo "Submitted all $SUBMITTED job(s)"
echo ""
FINAL_COUNT=$(count_jobs)
echo "Total jobs now running/pending: $FINAL_COUNT"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in: logs/"

