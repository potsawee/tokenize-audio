#!/bin/bash

# Submit pretraining data preparation jobs that are not currently in the queue
# Usage: ./submit_missing_shards.sh <shard_ids_file>
# Example: ./submit_missing_shards.sh shard_ids.txt
#
# The file should contain one shard ID per line:
# en001
# en002
# en003
# th001
# ...

set -euo pipefail

SHARD_IDS_FILE="${1:-}"

if [ -z "$SHARD_IDS_FILE" ]; then
    echo "Error: Shard IDs file not provided"
    echo "Usage: ./submit_missing_shards.sh <shard_ids_file>"
    echo "Example: ./submit_missing_shards.sh shard_ids.txt"
    exit 1
fi

if [ ! -f "$SHARD_IDS_FILE" ]; then
    echo "Error: File not found: $SHARD_IDS_FILE"
    exit 1
fi

# Get currently running/pending jobs
echo "Checking current job queue..."
CURRENT_JOBS=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || echo "")

if [ -z "$CURRENT_JOBS" ]; then
    echo "No jobs currently in queue"
    CURRENT_JOBS_ARRAY=()
else
    # Convert to array
    mapfile -t CURRENT_JOBS_ARRAY <<< "$CURRENT_JOBS"
    echo "Found ${#CURRENT_JOBS_ARRAY[@]} job(s) currently in queue"
fi

echo "============================================"

# Read file and determine missing shards
MISSING_SHARDS=()
while IFS= read -r SHARD_ID || [ -n "$SHARD_ID" ]; do
    # Skip empty lines and comments
    if [[ -z "$SHARD_ID" || "$SHARD_ID" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Trim whitespace
    SHARD_ID=$(echo "$SHARD_ID" | xargs)
    
    # Check if shard is in current jobs
    if [[ " ${CURRENT_JOBS_ARRAY[*]} " =~ " ${SHARD_ID} " ]]; then
        echo "Skipping $SHARD_ID (already in queue)"
    else
        MISSING_SHARDS+=("$SHARD_ID")
    fi
done < "$SHARD_IDS_FILE"

echo "============================================"
echo "Found ${#MISSING_SHARDS[@]} missing shard(s) to submit"

if [ ${#MISSING_SHARDS[@]} -eq 0 ]; then
    echo "All shards are already in the queue. Nothing to submit."
    exit 0
fi

# Shuffle the missing shards array
echo "Shuffling submission order..."
if command -v shuf &> /dev/null; then
    # Use shuf command if available
    mapfile -t MISSING_SHARDS < <(printf '%s\n' "${MISSING_SHARDS[@]}" | shuf)
else
    # Fallback: Fisher-Yates shuffle
    for ((i=${#MISSING_SHARDS[@]}-1; i>0; i--)); do
        j=$((RANDOM % (i+1)))
        temp="${MISSING_SHARDS[i]}"
        MISSING_SHARDS[i]="${MISSING_SHARDS[j]}"
        MISSING_SHARDS[j]="$temp"
    done
fi

echo "============================================"

# Submit missing shards
COUNT=0
TOTAL_MISSING=${#MISSING_SHARDS[@]}
for SHARD_ID in "${MISSING_SHARDS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL_MISSING] Submitting pretraining data prep for shard: $SHARD_ID"
    ./submit_shard.sh "$SHARD_ID"
    echo ""
done

echo "============================================"
echo "Submitted $COUNT job(s) successfully"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in: /sphinx/u/salt-checkpoints/yodas2-mm-pretrain/logs/"

