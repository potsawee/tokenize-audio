#!/bin/bash

# Submit shard processing jobs with a limit on concurrent jobs
# Usage: ./submit_all_shards.sh [max_concurrent_jobs]
# Example: ./submit_all_shards.sh 100
#
# This script reads from file_lists/train_files.txt and submits jobs
# while maintaining a maximum number of concurrent jobs in the queue.

set -euo pipefail

# SHARD_FILES_LIST="file_lists/train_files.txt"
SHARD_FILES_LIST="file_lists/pending_files.txt"
MAX_CONCURRENT_JOBS="${1:-100}"  # Default to 100 if not specified
CHECK_INTERVAL=30  # Seconds between queue checks

if [ ! -f "$SHARD_FILES_LIST" ]; then
    echo "Error: File not found: $SHARD_FILES_LIST"
    exit 1
fi

echo "============================================"
echo "Batch Job Submission with Concurrency Limit"
echo "============================================"
echo "Max concurrent jobs: $MAX_CONCURRENT_JOBS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Function to count current jobs in queue
count_current_jobs() {
    local count=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^mls" 2>/dev/null || true)
    echo "${count:-0}"
}

# Read all shard IDs into an array
SHARD_IDS=()
while IFS= read -r FILE_PATH || [ -n "$FILE_PATH" ]; do
    # Skip empty lines and comments
    if [[ -z "$FILE_PATH" || "$FILE_PATH" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Trim whitespace
    FILE_PATH=$(echo "$FILE_PATH" | xargs)
    
    # Extract shard ID from path
    SHARD_ID=$(basename "$FILE_PATH" .parquet)
    SHARD_IDS+=("$SHARD_ID")
done < "$SHARD_FILES_LIST"

TOTAL_SHARDS=${#SHARD_IDS[@]}
echo "Found $TOTAL_SHARDS total shards to process"
echo "============================================"
echo ""

# Get currently running/pending jobs
echo "Checking which shards are already in queue..."
CURRENT_JOBS=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || echo "")

if [ -z "$CURRENT_JOBS" ]; then
    CURRENT_JOBS_ARRAY=()
else
    mapfile -t CURRENT_JOBS_ARRAY <<< "$CURRENT_JOBS"
fi

# Filter out shards that are already in queue
# Extract job name from shard ID for comparison
SHARDS_TO_SUBMIT=()
for SHARD_ID in "${SHARD_IDS[@]}"; do
    # Extract numeric part (train-00000-of-01416 -> mls00000)
    SHARD_NUM=$(echo "$SHARD_ID" | grep -oP 'train-\K\d+' || echo "$SHARD_ID")
    JOB_NAME="mls${SHARD_NUM}"
    
    if [[ " ${CURRENT_JOBS_ARRAY[*]} " =~ " ${JOB_NAME} " ]]; then
        echo "  ✓ $SHARD_ID ($JOB_NAME already in queue)"
    else
        SHARDS_TO_SUBMIT+=("$SHARD_ID")
    fi
done

REMAINING=${#SHARDS_TO_SUBMIT[@]}
echo ""
echo "Shards already in queue: $((TOTAL_SHARDS - REMAINING))"
echo "Shards to submit: $REMAINING"
echo ""

if [ $REMAINING -eq 0 ]; then
    echo "All shards are already in the queue. Nothing to submit."
    exit 0
fi

echo "============================================"
echo "Starting batch submission..."
echo "============================================"
echo ""

SUBMITTED=0
for SHARD_ID in "${SHARDS_TO_SUBMIT[@]}"; do
    # Check current job count
    CURRENT_COUNT=$(count_current_jobs)
    
    # Wait if we've reached the limit
    while [ "$CURRENT_COUNT" -ge "$MAX_CONCURRENT_JOBS" ]; do
        echo "[$(date '+%H:%M:%S')] Queue full ($CURRENT_COUNT/$MAX_CONCURRENT_JOBS jobs). Waiting ${CHECK_INTERVAL}s..."
        sleep $CHECK_INTERVAL
        CURRENT_COUNT=$(count_current_jobs)
    done
    
    # Submit the job
    SHARD_NUM=$(echo "$SHARD_ID" | grep -oP 'train-\K\d+' || echo "$SHARD_ID")
    JOB_NAME="mls${SHARD_NUM}"
    echo "[$(date '+%H:%M:%S')] Submitting $JOB_NAME (Queue: $CURRENT_COUNT/$MAX_CONCURRENT_JOBS)"
    
    # Run submit_shard.sh and check if it succeeds
    if ./submit_shard.sh "$SHARD_ID" > /dev/null 2>&1; then
        SUBMITTED=$((SUBMITTED + 1))
    else
        echo "  ✗ Failed to submit $JOB_NAME"
    fi
    
    # Brief pause to avoid overwhelming the scheduler
    sleep 0.5
    
    # Show progress every 10 jobs
    if [ $((SUBMITTED % 10)) -eq 0 ]; then
        echo "  → Progress: $SUBMITTED/$REMAINING submitted"
    fi
done

echo ""
echo "============================================"
echo "Submission Complete!"
echo "============================================"
echo "Total submitted: $SUBMITTED jobs"
echo "Current queue size: $(count_current_jobs)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  watch -n 10 'squeue -u \$USER | grep mls'"
echo ""
echo "Check progress with:"
echo "  ./check_progress.sh"

