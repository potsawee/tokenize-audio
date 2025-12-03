#!/bin/bash

# Submit YODAS shard processing jobs with a limit on concurrent jobs
# Usage: 
#   ./submit_all_shards_conversational.sh [max_concurrent_jobs] [shard_list_file]
#   ./submit_all_shards_conversational.sh [shard_list_file]  (defaults to 100 max jobs)
# Examples: 
#   ./submit_all_shards_conversational.sh 100 file_lists/en_yodas.txt
#   ./submit_all_shards_conversational.sh file_lists/en_yodas.txt
#   ./submit_all_shards_conversational.sh 50 file_lists/en_yodas.txt
#
# This script reads from file_lists/en_yodas.txt (or specified file) and submits jobs
# while maintaining a maximum number of concurrent jobs in the queue.

set -euo pipefail

# Parse arguments intelligently
# If first arg is a file, use it as shard list, otherwise treat as max jobs
if [ -f "${1:-}" ]; then
    SHARD_FILES_LIST="$1"
    MAX_CONCURRENT_JOBS="${2:-100}"
elif [ -n "${1:-}" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_CONCURRENT_JOBS="$1"
    SHARD_FILES_LIST="${2:-file_lists/en_yodas.txt}"
else
    MAX_CONCURRENT_JOBS="${1:-100}"
    SHARD_FILES_LIST="${2:-file_lists/en_yodas.txt}"
fi

CHECK_INTERVAL=30  # Seconds between queue checks

if [ ! -f "$SHARD_FILES_LIST" ]; then
    echo "Error: File not found: $SHARD_FILES_LIST"
    exit 1
fi

echo "============================================"
echo "Batch Conversational Job Submission with Concurrency Limit"
echo "============================================"
echo "Shard list: $SHARD_FILES_LIST"
echo "Max concurrent jobs: $MAX_CONCURRENT_JOBS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo ""

# Function to count current Conversational jobs in queue (job names like ye_1361, yd_5, etc.)
count_current_jobs() {
    local count=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE "^y[a-z]_[0-9]+" 2>/dev/null || true)
    echo "${count:-0}"
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
done < "$SHARD_FILES_LIST"

TOTAL_SHARDS=${#SHARD_IDS[@]}
echo "Found $TOTAL_SHARDS total shards to process"
echo "============================================"
echo ""

# Get currently running/pending Conversational jobs (job names like ye_1361, yd_5, etc.)
echo "Checking which shards are already in queue..."
CURRENT_JOBS=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -E "^y[a-z]_[0-9]+" || echo "")

if [ -z "$CURRENT_JOBS" ]; then
    CURRENT_JOBS_ARRAY=()
else
    mapfile -t CURRENT_JOBS_ARRAY <<< "$CURRENT_JOBS"
fi

# Filter out shards that are already in queue
SHARDS_TO_SUBMIT=()
for SHARD_ID in "${SHARD_IDS[@]}"; do
    # Create job name (EN-B001361 -> ye_1361)
    LANG_CODE=$(echo "$SHARD_ID" | cut -c1-2 | tr '[:upper:]' '[:lower:]')
    NUMERIC_ID=$(echo "$SHARD_ID" | grep -oP '\d+' | head -1 | sed 's/^0*//')
    JOB_NAME="y${LANG_CODE:0:1}_${NUMERIC_ID}"
    
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
    LANG_CODE=$(echo "$SHARD_ID" | cut -c1-2 | tr '[:upper:]' '[:lower:]')
    NUMERIC_ID=$(echo "$SHARD_ID" | grep -oP '\d+' | head -1 | sed 's/^0*//')
    JOB_NAME="y${LANG_CODE:0:1}_${NUMERIC_ID}"
    echo "[$(date '+%H:%M:%S')] Submitting $JOB_NAME ($SHARD_ID) (Queue: $CURRENT_COUNT/$MAX_CONCURRENT_JOBS)"
    
    # Run submit_shard_conversational.sh and check if it succeeds
    if ./submit_shard_conversational.sh "$SHARD_ID" > /dev/null 2>&1; then
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
echo "  watch -n 10 'squeue -u \$USER | grep -E \"^y[a-z]_[0-9]+\"'"
echo ""
echo "Check progress with:"
echo "  python monitor_progress.py --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-conversational/progress-yodas --work-dir /sphinx/u/salt-checkpoints/emilia-mm-conversational/work-yodas"




