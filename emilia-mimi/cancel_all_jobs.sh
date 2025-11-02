#!/bin/bash

# Cancel all Emilia shard processing jobs
# Usage: ./cancel_all_jobs.sh
#
# This will cancel all jobs with names matching pattern: {lang}_{number} (e.g., en_0, de_123)

set -euo pipefail

echo "============================================"
echo "Cancel All Emilia Shard Jobs"
echo "============================================"
echo ""

# Get list of all emilia jobs (pattern: en_0, de_5, etc.)
JOBS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep -E "[a-z]{2}_[0-9]+" || echo "")

if [ -z "$JOBS" ]; then
    echo "No emilia jobs found in queue (pattern: {lang}_{number})."
    exit 0
fi

# Count jobs
JOB_COUNT=$(echo "$JOBS" | wc -l)

echo "Found $JOB_COUNT job(s) to cancel:"
echo ""
echo "$JOBS" | head -10
if [ "$JOB_COUNT" -gt 10 ]; then
    echo "... and $((JOB_COUNT - 10)) more"
fi
echo ""

# Ask for confirmation
read -p "Are you sure you want to cancel ALL $JOB_COUNT jobs? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled. No jobs were terminated."
    exit 0
fi

echo ""
echo "Cancelling jobs..."
echo ""

# Extract job IDs and cancel them one by one
JOB_IDS=$(echo "$JOBS" | awk '{print $1}')
CANCELLED=0
for JOB_ID in $JOB_IDS; do
    scancel "$JOB_ID"
    CANCELLED=$((CANCELLED + 1))
    if [ $((CANCELLED % 20)) -eq 0 ]; then
        echo "  Cancelled $CANCELLED/$JOB_COUNT jobs..."
    fi
done

# Wait a moment for cancellation to process
sleep 2

# Check remaining jobs
REMAINING=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE "^[a-z]{2}_[0-9]+" || echo "0")

echo "============================================"
echo "Cancellation Complete!"
echo "============================================"
echo "Jobs cancelled: $((JOB_COUNT - REMAINING))"
echo "Jobs remaining: $REMAINING"
echo ""

if [ "$REMAINING" -gt 0 ]; then
    echo "Note: Some jobs may still be terminating."
    echo "Check again with: squeue -u \$USER"
fi


