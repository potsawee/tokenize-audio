#!/bin/bash

# Cancel all Emilia-YODAS shard processing jobs
# Usage: ./cancel_all_jobs_yodas.sh
#
# This will cancel all jobs with names matching pattern: y{lang}_{number} (e.g., ye_1361, yd_123)

set -euo pipefail

echo "============================================"
echo "Cancel All Emilia-YODAS Shard Jobs"
echo "============================================"
echo ""

# Get list of all YODAS jobs (pattern: ye_1361, yd_5, etc.)
JOBS=$(squeue -u "$USER" -h -o "%i %j" 2>/dev/null | grep -E "y[a-z]_[0-9]+" || echo "")

if [ -z "$JOBS" ]; then
    echo "No YODAS jobs found in queue (pattern: y{lang}_{number})."
    exit 0
fi

# Count jobs
JOB_COUNT=$(echo "$JOBS" | wc -l)

echo "Found $JOB_COUNT job(s) to cancel:"
echo ""
echo "$JOBS"
echo ""

# Ask for confirmation
read -p "Cancel all these jobs? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled. No jobs were terminated."
    exit 0
fi

echo ""
echo "Cancelling jobs..."
echo ""

# Cancel each job
while IFS= read -r line; do
    JOB_ID=$(echo "$line" | awk '{print $1}')
    JOB_NAME=$(echo "$line" | awk '{print $2}')
    
    echo "Cancelling job $JOB_ID ($JOB_NAME)..."
    scancel "$JOB_ID"
done <<< "$JOBS"

# Wait a moment for cancellation to process
sleep 2

# Check remaining jobs
REMAINING=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE "^y[a-z]_[0-9]+" || echo "0")

echo "============================================"
echo "Cancellation Complete!"
echo "============================================"
echo "Jobs cancelled: $((JOB_COUNT - REMAINING))"
echo "Jobs still running: $REMAINING"
echo ""

if [ "$REMAINING" -gt 0 ]; then
    echo "Some jobs may still be shutting down. Check again in a moment:"
    echo "  squeue -u \$USER | grep -E \"^y[a-z]_[0-9]+\""
fi

