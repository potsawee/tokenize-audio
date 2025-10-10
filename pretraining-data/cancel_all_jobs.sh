#!/bin/bash

# Cancel all pretraining data preparation jobs in the sc-loprio partition
# Usage: ./cancel_all_jobs.sh [OPTIONS]
# Options:
#   --pending    Cancel only pending jobs
#   --running    Cancel only running jobs
#   --all        Cancel all jobs (default)

set -euo pipefail

STATE="${1:-all}"

echo "Checking pretraining data preparation jobs for user: $USER in partition sc-loprio"
echo "============================================"

# Get list of job IDs, excluding interactive jobs
JOB_IDS=$(squeue -u $USER -p sc-loprio -h -o "%i %j" | grep -v "interactive" | awk '{print $1}')

# Count jobs (excluding interactive)
JOB_COUNT=$(echo "$JOB_IDS" | grep -v "^$" | wc -l)

# Show jobs that will be canceled
echo "Jobs that will be CANCELED:"
if [ "$JOB_COUNT" -eq 0 ]; then
    echo "  (none)"
else
    squeue -u $USER -p sc-loprio -h -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" | grep -v "interactive" || echo "  (none)"
fi
echo ""

# Show interactive jobs that will be KEPT
echo "Interactive jobs that will be KEPT (not canceled):"
INTERACTIVE_JOBS=$(squeue -u $USER -p sc-loprio -h -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" | grep "interactive" || echo "")
if [ -z "$INTERACTIVE_JOBS" ]; then
    echo "  (none)"
else
    echo "$INTERACTIVE_JOBS"
fi
echo ""

if [ "$JOB_COUNT" -eq 0 ]; then
    echo "No non-interactive jobs to cancel."
    exit 0
fi

echo "============================================"
echo "Total: $JOB_COUNT job(s) to cancel (interactive sessions excluded)"
echo ""

# Ask for confirmation
read -p "Are you sure you want to cancel these jobs? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled. No jobs were terminated."
    exit 0
fi

echo ""
echo "Cancelling jobs (excluding interactive sessions)..."

# Cancel jobs based on state filter
case "$STATE" in
    --pending)
        for JOB_ID in $JOB_IDS; do
            JOB_STATE=$(squeue -j $JOB_ID -h -o "%t" 2>/dev/null || echo "")
            if [ "$JOB_STATE" = "PD" ]; then
                scancel $JOB_ID
            fi
        done
        echo "Cancelled PENDING jobs (excluding interactive)"
        ;;
    --running)
        for JOB_ID in $JOB_IDS; do
            JOB_STATE=$(squeue -j $JOB_ID -h -o "%t" 2>/dev/null || echo "")
            if [ "$JOB_STATE" = "R" ]; then
                scancel $JOB_ID
            fi
        done
        echo "Cancelled RUNNING jobs (excluding interactive)"
        ;;
    --all|*)
        for JOB_ID in $JOB_IDS; do
            scancel $JOB_ID 2>/dev/null || true
        done
        echo "Cancelled all non-interactive jobs in sc-loprio partition"
        ;;
esac

echo ""
echo "Remaining jobs:"
squeue -u $USER -p sc-loprio -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" || echo "No jobs remaining"

