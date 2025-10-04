#!/bin/bash

# Cancel all jobs in the sc-loprio partition
# Usage: ./cancel_all_jobs.sh [OPTIONS]
# Options:
#   --pending    Cancel only pending jobs
#   --running    Cancel only running jobs
#   --all        Cancel all jobs (default)

set -euo pipefail

STATE="${1:-all}"

echo "Checking jobs for user: $USER in partition sc-loprio"
echo "============================================"

# Show current jobs before canceling
echo "Current jobs:"
squeue -u $USER -p sc-loprio -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" || echo "No jobs found"
echo ""

# Count jobs
JOB_COUNT=$(squeue -u $USER -p sc-loprio -h | wc -l)

if [ "$JOB_COUNT" -eq 0 ]; then
    echo "No jobs to cancel."
    exit 0
fi

echo "Found $JOB_COUNT job(s) to cancel"
echo ""

# Ask for confirmation
read -p "Are you sure you want to cancel these jobs? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled. No jobs were terminated."
    exit 0
fi

echo ""
echo "Cancelling jobs..."

case "$STATE" in
    --pending)
        scancel -u $USER -p sc-loprio --state=PENDING
        echo "Cancelled all PENDING jobs in sc-loprio"
        ;;
    --running)
        scancel -u $USER -p sc-loprio --state=RUNNING
        echo "Cancelled all RUNNING jobs in sc-loprio"
        ;;
    --all|*)
        scancel -u $USER -p sc-loprio
        echo "Cancelled all jobs in sc-loprio"
        ;;
esac

echo ""
echo "Remaining jobs:"
squeue -u $USER -p sc-loprio -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" || echo "No jobs remaining"

