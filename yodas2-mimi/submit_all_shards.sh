#!/bin/bash

# Submit multiple shard processing jobs from a file
# Usage: ./submit_all_shards.sh <shard_ids_file>
# Example: ./submit_all_shards.sh shard_ids.txt
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
    echo "Usage: ./submit_all_shards.sh <shard_ids_file>"
    echo "Example: ./submit_all_shards.sh shard_ids.txt"
    exit 1
fi

if [ ! -f "$SHARD_IDS_FILE" ]; then
    echo "Error: File not found: $SHARD_IDS_FILE"
    exit 1
fi

# Count total shards
TOTAL_SHARDS=$(grep -v '^[[:space:]]*$' "$SHARD_IDS_FILE" | grep -v '^#' | wc -l)
echo "Found $TOTAL_SHARDS shard(s) to submit"
echo "============================================"

# Read file line by line and submit jobs
COUNT=0
while IFS= read -r SHARD_ID || [ -n "$SHARD_ID" ]; do
    # Skip empty lines and comments
    if [[ -z "$SHARD_ID" || "$SHARD_ID" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Trim whitespace
    SHARD_ID=$(echo "$SHARD_ID" | xargs)
    
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL_SHARDS] Submitting shard: $SHARD_ID"
    ./submit_shard.sh "$SHARD_ID"
    echo ""
done < "$SHARD_IDS_FILE"

echo "============================================"
echo "Submitted $COUNT job(s) successfully"
echo ""
echo "Check job status with: squeue -u \$USER"
echo "Monitor logs in: logs/"

