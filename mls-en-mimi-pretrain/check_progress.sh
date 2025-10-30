#!/bin/bash

# Check progress of all shards
# Usage: ./check_progress.sh

PROGRESS_DIR="/sphinx/u/salt-checkpoints/mls-mm-pretrain/progress"
FILE_LIST="file_lists/train_files.txt"

if [ ! -f "$FILE_LIST" ]; then
    echo "Error: File not found: $FILE_LIST"
    exit 1
fi

echo "============================================"
echo "Shard Processing Progress Summary"
echo "============================================"
echo ""

TOTAL_SHARDS=0
COMPLETED_SHARDS=0
IN_PROGRESS_SHARDS=0
NOT_STARTED_SHARDS=0

while IFS= read -r FILE_PATH || [ -n "$FILE_PATH" ]; do
    # Skip empty lines and comments
    if [[ -z "$FILE_PATH" || "$FILE_PATH" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # Trim whitespace
    FILE_PATH=$(echo "$FILE_PATH" | xargs)
    
    # Extract shard ID
    SHARD_ID=$(basename "$FILE_PATH" .parquet)
    
    TOTAL_SHARDS=$((TOTAL_SHARDS + 1))
    
    # Check progress file
    PROGRESS_FILE="${PROGRESS_DIR}/progress_${SHARD_ID}.json"
    
    if [ -f "$PROGRESS_FILE" ]; then
        # Parse JSON to get processed_count and total_count (more precise extraction)
        PROCESSED=$(grep '"processed_count":' "$PROGRESS_FILE" | grep -oP '\d+')
        TOTAL=$(grep '"total_count":' "$PROGRESS_FILE" | grep -oP '\d+')
        
        if [ -n "$PROCESSED" ] && [ -n "$TOTAL" ]; then
            if [ "$PROCESSED" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
                COMPLETED_SHARDS=$((COMPLETED_SHARDS + 1))
            elif [ "$PROCESSED" -gt 0 ]; then
                IN_PROGRESS_SHARDS=$((IN_PROGRESS_SHARDS + 1))
            else
                NOT_STARTED_SHARDS=$((NOT_STARTED_SHARDS + 1))
            fi
        else
            NOT_STARTED_SHARDS=$((NOT_STARTED_SHARDS + 1))
        fi
    else
        NOT_STARTED_SHARDS=$((NOT_STARTED_SHARDS + 1))
    fi
done < "$FILE_LIST"

echo "Total Shards:        $TOTAL_SHARDS"
echo "Completed:           $COMPLETED_SHARDS"
echo "In Progress:         $IN_PROGRESS_SHARDS"
echo "Not Started:         $NOT_STARTED_SHARDS"
echo ""
# Use bash arithmetic instead of bc
if [ $TOTAL_SHARDS -gt 0 ]; then
    COMPLETION_RATE=$((COMPLETED_SHARDS * 100 / TOTAL_SHARDS))
    echo "Completion Rate:     ${COMPLETION_RATE}%"
else
    echo "Completion Rate:     0%"
fi
echo ""
echo "============================================"
echo ""
echo "To see detailed progress for specific shards:"
echo "  cat ${PROGRESS_DIR}/progress_<shard_id>.json"
echo ""
echo "To check running jobs:"
echo "  squeue -u \$USER"

