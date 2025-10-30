#!/bin/bash
#
# Write shard paths that are NOT done yet (not started or in progress)
# in the same format as file_lists/train_files.txt.
#
# Usage:
#   ./write_pending_shards.sh [OUTPUT_FILE]
# Example:
#   ./write_pending_shards.sh file_lists/pending_files.txt
#
# Notes:
# - Uses the same PROGRESS_DIR and FILE_LIST as your summary script.
# - "Done" means processed_count == total_count and total_count > 0.

PROGRESS_DIR="/sphinx/u/salt-checkpoints/mls-mm-pretrain/progress"
FILE_LIST="file_lists/train_files.txt"
OUTPUT_FILE="${1:-file_lists/pending_files.txt}"

if [ ! -f "$FILE_LIST" ]; then
    echo "Error: File not found: $FILE_LIST" >&2
    exit 1
fi

# Start fresh
> "$OUTPUT_FILE" || { echo "Error: Cannot write to $OUTPUT_FILE" >&2; exit 1; }

TOTAL_SHARDS=0
PENDING_SHARDS=0

while IFS= read -r FILE_PATH || [ -n "$FILE_PATH" ]; do
    # Skip empty lines and comments
    if [[ -z "$FILE_PATH" || "$FILE_PATH" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Trim whitespace
    FILE_PATH=$(echo "$FILE_PATH" | xargs)

    # Expect parquet file; derive shard id
    SHARD_ID=$(basename "$FILE_PATH" .parquet)

    TOTAL_SHARDS=$((TOTAL_SHARDS + 1))

    PROGRESS_FILE="${PROGRESS_DIR}/progress_${SHARD_ID}.json"

    # Default: treat as pending unless proven completed
    IS_COMPLETED=0

    if [ -f "$PROGRESS_FILE" ]; then
        # Extract numbers robustly (first match). Avoid needing jq.
        PROCESSED=$(grep -m1 -oP '"processed_count"\s*:\s*\K\d+' "$PROGRESS_FILE")
        TOTAL=$(grep -m1 -oP '"total_count"\s*:\s*\K\d+' "$PROGRESS_FILE")

        if [ -n "$PROCESSED" ] && [ -n "$TOTAL" ]; then
            if [ "$TOTAL" -gt 0 ] && [ "$PROCESSED" -eq "$TOTAL" ]; then
                IS_COMPLETED=1
            fi
        fi
    fi

    if [ "$IS_COMPLETED" -eq 0 ]; then
        echo "$FILE_PATH" >> "$OUTPUT_FILE"
        PENDING_SHARDS=$((PENDING_SHARDS + 1))
    fi
done < "$FILE_LIST"

echo "Wrote $PENDING_SHARDS pending shard path(s) to: $OUTPUT_FILE (from $TOTAL_SHARDS total)"