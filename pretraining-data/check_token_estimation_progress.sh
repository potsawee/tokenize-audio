#!/bin/bash

# Check progress of token estimation jobs
# Usage: ./check_token_estimation_progress.sh

set -euo pipefail

OUTPUT_DIR="./token_stats_by_language"
TOTAL_LANGUAGES=145  # Total number of languages in stats.md

echo "=========================================="
echo "Token Estimation Progress"
echo "=========================================="

# Count completed languages
if [ -d "$OUTPUT_DIR" ]; then
    COMPLETED=$(find "$OUTPUT_DIR" -name "*.json" -type f | wc -l)
else
    COMPLETED=0
fi

REMAINING=$((TOTAL_LANGUAGES - COMPLETED))
PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL_LANGUAGES)*100}")

echo "Completed: $COMPLETED / $TOTAL_LANGUAGES ($PERCENT%)"
echo "Remaining: $REMAINING"
echo ""

# Check running jobs
RUNNING_JOBS=$(squeue -u "$USER" -h -o "%j %T" 2>/dev/null | grep -c "^est_" || echo "0")
echo "Jobs currently running/pending: $RUNNING_JOBS"
echo ""

# Show recently completed languages
if [ -d "$OUTPUT_DIR" ] && [ $COMPLETED -gt 0 ]; then
    echo "=========================================="
    echo "Recently completed (last 10):"
    echo "=========================================="
    ls -lt "$OUTPUT_DIR"/*.json 2>/dev/null | head -10 | while read -r line; do
        FILE=$(echo "$line" | awk '{print $NF}')
        LANG=$(basename "$FILE" .json)
        DATETIME=$(echo "$line" | awk '{print $6, $7, $8}')
        
        # Try to extract document count and token count from JSON
        if command -v jq &> /dev/null && [ -f "$FILE" ]; then
            DOCS=$(jq -r '.statistics.num_documents // .extrapolated_statistics.num_documents // "N/A"' "$FILE" 2>/dev/null || echo "N/A")
            TOKENS=$(jq -r '.statistics.total_tokens // .extrapolated_statistics.total_tokens // "N/A"' "$FILE" 2>/dev/null || echo "N/A")
            
            # Format tokens with commas
            if [ "$TOKENS" != "N/A" ]; then
                TOKENS=$(printf "%'d" "$TOKENS" 2>/dev/null || echo "$TOKENS")
            fi
            
            printf "%-6s %s - %s docs, %s tokens\n" "$LANG" "$DATETIME" "$DOCS" "$TOKENS"
        else
            printf "%-6s %s\n" "$LANG" "$DATETIME"
        fi
    done
fi

echo ""
echo "=========================================="
echo "Commands:"
echo "  squeue -u \$USER | grep est_     # Check running jobs"
echo "  ls $OUTPUT_DIR/*.json | wc -l    # Count completed"
echo "  ./submit_all_estimate_tokens.sh  # Submit remaining"

