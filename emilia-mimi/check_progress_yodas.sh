#!/bin/bash

# Check progress for Emilia-YODAS shards
# Usage: ./check_progress_yodas.sh [options]
# Example: ./check_progress_yodas.sh --verbose

cd "$(dirname "$0")"

python monitor_progress.py \
    --shard-list file_lists/en_yodas.txt \
    --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress-yodas \
    --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work-yodas "$@"

