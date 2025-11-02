#!/bin/bash

# Quick script to check processing progress
# Usage: ./check_progress.sh [--verbose]

VERBOSE=""
if [[ "$1" == "--verbose" || "$1" == "-v" ]]; then
    VERBOSE="--verbose"
fi

python monitor_progress.py \
    --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress \
    --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work \
    $VERBOSE

