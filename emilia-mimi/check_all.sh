#!/bin/bash

# Convenient wrapper to check all language/split completeness
# Usage: 
#   ./check_all.sh                    # Basic check
#   ./check_all.sh -v                 # Verbose (show missing shards)
#   ./check_all.sh --save-missing missing_shards/  # Save missing lists

python check_all_completeness.py \
    --hf-repo-id potsawee/emilia-mm-pretrain \
    --file-lists-dir file_lists \
    --languages en,de,fr,ja,ko,zh \
    --splits Emilia,Emilia-YODAS \
    "$@"

