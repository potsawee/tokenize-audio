#!/usr/bin/env python3
"""
Check if all shards from shard_ids.txt have been successfully uploaded to HF.
"""

import argparse
from pathlib import Path
from typing import List, Set

from huggingface_hub import HfApi, list_repo_files


def load_shard_ids(shard_ids_file: Path) -> List[str]:
    """Load shard IDs from file, ignoring comments and empty lines."""
    shard_ids = []
    with open(shard_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            shard_ids.append(line)
    return shard_ids


def get_uploaded_shards(repo_id: str) -> Set[str]:
    """Get set of shards that have been uploaded to HF."""
    try:
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        uploaded_shards = set()
        for file in files:
            # Extract shard_id from paths like "en001/00000000.parquet"
            if '/' in file:
                shard_id = file.split('/')[0]
                uploaded_shards.add(shard_id)
        return uploaded_shards
    except Exception as e:
        print(f"Error listing files from {repo_id}: {e}")
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Check if all shards are uploaded to HF"
    )
    parser.add_argument(
        "--shard-ids-file",
        type=str,
        default="shard_ids.txt",
        help="Path to shard_ids.txt file"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="potsawee/yodas2-mimi-asr",
        help="HuggingFace dataset repo ID to check"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Checking upload status for yodas2-mimi-asr dataset")
    print("=" * 80)
    print(f"Repository: {args.repo_id}")
    print(f"Shard IDs file: {args.shard_ids_file}")
    print()
    
    # Load expected shard IDs
    shard_ids_file = Path(args.shard_ids_file)
    expected_shards = set(load_shard_ids(shard_ids_file))
    print(f"Expected shards: {len(expected_shards)}")
    
    # Get uploaded shards
    print("Fetching uploaded shards from HuggingFace...")
    uploaded_shards = get_uploaded_shards(args.repo_id)
    print(f"Uploaded shards: {len(uploaded_shards)}")
    print()
    
    # Find missing and extra shards
    missing_shards = expected_shards - uploaded_shards
    extra_shards = uploaded_shards - expected_shards
    
    # Print results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    
    if not missing_shards:
        print("✓ All shards successfully uploaded!")
    else:
        print(f"✗ Missing {len(missing_shards)} shards:")
        for shard_id in sorted(missing_shards):
            print(f"  - {shard_id}")
    
    print()
    
    if extra_shards:
        print(f"⚠ Found {len(extra_shards)} extra shards (not in shard_ids.txt):")
        for shard_id in sorted(extra_shards):
            print(f"  - {shard_id}")
    
    print()
    print("=" * 80)
    print(f"Summary: {len(uploaded_shards)}/{len(expected_shards)} shards uploaded")
    print("=" * 80)
    
    # Exit with error code if there are missing shards
    if missing_shards:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()

