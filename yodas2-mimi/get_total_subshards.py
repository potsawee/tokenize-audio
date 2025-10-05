#!/usr/bin/env python3
"""
Query HuggingFace to get the total number of sub-shards for each shard.
This creates a cache file to avoid repeated API calls.
"""

import argparse
import json
import time
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm


def get_subshard_count(api: HfApi, shard_id: str) -> int:
    """Get the number of sub-shards for a given shard by counting audio tar.gz files."""
    try:
        path = f"data/{shard_id}/audio"
        files = list(api.list_repo_tree(
            repo_id="espnet/yodas2",
            repo_type="dataset",
            path_in_repo=path,
        ))
        # Count .tar.gz files (sub-shards)
        tar_files = [f for f in files if hasattr(f, 'rfilename') and f.rfilename.endswith('.tar.gz')]
        return len(tar_files)
    except Exception as e:
        print(f"  Warning: Could not get count for {shard_id}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Get total sub-shard counts from HuggingFace")
    parser.add_argument("--shard-ids-file", type=str, default="shard_ids.txt", help="File containing shard IDs")
    parser.add_argument("--output", type=str, default="subshard_counts.json", help="Output cache file")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh even if cache exists")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    
    # Load existing cache if it exists and not forcing refresh
    if output_path.exists() and not args.force_refresh:
        print(f"Loading existing cache from {output_path}")
        with open(output_path, 'r') as f:
            cache = json.load(f)
        print(f"Loaded counts for {len(cache)} shards")
        return cache
    
    # Read shard IDs
    with open(args.shard_ids_file, 'r') as f:
        shard_ids = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Querying HuggingFace for {len(shard_ids)} shards...")
    print("This may take a few minutes due to API rate limiting...")
    
    api = HfApi()
    counts = {}
    
    for shard_id in tqdm(shard_ids, desc="Querying shards"):
        count = get_subshard_count(api, shard_id)
        counts[shard_id] = count
        time.sleep(0.3)  # Rate limiting
    
    # Save cache
    with open(output_path, 'w') as f:
        json.dump(counts, f, indent=2)
    
    print(f"\nSaved counts to {output_path}")
    
    # Print summary
    total_subshards = sum(counts.values())
    shards_with_data = sum(1 for c in counts.values() if c > 0)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total shards: {len(counts)}")
    print(f"  Shards with data: {shards_with_data}")
    print(f"  Total sub-shards: {total_subshards:,}")
    print(f"  Average per shard: {total_subshards / shards_with_data:.1f}")
    print(f"  Max sub-shards: {max(counts.values())}")
    print(f"  Min sub-shards: {min(c for c in counts.values() if c > 0)}")
    print("=" * 60)
    
    return counts


if __name__ == "__main__":
    main()

