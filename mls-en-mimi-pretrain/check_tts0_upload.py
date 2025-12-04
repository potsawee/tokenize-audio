#!/usr/bin/env python3
"""Check if all shards are successfully uploaded to HuggingFace."""

from huggingface_hub import HfApi

TARGET_REPO_ID = "potsawee/mls-en-mm-tts0"
TOTAL_SHARDS = 183  # 0 to 182 inclusive


def main():
    api = HfApi()
    
    print(f"Checking repository: {TARGET_REPO_ID}")
    print(f"Expected shards: {TOTAL_SHARDS} (0-182)")
    print("-" * 60)
    
    # Get all files in the repo
    repo_files = set(api.list_repo_files(repo_id=TARGET_REPO_ID, repo_type="dataset"))
    
    missing_shards = []
    found_shards = []
    
    for shard_idx in range(TOTAL_SHARDS):
        shard_filename = f"train-{shard_idx:04d}-of-0182.parquet"
        repo_path = f"data/{shard_filename}"
        
        if repo_path in repo_files:
            found_shards.append(shard_idx)
        else:
            missing_shards.append(shard_idx)
    
    print(f"Found: {len(found_shards)}/{TOTAL_SHARDS} shards")
    
    if missing_shards:
        print(f"\nMissing {len(missing_shards)} shards:")
        for idx in missing_shards:
            print(f"  - train-{idx:04d}-of-0182.parquet")
    else:
        print("\n✓ All shards successfully uploaded!")


if __name__ == "__main__":
    main()
    # python check_tts0_upload.py
