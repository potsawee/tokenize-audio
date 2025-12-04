#!/usr/bin/env python3
"""
Build mls-en-mm-tts0 dataset from mls-en-mm-pretrain.

This script:
1. Loops over all shards (train-0000-of-0182 to train-0182-of-0182)
2. For each shard, checks if it's already processed in the target HF repo
3. Downloads unprocessed shards from the source HF repo
4. Filters to keep only "type1" examples (id ends with "_type1")
5. Removes "_type1" suffix from the id column
6. Processes text by appending "[0]" after <|text_start|>
7. Uploads the filtered shard to the target HF repo
8. Cleans up temporary files
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

SOURCE_REPO_ID = "potsawee/mls-en-mm-pretrain"
TARGET_REPO_ID = "potsawee/mls-en-mm-tts0"
TOTAL_SHARDS = 183  # 0 to 182 inclusive


def file_exists_in_repo(repo_id: str, file_path: str) -> bool:
    """
    Check if a file exists in the HuggingFace repository.
    
    Args:
        repo_id: HuggingFace repository ID
        file_path: Path to file in repository (e.g., "data/train-0000-of-0182.parquet")
        
    Returns:
        True if file exists, False otherwise
    """
    api = HfApi()
    repo_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset"
    )
    return file_path in repo_files


def get_shard_filename(shard_idx: int) -> str:
    """Get the parquet filename for a given shard index."""
    return f"train-{shard_idx:04d}-of-0182.parquet"


def process_shard(
    shard_idx: int,
    work_dir: Path,
    source_repo_id: str = SOURCE_REPO_ID,
    target_repo_id: str = TARGET_REPO_ID,
):
    """
    Process a single shard: download, filter, transform, and upload.
    
    Args:
        shard_idx: Shard index (0-182)
        work_dir: Working directory for temporary files
        source_repo_id: Source HuggingFace repository ID
        target_repo_id: Target HuggingFace repository ID
    """
    shard_filename = get_shard_filename(shard_idx)
    repo_path = f"data/{shard_filename}"
    
    logger.info(f"Processing shard {shard_idx}: {shard_filename}")
    
    # Create work directory for this shard
    shard_work_dir = work_dir / f"shard_{shard_idx:04d}"
    shard_work_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download the parquet file from source repo
    logger.info(f"Downloading {repo_path} from {source_repo_id}")
    local_parquet_path = hf_hub_download(
        repo_id=source_repo_id,
        filename=repo_path,
        repo_type="dataset",
        local_dir=shard_work_dir,
    )
    logger.info(f"Downloaded to {local_parquet_path}")
    
    # Step 2: Load and filter the parquet file
    logger.info("Loading parquet file")
    df = pd.read_parquet(local_parquet_path)
    original_count = len(df)
    logger.info(f"Loaded {original_count} rows")
    
    # Filter to keep only type1 examples
    df_filtered = df[df['id'].str.endswith('_type1')].copy()
    filtered_count = len(df_filtered)
    logger.info(f"Filtered to {filtered_count} type1 rows (removed {original_count - filtered_count} type2 rows)")
    
    # Step 3: Remove "_type1" suffix from id column
    df_filtered['id'] = df_filtered['id'].str.replace('_type1$', '', regex=True)
    
    # Step 4: Process text - append "[0]" after <|text_start|>
    df_filtered['text'] = df_filtered['text'].str.replace(
        '<|text_start|>', 
        '<|text_start|>[0]', 
        regex=False
    )
    
    # Step 5: Save the processed parquet file
    output_parquet_path = shard_work_dir / shard_filename
    df_filtered.to_parquet(output_parquet_path, index=False)
    logger.info(f"Saved processed parquet to {output_parquet_path}")
    
    # Step 6: Upload to target repo
    logger.info(f"Uploading to {target_repo_id} as {repo_path}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(output_parquet_path),
        path_in_repo=repo_path,
        repo_id=target_repo_id,
        repo_type="dataset",
        commit_message=f"Add {shard_filename}",
    )
    logger.info(f"Successfully uploaded {shard_filename}")
    
    # Step 7: Clean up
    logger.info(f"Cleaning up work directory: {shard_work_dir}")
    shutil.rmtree(shard_work_dir, ignore_errors=True)
    logger.info("Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Build mls-en-mm-tts0 dataset from mls-en-mm-pretrain"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work_tts0",
        help="Working directory for temporary files"
    )
    parser.add_argument(
        "--source-repo",
        type=str,
        default=SOURCE_REPO_ID,
        help=f"Source HuggingFace repository ID (default: {SOURCE_REPO_ID})"
    )
    parser.add_argument(
        "--target-repo",
        type=str,
        default=TARGET_REPO_ID,
        help=f"Target HuggingFace repository ID (default: {TARGET_REPO_ID})"
    )
    parser.add_argument(
        "--start-shard",
        type=int,
        default=0,
        help="Start shard index (default: 0)"
    )
    parser.add_argument(
        "--end-shard",
        type=int,
        default=182,
        help="End shard index inclusive (default: 182)"
    )
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Source repo: {args.source_repo}")
    logger.info(f"Target repo: {args.target_repo}")
    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Processing shards {args.start_shard} to {args.end_shard}")
    
    processed_count = 0
    skipped_count = 0
    
    for shard_idx in range(args.start_shard, args.end_shard + 1):
        shard_filename = get_shard_filename(shard_idx)
        repo_path = f"data/{shard_filename}"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Shard {shard_idx}/{args.end_shard}: {shard_filename}")
        logger.info(f"{'='*60}")
        
        # Check if already processed
        if file_exists_in_repo(args.target_repo, repo_path):
            logger.info(f"Skipping - already exists in target repo: {repo_path}")
            skipped_count += 1
            continue
        
        # Process the shard
        process_shard(
            shard_idx=shard_idx,
            work_dir=work_dir,
            source_repo_id=args.source_repo,
            target_repo_id=args.target_repo,
        )
        processed_count += 1
    
    logger.info(f"\n{'='*60}")
    logger.info("All shards completed!")
    logger.info(f"  Processed: {processed_count}")
    logger.info(f"  Skipped (already exists): {skipped_count}")
    logger.info(f"  Total: {processed_count + skipped_count}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
    
    # Usage:
    # python build_mls_en_mm_tts0.py --work-dir /sphinx/u/salt-checkpoints/mls-en-mm-tts0
    # 
    # To process specific range of shards:
    # python build_mls_en_mm_tts0.py --work-dir /sphinx/u/salt-checkpoints/mls-en-mm-tts0 --start-shard 0 --end-shard 182

