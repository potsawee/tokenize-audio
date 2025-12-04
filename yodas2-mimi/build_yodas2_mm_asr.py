#!/usr/bin/env python3
"""
Build the yodas2-mimi-asr dataset by filtering type2 examples from yodas2-mimi-pretrain.

This script:
1. Reads shard IDs from shard_ids.txt
2. For each shard, checks if it's already processed in the target HF repo
3. Downloads parquet files from the source HF repo
4. Filters to keep only "type2" examples
5. Removes "_type2" suffix from the id column
6. Pushes to the new HF repo with the same directory structure
7. Cleans up temporary files
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from tqdm import tqdm


def setup_logging(log_file: Optional[Path] = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def load_shard_ids(shard_ids_file: Path) -> List[str]:
    """
    Load shard IDs from file, ignoring comments and empty lines.
    
    Args:
        shard_ids_file: Path to the shard_ids.txt file
        
    Returns:
        List of shard IDs
    """
    shard_ids = []
    with open(shard_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            shard_ids.append(line)
    
    logger.info(f"Loaded {len(shard_ids)} shard IDs from {shard_ids_file}")
    return shard_ids


def check_shard_exists_on_hf(api: HfApi, repo_id: str, shard_id: str) -> bool:
    """
    Check if a shard directory exists on HuggingFace.
    
    Args:
        api: HuggingFace API instance
        repo_id: HuggingFace repo ID
        shard_id: Shard ID to check (e.g., "en001")
        
    Returns:
        True if the shard directory exists, False otherwise
    """
    try:
        # List files in the repo and check if any file starts with shard_id/
        files = list_repo_files(repo_id=repo_id, repo_type="dataset")
        shard_prefix = f"{shard_id}/"
        for file in files:
            if file.startswith(shard_prefix):
                return True
        return False
    except Exception as e:
        logger.debug(f"Error checking if shard {shard_id} exists on HF: {e}")
        return False


def get_parquet_files_for_shard(api: HfApi, repo_id: str, shard_id: str) -> List[str]:
    """
    Get list of parquet files for a shard from HuggingFace.
    
    Args:
        api: HuggingFace API instance
        repo_id: HuggingFace repo ID
        shard_id: Shard ID (e.g., "en001")
        
    Returns:
        List of parquet file paths in the repo
    """
    files = list_repo_files(repo_id=repo_id, repo_type="dataset")
    shard_prefix = f"{shard_id}/"
    parquet_files = [f for f in files if f.startswith(shard_prefix) and f.endswith('.parquet')]
    return sorted(parquet_files)


def download_parquet_files(
    api: HfApi,
    repo_id: str,
    parquet_files: List[str],
    work_dir: Path
) -> List[Path]:
    """
    Download parquet files to the work directory.
    
    Args:
        api: HuggingFace API instance
        repo_id: HuggingFace repo ID
        parquet_files: List of parquet file paths in the repo
        work_dir: Working directory to download files to
        
    Returns:
        List of local file paths
    """
    local_paths = []
    
    for parquet_file in tqdm(parquet_files, desc="Downloading parquet files"):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=parquet_file,
                repo_type="dataset",
                local_dir=work_dir
            )
            local_paths.append(Path(local_path))
        except Exception as e:
            logger.error(f"Failed to download {parquet_file}: {e}")
            raise
    
    return local_paths


def filter_type2_examples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to keep only type2 examples and clean up the id column.
    
    Args:
        df: Input dataframe with 'id' column
        
    Returns:
        Filtered dataframe with only type2 examples and cleaned id column
    """
    # Filter to keep only type2 examples (id ends with "_type2")
    type2_mask = df['id'].str.endswith('_type2')
    df_filtered = df[type2_mask].copy()
    
    # Remove "_type2" suffix from the id column
    df_filtered['id'] = df_filtered['id'].str.replace('_type2$', '', regex=True)
    
    return df_filtered


def process_shard(
    shard_id: str,
    source_repo_id: str,
    target_repo_id: str,
    work_dir: Path,
    api: HfApi
) -> bool:
    """
    Process a single shard: download, filter, and upload.
    
    Args:
        shard_id: Shard ID (e.g., "en001")
        source_repo_id: Source HuggingFace repo ID
        target_repo_id: Target HuggingFace repo ID
        work_dir: Working directory for temporary files
        api: HuggingFace API instance
        
    Returns:
        True if successful, False otherwise
    """
    shard_work_dir = work_dir / shard_id
    shard_work_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Get list of parquet files for this shard
        logger.info(f"Getting list of parquet files for shard {shard_id}")
        parquet_files = get_parquet_files_for_shard(api, source_repo_id, shard_id)
        
        if not parquet_files:
            logger.warning(f"No parquet files found for shard {shard_id}")
            return True  # Consider this as success (nothing to process)
        
        logger.info(f"Found {len(parquet_files)} parquet files for shard {shard_id}")
        
        # Step 2: Download all parquet files
        logger.info(f"Downloading parquet files for shard {shard_id}")
        local_paths = download_parquet_files(api, source_repo_id, parquet_files, work_dir)
        
        # Step 3: Process each parquet file
        output_dir = work_dir / "output" / shard_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_original = 0
        total_filtered = 0
        
        for local_path in tqdm(local_paths, desc=f"Processing {shard_id}"):
            # Read the parquet file
            df = pd.read_parquet(local_path)
            original_count = len(df)
            total_original += original_count
            
            # Filter to type2 examples
            df_filtered = filter_type2_examples(df)
            filtered_count = len(df_filtered)
            total_filtered += filtered_count
            
            logger.debug(f"{local_path.name}: {original_count} -> {filtered_count} examples")
            
            # Save the filtered parquet file
            # Use the same filename structure
            output_path = output_dir / local_path.name
            df_filtered.to_parquet(output_path, index=False)
        
        logger.info(f"Shard {shard_id}: {total_original} -> {total_filtered} examples ({total_filtered/total_original*100:.1f}%)")
        
        # Step 4: Upload to HuggingFace
        logger.info(f"Uploading filtered shard {shard_id} to {target_repo_id}")
        
        api.upload_folder(
            folder_path=str(output_dir),
            path_in_repo=shard_id,
            repo_id=target_repo_id,
            repo_type="dataset",
            commit_message=f"Add filtered shard {shard_id} (type2 only)"
        )
        
        logger.info(f"Successfully uploaded shard {shard_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to process shard {shard_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Step 5: Cleanup
        logger.info(f"Cleaning up temporary files for shard {shard_id}")
        
        # Clean up downloaded files
        downloaded_shard_dir = work_dir / shard_id
        if downloaded_shard_dir.exists():
            shutil.rmtree(downloaded_shard_dir)
        
        # Clean up output files
        output_shard_dir = work_dir / "output" / shard_id
        if output_shard_dir.exists():
            shutil.rmtree(output_shard_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Build yodas2-mimi-asr dataset by filtering type2 examples"
    )
    # Mutually exclusive options for specifying shards
    parser.add_argument(
        "--shard-id",
        type=str,
        default=None,
        help="Process a single shard (e.g., --shard-id en001)"
    )
    parser.add_argument(
        "--shard-ids-file",
        type=str,
        default=None,
        help="Path to shard_ids.txt file (default: shard_ids.txt)"
    )
    parser.add_argument(
        "--source-repo",
        type=str,
        default="potsawee/yodas2-mimi-pretrain",
        help="Source HuggingFace dataset repo ID"
    )
    parser.add_argument(
        "--target-repo",
        type=str,
        default="potsawee/yodas2-mimi-asr",
        help="Target HuggingFace dataset repo ID"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work_asr",
        help="Working directory for temporary files"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional log file path"
    )
    parser.add_argument(
        "--start-from",
        type=str,
        default=None,
        help="Start processing from this shard ID (skip earlier shards, only works with --shard-ids-file)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    log_file = Path(args.log_file) if args.log_file else None
    logger = setup_logging(log_file)
    
    logger.info("=" * 80)
    logger.info("Building yodas2-mimi-asr dataset")
    logger.info("=" * 80)
    logger.info(f"Source repo: {args.source_repo}")
    logger.info(f"Target repo: {args.target_repo}")
    logger.info(f"Work directory: {args.work_dir}")
    
    # Initialize HuggingFace API
    api = HfApi()
    
    # Create work directory
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine shard IDs to process
    if args.shard_id:
        # Option 2: Process a single shard
        shard_ids = [args.shard_id]
        logger.info(f"Processing single shard: {args.shard_id}")
    else:
        # Option 1: Load shard IDs from file
        shard_ids_file = Path(args.shard_ids_file) if args.shard_ids_file else Path("shard_ids.txt")
        shard_ids = load_shard_ids(shard_ids_file)
        
        # Handle start-from option (only applicable when using file)
        if args.start_from:
            if args.start_from in shard_ids:
                start_idx = shard_ids.index(args.start_from)
                shard_ids = shard_ids[start_idx:]
                logger.info(f"Starting from shard {args.start_from} ({len(shard_ids)} shards remaining)")
            else:
                logger.warning(f"Start-from shard {args.start_from} not found in shard_ids.txt")
    
    # Process each shard
    processed = 0
    skipped = 0
    failed = 0
    
    for shard_id in shard_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing shard: {shard_id} ({processed + skipped + failed + 1}/{len(shard_ids)})")
        logger.info(f"{'='*60}")
        
        # Check if shard already exists in target repo
        if check_shard_exists_on_hf(api, args.target_repo, shard_id):
            logger.info(f"Shard {shard_id} already exists in target repo, skipping")
            skipped += 1
            continue
        
        # Process the shard
        success = process_shard(
            shard_id=shard_id,
            source_repo_id=args.source_repo,
            target_repo_id=args.target_repo,
            work_dir=work_dir,
            api=api
        )
        
        if success:
            processed += 1
        else:
            failed += 1
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED")
    logger.info("=" * 80)
    logger.info(f"✓ Processed: {processed} shards")
    logger.info(f"○ Skipped (already existed): {skipped} shards")
    if failed:
        logger.warning(f"✗ Failed: {failed} shards")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
    # usage (option 1 - process all shards from file):
    #   python build_yodas2_mm_asr.py --shard-ids-file shard_ids.txt --source-repo potsawee/yodas2-mm-pretrain --target-repo potsawee/yodas2-mm-asr --work-dir /sphinx/u/salt-checkpoints/yodas2-mm-asr/work_asr --log-file ./logs_asr/build_yodas2_mm_asr.log
    # usage (option 2 - process a single shard) it takes precedence over --shard-ids-file:
    #   python build_yodas2_mm_asr.py --shard-id en001 --source-repo potsawee/yodas2-mm-pretrain --target-repo potsawee/yodas2-mm-asr --work-dir /sphinx/u/salt-checkpoints/yodas2-mm-asr/work_asr

