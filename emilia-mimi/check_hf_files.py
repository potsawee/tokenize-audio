#!/usr/bin/env python3
"""
Check if parquet files exist on HuggingFace repository based on a file list.
This helps verify which shards have been successfully processed and uploaded.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set, Tuple

from huggingface_hub import HfApi
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_file_list(file_list_path: Path) -> List[str]:
    """Read shard IDs from a file list."""
    if not file_list_path.exists():
        logger.error(f"File list not found: {file_list_path}")
        sys.exit(1)
    
    with open(file_list_path, 'r') as f:
        shard_ids = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Read {len(shard_ids)} shard IDs from {file_list_path}")
    return shard_ids


def get_hf_repo_files(hf_repo_id: str) -> Set[str]:
    """Get all files in the HuggingFace repository."""
    logger.info(f"Fetching file list from HuggingFace repo: {hf_repo_id}")
    api = HfApi()
    
    files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    logger.info(f"Found {len(files)} total files in repo")
    
    return set(files)


def check_shard_exists(
    shard_id: str,
    split: str,
    repo_files: Set[str]
) -> bool:
    """Check if a shard's parquet file exists in the repository."""
    lang = shard_id.split("-")[0]
    repo_path = f"{split}/{lang}/{shard_id}.parquet"
    return repo_path in repo_files


def check_files(
    file_list_path: Path,
    hf_repo_id: str,
    split: str,
    verbose: bool = False,
    show_missing: bool = False,
    show_existing: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Check which files from the list exist on HuggingFace.
    
    Returns:
        Tuple of (existing_shards, missing_shards)
    """
    shard_ids = read_file_list(file_list_path)
    repo_files = get_hf_repo_files(hf_repo_id)
    
    existing = []
    missing = []
    
    logger.info("Checking shard existence...")
    
    iterator = tqdm(shard_ids, desc="Checking shards") if not verbose else shard_ids
    
    for shard_id in iterator:
        lang = shard_id.split("-")[0]
        repo_path = f"{split}/{lang}/{shard_id}.parquet"
        
        if repo_path in repo_files:
            existing.append(shard_id)
            if verbose or show_existing:
                print(f"✓ {shard_id} -> {repo_path}")
        else:
            missing.append(shard_id)
            if verbose or show_missing:
                print(f"✗ {shard_id} -> {repo_path}")
    
    return existing, missing


def print_summary(
    existing: List[str],
    missing: List[str],
    file_list_path: Path
):
    """Print summary of check results."""
    total = len(existing) + len(missing)
    existing_pct = (len(existing) / total * 100) if total > 0 else 0
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"File list: {file_list_path}")
    print(f"Total shards: {total}")
    print(f"Existing on HF: {len(existing)} ({existing_pct:.1f}%)")
    print(f"Missing from HF: {len(missing)} ({100 - existing_pct:.1f}%)")
    print("=" * 80)


def save_missing_list(missing: List[str], output_path: Path):
    """Save list of missing shards to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for shard_id in missing:
            f.write(f"{shard_id}\n")
    logger.info(f"Saved {len(missing)} missing shard IDs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check if parquet files exist on HuggingFace repository based on a file list"
    )
    parser.add_argument(
        "--file-list",
        type=str,
        required=True,
        help="Path to file containing shard IDs (one per line)"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="potsawee/emilia-mm-pretrain",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Emilia",
        help="Split name (e.g., Emilia, Emilia-YODAS)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show status for each shard"
    )
    parser.add_argument(
        "--show-missing",
        action="store_true",
        help="Show only missing shards"
    )
    parser.add_argument(
        "--show-existing",
        action="store_true",
        help="Show only existing shards"
    )
    parser.add_argument(
        "--save-missing",
        type=str,
        help="Save missing shard IDs to this file"
    )
    
    args = parser.parse_args()
    
    file_list_path = Path(args.file_list)
    
    existing, missing = check_files(
        file_list_path=file_list_path,
        hf_repo_id=args.hf_repo_id,
        split=args.split,
        verbose=args.verbose,
        show_missing=args.show_missing,
        show_existing=args.show_existing
    )
    
    print_summary(existing, missing, file_list_path)
    
    if args.save_missing and missing:
        save_missing_path = Path(args.save_missing)
        save_missing_list(missing, save_missing_path)
    
    # Exit with non-zero status if there are missing files
    if missing:
        sys.exit(1)
    else:
        print("\n✓ All shards exist on HuggingFace!")
        sys.exit(0)


if __name__ == "__main__":
    main()

    # usage: python check_hf_files.py --file-list file_lists/en.txt --hf-repo-id potsawee/emilia-mm-pretrain --split Emilia

