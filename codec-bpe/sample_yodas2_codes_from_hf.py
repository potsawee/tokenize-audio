#!/usr/bin/env python3
"""
Sample a subset of Yodas2 dataset codes from HuggingFace for training acoustic tokenizer.

This script:
1. Reads shard IDs and subshard counts
2. Samples up to N subshards per shard (default: 5)
3. Downloads JSON files from HuggingFace
4. Extracts codes and saves as individual .npy files
5. Supports resumability via progress tracking

python sample_yodas2_codes_from_hf.py \
  --shard-ids-file ../yodas2-mimi/shard_ids.txt \
  --subshard-counts-file ../yodas2-mimi/subshard_counts.json \
  --max-subshards-per-shard 5 \
  --repo-id potsawee/yodas2-mm \
  --output-dir ./yodas2_mimi_8cb_samples \
  --subshard-list-file ./subset_subshard_ids_for_bpe.txt \
  --progress-file ./sample_progress.txt \
  --seed 42 \
  --num-target-codebooks 8

"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import requests
from tqdm import tqdm


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def read_shard_ids(shard_ids_file: Path) -> List[str]:
    """
    Read shard IDs from file, ignoring comments and empty lines.
    
    Args:
        shard_ids_file: Path to shard_ids.txt
        
    Returns:
        List of shard IDs
    """
    shard_ids = []
    with open(shard_ids_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                shard_ids.append(line)
    logger.info(f"Read {len(shard_ids)} shard IDs from {shard_ids_file}")
    return shard_ids


def read_subshard_counts(subshard_counts_file: Path) -> Dict[str, int]:
    """
    Read subshard counts from JSON file.
    
    Args:
        subshard_counts_file: Path to subshard_counts.json
        
    Returns:
        Dictionary mapping shard_id to number of subshards
    """
    with open(subshard_counts_file, 'r') as f:
        counts = json.load(f)
    logger.info(f"Read subshard counts for {len(counts)} shards from {subshard_counts_file}")
    return counts


def sample_subshards(
    shard_ids: List[str],
    subshard_counts: Dict[str, int],
    max_subshards_per_shard: int = 5,
    seed: int = 42
) -> List[Tuple[str, str]]:
    """
    Sample subshards from each shard.
    
    Args:
        shard_ids: List of shard IDs
        subshard_counts: Dictionary of subshard counts per shard
        max_subshards_per_shard: Maximum number of subshards to sample per shard
        seed: Random seed for reproducibility
        
    Returns:
        List of (shard_id, subshard_id) tuples
    """
    random.seed(seed)
    selected_subshards = []
    
    for shard_id in shard_ids:
        if shard_id not in subshard_counts:
            logger.warning(f"Shard {shard_id} not found in subshard_counts.json, skipping")
            continue
        
        num_subshards = subshard_counts[shard_id]
        
        # Generate all available subshard IDs (00000000, 00000001, ...)
        available_subshards = [f"{i:08d}" for i in range(num_subshards)]
        
        # Sample up to max_subshards_per_shard
        num_to_sample = min(max_subshards_per_shard, num_subshards)
        sampled = random.sample(available_subshards, num_to_sample)
        
        for subshard_id in sampled:
            selected_subshards.append((shard_id, subshard_id))
        
        logger.info(f"Shard {shard_id}: sampled {num_to_sample}/{num_subshards} subshards")
    
    logger.info(f"Total subshards selected: {len(selected_subshards)}")
    return selected_subshards


def save_subshard_list(subshards: List[Tuple[str, str]], output_file: Path):
    """
    Save list of selected subshards to file.
    
    Args:
        subshards: List of (shard_id, subshard_id) tuples
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for shard_id, subshard_id in subshards:
            f.write(f"{shard_id}\t{subshard_id}\n")
    logger.info(f"Saved {len(subshards)} subshard selections to {output_file}")


def load_subshard_list(input_file: Path) -> List[Tuple[str, str]]:
    """
    Load list of selected subshards from file.
    
    Args:
        input_file: Input file path
        
    Returns:
        List of (shard_id, subshard_id) tuples
    """
    subshards = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    subshards.append((parts[0], parts[1]))
    logger.info(f"Loaded {len(subshards)} subshard selections from {input_file}")
    return subshards


def download_json_from_hf(
    repo_id: str,
    shard_id: str,
    subshard_id: str,
    max_retries: int = 3
) -> List[Dict]:
    """
    Download JSON file from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "potsawee/yodas2-mm")
        shard_id: Shard ID (e.g., "en000")
        subshard_id: Subshard ID (e.g., "00000000")
        max_retries: Maximum number of retry attempts
        
    Returns:
        JSON data as list of dictionaries
    """
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{shard_id}/{subshard_id}.json"
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1}/{max_retries} failed for {shard_id}/{subshard_id}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Failed to download {shard_id}/{subshard_id} after {max_retries} attempts")
                raise


def extract_and_save_codes(
    json_data: List[Dict],
    shard_id: str,
    subshard_id: str,
    output_dir: Path,
    num_target_codebooks: int = None
) -> int:
    """
    Extract codes from JSON data and save as individual .npy files.
    
    Args:
        json_data: JSON data containing audio entries with codes
        shard_id: Shard ID
        subshard_id: Subshard ID
        output_dir: Output directory for .npy files
        num_target_codebooks: Number of codebooks to keep (default: None, keeps all)
        
    Returns:
        Number of code files saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    num_saved = 0
    
    # JSON data is a list of entries
    for entry in json_data:
        if "codes" not in entry:
            logger.warning(f"Entry in {shard_id}/{subshard_id} missing 'codes' field, skipping")
            continue
        
        codes_dict = entry["codes"]
        
        # codes_dict is a dictionary: {chunk_id: codes_list}
        # codes_list is a 2D array: [[codebook_0_values], [codebook_1_values], ...]
        for chunk_id, codes_list in codes_dict.items():
            # Convert to numpy array
            codes_array = np.array(codes_list, dtype=np.uint16)  # shape: (num_codebooks, num_frames)
            
            # Select only the first N codebooks if specified
            if num_target_codebooks is not None:
                codes_array = codes_array[:num_target_codebooks, :]
            
            # Add batch dimension to make it (1, num_codebooks, num_frames)
            codes_array = np.expand_dims(codes_array, axis=0)

            # Save as .npy file with format: {shard_id}_{subshard_id}_{chunk_id}.npy
            filename = f"{shard_id}_{subshard_id}_{chunk_id}.npy"
            output_path = output_dir / filename
            
            np.save(output_path, codes_array)
            num_saved += 1
    
    return num_saved


def load_progress(progress_file: Path) -> Set[Tuple[str, str]]:
    """
    Load processing progress.
    
    Args:
        progress_file: Path to progress file
        
    Returns:
        Set of (shard_id, subshard_id) tuples that have been processed
    """
    if not progress_file.exists():
        return set()
    
    processed = set()
    with open(progress_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    processed.add((parts[0], parts[1]))
    
    logger.info(f"Loaded progress: {len(processed)} subshards already processed")
    return processed


def save_progress(progress_file: Path, shard_id: str, subshard_id: str):
    """
    Save processing progress.
    
    Args:
        progress_file: Path to progress file
        shard_id: Shard ID
        subshard_id: Subshard ID
    """
    with open(progress_file, 'a') as f:
        f.write(f"{shard_id}\t{subshard_id}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sample Yodas2 codes from HuggingFace for acoustic tokenizer training"
    )
    parser.add_argument(
        "--shard-ids-file",
        type=str,
        default="yodas2-mimi/shard_ids.txt",
        help="Path to shard IDs file (default: yodas2-mimi/shard_ids.txt)"
    )
    parser.add_argument(
        "--subshard-counts-file",
        type=str,
        default="yodas2-mimi/subshard_counts.json",
        help="Path to subshard counts JSON file (default: yodas2-mimi/subshard_counts.json)"
    )
    parser.add_argument(
        "--max-subshards-per-shard",
        type=int,
        default=5,
        help="Maximum number of subshards to sample per shard (default: 5)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="potsawee/yodas2-mm",
        help="HuggingFace repo ID (default: potsawee/yodas2-mm)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./yodas2_codes_samples",
        help="Output directory for .npy code files (default: ./yodas2_codes_samples)"
    )
    parser.add_argument(
        "--subshard-list-file",
        type=str,
        default="./subset_subshard_ids_for_bpe.txt",
        help="File containing list of selected subshards (default: ./subset_subshard_ids_for_bpe.txt)"
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default="./sample_progress.txt",
        help="File for tracking processing progress (default: ./sample_progress.txt)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--num-target-codebooks",
        type=int,
        default=8,
        help="Number of codebooks to keep (default: None, keeps all codebooks)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Yodas2 Code Sampling Script")
    logger.info("=" * 80)
    
    # Convert to Path objects
    shard_ids_file = Path(args.shard_ids_file)
    subshard_counts_file = Path(args.subshard_counts_file)
    subshard_list_file = Path(args.subshard_list_file)
    output_dir = Path(args.output_dir)
    progress_file = Path(args.progress_file)
    
    # Step 1 & 2: Get or create list of subshards to download
    if subshard_list_file.exists():
        logger.info(f"Loading existing subshard list from {subshard_list_file}")
        selected_subshards = load_subshard_list(subshard_list_file)
    else:
        logger.info("Creating new subshard selection")
        # Read shard IDs and subshard counts
        shard_ids = read_shard_ids(shard_ids_file)
        subshard_counts = read_subshard_counts(subshard_counts_file)
        
        logger.info(f"Found {len(shard_ids)} shards in {shard_ids_file}")
        
        # Sample subshards
        selected_subshards = sample_subshards(
            shard_ids,
            subshard_counts,
            max_subshards_per_shard=args.max_subshards_per_shard,
            seed=args.seed
        )
        
        # Save selection
        save_subshard_list(selected_subshards, subshard_list_file)
    
    logger.info(f"Total subshards to process: {len(selected_subshards)}")
    
    # Load progress
    processed = load_progress(progress_file)
    remaining = [s for s in selected_subshards if s not in processed]
    
    logger.info(f"Already processed: {len(processed)}")
    logger.info(f"Remaining to process: {len(remaining)}")
    
    if not remaining:
        logger.info("All subshards already processed!")
        return
    
    # Step 3 & 4: Download and extract codes
    total_codes_saved = 0
    successful_subshards = 0
    failed_subshards = 0
    
    logger.info("=" * 80)
    logger.info("Starting download and extraction")
    logger.info("=" * 80)
    
    for shard_id, subshard_id in tqdm(remaining, desc="Processing subshards"):
        try:
            # Download JSON
            logger.info(f"Downloading {shard_id}/{subshard_id}")
            json_data = download_json_from_hf(
                repo_id=args.repo_id,
                shard_id=shard_id,
                subshard_id=subshard_id
            )
            
            # Extract and save codes
            logger.info(f"Extracting codes from {shard_id}/{subshard_id}")
            num_saved = extract_and_save_codes(
                json_data=json_data,
                shard_id=shard_id,
                subshard_id=subshard_id,
                output_dir=output_dir,
                num_target_codebooks=args.num_target_codebooks
            )
            
            total_codes_saved += num_saved
            successful_subshards += 1
            logger.info(f"Saved {num_saved} code files from {shard_id}/{subshard_id}")
            
            # Save progress
            save_progress(progress_file, shard_id, subshard_id)
            
        except Exception as e:
            failed_subshards += 1
            logger.error(f"Failed to process {shard_id}/{subshard_id}: {e}")
            continue
    
    # Print summary
    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successfully processed: {successful_subshards} subshards")
    logger.info(f"Failed: {failed_subshards} subshards")
    logger.info(f"Total code files saved: {total_codes_saved}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
