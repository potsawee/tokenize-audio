#!/usr/bin/env python3
"""
Create batch list files from the output directory structure.

This script scans the output directory to find all speaker_id-book_id pairs,
then creates batch files where each batch contains 10 unique speakers.
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def find_all_speaker_book_pairs(output_dir: Path) -> List[tuple]:
    """
    Find all speaker_id-book_id pairs in the output directory.
    
    Args:
        output_dir: Base directory containing speaker_id/book_id/ folders
        
    Returns:
        List of (speaker_id, book_id) tuples, sorted by speaker_id then book_id
    """
    pairs = []
    
    # First, get all speaker directories
    logger.info("Scanning for speaker directories...")
    speaker_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(speaker_dirs)} speaker directories")
    
    # Find all directories at depth 2 (speaker_id/book_id/)
    for speaker_dir in tqdm(speaker_dirs, desc="Scanning speakers"):
        speaker_id = speaker_dir.name
        
        for book_dir in sorted(speaker_dir.iterdir()):
            if not book_dir.is_dir():
                continue
            
            book_id = book_dir.name
            
            # Check if directory has JSON files
            json_files = list(book_dir.glob("*.json"))
            if json_files:
                pairs.append((speaker_id, book_id))
    
    # Sort by speaker_id (as string) then book_id (as string)
    pairs.sort(key=lambda x: (x[0], x[1]))
    
    logger.info(f"Found {len(pairs)} speaker-book pairs")
    return pairs


def group_by_speakers(pairs: List[tuple], speakers_per_batch: int = 10) -> List[List[tuple]]:
    """
    Group speaker-book pairs into batches with N unique speakers each.
    
    Args:
        pairs: List of (speaker_id, book_id) tuples
        speakers_per_batch: Number of unique speakers per batch
        
    Returns:
        List of batches, where each batch is a list of (speaker_id, book_id) tuples
    """
    # Group pairs by speaker_id
    speaker_to_books = defaultdict(list)
    for speaker_id, book_id in pairs:
        speaker_to_books[speaker_id].append(book_id)
    
    # Get sorted list of speakers
    sorted_speakers = sorted(speaker_to_books.keys())
    logger.info(f"Found {len(sorted_speakers)} unique speakers")
    
    # Create batches
    batches = []
    current_batch = []
    speakers_in_batch = 0
    
    for speaker_id in sorted_speakers:
        # Add all books for this speaker to current batch
        for book_id in sorted(speaker_to_books[speaker_id]):
            current_batch.append((speaker_id, book_id))
        
        speakers_in_batch += 1
        
        # If we've reached the target number of speakers, start a new batch
        if speakers_in_batch >= speakers_per_batch:
            batches.append(current_batch)
            current_batch = []
            speakers_in_batch = 0
    
    # Add remaining pairs as the last batch
    if current_batch:
        batches.append(current_batch)
    
    logger.info(f"Created {len(batches)} batches")
    for i, batch in enumerate(batches):
        unique_speakers = len(set(pair[0] for pair in batch))
        logger.info(f"  Batch {i}: {len(batch)} pairs from {unique_speakers} speakers")
    
    return batches


def write_batch_files(batches: List[List[tuple]], output_dir: Path):
    """
    Write batch files to the output directory.
    
    Args:
        batches: List of batches to write
        output_dir: Directory to write batch files to
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_batches = len(batches)
    total_files_minus_one = num_batches - 1
    
    for i, batch in tqdm(enumerate(batches), desc="Writing batch files", total=num_batches):
        filename = f"train-{i:04d}-of-{total_files_minus_one:04d}.txt"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            for speaker_id, book_id in batch:
                f.write(f"{speaker_id}-{book_id}\n")
        
        logger.debug(f"Wrote {filepath} with {len(batch)} pairs")


def main():
    parser = argparse.ArgumentParser(
        description="Create batch list files from output directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory with speaker_id/book_id/ folders"
    )
    parser.add_argument(
        "--list-dir",
        type=str,
        default="stage2/list_of_batches",
        help="Directory to write batch list files (default: stage2/list_of_batches)"
    )
    parser.add_argument(
        "--speakers-per-batch",
        type=int,
        default=10,
        help="Number of unique speakers per batch (default: 10)"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    list_dir = Path(args.list_dir)
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return
    
    # Find all speaker-book pairs
    pairs = find_all_speaker_book_pairs(output_dir)
    
    if not pairs:
        logger.error("No speaker-book pairs found")
        return
    
    # Group into batches
    batches = group_by_speakers(pairs, args.speakers_per_batch)
    
    # Write batch files
    write_batch_files(batches, list_dir)
    
    logger.info(f"\nDone! Created {len(batches)} batch files in {list_dir}")


if __name__ == "__main__":
    main()

    # usage: python create_batch_lists.py --output-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train --list-dir list_of_batches --speakers-per-batch 30

