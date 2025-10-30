#!/usr/bin/env python3
"""
Merge audio_str from multiple JSON files and upload to HuggingFace.

This script:
1. Processes all .txt list files in a directory
2. For each list file, reads the speaker_id-book_id folders
3. Groups JSON files by original_path
4. Orders by begin_time and merges audio_str and transcript
5. Creates text-first and audio-first interleaved documents
6. Uploads each batch as a parquet file to HuggingFace (e.g., temp000.parquet)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from datasets import Dataset
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Special tokens
BEGIN_TEXT = "<|begin_of_text|>"
END_TEXT = "<|end_of_text|>"
TEXT_START = "<|text_start|>"
TEXT_END = "<|text_end|>"
AUDIO_START = "<|audio_start|>"
AUDIO_END = "<|audio_end|>"

# Time tolerance for consecutive chunks (in seconds)
TIME_TOLERANCE = 0.2


def read_folder_list(list_file: Path) -> List[tuple]:
    """
    Read list of speaker_id-book_id from file.
    
    Args:
        list_file: Path to file containing speaker_id-book_id per line
        
    Returns:
        List of (speaker_id, book_id) tuples
    """
    folders = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('-')
            if len(parts) >= 2:
                speaker_id = parts[0]
                book_id = '-'.join(parts[1:])  # Handle book_ids with dashes
                folders.append((speaker_id, book_id))
    return folders


def read_json_files(output_dir: Path, speaker_id: str, book_id: str) -> List[Dict]:
    """
    Read all JSON files from a speaker_id/book_id folder.
    
    Args:
        output_dir: Base output directory
        speaker_id: Speaker ID
        book_id: Book ID
        
    Returns:
        List of JSON data dictionaries
    """
    folder_path = output_dir / speaker_id / book_id
    if not folder_path.exists():
        logger.warning(f"Folder does not exist: {folder_path}")
        return []
    
    json_files = list(folder_path.glob("*.json"))
    data = []
    skipped = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data.append(json.load(f))
        except Exception as e:
            logger.warning(f"Error loading JSON file {json_file}: {e}")
            skipped += 1

    logger.info(f"Read {len(data)} JSON files from {speaker_id}/{book_id} (skipped {skipped} empty files)")
    return data


def group_by_original_path(json_data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group JSON entries by their original_path and sort by begin_time.
    
    Entries are sorted by begin_time to ensure chronological order when merging
    audio chunks from the same document.
    
    Args:
        json_data: List of JSON data dictionaries
        
    Returns:
        Dictionary mapping original_path to sorted list of entries (by begin_time)
    """
    grouped = defaultdict(list)
    for entry in json_data:
        original_path = entry.get('original_path', '')
        grouped[original_path].append(entry)
    
    # Sort each group by begin_time to ensure chronological order
    for original_path in grouped:
        grouped[original_path].sort(key=lambda x: float(x.get('begin_time', 0)))
    
    return dict(grouped)


def split_consecutive_chunks(entries: List[Dict], tolerance: float = TIME_TOLERANCE) -> List[List[Dict]]:
    """
    Split entries into consecutive segments based on time continuity.
    
    Chunks are considered consecutive if the next chunk's begin_time is approximately
    equal to the current chunk's end_time (within tolerance).
    
    Args:
        entries: Sorted list of entries (sorted by begin_time)
        tolerance: Time tolerance in seconds for considering chunks consecutive
        
    Returns:
        List of consecutive segments, where each segment is a list of entries
    """
    if not entries:
        return []
    
    segments = []
    current_segment = [entries[0]]
    
    for i in range(1, len(entries)):
        prev_entry = entries[i - 1]
        curr_entry = entries[i]
        
        prev_end_time = float(prev_entry.get('end_time', 0))
        curr_begin_time = float(curr_entry.get('begin_time', 0))
        
        # Check if chunks are consecutive (within tolerance)
        time_gap = abs(curr_begin_time - prev_end_time)
        
        if time_gap <= tolerance:
            # Chunks are consecutive, add to current segment
            current_segment.append(curr_entry)
        else:
            # Gap detected, start a new segment
            segments.append(current_segment)
            current_segment = [curr_entry]
    
    # Add the last segment
    if current_segment:
        segments.append(current_segment)
    
    return segments


def create_interleaved_documents(grouped_data: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Create text-first and audio-first interleaved documents.
    
    Only consecutive chunks (where next begin_time ≈ current end_time) are grouped
    into the same document. Non-consecutive chunks create separate documents.
    
    Args:
        grouped_data: Dictionary mapping original_path to sorted list of entries
        
    Returns:
        List of document dictionaries with 'text_first' and 'audio_first' fields
    """
    documents = []
    
    for original_path, entries in grouped_data.items():
        if not entries:
            continue
        
        # Split into consecutive segments
        consecutive_segments = split_consecutive_chunks(entries)
        
        # Create documents for each consecutive segment
        for segment_idx, segment_entries in enumerate(consecutive_segments):
            if not segment_entries:
                continue
            
            # Build text-first document
            text_first_parts = [BEGIN_TEXT]
            audio_first_parts = [BEGIN_TEXT]
            
            for entry in segment_entries:
                transcript = entry['transcript'].strip()
                audio_str = entry['audio_str'].strip()
                
                # Text-first: text then audio
                text_first_parts.append(TEXT_START)
                text_first_parts.append(transcript)
                text_first_parts.append(TEXT_END)
                text_first_parts.append(AUDIO_START)
                text_first_parts.append(audio_str)
                text_first_parts.append(AUDIO_END)
                
                # Audio-first: audio then text
                audio_first_parts.append(AUDIO_START)
                audio_first_parts.append(audio_str)
                audio_first_parts.append(AUDIO_END)
                audio_first_parts.append(TEXT_START)
                audio_first_parts.append(transcript)
                audio_first_parts.append(TEXT_END)
            
            text_first_parts.append(END_TEXT)
            audio_first_parts.append(END_TEXT)
            
            # Create document IDs that include segment index for uniqueness
            first_entry = segment_entries[0]
            base_id = first_entry['entry_id']
            segment_suffix = f"_seg{segment_idx}" if len(consecutive_segments) > 1 else ""
            
            # Create document
            doc_text_first = {
                'id': f"{base_id}{segment_suffix}_type1",
                'original_path': original_path,
                'text': ''.join(text_first_parts),
                'segment_index': segment_idx,
                'num_segments': len(segment_entries),
                'speaker_id': first_entry.get('speaker_id', ''),
                'book_id': first_entry.get('book_id', ''),
            }
            doc_audio_first = {
                'id': f"{base_id}{segment_suffix}_type2",
                'original_path': original_path,
                'text': ''.join(audio_first_parts),
                'segment_index': segment_idx,
                'num_segments': len(segment_entries),
                'speaker_id': first_entry.get('speaker_id', ''),
                'book_id': first_entry.get('book_id', ''),
            }
            documents.append(doc_text_first)
            documents.append(doc_audio_first)
    
    return documents


def file_exists_in_repo(hf_repo_id: str, file_path: str) -> bool:
    """
    Check if a file exists in the HuggingFace repository.
    
    Args:
        hf_repo_id: HuggingFace repository ID
        file_path: Path to file in repository (e.g., "data/temp000.parquet")
        
    Returns:
        True if file exists, False otherwise
    """
    api = HfApi()
    repo_files = api.list_repo_files(
        repo_id=hf_repo_id,
        repo_type="dataset"
    )
    return file_path in repo_files


def process_batch(
    list_file: Path,
    output_dir: Path,
    hf_repo_id: str
):
    """
    Process a batch of folders and upload to HuggingFace.
    
    Args:
        list_file: Path to file with list of speaker_id-book_id
        output_dir: Base directory with JSON files
        hf_repo_id: HuggingFace repository ID
    """
    # Extract batch name from list file (e.g., "temp000" from "list_of_batches/temp000.txt")
    batch_name = list_file.stem  # Gets filename without extension
    logger.info(f"Processing batch {batch_name} from {list_file}")
    
    # Read list of folders
    folders = read_folder_list(list_file)
    logger.info(f"Found {len(folders)} folders to process")
    
    # Process each folder
    all_documents = []
    for speaker_id, book_id in folders:
        # Read JSON files
        json_data = read_json_files(output_dir, speaker_id, book_id)
        if not json_data:
            continue
        
        # Group by original_path
        grouped = group_by_original_path(json_data)
        logger.info(f"  {speaker_id}/{book_id}: {len(grouped)} unique original_paths")
        
        # Create interleaved documents
        documents = create_interleaved_documents(grouped)
        all_documents.extend(documents)
    
    logger.info(f"Total documents created: {len(all_documents)}")
    
    if not all_documents:
        logger.warning("No documents to upload")
        return
    
    # Convert to dataset (directly from dict, not through pandas)
    dataset = Dataset.from_dict({
        'id': [doc['id'] for doc in all_documents],
        'original_path': [doc['original_path'] for doc in all_documents],
        'text': [doc['text'] for doc in all_documents],
        'segment_index': [doc['segment_index'] for doc in all_documents],
        'num_segments': [doc['num_segments'] for doc in all_documents],
        'speaker_id': [doc['speaker_id'] for doc in all_documents],
        'book_id': [doc['book_id'] for doc in all_documents],
    })
    
    # Save parquet file locally with specific name
    parquet_filename = f"{batch_name}.parquet"
    temp_parquet_path = Path(f"./tmp/{parquet_filename}")
    logger.info(f"Saving parquet file: {temp_parquet_path}")
    dataset.to_parquet(str(temp_parquet_path))
    
    # Upload to HuggingFace with specific filename
    logger.info(f"Uploading to {hf_repo_id} as data/{parquet_filename}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(temp_parquet_path),
        path_in_repo=f"data/{parquet_filename}",
        repo_id=hf_repo_id,
        repo_type="dataset",
        commit_message=f"Add batch {batch_name}",
    )
    
    # Clean up temporary file
    temp_parquet_path.unlink()
    logger.info(f"Successfully uploaded batch {batch_name} as {parquet_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge audio_str and upload to HuggingFace"
    )
    parser.add_argument(
        "--list-dir",
        type=str,
        required=True,
        help="Directory containing list files (e.g., stage2/list_of_batches/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base directory with JSON files (e.g., /sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train)"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., username/dataset-name)"
    )
    
    args = parser.parse_args()
    
    list_dir = Path(args.list_dir)
    output_dir = Path(args.output_dir)
    
    # Get all .txt files in the directory
    list_files = sorted(list_dir.glob("*.txt"))
    
    if not list_files:
        logger.error(f"No .txt files found in {list_dir}")
        return
    
    logger.info(f"Found {len(list_files)} list files to process")
    
    # Process each list file
    skipped_count = 0
    
    for i, list_file in enumerate(list_files, 1):
        batch_name = list_file.stem
        parquet_path = f"data/{batch_name}.parquet"
        
        # Check if file already exists in repo
        if file_exists_in_repo(args.hf_repo_id, parquet_path):
            logger.info(f"\n{'='*60}")
            logger.info(f"Skipping batch {i}/{len(list_files)}: {list_file.name}")
            logger.info(f"File already exists in repo: {parquet_path}")
            logger.info(f"{'='*60}")
            skipped_count += 1
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing batch {i}/{len(list_files)}: {list_file.name}")
        logger.info(f"{'='*60}")
        
        process_batch(
            list_file=list_file,
            output_dir=output_dir,
            hf_repo_id=args.hf_repo_id
        )
    

    logger.info(f"\n{'='*60}")
    logger.info(f"All batches completed: {len(list_files)} batches processed")
    logger.info(f"  Processed: {len(list_files) - skipped_count}")
    logger.info(f"  Skipped (already exists): {skipped_count}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

    # usage: python merge_and_upload.py --list-dir list_of_batches --output-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train --hf-repo-id potsawee/mls-en-mm-pretrain

