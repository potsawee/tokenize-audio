#!/usr/bin/env python3
"""
Prepare pretraining data from Yodas2-Mimi tokenized dataset.

This script:
1. Downloads tokenized data from HuggingFace (shard/subshard.json format)
2. Converts audio codes to unicode strings
3. Creates interleaved text-audio documents in two formats:
   - Type 1: text -> audio (text comes first)
   - Type 2: audio -> text (audio comes first)
4. Outputs in FineWeb format (one document per row)
5. Tracks progress for resumability
6. Optionally uploads results to HuggingFace


# Example usage:
python prepare_pretraining_data.py \
    --shard-id en000 \
    --source-repo-id potsawee/yodas2-mm \
    --dest-repo-id potsawee/yodas2-mm-pretrain \
    --work-dir ./work \
    --output-dir ./output \
    --progress-dir ./progress \
    --num-codebooks 8 \
    --codebook-size 2048 \
    --unicode-offset 0xe000 \
    --parquet-batch-size 10000 \
    --upload-batch-size 1 \
    --checkpoint-interval 5

# 20000 entries -> 2-3GB per parquet file (similar to fineweb)
# Checkpoint saved every 10 subshards for resumability
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, CommitOperationAdd
from tqdm import tqdm

from converter import codes_to_chars, UNICODE_OFFSET_LARGE


def setup_logging(shard_id: Optional[str] = None):
    """Setup logging with shard-specific log file to avoid conflicts."""
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    
    if shard_id:
        log_file = log_dir / f"prepare_{shard_id}.log"
    else:
        log_file = log_dir / "prepare_pretraining_data.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)


SPECIAL_TOKENS = {
    "begin_of_text": "<|begin_of_text|>",
    "end_of_text": "<|end_of_text|>",
    "text_start": "<|text_start|>",
    "text_end": "<|text_end|>",
    "audio_start": "<|audio_start|>",
    "audio_end": "<|audio_end|>",
}


class HuggingFaceManager:
    """Handle downloading from source repo and uploading to destination repo."""
    
    def __init__(
        self, 
        source_repo_id: str,
        dest_repo_id: Optional[str] = None,
        enable_upload: bool = True
    ):
        self.source_repo_id = source_repo_id
        self.dest_repo_id = dest_repo_id
        self.enable_upload = enable_upload and (dest_repo_id is not None)
        self.api = HfApi()
        
        logger.info(f"Source repo: {source_repo_id}")
        if self.enable_upload:
            logger.info(f"Destination repo: {dest_repo_id}")
        else:
            logger.info("Upload disabled")
    
    def download_subshard(self, shard_id: str, subshard_id: str, dest_path: Path, max_retries: int = 3) -> bool:
        """Download a subshard JSON file from HuggingFace."""
        # Try HF API first, then fallback to direct URL
        hf_path = f"{shard_id}/{subshard_id}.json"
        
        for attempt in range(max_retries):
            try:
                # Try using HF API
                logger.info(f"Downloading {hf_path} from {self.source_repo_id} (attempt {attempt + 1}/{max_retries})")
                self.api.hf_hub_download(
                    repo_id=self.source_repo_id,
                    filename=hf_path,
                    repo_type="dataset",
                    local_dir=dest_path.parent.parent,
                    # local_dir_use_symlinks=True,
                )
                
                # Check if file was downloaded
                expected_path = dest_path.parent.parent / hf_path
                if expected_path.exists():
                    # Move to expected location if different
                    if expected_path != dest_path:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        expected_path.rename(dest_path)
                    logger.info(f"Successfully downloaded {hf_path}")
                    return True
                else:
                    logger.warning(f"File not found at expected location: {expected_path}")
                    
            except Exception as e:
                logger.warning(f"HF API download failed (attempt {attempt + 1}): {e}")
                
                # Fallback to direct URL
                try:
                    url = f"https://huggingface.co/datasets/{self.source_repo_id}/resolve/main/{hf_path}"
                    logger.info(f"Trying direct URL: {url}")
                    
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()
                    
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(dest_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    logger.info(f"Successfully downloaded via direct URL")
                    return True
                except Exception as e2:
                    logger.warning(f"Direct URL download failed (attempt {attempt + 1}): {e2}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"Failed to download {hf_path} after {max_retries} attempts")
        return False
    
    def file_exists_on_dest(self, path_in_repo: str) -> bool:
        """Check if a file exists in destination repo."""
        if not self.enable_upload:
            return False
        
        try:
            return self.api.file_exists(
                repo_id=self.dest_repo_id,
                filename=path_in_repo,
                repo_type="dataset",
            )
        except Exception as e:
            logger.debug(f"Error checking if {path_in_repo} exists: {e}")
            return False
    
    def upload_file(self, local_path: Path, path_in_repo: str) -> bool:
        """Upload a file to destination repo."""
        if not self.enable_upload:
            logger.info(f"Upload disabled, keeping local file: {local_path}")
            return True
        
        try:
            logger.info(f"Uploading {local_path} to {self.dest_repo_id}/{path_in_repo}")
            
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=self.dest_repo_id,
                repo_type="dataset",
            )
            
            logger.info(f"Successfully uploaded {path_in_repo}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
    
    def upload_batch(self, file_paths: List[Tuple[Path, str]]) -> bool:
        """Upload multiple files in a single commit."""
        if not self.enable_upload:
            logger.info(f"Upload disabled, keeping {len(file_paths)} local files")
            return True
        
        if not file_paths:
            return True
        
        try:
            logger.info(f"Batch uploading {len(file_paths)} files to {self.dest_repo_id}")
            
            operations = []
            for local_path, path_in_repo in file_paths:
                if local_path.exists():
                    operations.append(
                        CommitOperationAdd(
                            path_in_repo=path_in_repo,
                            path_or_fileobj=str(local_path)
                        )
                    )
            
            if operations:
                self.api.create_commit(
                    repo_id=self.dest_repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Upload {len(operations)} processed files",
                )
                
                logger.info(f"Successfully uploaded {len(operations)} files")
            
            return True
        except Exception as e:
            logger.error(f"Failed to batch upload: {e}")
            return False


class PretrainingDataProcessor:
    """Process tokenized data into pretraining format."""
    
    def __init__(
        self,
        num_codebooks: int = 8,
        codebook_size: int = 2048,
        unicode_offset: int = UNICODE_OFFSET_LARGE,
    ):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.unicode_offset = unicode_offset
        
        logger.info(f"Processor config: num_codebooks={num_codebooks}, codebook_size={codebook_size}, unicode_offset={hex(unicode_offset)}")
    
    def convert_codes_to_string(self, codes: List[List[int]]) -> str:
        """Convert audio codes to unicode string."""
        # codes shape: [num_codebooks, seq_length]
        codes_array = np.array(codes, dtype=np.int32)
        
        # Only keep first num_codebooks
        if codes_array.shape[0] > self.num_codebooks:
            codes_array = codes_array[:self.num_codebooks, :]
        
        # Convert to unicode string
        audio_str = codes_to_chars(
            codes_array,
            codebook_size=self.codebook_size,
            unicode_offset=self.unicode_offset,
        )
        
        return audio_str
    
    def create_interleaved_text_type1(self, chunks: List[Tuple[str, str]]) -> str:
        """
        Create type1 interleaved text: text -> audio.
        
        Format: <|begin_of_text|><|text_start|>text1<|text_end|><|audio_start|>audio1<|audio_end|>...<|end_of_text|>
        """
        parts = [SPECIAL_TOKENS["begin_of_text"]]
        
        for text, audio_str in chunks:
            parts.append(SPECIAL_TOKENS["text_start"])
            parts.append(text)
            parts.append(SPECIAL_TOKENS["text_end"])
            parts.append(SPECIAL_TOKENS["audio_start"])
            parts.append(audio_str)
            parts.append(SPECIAL_TOKENS["audio_end"])
        
        parts.append(SPECIAL_TOKENS["end_of_text"])
        
        return "".join(parts)
    
    def create_interleaved_text_type2(self, chunks: List[Tuple[str, str]]) -> str:
        """
        Create type2 interleaved text: audio -> text.
        
        Format: <|begin_of_text|><|audio_start|>audio1<|audio_end|><|text_start|>text1<|text_end|>...<|end_of_text|>
        """
        parts = [SPECIAL_TOKENS["begin_of_text"]]
        
        for text, audio_str in chunks:
            parts.append(SPECIAL_TOKENS["audio_start"])
            parts.append(audio_str)
            parts.append(SPECIAL_TOKENS["audio_end"])
            parts.append(SPECIAL_TOKENS["text_start"])
            parts.append(text)
            parts.append(SPECIAL_TOKENS["text_end"])
        
        parts.append(SPECIAL_TOKENS["end_of_text"])
        
        return "".join(parts)
    
    def process_document(
        self, 
        document: Dict, 
        shard_id: str, 
        subshard_id: str
    ) -> Tuple[Dict, Dict]:
        """
        Process a single document into two FineWeb format entries (type1 and type2).
        
        Returns:
            Tuple of (type1_entry, type2_entry)
        """
        
        if "codes" in document and "text" in document:
            audio_id = document["audio_id"]
            text_dict = document["text"]
            codes_dict = document["codes"]
        else:
            logger.warning(f"Document {document['audio_id']} has no codes or text, skipping")
            return None, None
        
        # Get chunk IDs in order (they should already be ordered by timestamp)
        chunk_ids = list(text_dict.keys())
        
        # Prepare chunks (text, audio_str) pairs
        chunks = []
        for chunk_id in chunk_ids:
            if chunk_id not in codes_dict:
                logger.warning(f"Chunk {chunk_id} has text but no codes, skipping")
                continue
            
            text = text_dict[chunk_id]
            codes = codes_dict[chunk_id]
            
            # Skip empty codes
            if not codes or len(codes) == 0:
                logger.warning(f"Chunk {chunk_id} has empty codes, skipping")
                continue
            
            # Convert codes to string
            audio_str = self.convert_codes_to_string(codes)
            
            chunks.append((text, audio_str))
        
        if not chunks:
            logger.warning(f"Document {audio_id} has no valid chunks")
            return None, None
        
        # Create type1 and type2 interleaved texts
        type1_text = self.create_interleaved_text_type1(chunks)
        type2_text = self.create_interleaved_text_type2(chunks)
        
        split_name = f"{shard_id}/{subshard_id}"
        
        type1_entry = {
            "id": f"{audio_id}_type1",
            "split": split_name,
            "text": type1_text,
        }
        
        type2_entry = {
            "id": f"{audio_id}_type2",
            "split": split_name,
            "text": type2_text,
        }
        
        return type1_entry, type2_entry
    
    def process_subshard(
        self,
        subshard_data: List[Dict],
        shard_id: str,
        subshard_id: str,
    ) -> List[Dict]:
        """Process an entire subshard into FineWeb format."""
        results = []
        
        for document in tqdm(subshard_data, desc=f"Processing {shard_id}/{subshard_id}"):
            type1_entry, type2_entry = self.process_document(document, shard_id, subshard_id)
            
            if type1_entry and type2_entry:
                results.append(type1_entry)
                results.append(type2_entry)
        
        logger.info(f"Processed {len(subshard_data)} documents into {len(results)} entries")
        return results


class ShardProcessor:
    """Process an entire shard with progress tracking."""
    
    def __init__(
        self,
        shard_id: str,
        source_repo_id: str,
        dest_repo_id: Optional[str],
        work_dir: Path,
        output_dir: Path,
        progress_dir: Path,
        num_codebooks: int = 8,
        codebook_size: int = 2048,
        unicode_offset: int = UNICODE_OFFSET_LARGE,
        upload_batch_size: int = 5,
        max_consecutive_missing: int = 5,
        enable_upload: bool = True,
        parquet_batch_size: int = 20000,
        checkpoint_interval: int = 10,
    ):
        self.shard_id = shard_id
        self.work_dir = work_dir / shard_id
        self.output_dir = output_dir
        self.progress_dir = progress_dir
        self.progress_file = progress_dir / f"{shard_id}_progress.json"
        self.upload_batch_size = upload_batch_size
        self.max_consecutive_missing = max_consecutive_missing
        self.parquet_batch_size = parquet_batch_size
        self.checkpoint_interval = checkpoint_interval
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize HuggingFace manager
        self.hf_manager = HuggingFaceManager(
            source_repo_id=source_repo_id,
            dest_repo_id=dest_repo_id,
            enable_upload=enable_upload,
        )
        
        # Initialize processor
        self.processor = PretrainingDataProcessor(
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            unicode_offset=unicode_offset,
        )
        
        # Load progress
        self.progress = self.load_progress()
        
        # Track pending uploads and accumulated data for parquet batching
        self.pending_uploads = []
        self.accumulated_data = []
        self.parquet_counter = 0
        
        # Track which subshards are in each parquet file (for marking as completed after upload)
        self.parquet_to_subshards = {}  # {parquet_filename: [subshard_ids]}
        self.current_batch_subshards = []  # subshards in current accumulated batch
        
        # Checkpoint file for resumability during accumulation
        self.checkpoint_file = self.work_dir / "accumulation_checkpoint.parquet"
        self.checkpoint_meta_file = self.work_dir / "accumulation_checkpoint_meta.json"
        
        # Track subshards processed since last checkpoint
        self.subshards_since_checkpoint = 0
        
        # Clean up any leftover temp files from previous crashed runs
        self.cleanup_temp_files()
        
        # Initialize parquet counter based on existing files to avoid overwriting
        self.initialize_parquet_counter()
        
        # Load any existing checkpoint on startup (may override parquet_counter)
        self.load_checkpoint()
        
        # Scan for local files that need uploading
        if enable_upload:
            self.scan_and_queue_local_files()
    
    def cleanup_temp_files(self):
        """Clean up leftover temporary parquet files from previous crashed runs."""
        shard_output_dir = self.output_dir / self.shard_id
        if not shard_output_dir.exists():
            return
        
        temp_files = list(shard_output_dir.glob(".tmp_*.parquet"))
        if temp_files:
            logger.info(f"Cleaning up {len(temp_files)} leftover temp files from previous runs")
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    logger.debug(f"Deleted temp file: {temp_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file.name}: {e}")
    
    def initialize_parquet_counter(self):
        """
        Initialize parquet counter from progress file and existing local files.
        Uses the maximum of both to ensure we never reuse a parquet ID.
        """
        # Get counter from progress file (persists across runs even if files deleted)
        saved_counter = self.progress.get("parquet_counter", 0)
        
        # Scan for existing local parquet files (safety net)
        file_counter = 0
        shard_output_dir = self.output_dir / self.shard_id
        if shard_output_dir.exists():
            existing_files = sorted(shard_output_dir.glob("*.parquet"))
            if existing_files:
                max_counter = -1
                for parquet_file in existing_files:
                    # Skip checkpoint and temp files
                    if parquet_file.name.startswith('.checkpoint') or parquet_file.name.startswith('.tmp_'):
                        continue
                    
                    # Extract counter from filename (e.g., "00000042.parquet" -> 42)
                    try:
                        counter = int(parquet_file.stem)
                        max_counter = max(max_counter, counter)
                    except ValueError:
                        logger.warning(f"Ignoring non-standard parquet filename: {parquet_file.name}")
                        continue
                
                if max_counter >= 0:
                    file_counter = max_counter + 1
                    logger.info(f"Found existing parquet files up to {max_counter:08d}.parquet")
        
        # Use the maximum to ensure we never overwrite
        self.parquet_counter = max(saved_counter, file_counter)
        
        if saved_counter > 0:
            logger.info(f"Loaded parquet counter from progress: {saved_counter}")
        if file_counter > 0:
            logger.info(f"Scanned local files, next counter: {file_counter}")
        
        logger.info(f"Starting parquet counter at: {self.parquet_counter}")
    
    def load_progress(self) -> Dict:
        """Load processing progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "shard_id": self.shard_id,
            "completed_subshards": [],
            "failed_subshards": [],
            "parquet_counter": 0
        }
    
    def save_progress(self):
        """Save processing progress."""
        # Always save current parquet_counter to persist across runs
        self.progress["parquet_counter"] = self.parquet_counter
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def load_checkpoint(self):
        """Load checkpoint of accumulated data from previous run if exists."""
        # Clean up any leftover temporary checkpoint files from previous crashed runs
        for temp_file in self.work_dir.glob(".checkpoint_tmp_*.parquet"):
            temp_file.unlink()
            logger.debug(f"Cleaned up leftover temp checkpoint: {temp_file}")
        for temp_file in self.work_dir.glob(".checkpoint_meta_tmp_*.json"):
            temp_file.unlink()
            logger.debug(f"Cleaned up leftover temp metadata: {temp_file}")
        
        if self.checkpoint_file.exists() and self.checkpoint_meta_file.exists():
            try:
                logger.info(f"Found checkpoint file, loading accumulated data...")
                
                # Load the accumulated data
                df = pd.read_parquet(self.checkpoint_file)
                self.accumulated_data = df.to_dict('records')
                
                # Load metadata (subshard tracking and parquet counter)
                with open(self.checkpoint_meta_file, 'r') as f:
                    meta = json.load(f)
                    self.current_batch_subshards = meta.get('subshards', [])
                    checkpoint_counter = meta.get('parquet_counter', 0)
                    # Use the max to avoid overwriting existing files
                    self.parquet_counter = max(self.parquet_counter, checkpoint_counter)
                
                logger.info(f"Loaded checkpoint: {len(self.accumulated_data)} entries from {len(self.current_batch_subshards)} subshards")
                logger.info(f"Continuing from parquet counter: {self.parquet_counter}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint (possibly corrupted): {e}")
                logger.info("Cleaning up corrupted checkpoint and starting fresh")
                
                # Clean up corrupted checkpoint files
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
                    logger.info(f"Deleted corrupted checkpoint: {self.checkpoint_file}")
                if self.checkpoint_meta_file.exists():
                    self.checkpoint_meta_file.unlink()
                    logger.info(f"Deleted corrupted metadata: {self.checkpoint_meta_file}")
                
                # Start fresh
                self.accumulated_data = []
                self.current_batch_subshards = []
                self.parquet_counter = 0
        else:
            logger.info("No checkpoint found, starting fresh")
    
    def save_checkpoint(self):
        """Save current accumulated data as checkpoint for resumability (atomic write)."""
        if not self.accumulated_data:
            # No data to checkpoint, clean up any existing checkpoint
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.checkpoint_meta_file.exists():
                self.checkpoint_meta_file.unlink()
            return
        
        try:
            # Use temporary files for atomic writes (prevents corruption on preemption)
            temp_checkpoint = self.work_dir / f".checkpoint_tmp_{os.getpid()}.parquet"
            temp_meta = self.work_dir / f".checkpoint_meta_tmp_{os.getpid()}.json"
            
            # Save the accumulated data to temp file
            df = pd.DataFrame(self.accumulated_data)
            df.to_parquet(temp_checkpoint, engine='pyarrow', compression='snappy', index=False)
            
            # Save metadata to temp file
            meta = {
                'subshards': self.current_batch_subshards,
                'parquet_counter': self.parquet_counter,
                'entry_count': len(self.accumulated_data)
            }
            with open(temp_meta, 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Atomic rename (this is atomic on POSIX filesystems)
            temp_checkpoint.rename(self.checkpoint_file)
            temp_meta.rename(self.checkpoint_meta_file)
            
            logger.debug(f"Saved checkpoint: {len(self.accumulated_data)} entries")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            # Clean up temp files if they exist
            if temp_checkpoint.exists():
                temp_checkpoint.unlink()
            if temp_meta.exists():
                temp_meta.unlink()
    
    def clear_checkpoint(self):
        """Clear checkpoint files after successful parquet save."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.debug("Cleared checkpoint file")
        if self.checkpoint_meta_file.exists():
            self.checkpoint_meta_file.unlink()
            logger.debug("Cleared checkpoint metadata")
    
    def scan_and_queue_local_files(self):
        """Scan for local files that need uploading and extract which subshards they contain."""
        shard_output_dir = self.output_dir / self.shard_id
        if not shard_output_dir.exists():
            return
        
        local_files = sorted(shard_output_dir.glob("*.parquet"))
        if not local_files:
            return
        
        logger.info(f"Scanning {len(local_files)} local files for pending uploads")
        
        for local_file in local_files:
            parquet_filename = local_file.name
            hf_path = f"{self.shard_id}/{parquet_filename}"
            
            # Read parquet file to determine which subshards it contains
            subshard_ids = []
            try:
                df = pd.read_parquet(local_file)
                # Extract unique subshard IDs from "split" field (format: "shard_id/subshard_id")
                unique_splits = df['split'].unique()
                subshard_ids = [split.split('/')[1] for split in unique_splits if '/' in split]
                logger.info(f"File {parquet_filename} contains {len(subshard_ids)} subshards: {subshard_ids}")
            except Exception as e:
                logger.warning(f"Failed to read {parquet_filename} to extract subshard IDs: {e}")
            
            # Check if already on destination
            if self.hf_manager.file_exists_on_dest(hf_path):
                logger.info(f"File {parquet_filename} already uploaded to HF")
                
                # Mark subshards as completed (handles race condition where upload succeeded but progress not saved)
                if subshard_ids:
                    for subshard_id in subshard_ids:
                        if subshard_id not in self.progress["completed_subshards"]:
                            self.progress["completed_subshards"].append(subshard_id)
                    logger.info(f"Marked {len(subshard_ids)} subshards as completed from already-uploaded file")
                    self.save_progress()
                
                # Delete local file since it's already on HF
                local_file.unlink()
                logger.info(f"Deleted local file {parquet_filename} (already on HF)")
                continue
            
            # File not on HF yet - queue for upload
            if subshard_ids:
                self.parquet_to_subshards[parquet_filename] = subshard_ids
            
            if parquet_filename not in self.pending_uploads:
                self.pending_uploads.append(parquet_filename)
                logger.info(f"Queued {parquet_filename} for upload")
        
        if self.pending_uploads:
            logger.info(f"Found {len(self.pending_uploads)} files to upload")
            self.batch_upload_pending(force=False)
    
    def get_subshard_list(self) -> List[str]:
        """Get list of subshards to process."""
        return [f"{i:08d}" for i in range(1000)]
    
    def is_subshard_available(self, subshard_id: str) -> bool:
        """Check if a subshard exists in source repo."""
        # We'll try to download it - if it fails, it doesn't exist
        return True  # Optimistic - will fail gracefully during download
    
    def is_subshard_completed(self, subshard_id: str) -> bool:
        """
        Check if subshard is already processed.
        
        A subshard is completed if:
        1. It's in the progress file (uploaded to HF), OR
        2. It's in the current checkpoint (being accumulated)
        """
        if subshard_id in self.progress["completed_subshards"]:
            return True
        
        # Also check if it's in the current accumulation batch
        if subshard_id in self.current_batch_subshards:
            return True
        
        return False
    
    def save_parquet_batch(self, data: List[Dict], subshard_ids: List[str], force: bool = False) -> Optional[str]:
        """
        Save accumulated data to a parquet file and track which subshards are in it.
        
        Args:
            data: List of document entries
            subshard_ids: List of subshard IDs that contributed to this batch
            force: Force save even if below batch size threshold
        
        Returns:
            Parquet filename if saved, None otherwise
        """
        if not data:
            return None
        
        if len(data) < self.parquet_batch_size and not force:
            return None
        
        # Create parquet filename
        parquet_filename = f"{self.parquet_counter:08d}.parquet"
        output_path = self.output_dir / self.shard_id / parquet_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use atomic write: write to temp file first, then rename
        # This prevents uploading incomplete files if job is killed during write
        temp_path = output_path.parent / f".tmp_{os.getpid()}_{parquet_filename}"
        
        try:
            # Convert to DataFrame and save as parquet to temp file
            df = pd.DataFrame(data)
            df.to_parquet(temp_path, engine='pyarrow', compression='snappy', index=False)
            
            # Atomic rename (atomic on POSIX filesystems)
            temp_path.rename(output_path)
            
            logger.info(f"Saved {len(data)} entries from {len(subshard_ids)} subshards to {parquet_filename}")
        except Exception as e:
            # Clean up temp file if something went wrong
            if temp_path.exists():
                temp_path.unlink()
            logger.error(f"Failed to save parquet file {parquet_filename}: {e}")
            raise
        
        # Track which subshards are in this parquet file
        self.parquet_to_subshards[parquet_filename] = subshard_ids.copy()
        
        self.parquet_counter += 1
        
        # Clear checkpoint after successful parquet save
        self.clear_checkpoint()
        
        return parquet_filename
    
    def batch_upload_pending(self, force: bool = False):
        """Upload pending files in batch and mark subshards as completed only after successful upload."""
        if not self.hf_manager.enable_upload:
            return
        
        if not self.pending_uploads:
            return
        
        if len(self.pending_uploads) >= self.upload_batch_size or force:
            logger.info(f"Batch uploading {len(self.pending_uploads)} files")
            
            file_paths = []
            for parquet_filename in self.pending_uploads:
                local_path = self.output_dir / self.shard_id / parquet_filename
                hf_path = f"{self.shard_id}/{parquet_filename}"
                if local_path.exists():
                    file_paths.append((local_path, hf_path))
            
            success = self.hf_manager.upload_batch(file_paths)
            
            if success:
                # Mark subshards as completed ONLY AFTER successful upload
                for parquet_filename in self.pending_uploads:
                    if parquet_filename in self.parquet_to_subshards:
                        subshard_ids = self.parquet_to_subshards[parquet_filename]
                        for subshard_id in subshard_ids:
                            if subshard_id not in self.progress["completed_subshards"]:
                                self.progress["completed_subshards"].append(subshard_id)
                        logger.info(f"Marked {len(subshard_ids)} subshards as completed after upload of {parquet_filename}")
                        # Clean up tracking
                        del self.parquet_to_subshards[parquet_filename]
                
                # Save progress (including parquet_counter) after marking subshards as completed
                self.save_progress()
                
                # Delete local files after successful upload to save disk space
                # The parquet_counter is saved in progress file, so we won't reuse IDs on resume
                for local_path, _ in file_paths:
                    if local_path.exists():
                        local_path.unlink()
                        logger.debug(f"Deleted local file after upload: {local_path}")
                
                logger.info(f"Uploaded and deleted {len(file_paths)} files")
                self.pending_uploads.clear()
            else:
                logger.error("Batch upload failed, keeping files for retry")
    
    def process_subshard(self, subshard_id: str) -> Optional[List[Dict]]:
        """
        Process a single subshard and return the processed data.
        
        Returns:
            List of processed entries, or None if failed
        """
        logger.info(f"Processing subshard {self.shard_id}/{subshard_id}")
        
        # Download subshard from source
        source_path = self.work_dir / f"{subshard_id}.json"
        if not source_path.exists():
            if not self.hf_manager.download_subshard(self.shard_id, subshard_id, source_path):
                return None
        
        # Load data
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                subshard_data = json.load(f)
            logger.info(f"Loaded {len(subshard_data)} documents from {subshard_id}")
        except Exception as e:
            logger.error(f"Failed to load {source_path}: {e}")
            return None
        
        # Process
        results = self.processor.process_subshard(subshard_data, self.shard_id, subshard_id)
        
        # Cleanup source file
        if source_path.exists():
            source_path.unlink()
            logger.debug(f"Deleted source file: {source_path}")
        
        return results
    
    def process(self):
        """Process the entire shard, accumulating data and saving as parquet files."""
        logger.info(f"Starting processing of shard {self.shard_id}")
        logger.info(f"Parquet batch size: {self.parquet_batch_size} subshards per file")
        
        all_subshards = self.get_subshard_list()
        logger.info(f"Total subshards to check: {len(all_subshards)}")
        logger.info(f"Already completed: {len(self.progress['completed_subshards'])}")
        
        consecutive_missing = 0
        
        for subshard_id in all_subshards:
            # Check if already completed
            if self.is_subshard_completed(subshard_id):
                logger.info(f"Skipping completed subshard {subshard_id}")
                consecutive_missing = 0
                continue
            
            # Try to process
            results = self.process_subshard(subshard_id)
            
            if results is None:
                consecutive_missing += 1
                logger.warning(f"Failed to process {subshard_id} (consecutive missing: {consecutive_missing}/{self.max_consecutive_missing})")
                
                if consecutive_missing >= self.max_consecutive_missing:
                    logger.info(f"Reached {self.max_consecutive_missing} consecutive failures, stopping")
                    break
                
                continue
            
            # Success - accumulate data (DON'T mark as completed yet!)
            consecutive_missing = 0
            self.accumulated_data.extend(results)
            self.current_batch_subshards.append(subshard_id)
            logger.info(f"Processed {subshard_id}, accumulated {len(self.accumulated_data)} entries from {len(self.current_batch_subshards)} subshards")
            
            # Increment checkpoint counter
            self.subshards_since_checkpoint += 1
            
            # Save checkpoint every N subshards for resumability
            if self.subshards_since_checkpoint >= self.checkpoint_interval:
                self.save_checkpoint()
                self.subshards_since_checkpoint = 0
                logger.info(f"Checkpoint saved (every {self.checkpoint_interval} subshards)")
            
            # Try to save parquet batch if we have enough data
            parquet_filename = self.save_parquet_batch(
                self.accumulated_data, 
                self.current_batch_subshards,
                force=False
            )
            if parquet_filename:
                self.pending_uploads.append(parquet_filename)
                self.accumulated_data = []  # Clear accumulated data
                self.current_batch_subshards = []  # Clear subshard tracking
                self.subshards_since_checkpoint = 0  # Reset checkpoint counter
                logger.info(f"Saved parquet file, pending uploads: {len(self.pending_uploads)}")
            
            # Batch upload (subshards will be marked as completed after successful upload)
            self.batch_upload_pending(force=False)
        
        # Save any remaining accumulated data
        if self.accumulated_data:
            logger.info(f"Saving remaining {len(self.accumulated_data)} entries from {len(self.current_batch_subshards)} subshards")
            parquet_filename = self.save_parquet_batch(
                self.accumulated_data, 
                self.current_batch_subshards,
                force=True
            )
            if parquet_filename:
                self.pending_uploads.append(parquet_filename)
            self.accumulated_data = []
            self.current_batch_subshards = []
        
        # Final upload
        if self.pending_uploads:
            logger.info(f"Uploading remaining {len(self.pending_uploads)} files")
            self.batch_upload_pending(force=True)
        
        # Clear checkpoint (everything should be uploaded now)
        self.clear_checkpoint()
        
        # Summary
        logger.info("=" * 80)
        logger.info(f"COMPLETED PROCESSING SHARD {self.shard_id}")
        logger.info("=" * 80)
        logger.info(f"✓ Successfully processed: {len(self.progress['completed_subshards'])} subshards")
        logger.info(f"✓ Created {self.parquet_counter} parquet files")
        
        if self.progress['failed_subshards']:
            logger.warning(f"✗ Failed: {len(self.progress['failed_subshards'])} subshards")
        
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Prepare pretraining data from Yodas2-Mimi tokenized dataset")
    parser.add_argument("--shard-id", type=str, required=True, help="Shard ID (e.g., en000)")
    parser.add_argument("--source-repo-id", type=str, required=True, help="Source HuggingFace repo ID (e.g., potsawee/yodas2-mm)")
    parser.add_argument("--dest-repo-id", type=str, default=None, help="Destination HuggingFace repo ID for output")
    parser.add_argument("--work-dir", type=str, default="./work", help="Working directory for temporary files")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--progress-dir", type=str, default="./progress", help="Progress tracking directory")
    parser.add_argument("--num-codebooks", type=int, default=8, help="Number of codebooks to keep (default: 8)")
    parser.add_argument("--codebook-size", type=int, default=2048, help="Codebook size (default: 2048)")
    parser.add_argument("--unicode-offset", type=lambda x: int(x, 0), default=UNICODE_OFFSET_LARGE, help="Unicode offset for code conversion (default: 0xe000)")
    parser.add_argument("--parquet-batch-size", type=int, default=20000, help="Number of entries to batch into one parquet file (default: 20000)")
    parser.add_argument("--upload-batch-size", type=int, default=5, help="Number of parquet files to batch for upload (default: 10)")
    parser.add_argument("--checkpoint-interval", type=int, default=10, help="Save checkpoint every N subshards (default: 10)")
    parser.add_argument("--max-consecutive-missing", type=int, default=5, help="Max consecutive missing subshards (default: 5)")
    parser.add_argument("--no-upload", action="store_true", help="Disable uploading to destination repo")
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.shard_id)
    
    # Process shard
    processor = ShardProcessor(
        shard_id=args.shard_id,
        source_repo_id=args.source_repo_id,
        dest_repo_id=args.dest_repo_id,
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir),
        progress_dir=Path(args.progress_dir),
        num_codebooks=args.num_codebooks,
        codebook_size=args.codebook_size,
        unicode_offset=args.unicode_offset,
        parquet_batch_size=args.parquet_batch_size,
        upload_batch_size=args.upload_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        max_consecutive_missing=args.max_consecutive_missing,
        enable_upload=not args.no_upload,
    )
    
    processor.process()


if __name__ == "__main__":
    main()

