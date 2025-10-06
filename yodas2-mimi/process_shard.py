#!/usr/bin/env python3
"""
Process a single shard of Yodas2 data by encoding audio chunks with Mimi.

This script:
1. Downloads one sub-shard (tar.gz + json) at a time
2. Extracts audio files
3. Encodes audio chunks with Mimi
4. Adds "codes" field to JSON metadata
5. Saves processed results
6. Cleans up to save storage
7. Tracks progress for resumability
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import requests
import torch
from huggingface_hub import HfApi, CommitOperationAdd
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm


def setup_logging(shard_id: str = None):
    """Setup logging with shard-specific log file to avoid conflicts."""
    log_dir = Path("/sphinx/u/salt-checkpoints/yodas2-mm/logs")
    log_dir.mkdir(exist_ok=True)
    
    if shard_id:
        log_file = log_dir / f"process_{shard_id}.log"
    else:
        log_file = log_dir / "process_shard.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ],
        force=True  # Override any existing config
    )
    return logging.getLogger(__name__)

# Initialize with default logger (will be reconfigured in main())
logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """Handle uploading processed files to HuggingFace and managing local storage."""
    
    def __init__(self, repo_id: str, enabled: bool = True):
        self.repo_id = repo_id
        self.enabled = enabled
        if self.enabled:
            self.api = HfApi()
            logger.info(f"HuggingFace uploader initialized for repo: {repo_id}")
        else:
            logger.info("HuggingFace uploader disabled")
    
    def file_exists_on_hf(self, path_in_repo: str) -> bool:
        """Check if a file already exists on HuggingFace."""
        if not self.enabled:
            return False
        
        try:
            # Use file_exists API method
            return self.api.file_exists(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="dataset",
            )
        except Exception as e:
            logger.debug(f"Error checking if {path_in_repo} exists on HF: {e}")
            return False
    
    def upload_and_delete(self, local_path: Path, path_in_repo: str) -> bool:
        """
        Upload a file to HuggingFace and delete the local copy.
        
        Args:
            local_path: Path to local file
            path_in_repo: Path within the HF repo (e.g., "en000/00000000.json")
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.info(f"Upload disabled, keeping local file: {local_path}")
            return True
        
        try:
            logger.info(f"Uploading {local_path} to HF repo {self.repo_id} as {path_in_repo}")
            
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            
            logger.info(f"Successfully uploaded {path_in_repo}")
            
            # Delete local file to save space
            if local_path.exists():
                local_path.unlink()
                logger.info(f"Deleted local file: {local_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to HF: {e}")
            return False
    
    def upload_folder_and_delete(self, file_paths: List[Tuple[Path, str]]) -> bool:
        """
        Upload multiple files to HuggingFace in a SINGLE commit and delete local copies.
        This avoids rate limits by batching all uploads into one commit.
        
        Args:
            file_paths: List of tuples (local_path, path_in_repo)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.info(f"Upload disabled, keeping {len(file_paths)} local files")
            return True
        
        if not file_paths:
            return True
        
        try:
            logger.info(f"Batch uploading {len(file_paths)} files to HF repo {self.repo_id} in a SINGLE commit")
            
            # Create commit operations for all files
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
                # Create a single commit with all files
                commit_message = f"Upload {len(operations)} processed sub-shards"
                
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=commit_message,
                )
                
                logger.info(f"Successfully uploaded {len(operations)} files in 1 commit")
                
                # Delete local files after successful upload
                for local_path, _ in file_paths:
                    if local_path.exists():
                        local_path.unlink()
                        logger.debug(f"Deleted local file: {local_path}")
                
                logger.info(f"Deleted {len(file_paths)} local files to save space")
            
            return True
        except Exception as e:
            logger.error(f"Failed to batch upload files to HF: {e}")
            return False


class MimiEncoder:
    """Wrapper for Mimi model encoding."""
    
    def __init__(self, model_id: str = "kyutai/mimi", device: str = "cuda"):
        logger.info(f"Loading Mimi model: {model_id}")
        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = MimiModel.from_pretrained(model_id)
        self.model = self.model.to(device)
        self.model.eval()
        logger.info("Mimi model loaded successfully")
    
    def encode_audio_chunk(self, audio_array: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """
        Encode an audio chunk to Mimi tokens.
        
        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate (default 24000)
            
        Returns:
            Audio codes as numpy array of shape (num_codebooks, num_frames)
        """
        with torch.no_grad():
            inputs = self.feature_extractor(
                raw_audio=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            encoder_outputs = self.model.encode(
                inputs["input_values"],
                inputs["padding_mask"]
            )
            audio_codes = encoder_outputs.audio_codes
            return audio_codes.cpu().numpy()[0]  # Remove batch dimension
    
    def encode_audio_batch(self, audio_arrays: List[np.ndarray], sample_rate: int = 24000) -> List[np.ndarray]:
        """
        Encode multiple audio chunks in a batch with proper padding and masking.
        NO TRUNCATION - all audio is preserved with padding, then trimmed to actual lengths.
        
        Args:
            audio_arrays: List of audio data arrays (variable lengths)
            sample_rate: Sample rate (default 24000)
            
        Returns:
            List of audio codes as numpy arrays, one per input (trimmed to actual lengths)
        """
        if len(audio_arrays) == 0:
            return []
        
        if len(audio_arrays) == 1:
            return [self.encode_audio_chunk(audio_arrays[0], sample_rate)]
        
        with torch.no_grad():
            # Track original lengths to trim padding later
            original_lengths = [len(audio) for audio in audio_arrays]
            
            # Use feature extractor with padding=True
            # This will pad shorter sequences to match the longest in the batch
            # and create appropriate padding masks
            inputs = self.feature_extractor(
                raw_audio=audio_arrays,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True  # Pad to longest sequence and create padding_mask
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Encode with padding mask (feature extractor creates it automatically)
            encoder_outputs = self.model.encode(
                input_values=inputs["input_values"],
                padding_mask=inputs["padding_mask"]
            )
            audio_codes = encoder_outputs.audio_codes  # [batch_size, num_codebooks, max_frames]
            
            # Trim each result to its actual length (remove padding)
            # The model's frame rate is typically 12.5 Hz for Mimi (24000 / 12.5)
            frame_rate = sample_rate / 12.5  # Mimi uses 12.5 hop length
            results = []
            for i, orig_length in enumerate(original_lengths):
                # Calculate actual number of frames for this audio
                actual_frames = int(np.ceil(orig_length / frame_rate))
                # Trim the codes to actual length
                trimmed_codes = audio_codes[i, :, :actual_frames].cpu().numpy()
                results.append(trimmed_codes)
            return results


class SubShardProcessor:
    """Process a single sub-shard (one tar.gz + json pair)."""
    
    def __init__(
        self,
        shard_id: str,
        subshard_id: str,
        work_dir: Path,
        output_dir: Path,
        encoder: MimiEncoder,
        hf_uploader: Optional[HuggingFaceUploader] = None,
        num_workers: int = 1,
        batch_size: int = 32,
        save_every: int = 64,
        base_url: str = "https://huggingface.co/datasets/espnet/yodas2/raw/main/data",
        max_chunk_duration: float = 60.0
    ):
        self.shard_id = shard_id
        self.subshard_id = subshard_id
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.encoder = encoder
        self.hf_uploader = hf_uploader
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_every = save_every
        self.base_url = base_url
        self.max_chunk_duration = max_chunk_duration
        
        self.audio_url = f"{base_url}/{shard_id}/audio/{subshard_id}.tar.gz".replace("raw", "resolve") + "?download=true"
        self.text_url = f"{base_url}/{shard_id}/text/{subshard_id}.json"
        
        self.audio_tar_path = work_dir / f"{subshard_id}.tar.gz"
        self.text_json_path = work_dir / f"{subshard_id}.json"
        self.audio_extract_dir = work_dir / f"{subshard_id}_audio"
        
    def download_file(self, url: str, dest_path: Path, max_retries: int = 3) -> bool:
        """Download a file with retry logic."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {url} to {dest_path} (attempt {attempt + 1}/{max_retries})")
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(dest_path, 'wb') as f, tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=dest_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                
                logger.info(f"Successfully downloaded {dest_path.name}")
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to download {url} after {max_retries} attempts")
                    return False
        return False
    
    def extract_audio_tar(self) -> bool:
        """Extract audio tar.gz file and create completion marker."""
        try:
            logger.info(f"Extracting {self.audio_tar_path}")
            self.audio_extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.audio_tar_path, 'r:gz') as tar:
                tar.extractall(path=self.audio_extract_dir)
            
            # Create marker file to indicate successful extraction
            extraction_marker = self.audio_extract_dir / ".extraction_complete"
            extraction_marker.touch()
            
            logger.info(f"Extracted to {self.audio_extract_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract {self.audio_tar_path}: {e}")
            return False
    
    def load_text_metadata(self) -> Optional[List[Dict]]:
        """Load text metadata JSON."""
        try:
            with open(self.text_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded metadata with {len(data)} entries")
            return data
        except Exception as e:
            logger.error(f"Failed to load {self.text_json_path}: {e}")
            return None

    
    def process_audio_entry(self, entry: Dict, sample_rate: int = 24000) -> Dict:
        """Process a single audio entry (one audio file with multiple chunks) WITH BATCHING."""
        audio_id = entry["audio_id"]
        text_dict = entry["text"]
        
        # Find the audio file
        audio_files = list(self.audio_extract_dir.rglob(f"{audio_id}.wav"))
        if not audio_files:
            logger.warning(f"Audio file not found for {audio_id}")
            return entry
        
        audio_path = audio_files[0]
        logger.info(f"Processing {audio_id} with {len(text_dict)} chunks (batch_size={self.batch_size})")
        
        # Load the entire audio file once
        try:
            audio_array, sr = librosa.load(audio_path, sr=sample_rate)
            logger.info(f"Loaded {audio_path}: {len(audio_array)} samples at {sr}Hz")
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            return entry
        
        # Collect all chunks first - split long chunks into sub-chunks
        all_chunks = []
        all_chunk_ids = []
        long_chunks_split = 0
        
        for chunk_id, text in text_dict.items():
            # Parse chunk_id: {audio_id}-{index:05d}-{start_cs:08d}-{end_cs:08d}
            # Note: timestamps are in centiseconds (10ms units), NOT milliseconds
            # Use rsplit to split from right, since audio_id may contain hyphens
            parts = chunk_id.rsplit('-', 3)
            assert len(parts) == 4, f"Invalid chunk_id format: {chunk_id}"
            
            # parts[0] = audio_id (may contain hyphens)
            # parts[1] = index (5 digits)
            # parts[2] = start_cs (8 digits, in centiseconds)
            # parts[3] = end_cs (8 digits, in centiseconds)
            start_cs = int(parts[2])
            end_cs = int(parts[3])

            if start_cs < end_cs:
                pass
            elif start_cs == end_cs:
                # there are broken segments like (same start and end):
                # Y65x5_9PNO8-00026-00003279-00003279 in en000-00000000
                continue
            else:
                raise ValueError(f"Invalid chunk_id format: {chunk_id}")

            # Extract audio segment by slicing the array
            # Convert centiseconds to samples (1 centisecond = 10ms = sample_rate/100 samples)
            start_sample = int(start_cs * sample_rate / 100)
            end_sample = int(end_cs * sample_rate / 100)
            audio_segment = audio_array[start_sample:end_sample]
            
            if len(audio_segment) == 0:
                # some video seems shorter than what the trascript is available for
                # "Yg-Y2--S7q8" is 18min, but the transcript is available for 31.7min
                continue

            # Check duration - if too long, mark for separate processing
            duration_seconds = len(audio_segment) / sample_rate
            if duration_seconds > self.max_chunk_duration:
                logger.warning(f"Chunk {chunk_id} is {duration_seconds:.2f}s (exceeds max={self.max_chunk_duration}s), will split and process separately")
                long_chunks_split += 1
                # Mark with special flag for separate processing
                all_chunks.append(("LONG_CHUNK", audio_segment))
                all_chunk_ids.append(chunk_id)
            else:
                all_chunks.append(audio_segment)
                all_chunk_ids.append(chunk_id)
        
        if long_chunks_split > 0:
            logger.info(f"Found {long_chunks_split} long chunks for {audio_id} that will be split and processed separately")
        
        # Encode in batches with proper padding and masking
        codes_dict = {}
        num_chunks = len(all_chunks)
        
        i = 0
        while i < num_chunks:
            # Check if current chunk is a long chunk that needs splitting
            current_chunk = all_chunks[i]
            current_id = all_chunk_ids[i]
            
            if isinstance(current_chunk, tuple) and current_chunk[0] == "LONG_CHUNK":
                # Process long chunk separately by splitting it
                audio_segment = current_chunk[1]
                duration_seconds = len(audio_segment) / sample_rate
                
                logger.info(f"Processing long chunk {current_id} ({duration_seconds:.2f}s) by splitting into sub-chunks")
                
                # Split into sub-chunks
                max_samples = int(self.max_chunk_duration * sample_rate)
                sub_chunks = []
                for start_idx in range(0, len(audio_segment), max_samples):
                    end_idx = min(start_idx + max_samples, len(audio_segment))
                    sub_chunks.append(audio_segment[start_idx:end_idx])
                
                logger.info(f"Split into {len(sub_chunks)} sub-chunks")
                
                # Encode each sub-chunk individually and concatenate
                sub_codes = []
                for j, sub_chunk in enumerate(sub_chunks):
                    sub_duration = len(sub_chunk) / sample_rate
                    logger.debug(f"Encoding sub-chunk {j+1}/{len(sub_chunks)} (duration={sub_duration:.2f}s)")
                    codes = self.encoder.encode_audio_chunk(sub_chunk, sample_rate=sample_rate)
                    sub_codes.append(codes)
                
                # Concatenate along time dimension (axis 1)
                concatenated_codes = np.concatenate(sub_codes, axis=1)
                codes_uint16 = concatenated_codes.astype(np.uint16)
                codes_dict[current_id] = codes_uint16.tolist()
                
                # Clean up GPU memory after processing long chunk (safety measure for rare edge case)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Completed long chunk {current_id}, final shape: {concatenated_codes.shape}")
                i += 1
            else:
                # Process normal batch
                batch_end = min(i + self.batch_size, num_chunks)
                batch_chunks = []
                batch_ids = []
                
                # Collect batch, stopping if we hit a long chunk
                for j in range(i, batch_end):
                    chunk = all_chunks[j]
                    if isinstance(chunk, tuple) and chunk[0] == "LONG_CHUNK":
                        # Stop batch here, will process long chunk in next iteration
                        break
                    batch_chunks.append(chunk)
                    batch_ids.append(all_chunk_ids[j])
                
                if batch_chunks:
                    # Log batch info for debugging
                    batch_lengths = [len(chunk) / sample_rate for chunk in batch_chunks]
                    max_length = max(batch_lengths)
                    logger.debug(f"Batch: {len(batch_chunks)} chunks, max_duration={max_length:.2f}s")
                    
                    # Encode batch with proper padding and masking (NO TRUNCATION)
                    batch_codes = self.encoder.encode_audio_batch(batch_chunks, sample_rate=sample_rate)
                    
                    # Store results
                    for chunk_id, codes in zip(batch_ids, batch_codes):
                        # Convert to uint16 for efficient storage (codes are in range [0, 2047])
                        # This uses 2 bytes per code instead of 4 or 8 bytes
                        codes_uint16 = codes.astype(np.uint16)
                        codes_dict[chunk_id] = codes_uint16.tolist()
                
                i += len(batch_chunks)
        
        # Add codes to entry (may be empty if all chunks were filtered out)
        entry["codes"] = codes_dict
        
        if not codes_dict:
            logger.warning(f"Audio {audio_id} has 0 valid chunks after filtering (all chunks were empty/invalid)")
        
        return entry
    
    def get_output_path(self) -> Path:
        """Get the output path for this sub-shard."""
        return self.output_dir / self.shard_id / f"{self.subshard_id}.json"
    
    def get_hf_path(self) -> str:
        """Get the path in HF repo for this sub-shard."""
        return f"{self.shard_id}/{self.subshard_id}.json"
    
    def is_already_on_hf(self) -> bool:
        """Check if this sub-shard is already uploaded to HuggingFace."""
        if not self.hf_uploader or not self.hf_uploader.enabled:
            return False
        return self.hf_uploader.file_exists_on_hf(self.get_hf_path())
    
    def load_existing_output(self) -> Optional[Dict[str, Dict]]:
        """Load existing partial output if it exists."""
        output_path = self.get_output_path()
        if not output_path.exists():
            return None
        
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Convert list to dict indexed by audio_id for faster lookup
            return {entry["audio_id"]: entry for entry in data}
        except Exception as e:
            logger.warning(f"Could not load existing output: {e}")
            return None
    
    def save_incremental_output(self, processed_metadata: List[Dict]):
        """Save processed results incrementally."""
        output_path = self.get_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_metadata, f, ensure_ascii=False, indent=2)
    
    def process(self) -> bool:
        """Process the entire sub-shard with resumability and parallel workers."""
        logger.info(f"Processing sub-shard {self.shard_id}/{self.subshard_id} with {self.num_workers} workers, batch_size={self.batch_size}")
        
        # Check if extraction is complete (not just if directory exists)
        extraction_marker = self.audio_extract_dir / ".extraction_complete"
        audio_already_extracted = extraction_marker.exists()
        
        if not audio_already_extracted:
            # Clean up incomplete extraction if directory exists
            if self.audio_extract_dir.exists():
                logger.warning(f"Found incomplete extraction at {self.audio_extract_dir}, cleaning up")
                shutil.rmtree(self.audio_extract_dir)
            
            # Step 1 & 2: Download and extract with retry logic (for corrupted downloads)
            max_retries = 3
            extraction_success = False
            
            for attempt in range(1, max_retries + 1):
                # Step 1: Download audio tar.gz (or reuse if already downloaded)
                if not self.audio_tar_path.exists():
                    logger.info(f"Downloading tar.gz (attempt {attempt}/{max_retries})")
                    if not self.download_file(self.audio_url, self.audio_tar_path):
                        logger.error(f"Download failed on attempt {attempt}/{max_retries}")
                        if attempt < max_retries:
                            time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        else:
                            return False
                else:
                    if attempt == 1:
                        logger.info(f"Audio tar.gz already downloaded at {self.audio_tar_path}, reusing")
                    else:
                        logger.info(f"Retry {attempt}/{max_retries}: Re-downloaded tar.gz")
                
                # Step 2: Extract audio
                if self.extract_audio_tar():
                    extraction_success = True
                    break
                else:
                    # Extraction failed - delete corrupted tar.gz and retry
                    logger.warning(f"Extraction failed on attempt {attempt}/{max_retries}, deleting potentially corrupted tar.gz")
                    if self.audio_tar_path.exists():
                        self.audio_tar_path.unlink()
                    # Also clean up any partial extraction
                    if self.audio_extract_dir.exists():
                        shutil.rmtree(self.audio_extract_dir)
                    
                    if attempt < max_retries:
                        logger.info(f"Will retry download and extraction...")
                        time.sleep(2)  # Brief pause before retry
                    else:
                        logger.error(f"Extraction failed after {max_retries} attempts, giving up")
                        return False
            
            if not extraction_success:
                return False
            
            # Step 2a: Delete tar.gz immediately to save disk space
            logger.info(f"Deleting {self.audio_tar_path} to save disk space")
            if self.audio_tar_path.exists():
                self.audio_tar_path.unlink() # delete the tar.gz file to save disk space
        else:
            logger.info(f"Audio already extracted at {self.audio_extract_dir}, skipping download")
        
        # Step 3: Download and load metadata
        if not self.text_json_path.exists():
            if not self.download_file(self.text_url, self.text_json_path):
                return False
        
        metadata = self.load_text_metadata()
        if metadata is None:
            return False
        
        # Step 4: Load existing partial output (if resuming)
        existing_output = self.load_existing_output()
        # Only consider entries that have "codes" field as completed (even if empty)
        # Empty codes dict means the entry was processed but had no valid chunks
        if existing_output:
            completed_audio_ids = set(
                audio_id for audio_id, entry in existing_output.items()
                if "codes" in entry  # Has been processed (codes may be empty if no valid chunks)
            )
        else:
            completed_audio_ids = set()
        
        if completed_audio_ids:
            logger.info(f"Resuming: {len(completed_audio_ids)} audio files already processed")
        
        # Count incomplete entries that will be reprocessed
        if existing_output:
            incomplete_count = len(existing_output) - len(completed_audio_ids)
            if incomplete_count > 0:
                logger.info(f"Found {incomplete_count} incomplete entries that will be reprocessed")
        
        # Step 5: Process each audio entry (with parallel workers)
        # Separate already-completed entries from entries to process
        processed_metadata = []
        entries_to_process = []
        
        for entry in metadata:
            audio_id = entry["audio_id"]
            if audio_id in completed_audio_ids:
                processed_metadata.append(existing_output[audio_id])
            else:
                entries_to_process.append(entry)
        
        logger.info(f"Processing {len(entries_to_process)} new audio files with {self.num_workers} workers")
        
        if entries_to_process:
            if self.num_workers == 1:
                # Sequential processing
                for entry in tqdm(entries_to_process, desc=f"Processing {self.subshard_id}"):
                    processed_entry = self.process_audio_entry(entry)
                    processed_metadata.append(processed_entry)
                    
                    # Save every N audio files
                    if len(processed_metadata) % self.save_every == 0:
                        self.save_incremental_output(processed_metadata)
            else:
                # Parallel processing with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    # Submit all tasks with index to preserve order
                    future_to_index = {
                        executor.submit(self.process_audio_entry, entry): (idx, entry)
                        for idx, entry in enumerate(entries_to_process)
                    }
                    
                    # Store results with their original index
                    results = {}
                    
                    # Process completed tasks as they finish
                    with tqdm(total=len(entries_to_process), desc=f"Processing {self.subshard_id}") as pbar:
                        for future in as_completed(future_to_index):
                            idx, entry = future_to_index[future]
                            # Don't catch exceptions - let them crash for debugging
                            processed_entry = future.result()  # Will raise if processing failed
                            results[idx] = processed_entry
                            pbar.update(1)
                    
                    # Add results in original order
                    for idx in sorted(results.keys()):
                        processed_metadata.append(results[idx])
                        
                        # Save every N audio files
                        if len(processed_metadata) % self.save_every == 0:
                            self.save_incremental_output(processed_metadata)
        
        # Step 6: Final save
        logger.info(f"Saving final results for {self.subshard_id}")
        self.save_incremental_output(processed_metadata)
        
        # Step 7: Cleanup temporary files (audio files only, keep output JSON for batch upload)
        self.cleanup()
        
        return True
    
    def cleanup(self):
        """Clean up downloaded and extracted files."""
        logger.info("Cleaning up temporary files")
        if self.audio_tar_path.exists():
            self.audio_tar_path.unlink()
        if self.text_json_path.exists():
            self.text_json_path.unlink()
        if self.audio_extract_dir.exists():
            shutil.rmtree(self.audio_extract_dir)


class ShardProcessor:
    """Process an entire shard with progress tracking and resumability."""
    
    def __init__(
        self,
        shard_id: str,
        work_dir: Path,
        output_dir: Path,
        progress_dir: Path,
        device: str = "cuda",
        num_workers: int = 1,
        batch_size: int = 32,
        save_every: int = 64,
        hf_repo_id: Optional[str] = None,
        max_chunk_duration: float = 60.0,
        upload_batch_size: int = 10,
        max_consecutive_missing: int = 5,
    ):
        self.shard_id = shard_id
        self.work_dir = work_dir / shard_id
        self.output_dir = output_dir
        self.progress_dir = progress_dir
        self.progress_file = progress_dir / f"{shard_id}_progress.json"
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_every = save_every
        self.max_chunk_duration = max_chunk_duration
        self.upload_batch_size = upload_batch_size
        self.max_consecutive_missing = max_consecutive_missing
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder once for the entire shard
        self.encoder = MimiEncoder(device=device)
        
        # Initialize HuggingFace uploader if repo_id is provided
        if hf_repo_id:
            self.hf_uploader = HuggingFaceUploader(repo_id=hf_repo_id, enabled=True)
        else:
            self.hf_uploader = None
        
        # Track pending uploads (sub-shards that have been processed but not uploaded yet)
        self.pending_uploads = []
        
        # Load or initialize progress
        self.progress = self.load_progress()
        
        # Scan for existing complete local files that need uploading
        if self.hf_uploader and self.hf_uploader.enabled:
            self.scan_and_queue_local_files()
    
    def is_json_complete(self, json_path: Path) -> bool:
        """
        Check if a JSON file is complete (all entries have been processed).
        
        An entry is considered processed if it has a "codes" field, even if empty.
        Empty codes dict means all chunks were filtered out as invalid.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            True if complete, False otherwise
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list) or len(data) == 0:
                return False
            
            # Check if all entries have the "codes" field (even if empty)
            # An empty codes dict means the entry was processed but had no valid chunks
            for entry in data:
                if not isinstance(entry, dict):
                    return False
                if "codes" not in entry:
                    return False
                # Note: We allow empty codes dict - it means processed but no valid chunks
            
            return True
        except Exception as e:
            logger.warning(f"Failed to validate {json_path}: {e}")
            return False
    
    def cleanup_work_files_for_subshard(self, subshard_id: str):
        """
        Clean up work directory files for a completed sub-shard.
        
        Args:
            subshard_id: The sub-shard ID (e.g., "00000000")
        """
        # Clean up audio extraction directory
        audio_extract_dir = self.work_dir / f"{subshard_id}_audio"
        if audio_extract_dir.exists():
            shutil.rmtree(audio_extract_dir)
            logger.debug(f"Cleaned up work audio dir: {audio_extract_dir}")
        
        # Clean up downloaded tar.gz
        audio_tar_path = self.work_dir / f"{subshard_id}.tar.gz"
        if audio_tar_path.exists():
            audio_tar_path.unlink()
            logger.debug(f"Cleaned up work tar.gz: {audio_tar_path}")
        
        # Clean up text json
        text_json_path = self.work_dir / f"{subshard_id}.json"
        if text_json_path.exists():
            text_json_path.unlink()
            logger.debug(f"Cleaned up work json: {text_json_path}")
    
    def scan_and_queue_local_files(self):
        """
        Scan for existing complete local output files that haven't been uploaded to HF yet.
        Only complete files (with all codes) will be queued for upload.
        Also cleans up work directory files for completed sub-shards.
        """
        shard_output_dir = self.output_dir / self.shard_id
        if not shard_output_dir.exists():
            return
        
        # Find all .json files in the shard output directory
        local_files = sorted(shard_output_dir.glob("*.json"))
        
        if not local_files:
            return
        
        logger.info(f"Scanning {len(local_files)} local output files for pending uploads")
        
        complete_count = 0
        queued_count = 0
        cleaned_count = 0
        
        # Check each file
        for local_file in local_files:
            subshard_id = local_file.stem  # e.g., "00000000" from "00000000.json"
            
            # First check if it's complete
            if not self.is_json_complete(local_file):
                logger.warning(f"Skipping incomplete file: {subshard_id} ({local_file})")
                continue
            
            complete_count += 1
            logger.info(f"Found complete local file: {subshard_id}")
            
            # Clean up work directory files for this complete sub-shard
            self.cleanup_work_files_for_subshard(subshard_id)
            cleaned_count += 1
            
            # Check if already on HF
            hf_path = f"{self.shard_id}/{subshard_id}.json"
            if self.hf_uploader.file_exists_on_hf(hf_path):
                logger.info(f"File {subshard_id} already on HF, skipping upload")
                continue
            
            # File is complete and not on HF - queue it for upload
            if subshard_id not in self.pending_uploads:
                self.pending_uploads.append(subshard_id)
                queued_count += 1
                logger.info(f"Queued {subshard_id} for upload (pending: {len(self.pending_uploads)})")
                
                # Add to completed list if not already there
                if subshard_id not in self.progress["completed_subshards"]:
                    self.progress["completed_subshards"].append(subshard_id)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up work files for {cleaned_count} completed sub-shards")
        
        if queued_count > 0:
            logger.info(f"Found {complete_count} complete local files, queued {queued_count} for upload")
            # Save progress with updated completed list
            self.save_progress()
            # Try to upload them in batches
            self.batch_upload_pending(force=False)
        else:
            logger.info(f"Found {complete_count} complete local files, all already uploaded")
    
    def load_progress(self) -> Dict:
        """Load processing progress."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "shard_id": self.shard_id,
            "completed_subshards": [],
            "failed_subshards": []
        }
    
    def save_progress(self):
        """Save processing progress."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def get_subshard_list(self) -> List[str]:
        """
        Get list of sub-shards to process.
        
        For now, this returns a range. In practice, you might want to query
        the HuggingFace API or have a predefined list.
        """
        # Yodas2 typically has sub-shards numbered from 000000 onwards
        # You may need to adjust this range based on the actual data
        return [f"{i:08d}" for i in range(1000)]  # Adjust as needed
    
    def is_subshard_available(self, subshard_id: str, max_retries: int = 3) -> bool:
        """
        Check if a sub-shard exists on HuggingFace with retry logic.
        
        Args:
            subshard_id: Sub-shard ID to check
            max_retries: Maximum number of retry attempts
            
        Returns:
            True if available, False if confirmed not available
        """
        base_url = "https://huggingface.co/datasets/espnet/yodas2/raw/main/data"
        audio_url = f"{base_url}/{self.shard_id}/audio/{subshard_id}.tar.gz"
        
        for attempt in range(max_retries):
            try:
                response = requests.head(audio_url, timeout=10)
                if response.status_code == 200:
                    return True
                elif response.status_code == 404:
                    # Confirmed not found
                    return False
                else:
                    # Other status codes - retry
                    logger.warning(f"Availability check for {subshard_id} returned status {response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout checking availability of {subshard_id} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error checking availability of {subshard_id}: {e} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected error checking availability of {subshard_id}: {e}")
                return False
        
        # After all retries failed, assume not available
        logger.warning(f"Could not confirm availability of {subshard_id} after {max_retries} attempts, assuming not available")
        return False
    
    def is_subshard_completed(self, subshard_id: str) -> bool:
        """Check if a sub-shard is already completed (either locally or on HF)."""
        # Check if in completed list
        if subshard_id in self.progress["completed_subshards"]:
            # Verify it actually exists on HF
            if self.hf_uploader and self.hf_uploader.enabled:
                hf_path = f"{self.shard_id}/{subshard_id}.json"
                if self.hf_uploader.file_exists_on_hf(hf_path):
                    return True
            # Or exists locally AND is complete (has all codes)
            local_path = self.output_dir / self.shard_id / f"{subshard_id}.json"
            if local_path.exists() and self.is_json_complete(local_path):
                return True
        return False
    
    def batch_upload_pending(self, force=False):
        """Upload pending sub-shards to HF and delete local files."""
        if not self.hf_uploader or not self.hf_uploader.enabled:
            logger.debug(f"Skipping upload: HF uploader not enabled (pending: {len(self.pending_uploads)})")
            return
        
        if not self.pending_uploads:
            logger.debug("Skipping upload: No pending uploads")
            return
        
        logger.debug(f"Checking batch upload: {len(self.pending_uploads)} pending, threshold: {self.upload_batch_size}, force: {force}")
        
        # Upload if we have enough pending or if forced (end of shard)
        if len(self.pending_uploads) >= self.upload_batch_size or force:
            logger.info(f"Batch uploading {len(self.pending_uploads)} sub-shards to HuggingFace")
            
            # Prepare file paths for batch upload
            file_paths = []
            for subshard_id in self.pending_uploads:
                local_path = self.output_dir / self.shard_id / f"{subshard_id}.json"
                hf_path = f"{self.shard_id}/{subshard_id}.json"
                if local_path.exists():
                    file_paths.append((local_path, hf_path))
            
            # Batch upload
            success = self.hf_uploader.upload_folder_and_delete(file_paths)
            
            if success:
                logger.info(f"Successfully uploaded and deleted {len(file_paths)} files")
                self.pending_uploads.clear()
            else:
                logger.error("Batch upload failed, keeping files for retry")
    
    def process(self):
        """Process the entire shard."""
        logger.info(f"Starting processing of shard {self.shard_id}")
        logger.info(f"Work directory: {self.work_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Max consecutive missing: {self.max_consecutive_missing}")
        
        # Get list of sub-shards
        all_subshards = self.get_subshard_list()
        completed = set(self.progress["completed_subshards"])
        
        logger.info(f"Total sub-shards to check: {len(all_subshards)}")
        logger.info(f"Already completed: {len(completed)}")

        consecutive_missing = 0
        
        for subshard_id in all_subshards:
            # Check if already completed (either on HF or locally)
            if self.is_subshard_completed(subshard_id):
                logger.info(f"Skipping already completed sub-shard {subshard_id}")
                consecutive_missing = 0  # Reset counter on successful find
                continue
            
            # Check if sub-shard exists on source (with retry logic)
            if not self.is_subshard_available(subshard_id):
                consecutive_missing += 1
                logger.warning(f"Sub-shard {subshard_id} not available (consecutive missing: {consecutive_missing}/{self.max_consecutive_missing})")
                
                if consecutive_missing >= self.max_consecutive_missing:
                    logger.info(f"Reached {self.max_consecutive_missing} consecutive missing sub-shards, stopping enumeration")
                    break
                else:
                    logger.info(f"Continuing to check next sub-shard (allowing for gaps)")
                    continue
            
            # Reset consecutive missing counter when we find an available sub-shard
            consecutive_missing = 0
            logger.info(f"Processing sub-shard {subshard_id}")
            
            processor = SubShardProcessor(
                shard_id=self.shard_id,
                subshard_id=subshard_id,
                work_dir=self.work_dir,
                output_dir=self.output_dir,
                encoder=self.encoder,
                hf_uploader=self.hf_uploader,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                save_every=self.save_every,
                max_chunk_duration=self.max_chunk_duration,
            )
            
            success = processor.process()
            
            if success:
                self.progress["completed_subshards"].append(subshard_id)
                self.pending_uploads.append(subshard_id)
                logger.info(f"Successfully completed sub-shard {subshard_id} (pending uploads: {len(self.pending_uploads)}/{self.upload_batch_size})")
                
                # Batch upload every N sub-shards
                self.batch_upload_pending(force=False)
            else:
                self.progress["failed_subshards"].append(subshard_id)
                logger.error(f"Failed to process sub-shard {subshard_id}")
            
            # Save progress after each sub-shard
            self.save_progress()
        
        # Final batch upload for any remaining sub-shards
        if self.pending_uploads:
            logger.info(f"Uploading remaining {len(self.pending_uploads)} sub-shards")
            self.batch_upload_pending(force=True)
        
        # Print final summary
        logger.info("=" * 80)
        logger.info(f"COMPLETED PROCESSING SHARD {self.shard_id}")
        logger.info("=" * 80)
        logger.info(f"✓ Successfully processed and uploaded: {len(self.progress['completed_subshards'])} sub-shards")
        
        if self.progress['failed_subshards']:
            failed_count = len(self.progress['failed_subshards'])
            logger.warning(f"✗ Failed to process (after max retries): {failed_count} sub-shards")
            logger.warning(f"  Failed sub-shard IDs: {', '.join(self.progress['failed_subshards'])}")
            logger.warning(f"  These sub-shards will be retried when you restart this job")
            logger.warning(f"  Common causes: corrupted source files, network issues, or OOM errors")
        else:
            logger.info("✓ All available sub-shards processed successfully!")
        
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Process a Yodas2 shard with Mimi encoding (with batching)")
    parser.add_argument("--shard-id", type=str, required=True, help="Shard ID (e.g., en000)")
    parser.add_argument("--work-dir", type=str, default="./work", help="Working directory for temporary files")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for processed files")
    parser.add_argument("--progress-dir", type=str, default="./progress", help="Directory for progress tracking")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Mimi model (cuda or cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers for processing audio files (default: 4)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for encoding chunks (default: 32)")
    parser.add_argument("--save-every", type=int, default=64, help="Save progress every N audio files (default: 64)")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="HuggingFace dataset repo ID for automatic upload (e.g., potsawee/yodas2-mm)")
    parser.add_argument("--max-chunk-duration", type=float, default=60.0, help="Maximum duration of a chunk in seconds to prevent OOM (default: 60.0)")
    parser.add_argument("--upload-batch-size", type=int, default=10, help="Number of sub-shards to batch together for HF upload to avoid rate limits (default: 10)")
    parser.add_argument("--max-consecutive-missing", type=int, default=5, help="Maximum number of consecutive missing sub-shards before stopping enumeration (default: 5)")
    
    args = parser.parse_args()
    
    # Setup logging with shard-specific log file
    global logger
    logger = setup_logging(args.shard_id)
    
    processor = ShardProcessor(
        shard_id=args.shard_id,
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir),
        progress_dir=Path(args.progress_dir),
        device=args.device,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        save_every=args.save_every,
        hf_repo_id=args.hf_repo_id,
        max_chunk_duration=args.max_chunk_duration,
        upload_batch_size=args.upload_batch_size,
        max_consecutive_missing=args.max_consecutive_missing,
    )
    
    processor.process()

    # usage: python process_shard.py --shard-id en000 --work-dir ./work --output-dir ./output --progress-dir ./progress --device cuda --num-workers 1 --batch-size 16 --save-every 48 --hf-repo-id potsawee/yodas2-mm

if __name__ == "__main__":
    main()

