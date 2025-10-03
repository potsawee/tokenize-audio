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
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm


def setup_logging(shard_id: str = None):
    """Setup logging with shard-specific log file to avoid conflicts."""
    log_dir = Path("logs")
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
        num_workers: int = 1,
        batch_size: int = 32,
        save_every: int = 64,
        base_url: str = "https://huggingface.co/datasets/espnet/yodas2/raw/main/data"
    ):
        self.shard_id = shard_id
        self.subshard_id = subshard_id
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.encoder = encoder
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_every = save_every
        self.base_url = base_url
        
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
        """Extract audio tar.gz file."""
        try:
            logger.info(f"Extracting {self.audio_tar_path}")
            self.audio_extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(self.audio_tar_path, 'r:gz') as tar:
                tar.extractall(path=self.audio_extract_dir)
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
        
        # Collect all chunks first
        all_chunks = []
        all_chunk_ids = []
        
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
            
            if len(audio_segment) > 0:
                pass
            else:
                # some video seems shorter than what the trascript is available for
                # "Yg-Y2--S7q8" is 18min, but the transcript is available for 31.7min
                continue

            all_chunks.append(audio_segment)
            all_chunk_ids.append(chunk_id)
        
        # Encode in batches with proper padding and masking
        codes_dict = {}
        num_chunks = len(all_chunks)
        
        for i in tqdm(range(0, num_chunks, self.batch_size), 
                      desc=f"Batch encoding {audio_id}", 
                      leave=False):
            batch_chunks = all_chunks[i:i+self.batch_size]
            batch_ids = all_chunk_ids[i:i+self.batch_size]
            
            # Encode batch with proper padding and masking (NO TRUNCATION)
            batch_codes = self.encoder.encode_audio_batch(batch_chunks, sample_rate=sample_rate)
            
            # Store results
            for chunk_id, codes in zip(batch_ids, batch_codes):
                # Convert to uint16 for efficient storage (codes are in range [0, 2047])
                # This uses 2 bytes per code instead of 4 or 8 bytes
                codes_uint16 = codes.astype(np.uint16)
                codes_dict[chunk_id] = codes_uint16.tolist()
        
        # Add codes to entry
        entry["codes"] = codes_dict
        return entry
    
    def get_output_path(self) -> Path:
        """Get the output path for this sub-shard."""
        return self.output_dir / self.shard_id / f"{self.subshard_id}.json"
    
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
        
        # Check if extraction directory already exists (from previous interrupted run)
        audio_already_extracted = self.audio_extract_dir.exists()
        
        if not audio_already_extracted:
            # Step 1: Download audio tar.gz
            if not self.download_file(self.audio_url, self.audio_tar_path):
                return False
            
            # Step 2: Extract audio
            if not self.extract_audio_tar():
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
        completed_audio_ids = set(existing_output.keys()) if existing_output else set()
        
        if completed_audio_ids:
            logger.info(f"Resuming: {len(completed_audio_ids)} audio files already processed")
        
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
        
        # Step 7: Cleanup
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
    ):
        self.shard_id = shard_id
        self.work_dir = work_dir / shard_id
        self.output_dir = output_dir
        self.progress_dir = progress_dir
        self.progress_file = progress_dir / f"{shard_id}_progress.json"
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.save_every = save_every
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder once for the entire shard
        self.encoder = MimiEncoder(device=device)
        
        # Load or initialize progress
        self.progress = self.load_progress()
    
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
    
    def is_subshard_available(self, subshard_id: str) -> bool:
        """Check if a sub-shard exists on HuggingFace."""
        base_url = "https://huggingface.co/datasets/espnet/yodas2/raw/main/data"
        audio_url = f"{base_url}/{self.shard_id}/audio/{subshard_id}.tar.gz"
        try:
            response = requests.head(audio_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def process(self):
        """Process the entire shard."""
        logger.info(f"Starting processing of shard {self.shard_id}")
        logger.info(f"Work directory: {self.work_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Batch size: {self.batch_size}")
        
        # Get list of sub-shards
        all_subshards = self.get_subshard_list()
        completed = set(self.progress["completed_subshards"])
        
        logger.info(f"Total sub-shards to check: {len(all_subshards)}")
        logger.info(f"Already completed: {len(completed)}")

        for subshard_id in all_subshards:
            if subshard_id in completed:
                logger.info(f"Skipping already completed sub-shard {subshard_id}")
                continue
            
            # Check if sub-shard exists
            if not self.is_subshard_available(subshard_id):
                logger.info(f"Sub-shard {subshard_id} not available, stopping enumeration")
                break
            
            logger.info(f"Processing sub-shard {subshard_id}")
            
            processor = SubShardProcessor(
                shard_id=self.shard_id,
                subshard_id=subshard_id,
                work_dir=self.work_dir,
                output_dir=self.output_dir,
                encoder=self.encoder,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                save_every=self.save_every,
            )
            
            success = processor.process()
            
            if success:
                self.progress["completed_subshards"].append(subshard_id)
                logger.info(f"Successfully completed sub-shard {subshard_id}")
            else:
                self.progress["failed_subshards"].append(subshard_id)
                logger.error(f"Failed to process sub-shard {subshard_id}")
            
            # Save progress after each sub-shard
            self.save_progress()
        
        logger.info(f"Completed processing shard {self.shard_id}")
        logger.info(f"Successfully processed: {len(self.progress['completed_subshards'])} sub-shards")
        logger.info(f"Failed: {len(self.progress['failed_subshards'])} sub-shards")


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
    )
    
    processor.process()

    # usage: python process_shard.py --shard-id en000 --work-dir ./work --output-dir ./output --progress-dir ./progress --device cuda --num-workers 1 --batch-size 12 --save-every 48

if __name__ == "__main__":
    main()

