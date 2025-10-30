#!/usr/bin/env python3
"""
Process a single shard of MLS-en data by encoding audio chunks with Mimi.

This script:
1. Downloads one parquet file at a time (there are 1417 in total)
2. Extracts audio files from the parquet file
3. Encodes audio chunks with Mimi
4. Adds "codes" field to JSON metadata
5. Saves processed results
6. Cleans up to save storage
7. Tracks progress for resumability
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Optional
import re, unicodedata, hashlib, base64

import numpy as np
import torch
import shutil
from transformers import MimiModel, AutoFeatureExtractor
from datasets import load_dataset
from tqdm import tqdm
from utils import codes_to_chars, resample_audio

MLS_SAMPLE_RATE = 16000
MIMI_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 8 # we're using 8 coebooks instead of 32
CODEBOOK_SIZE = 2048

def setup_logging(shard_id: str = None):
    """Setup logging with shard-specific log file to avoid conflicts."""
    log_dir = Path("/sphinx/u/salt-checkpoints/mls-mm-pretrain/logs")
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


def canonicalize(text: str) -> str:
    # Normalize Unicode, trim, lowercase, collapse whitespace
    t = unicodedata.normalize("NFKC", text)
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def text_to_id(text: str, bits=128) -> str:
    """
    Deterministic content ID for a transcript.
    - version: string prefix so you can change the scheme later.
    - bits: 128 or 256 (use 128 for shorter IDs; 256 for maximum safety).
    """
    c = canonicalize(text).encode("utf-8")
    h = hashlib.sha256(c).digest()
    if bits == 128:
        h = h[:16]
    # URL-safe base64 without padding keeps it short and copyable
    b64 = base64.urlsafe_b64encode(h).decode("ascii").rstrip("=")
    return b64

class ParquetProcessor:
    """Process an entire parquet file with progress tracking and resumability."""
    
    def __init__(
        self,
        shard_id: str,
        work_dir: Path,
        output_dir: Path,
        progress_dir: Path,
        device: str = "cuda",
        progress_save_interval: int = 500,
    ):
        self.shard_id = shard_id
        self.work_dir = work_dir / shard_id
        self.output_dir = output_dir
        self.progress_dir = progress_dir
        self.progress_file = progress_dir / f"progress_{shard_id}.json"
        self.progress_save_interval = progress_save_interval
        
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoder once for the entire shard
        self.encoder = MimiEncoder(device=device)
    
    def _cleanup_work_dir(self):
        """Clean up the work directory for this shard."""
        if self.work_dir.exists():
            logger.info(f"Cleaning up work directory: {self.work_dir}")
            shutil.rmtree(self.work_dir, ignore_errors=True)
            
            # Verify cleanup
            if self.work_dir.exists():
                logger.warning(f"Work directory still exists after cleanup attempt: {self.work_dir}")
            else:
                logger.info(f"Successfully cleaned up work directory")
    
    def load_progress(self) -> dict:
        """Load progress from progress file."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                logger.info(f"Loaded progress: {progress['processed_count']}/{progress['total_count']} entries processed")
                logger.info(f"Resuming from index: {progress.get('last_processed_index', -1) + 1}")
                return progress
        return {
            "shard_id": self.shard_id,
            "processed_count": 0,
            "total_count": 0,
            "last_processed_index": -1
        }
    
    def save_progress(self, progress: dict):
        """Save progress to progress file."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)
        logger.info(f"Progress saved: {progress['processed_count']}/{progress['total_count']} entries processed")
        
    def process(self):
        """Process the shard with resumability."""
        logger.info(f"Processing parquet file {self.shard_id}")
        logger.info(f"Work directory: {self.work_dir}")
        logger.info(f"Output directory: {self.output_dir}")

        # Load progress
        progress = self.load_progress()
        start_index = progress.get('last_processed_index', -1) + 1
        
        # Step 1: Download the parquet file from HuggingFace
        data_files = f"hf://datasets/parler-tts/mls_eng/data/{self.shard_id}.parquet"
        ds = load_dataset(
            "parquet",
            data_files={"train": data_files},
            # it is going save it the downloaded parquet file in the cache directory {cache_dir}/parquet/xxx
            cache_dir=self.work_dir
        )["train"]
        # features: ['audio', 'original_path', 'begin_time', 'end_time', 'transcript', 'audio_duration', 'speaker_id', 'book_id'],

        # Update total count if not set
        total_entries = len(ds)
        if progress['total_count'] == 0:
            progress['total_count'] = total_entries
            self.save_progress(progress)
        
        # If we've already processed everything, clean up and exit
        if start_index >= total_entries:
            logger.info(f"All entries already processed ({progress['processed_count']}/{progress['total_count']})")
            self._cleanup_work_dir()
            return
        
        # Step 2: Process each audio entry starting from the last checkpoint
        logger.info(f"Starting from index {start_index} out of {total_entries}")
        entries_since_last_save = 0
        
        for idx in tqdm(range(start_index, total_entries), desc=f"Processing {self.shard_id}", initial=start_index, total=total_entries):
            entry = ds[idx]
            
            begin_time_00_str = f"{int(entry['begin_time'] * 100):08d}"
            end_time_00_str = f"{int(entry['end_time'] * 100):08d}"
            # there isn't an ID field, so we'll create one
            entry_id = f"{entry['speaker_id']}-{entry['book_id']}-{begin_time_00_str}-{end_time_00_str}-{text_to_id(entry['transcript'])}"

            output_prefix = self.output_dir / f"{entry['speaker_id']}" / f"{entry['book_id']}"
            output_path = output_prefix / f"{entry_id}.json"

            # check if the file already exists (shouldn't happen but just in case)
            if output_path.exists():
                logger.debug(f"Skipping {entry_id} because output file already exists")
                progress['processed_count'] += 1
                progress['last_processed_index'] = idx
                entries_since_last_save += 1
                
                # Save progress periodically
                if entries_since_last_save >= self.progress_save_interval:
                    self.save_progress(progress)
                    entries_since_last_save = 0
                continue

            transcript = entry['transcript']
            begin_time = entry['begin_time']
            end_time = entry['end_time']
            audio_duration = entry['audio_duration']
            audio = entry['audio']
            original_path = entry['original_path']

            speaker_id = entry['speaker_id']
            book_id = entry['book_id']

            audio_arr, audio_sr = audio['array'], audio['sampling_rate']
            # upsample from 16000 to 24000
            upsampled_audio_arr = resample_audio(audio_arr, audio_sr, MIMI_SAMPLE_RATE)
            audio_codes = self.encoder.encode_audio_chunk(upsampled_audio_arr, MIMI_SAMPLE_RATE)
            audio_codes = audio_codes[:NUM_CODEBOOKS, :]
            audio_str = codes_to_chars(audio_codes, codebook_size=CODEBOOK_SIZE)

            output_prefix.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump({
                    "entry_id": entry_id,
                    "original_path": original_path,
                    "speaker_id": speaker_id,
                    "book_id": book_id,
                    "transcript": transcript,
                    "begin_time": begin_time,
                    "end_time": end_time,
                    "audio_duration": audio_duration,
                    "audio_str": audio_str,
                }, f, ensure_ascii=False, indent=2)

            # Update progress
            progress['processed_count'] += 1
            progress['last_processed_index'] = idx
            entries_since_last_save += 1
            
            # Save progress periodically
            if entries_since_last_save >= self.progress_save_interval:
                self.save_progress(progress)
                entries_since_last_save = 0

        # Save final progress
        self.save_progress(progress)
        
        logger.info(f"Processed {progress['processed_count']}/{progress['total_count']} entries")
        logger.info(f"Finished Mimi encoding for shard {self.shard_id}")

        # Step 3: Once we're done, delete the cache (i.e., the work_dir)
        self._cleanup_work_dir()

def main():
    parser = argparse.ArgumentParser(description="Process a MLS-en shard with Mimi encoding (with batching)")
    parser.add_argument("--shard-id", type=str, required=True, help="Shard ID (e.g., train-00000-of-01416)")
    parser.add_argument("--work-dir", type=str, default="./work", help="Working directory for temporary files")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for processed files")
    parser.add_argument("--progress-dir", type=str, default="./progress", help="Directory for progress tracking")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Mimi model (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Setup logging with shard-specific log file
    global logger
    logger = setup_logging(args.shard_id)
    
    processor = ParquetProcessor(
        shard_id=args.shard_id,
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir),
        progress_dir=Path(args.progress_dir),
        device=args.device,
    )
    processor.process()

    # usage: python process_shard.py --shard-id train-00000-of-01416 --work-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/work --output-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train --progress-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/progress --device cuda
    
if __name__ == "__main__":
    main()

