#!/usr/bin/env python3
import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm
from transformers import AutoFeatureExtractor, MimiModel

from utils import codes_to_chars, resample_audio

# PEOPLES_SPEECH_SAMPLE_RATE = 16000 --> not used
MIMI_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048

HF_SOURCE_REPO = "MLCommons/peoples_speech"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Retry configuration for concurrent uploads
MAX_RETRIES = 10
BASE_DELAY = 5  # seconds
MAX_DELAY = 120  # seconds


def upload_with_retry(
    api: HfApi,
    path_or_fileobj: str,
    path_in_repo: str,
    repo_id: str,
    repo_type: str,
    commit_message: str,
) -> None:
    """
    Upload a file to HuggingFace Hub with retry logic for handling concurrent uploads.
    
    Uses exponential backoff with jitter to handle 409 Conflict errors that occur
    when multiple jobs try to commit to the same repository simultaneously.
    """
    for attempt in range(MAX_RETRIES):
        try:
            api.upload_file(
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
            )
            return  # Success
        except HfHubHTTPError as e:
            if e.response.status_code == 409:
                # Exponential backoff with jitter
                delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 5), MAX_DELAY)
                logger.warning(
                    f"Upload conflict (attempt {attempt + 1}/{MAX_RETRIES}). "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                raise  # Re-raise non-409 errors
    
    # All retries exhausted
    raise RuntimeError(
        f"Failed to upload {path_in_repo} after {MAX_RETRIES} attempts due to concurrent commits"
    )


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
            return audio_codes.cpu().numpy()[0]
    
    def encode_audio_batch(self, audio_arrays: List[np.ndarray], sample_rate: int = 24000) -> List[np.ndarray]:
        """
        Encode multiple audio chunks in a batch with proper padding and masking.
        """
        if len(audio_arrays) == 0:
            return []
        
        if len(audio_arrays) == 1:
            return [self.encode_audio_chunk(audio_arrays[0], sample_rate)]
        
        with torch.no_grad():
            original_lengths = [len(audio) for audio in audio_arrays]
            
            inputs = self.feature_extractor(
                raw_audio=audio_arrays,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            encoder_outputs = self.model.encode(
                input_values=inputs["input_values"],
                padding_mask=inputs["padding_mask"]
            )
            audio_codes = encoder_outputs.audio_codes
            
            frame_rate = sample_rate / 12.5
            results = []
            for i, orig_length in enumerate(original_lengths):
                actual_frames = int(np.ceil(orig_length / frame_rate))
                trimmed_codes = audio_codes[i, :, :actual_frames].cpu().numpy()
                results.append(trimmed_codes)
            return results


def download_shard(split_name: str, shard_id: str, cache_dir: str = "./cache") -> str:
    """
    Download a specific shard from the HuggingFace dataset.
    
    Args:
        split_name: Split name like 'clean', 'clean_sa', 'dirty', ...
        shard_id: Shard ID like 'train-00061-of-00804'
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Path to the downloaded parquet file
    """
    repo_path = f"{split_name}/{shard_id}.parquet"
    
    logger.info(f"Downloading shard: {repo_path}")
    
    local_path = hf_hub_download(
        repo_id=HF_SOURCE_REPO,
        filename=repo_path,
        repo_type="dataset",
        cache_dir=cache_dir
    )
    
    return local_path


def process_shard(
    split_name: str,
    shard_id: str,
    mimi_encoder: MimiEncoder,
    hf_repo_id: str,
    output_dir: str = "./output",
    cache_dir: str = "./cache",
    batch_size: int = 16
):
    """
    Process a single shard: download, encode, and upload.
    
    Args:
        split_name: Split name like 'clean', 'clean_sa', 'dirty', ...
        shard_id: Shard ID to process
        mimi_encoder: MimiEncoder instance
        hf_repo_id: HuggingFace repo ID to upload to
        output_dir: Directory for output parquet files
        cache_dir: Directory for caching downloaded files
        batch_size: Batch size for encoding
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download the shard
    parquet_path = download_shard(split_name, shard_id, cache_dir)
    
    # Load the parquet file
    logger.info(f"Loading parquet file: {parquet_path}")
    dataset = Dataset.from_parquet(parquet_path)
    logger.info(f"Loaded {len(dataset)} samples from shard {shard_id}")
    
    # Encode all audio and store with metadata
    combined_data = []
    for batch_start in tqdm(range(0, len(dataset), batch_size), desc=f"Encoding {shard_id}"):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]
        
        # Prepare audio arrays
        audio_arrays = []
        for i in range(batch_end - batch_start):
            audio_data = batch["audio"][i]
            audio_array = np.array(audio_data["array"], dtype=np.float32)
            orig_sr = audio_data["sampling_rate"]
            
            # Resample if needed
            if orig_sr != MIMI_SAMPLE_RATE:
                audio_array = resample_audio(audio_array, orig_sr, MIMI_SAMPLE_RATE)
            
            audio_arrays.append(audio_array)
        
        # Batch encode
        audio_codes_batch = mimi_encoder.encode_audio_batch(audio_arrays, MIMI_SAMPLE_RATE)
        
        # Store encoded samples with metadata
        for i in range(batch_end - batch_start):
            audio_codes = audio_codes_batch[i][:NUM_CODEBOOKS, :]
            audio_str = codes_to_chars(audio_codes, codebook_size=CODEBOOK_SIZE)
            transcript = batch["text"][i]
            file_id = batch["id"][i]

            text_for_asr = f"<|begin_of_text|><|audio_start|>{audio_str}<|audio_end|><|text_start|>{transcript}<|text_end|><|end_of_text|>"
            text_for_tts = f"<|begin_of_text|><|text_start|>{transcript}<|text_end|><|audio_start|>{audio_str}<|audio_end|><|end_of_text|>"

            # Add TTS sample with _type1 suffix
            combined_data.append({
                "id": f"{file_id}_type1",
                "text": text_for_tts,
            })
            # Add ASR sample with _type2 suffix
            combined_data.append({
                "id": f"{file_id}_type2",
                "text": text_for_asr,
            })
    
    # Create output dataset
    output_dataset = Dataset.from_dict({
        "id": [item["id"] for item in combined_data],
        "text": [item["text"] for item in combined_data],
    })
    
    # Save and upload
    output_filename = f"{shard_id}.parquet"
    output_path = Path(output_dir) / output_filename
    output_dataset.to_parquet(str(output_path))
    
    logger.info(f"Uploading {output_filename} to {hf_repo_id}")
    api = HfApi()
    upload_with_retry(
        api=api,
        path_or_fileobj=str(output_path),
        path_in_repo=f"{split_name}/{output_filename}",
        repo_id=hf_repo_id,
        repo_type="dataset",
        commit_message=f"Add processed shard {shard_id}"
    )
    
    # Cleanup local files
    output_path.unlink()
    logger.info(f"Successfully processed shard {shard_id}: {len(combined_data)} samples")


def get_existing_shards(hf_repo_id: str, split_name: str) -> set:
    """
    Get set of already processed shard IDs from the HuggingFace repo.
    
    Args:
        hf_repo_id: HuggingFace repo ID
        split_name: Split name like 'clean', 'clean_sa', 'dirty', ...
    Returns:
        Set of shard IDs that have already been processed
    """
    api = HfApi()
    
    repo_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    
    existing_shards = set()
    for f in repo_files:
        if f.startswith(split_name) and f.endswith(".parquet") and split_name in f:
            shard_id = f.replace(split_name, "").strip("/").replace(".parquet", "")
            existing_shards.add(shard_id)
    return existing_shards


def main():
    parser = argparse.ArgumentParser(description="Process Peoples Speech shards with Mimi encoding")
    parser.add_argument("--shard-id", type=str, help="Single shard ID to process (e.g., train-00061-of-00804)")
    parser.add_argument("--split-name", type=str, required=True, help="Split name like 'clean', 'clean_sa', 'dirty', ...")
    parser.add_argument("--shard-id-list", type=str, help="Path to file containing shard IDs (one per line)")
    parser.add_argument("--hf-repo-id", type=str, required=True, help="HuggingFace repo ID for output dataset")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory for temporary output files")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Directory for caching downloaded files")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size for encoding (default: 16)")
    args = parser.parse_args()
    
    # Collect shard IDs to process
    shard_ids = []
    
    if args.shard_id:
        shard_ids.append(args.shard_id)
    
    if args.shard_id_list:
        with open(args.shard_id_list, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    shard_ids.append(line)
    
    if not shard_ids:
        logger.error("No shard IDs provided. Use --shard-id or --shard-id-list")
        sys.exit(1)
    
    logger.info(f"Found {len(shard_ids)} shards to process")
    # Skip already processed shards
    existing_shards = get_existing_shards(args.hf_repo_id, args.split_name)
    original_count = len(shard_ids)
    shard_ids = [s for s in shard_ids if s not in existing_shards]
    skipped = original_count - len(shard_ids)
    if skipped > 0:
        logger.info(f"Skipping {skipped} already processed shards")
    
    if not shard_ids:
        logger.info("All shards already processed. Nothing to do.")
        return
    
    # Initialize encoder
    mimi_encoder = MimiEncoder(device="cuda")
    
    # Process each shard
    for i, shard_id in enumerate(shard_ids):
        logger.info(f"Processing shard {i+1}/{len(shard_ids)}: {shard_id}")
        process_shard(
            split_name=args.split_name,
            shard_id=shard_id,
            mimi_encoder=mimi_encoder,
            hf_repo_id=args.hf_repo_id,
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size
        )
    
    logger.info(f"All {len(shard_ids)} shards processed successfully!")


if __name__ == "__main__":
    main()
    # Example usage:
    # python process_peoples_speech.py --split-name clean --shard-id train-00000-of-00804 --hf-repo-id potsawee/peoples-speech-mm-pretrain --output-dir /sphinx/u/salt-checkpoints/peoples-speech/output --cache-dir /sphinx/u/salt-checkpoints/peoples-speech/cache
    # python process_peoples_speech.py --split-name clean --shard-id-list shard_list.txt --hf-repo-id potsawee/peoples-speech-mm-pretrain --output-dir /sphinx/u/salt-checkpoints/peoples-speech/output --cache-dir /sphinx/u/salt-checkpoints/peoples-speech/cache

