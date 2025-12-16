#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from transformers import AutoFeatureExtractor, MimiModel

from utils import codes_to_chars, resample_audio

LIBRITTS_SAMPLE_RATE = 24000
MIMI_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048

HF_SOURCE_REPO = "parler-tts/libritts_r_filtered"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
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


def parse_shard_id(shard_id: str) -> tuple[str, str]:
    """
    Parse shard ID to determine the subset (clean/other) and parquet filename.
    
    Args:
        shard_id: Shard ID like 'train.clean.100-00000-of-00029' or 'train.other.500-00001-of-00102'
        
    Returns:
        Tuple of (subset_path, parquet_filename)
        e.g., ('clean', 'train.clean.100-00000-of-00029.parquet')
    """
    if "clean" in shard_id:
        subset = "clean"
    elif "other" in shard_id:
        subset = "other"
    else:
        raise ValueError(f"Cannot determine subset from shard_id: {shard_id}")
    
    parquet_filename = f"{shard_id}.parquet"
    return subset, parquet_filename


def download_shard(shard_id: str, cache_dir: str = "./cache") -> str:
    """
    Download a specific shard from the HuggingFace dataset.
    
    Args:
        shard_id: Shard ID like 'train.clean.100-00000-of-00029'
        cache_dir: Directory to cache downloaded files
        
    Returns:
        Path to the downloaded parquet file
    """
    subset, parquet_filename = parse_shard_id(shard_id)
    repo_path = f"{subset}/{parquet_filename}"
    
    logger.info(f"Downloading shard: {repo_path}")
    
    local_path = hf_hub_download(
        repo_id=HF_SOURCE_REPO,
        filename=repo_path,
        repo_type="dataset",
        cache_dir=cache_dir
    )
    
    return local_path


def process_shard(
    shard_id: str,
    mimi_encoder: MimiEncoder,
    hf_repo_id: str,
    output_dir: str = "./output",
    cache_dir: str = "./cache",
    batch_size: int = 16
):
    """
    Process a single shard: download, encode, and upload.
    Creates zero-shot TTS pairs from consecutive samples with same (speaker_id, chapter_id).
    
    Args:
        shard_id: Shard ID to process
        mimi_encoder: MimiEncoder instance
        hf_repo_id: HuggingFace repo ID to upload to
        output_dir: Directory for output parquet files
        cache_dir: Directory for caching downloaded files
        batch_size: Batch size for encoding
    """
    from collections import defaultdict
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Download the shard
    parquet_path = download_shard(shard_id, cache_dir)
    
    # Load the parquet file
    logger.info(f"Loading parquet file: {parquet_path}")
    dataset = Dataset.from_parquet(parquet_path)
    logger.info(f"Loaded {len(dataset)} samples from shard {shard_id}")
    
    # Step 1: Encode all audio and store with metadata
    encoded_samples = []
    
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
            
            encoded_samples.append({
                "id": batch["id"][i],
                "transcript": batch["text_normalized"][i],
                "speaker_id": batch["speaker_id"][i],
                "chapter_id": batch["chapter_id"][i],
                "audio_str": audio_str,
            })
    
    # Step 2: Group samples by (speaker_id, chapter_id)
    groups = defaultdict(list)
    for sample in encoded_samples:
        key = (sample["speaker_id"], sample["chapter_id"])
        groups[key].append(sample)
    
    logger.info(f"Found {len(groups)} unique (speaker_id, chapter_id) groups")
    
    # Step 3: Create zero-shot TTS pairs from consecutive samples in each group
    processed_data = []
    
    for (speaker_id, chapter_id), samples in groups.items():
        # Create pairs from consecutive samples
        for i in range(len(samples) - 1):
            sample_i = samples[i]
            sample_j = samples[i + 1]
            
            text_i = sample_i['transcript'].strip().strip('"').strip("'")
            text_j = sample_j['transcript'].strip().strip('"').strip("'")

            text_for_tts = (
                f"<|begin_of_text|>"
                f"<|text_start|>[0]{text_i}<|text_end|>"
                f"<|audio_start|>{sample_i['audio_str']}<|audio_end|>"
                f"<|text_start|>[0]{text_j}<|text_end|>"
                f"<|audio_start|>{sample_j['audio_str']}<|audio_end|>"
                f"<|end_of_text|>"
            )
            
            processed_data.append({
                "id": f"{sample_i['id']}#{sample_j['id']}",
                "text": text_for_tts,
                "speaker_id": speaker_id,
                "chapter_id": chapter_id,
            })
    
    logger.info(f"Created {len(processed_data)} zero-shot TTS pairs from {len(encoded_samples)} samples")
    
    # Create output dataset
    output_dataset = Dataset.from_dict({
        "id": [item["id"] for item in processed_data],
        "text": [item["text"] for item in processed_data],
        "speaker_id": [item["speaker_id"] for item in processed_data],
        "chapter_id": [item["chapter_id"] for item in processed_data],
    })
    
    # Save and upload
    output_filename = f"{shard_id}.parquet"
    output_path = Path(output_dir) / output_filename
    output_dataset.to_parquet(str(output_path))
    
    logger.info(f"Uploading {output_filename} to {hf_repo_id}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(output_path),
        path_in_repo=f"data/{output_filename}",
        repo_id=hf_repo_id,
        repo_type="dataset",
        commit_message=f"Add processed shard {shard_id}"
    )
    
    # Cleanup local files
    output_path.unlink()
    logger.info(f"Successfully processed shard {shard_id}: {len(processed_data)} zero-shot TTS pairs")


def get_existing_shards(hf_repo_id: str) -> set:
    """
    Get set of already processed shard IDs from the HuggingFace repo.
    
    Args:
        hf_repo_id: HuggingFace repo ID
        
    Returns:
        Set of shard IDs that have already been processed
    """
    api = HfApi()
    
    repo_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    
    existing_shards = set()
    for f in repo_files:
        if f.startswith("data/") and f.endswith(".parquet"):
            shard_id = f.replace("data/", "").replace(".parquet", "")
            existing_shards.add(shard_id)
    
    return existing_shards


def main():
    parser = argparse.ArgumentParser(description="Process LibriTTS-R shards with Mimi encoding")
    parser.add_argument("--shard-id", type=str, help="Single shard ID to process (e.g., train.clean.100-00000-of-00029)")
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
    existing_shards = get_existing_shards(args.hf_repo_id)
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
    # python process_libritts_r.py --shard-id train.clean.100-00000-of-00018 --hf-repo-id potsawee/libritts-r-mm-tts0
    # python process_libritts_r.py --shard-id-list shard_list.txt --hf-repo-id potsawee/libritts-r-mm-tts0

