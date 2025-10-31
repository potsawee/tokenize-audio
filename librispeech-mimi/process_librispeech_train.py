#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from typing import List
from pathlib import Path
import librosa
import numpy as np
import torch
from datasets import Dataset
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm
from huggingface_hub import HfApi
from utils import codes_to_chars, resample_audio

MLS_SAMPLE_RATE = 16000
MIMI_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 8 # we're using 8 coebooks instead of 32
CODEBOOK_SIZE = 2048

# Configure logging with immediate output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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

def check_existing_chunks(hf_repo_id: str, split_name: str) -> int:
    """
    Check how many parquet chunks already exist for a given split.
    
    Args:
        hf_repo_id: HuggingFace repo ID
        split_name: Split name (e.g., 'train_clean_100')
    
    Returns:
        Number of existing chunks (0 if none found or repo doesn't exist)
    """
    api = HfApi()
    
    repo_files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    
    # Look for parquet files matching the pattern: data/{split_name}-XXXXX-of-YYYYY.parquet
    parquet_files = [
        f for f in repo_files 
        if f.startswith(f"data/{split_name}-") and f.endswith(".parquet")
    ]
    
    logger.info(f"Found {len(parquet_files)} existing parquet files for {split_name}")
    return len(parquet_files)

def process_librispeech(data_path: str, hf_repo_id: str, split: str, chunk_size: int = 10000):
    mimi_encoder = MimiEncoder(device="cuda")
    split_name = split.replace("-", "_")

    # Load dataset
    logger.info(f"Loading dataset from {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    total_samples = len(data)
    logger.info(f"Total samples in dataset: {total_samples}")

    # Check for existing chunks to enable resumability
    existing_chunks = check_existing_chunks(hf_repo_id, split_name)
    start_index = existing_chunks * chunk_size
    
    if start_index > 0:
        logger.info(f"Found {existing_chunks} existing chunks. Resuming from entry {start_index}")
        if start_index >= total_samples:
            logger.info("All samples already processed. Nothing to do.")
            return
    else:
        logger.info("Starting from the beginning")

    # Create temporary directory for parquet files
    temp_dir = Path("./tmp_parquets")
    temp_dir.mkdir(exist_ok=True)
    
    # Process data in chunks
    for chunk_idx in range(existing_chunks, (total_samples + chunk_size - 1) // chunk_size):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_data = data[chunk_start:chunk_end]
        
        logger.info(f"Processing chunk {chunk_idx}: entries {chunk_start} to {chunk_end-1}")
        
        combined_data = []
        
        for sample in tqdm(chunk_data, desc=f"Processing samples [{chunk_start}:{chunk_end}]"):
            transcript = sample['transcript'].lower()
            file_path = sample['file_path']
            file_id = sample['file_path'].split("LibriSpeech")[-1][1:].replace(".flac", "")
            audio, sr = librosa.load(file_path, sr=None)
            audio_resampled = resample_audio(audio, orig_sr=sr, target_sr=MIMI_SAMPLE_RATE)

            audio_codes = mimi_encoder.encode_audio_chunk(audio_resampled, MIMI_SAMPLE_RATE)
            audio_codes = audio_codes[:NUM_CODEBOOKS, :]
            audio_str = codes_to_chars(audio_codes, codebook_size=CODEBOOK_SIZE)

            text_for_asr = f"<|audio_start|>{audio_str}<|audio_end|><|text_start|>{transcript}<|text_end|>"
            text_for_tts = f"<|text_start|>{transcript}<|text_end|><|audio_start|>{audio_str}<|audio_end|>"
            
            # Add TTS sample with _type1 suffix
            combined_data.append({
                "file_id": f"{file_id}_type1",
                "text": text_for_tts,
            })
            # Add ASR sample with _type2 suffix
            combined_data.append({
                "file_id": f"{file_id}_type2",
                "text": text_for_asr,
            })

        # Push this chunk to HuggingFace with explicit parquet naming
        logger.info(f"Pushing chunk {chunk_idx} to HuggingFace")
        
        # Calculate total chunks for naming
        total_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        # Save and upload combined chunk (contains both ASR and TTS samples)
        dataset = Dataset.from_dict({
            "file_id": [item["file_id"] for item in combined_data],
            "text": [item["text"] for item in combined_data]
        })
        parquet_name = f"{split_name}-{chunk_idx:05d}-of-{total_chunks:05d}.parquet"
        temp_path = temp_dir / parquet_name
        dataset.to_parquet(str(temp_path))
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(temp_path),
            path_in_repo=f"data/{parquet_name}",
            repo_id=hf_repo_id,
            repo_type="dataset",
            commit_message=f"Add {split_name} chunk {chunk_idx}"
        )
        temp_path.unlink()
        logger.info(f"Successfully uploaded chunk {chunk_idx} as {parquet_name} ({len(combined_data)} samples total: {len(combined_data)//2} TTS + {len(combined_data)//2} ASR)")
    
    # Clean up temporary directory
    if temp_dir.exists():
        temp_dir.rmdir()
    
    logger.info(f"All chunks processed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Process a MLS-en shard with Mimi encoding (with batching)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the LibriSpeech JSON dataset file")
    parser.add_argument("--hf-repo-id", type=str, required=True, help="HuggingFace repo ID for the dataset")
    parser.add_argument("--split", type=str, default="dev-clean", help="HF split to named")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of entries per parquet file (default: 10000)")
    args = parser.parse_args()
    process_librispeech(args.data_path, args.hf_repo_id, args.split, args.chunk_size)
    
if __name__ == "__main__":
    main()
    # usage: python process_librispeech_train.py --data-path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/train-clean-100.json --hf-repo-id potsawee/librispeech-mm-pretrain --split train-clean-100 --chunk-size 10000
    # usage: python process_librispeech_train.py --data-path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/train-clean-360.json --hf-repo-id potsawee/librispeech-mm-pretrain --split train-clean-360 --chunk-size 10000
    # usage: python process_librispeech_train.py --data-path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/train-other-500.json --hf-repo-id potsawee/librispeech-mm-pretrain --split train-other-500 --chunk-size 10000

