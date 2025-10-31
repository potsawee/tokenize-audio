#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from typing import List
import librosa
import numpy as np
import torch
from datasets import Dataset
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm
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
        
def process_librispeech(data_path: str, hf_repo_id: str, split: str):

    mimi_encoder = MimiEncoder(device="cuda")

    # Load dataset
    print(f"Loading dataset from {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    list_of_text_asr = []
    list_of_text_tts = []
    for sample in tqdm(data, desc="Processing samples"):
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
        list_of_text_asr.append({
            "file_id": file_id,
            "text": text_for_asr,
        })
        list_of_text_tts.append({
            "file_id": file_id,
            "text": text_for_tts,
        })

    # Push to HuggingFace
    split_name = split.replace("-", "_")
    
    logger.info(f"Pushing ASR data to {hf_repo_id} as split '{split_name}_asr'")
    dataset_asr = Dataset.from_dict({
        "file_id": [item["file_id"] for item in list_of_text_asr],
        "text": [item["text"] for item in list_of_text_asr]
    })
    dataset_asr.push_to_hub(hf_repo_id, split=f"{split_name}_asr")
    logger.info(f"Successfully pushed {len(list_of_text_asr)} ASR samples")
    
    logger.info(f"Pushing TTS data to {hf_repo_id} as split '{split_name}_tts'")
    dataset_tts = Dataset.from_dict({
        "file_id": [item["file_id"] for item in list_of_text_tts],
        "text": [item["text"] for item in list_of_text_tts]
    })
    dataset_tts.push_to_hub(hf_repo_id, split=f"{split_name}_tts")
    logger.info(f"Successfully pushed {len(list_of_text_tts)} TTS samples")


def main():
    parser = argparse.ArgumentParser(description="Process a MLS-en shard with Mimi encoding (with batching)")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the LibriSpeech JSON dataset file")
    parser.add_argument("--hf-repo-id", type=str, required=True, help="HuggingFace repo ID for the dataset")
    parser.add_argument("--split", type=str, default="dev-clean", help="HF split to named")
    args = parser.parse_args()
    process_librispeech(args.data_path, args.hf_repo_id, args.split)
    
if __name__ == "__main__":
    main()
    # usage: python process_librispeech.py --data-path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/dev-clean.json --hf-repo-id potsawee/librispeech-mm --split dev-clean
    # usage: python process_librispeech.py --data-path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/test-clean.json --hf-repo-id potsawee/librispeech-mm --split test-clean
    # usage: python process_librispeech.py --data-path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/dev-other.json --hf-repo-id potsawee/librispeech-mm --split dev-other
    # usage: python process_librispeech.py --data-path /nlp/scr/potsawee/workspace/data/LibriSpeech/data_json/test-other.json --hf-repo-id potsawee/librispeech-mm --split test-other

