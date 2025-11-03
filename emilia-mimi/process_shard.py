#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import hf_hub_download, HfApi
from transformers import MimiModel, AutoFeatureExtractor
from tqdm import tqdm
from utils import codes_to_chars, resample_audio

MIMI_SAMPLE_RATE = 24000
NUM_CODEBOOKS = 8 # we're using 8 coebooks instead of 32
CODEBOOK_SIZE = 2048

# Force unbuffered output from the start
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

# Configure logging with immediate output
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        FlushStreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print("=" * 80, flush=True)
print("EMILIA SHARD PROCESSOR STARTING", flush=True)
print("=" * 80, flush=True)

class MimiEncoder:
    """Wrapper for Mimi model encoding."""
    
    def __init__(self, model_id: str = "kyutai/mimi", device: str = "cuda"):
        print(f"Loading Mimi model: {model_id}", flush=True)
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

class EmiliaShardProcessor:
    """Process a single Emilia shard."""
    
    def __init__(
        self,
        split: str,
        shard_id: str,
        work_dir: Path,
        output_dir: Path,
        progress_dir: Path,
        batch_size: int,
        hf_repo_id: str,
        device: str = "cuda",
        cache_interval: int = 2048,
    ):
        assert split in ["Emilia", "Emilia-YODAS"]
        lang = shard_id.split("-")[0]
        assert lang in ["EN", "DE", "FR", "JA", "KO", "ZH"]

        self.split = split
        self.lang = lang
        self.shard_id = shard_id
        self.work_dir = work_dir
        self.output_dir = output_dir
        self.progress_dir = progress_dir
        self.device = device
        self.batch_size = batch_size
        self.hf_repo_id = hf_repo_id
        self.cache_interval = cache_interval

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

        self.mimi_encoder = MimiEncoder(device=device)

        # paths
        self.audio_tar_path = work_dir / shard_id / "shard.tar"
        self.audio_extract_dir = work_dir / shard_id / "extracted"
        self.extraction_marker = work_dir / shard_id / "extracted.complete"
        self.audio_cache_path = work_dir / shard_id / "audio_str_cache.json"
        self.progress_file = progress_dir / f"{shard_id}.json"

        # make dir for audio_tar_path, audio_extract_dir, etc
        self.audio_tar_path.parent.mkdir(parents=True, exist_ok=True)

    def download_shard(self, repo_id: str, filename: str, dest_path: Path, max_retries: int = 3) -> bool:
        """Download a file from HuggingFace with retry logic and authentication."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {filename} from {repo_id} to {dest_path} (attempt {attempt + 1}/{max_retries})")
                
                # Use huggingface_hub which handles authentication automatically
                # Download to a temporary location in the work directory
                downloaded_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=dest_path.parent,
                )
                
                # Move to the exact destination path if needed
                downloaded_path = Path(downloaded_path)
                if downloaded_path != dest_path:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    if downloaded_path.exists():
                        # Remove destination if it exists
                        if dest_path.exists() and dest_path.is_file():
                            os.remove(dest_path)
                        downloaded_path.rename(dest_path)
                    else:
                        logger.error(f"Downloaded file not found at {downloaded_path}")
                        return False
                
                logger.info(f"Successfully downloaded {dest_path.name}")
                return True
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to download {filename} after {max_retries} attempts")
                    return False
        return False

    def is_extraction_complete(self) -> bool:
        """Check if extraction is complete by checking for marker file."""
        return self.extraction_marker.exists()
    
    def save_audio_cache(self, dict_of_audio_str: Dict[str, str]) -> None:
        """Save audio processing cache to JSON file."""
        try:
            msg = f"Saving audio cache to {self.audio_cache_path} ({len(dict_of_audio_str)} files processed)"
            print(msg, flush=True)
            logger.info(msg)
            with open(self.audio_cache_path, 'w') as f:
                json.dump(dict_of_audio_str, f)
            print("Cache saved successfully", flush=True)
        except Exception as e:
            err_msg = f"Failed to save audio cache: {e}"
            print(err_msg, flush=True)
            logger.error(err_msg)
    
    def load_audio_cache(self) -> Dict[str, str]:
        """Load audio processing cache from JSON file."""
        if not self.audio_cache_path.exists():
            print(f"No existing audio cache found at {self.audio_cache_path}, starting from scratch", flush=True)
            logger.info("No existing audio cache found, starting from scratch")
            return {}
        
        try:
            with open(self.audio_cache_path, 'r') as f:
                cache = json.load(f)
            msg = f"Loaded audio cache with {len(cache)} already processed files from {self.audio_cache_path}"
            print(msg, flush=True)
            logger.info(msg)
            return cache
        except Exception as e:
            print(f"Failed to load audio cache: {e}, starting from scratch", flush=True)
            logger.error(f"Failed to load audio cache: {e}, starting from scratch")
            return {}
    
    def delete_audio_cache(self) -> None:
        """Delete audio cache file after successful completion."""
        if self.audio_cache_path.exists() and self.audio_cache_path.is_file():
            logger.info(f"Deleting audio cache {self.audio_cache_path}")
            os.remove(self.audio_cache_path)
    
    def is_shard_already_processed(self) -> bool:
        """Check if the shard has already been processed and uploaded to HuggingFace."""
        try:
            api = HfApi()
            repo_path = f"{self.split}/{self.lang}/{self.shard_id}.parquet"
            
            # List files in the repo
            files = api.list_repo_files(repo_id=self.hf_repo_id, repo_type="dataset")
            
            if repo_path in files:
                print(f"Shard already processed: {repo_path} exists in {self.hf_repo_id}", flush=True)
                logger.info(f"Shard already processed: {repo_path} exists in {self.hf_repo_id}")
                return True
            else:
                print(f"Shard not yet processed: {repo_path} not found in {self.hf_repo_id}", flush=True)
                logger.info(f"Shard not yet processed: {repo_path} not found in {self.hf_repo_id}")
                return False
        except Exception as e:
            print(f"Error checking if shard is processed: {e}, assuming not processed", flush=True)
            logger.warning(f"Error checking if shard is processed: {e}, assuming not processed")
            return False
    
    def cleanup_work_directory(self) -> None:
        """Clean up the work directory for this shard."""
        shard_work_dir = self.work_dir / self.shard_id
        
        if shard_work_dir.exists():
            print(f"Cleaning up work directory: {shard_work_dir}", flush=True)
            logger.info(f"Cleaning up work directory: {shard_work_dir}")
            
            # Delete tar file if it exists
            if self.audio_tar_path.exists():
                logger.info(f"Deleting tar file: {self.audio_tar_path}")
                os.remove(self.audio_tar_path)
            
            # Delete extraction directory if it exists
            if self.audio_extract_dir.exists():
                logger.info(f"Deleting extraction directory: {self.audio_extract_dir}")
                shutil.rmtree(self.audio_extract_dir)
            
            # Delete extraction marker if it exists
            if self.extraction_marker.exists():
                logger.info(f"Deleting extraction marker: {self.extraction_marker}")
                os.remove(self.extraction_marker)
            
            # Delete cache if it exists
            self.delete_audio_cache()
            
            # Delete the shard work directory if it's now empty
            if shard_work_dir.exists() and not any(shard_work_dir.iterdir()):
                logger.info(f"Deleting empty shard work directory: {shard_work_dir}")
                shard_work_dir.rmdir()
            
            print(f"Work directory cleanup complete", flush=True)
            logger.info("Work directory cleanup complete")
        else:
            logger.info(f"Work directory does not exist, nothing to clean up: {shard_work_dir}")
    
    def write_progress(self, status: str, num_samples: int = None) -> None:
        """Write progress status to progress file."""
        import datetime
        
        progress_data = {
            "shard_id": self.shard_id,
            "split": self.split,
            "lang": self.lang,
            "status": status,
            "timestamp": datetime.datetime.now().isoformat(),
            "hf_repo_id": self.hf_repo_id,
        }
        
        if num_samples is not None:
            progress_data["num_samples"] = num_samples
        
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        logger.info(f"Progress written to {self.progress_file}: {status}")
        print(f"Progress written to {self.progress_file}: {status}", flush=True)
    
    def extract_audio_tar(self) -> bool:
        """Extract audio tar file and create completion marker."""
        try:
            # Clean up incomplete extraction if directory exists but marker doesn't
            if self.audio_extract_dir.exists() and not self.extraction_marker.exists():
                logger.warning(f"Found incomplete extraction directory {self.audio_extract_dir}, cleaning up")
                shutil.rmtree(self.audio_extract_dir)
            
            if not self.audio_tar_path.exists():
                logger.error(f"Tar file {self.audio_tar_path} does not exist")
                return False
            
            # Get tar file size
            tar_size_mb = self.audio_tar_path.stat().st_size / (1024 * 1024)
            logger.info(f"Extracting {self.audio_tar_path} ({tar_size_mb:.2f} MB) to {self.audio_extract_dir}")
            
            self.audio_extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Remove marker if it exists (in case of incomplete previous extraction)
            if self.extraction_marker.exists() and self.extraction_marker.is_file():
                os.remove(self.extraction_marker)
            
            start_time = time.time()
            
            with tarfile.open(self.audio_tar_path, 'r') as tar:
                members = tar.getmembers()
                logger.info(f"Total files in archive: {len(members)}")
                
                # Extract all files with progress bar
                tar.extractall(path=self.audio_extract_dir, members=tqdm(
                    members, 
                    desc="Extracting files",
                    unit="files"
                ))
            
            extraction_time = time.time() - start_time
            logger.info(f"Extraction completed in {extraction_time:.1f} seconds")
            
            # Create completion marker
            self.extraction_marker.touch()
            
            # Delete tar file to save disk space
            if self.audio_tar_path.exists() and self.audio_tar_path.is_file():
                logger.info(f"Deleting tar file {self.audio_tar_path} to save {tar_size_mb:.2f} MB")
                os.remove(self.audio_tar_path)
            
            logger.info(f"Successfully extracted {self.audio_tar_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract {self.audio_tar_path}: {e}")
            # Remove marker if extraction failed
            if self.extraction_marker.exists() and self.extraction_marker.is_file():
                os.remove(self.extraction_marker)
        return False

    def process(self):
        # Step 0: Check if shard is already processed
        print("=" * 80, flush=True)
        print(f"Starting processing for shard: {self.shard_id}", flush=True)
        print("=" * 80, flush=True)
        
        if self.is_shard_already_processed():
            logger.info("Shard already fully processed, cleaning up any remaining local files")
            print("Shard already fully processed, cleaning up any remaining local files", flush=True)
            self.cleanup_work_directory()
            self.write_progress("completed_already_uploaded")
            print("=" * 80, flush=True)
            print("Processing complete (shard was already done)", flush=True)
            print("=" * 80, flush=True)
            return
        
        # Step1: Download the shard from Emilia
        repo_id = "amphion/Emilia-Dataset"
        filename = f"{self.split}/{self.lang}/{self.shard_id}.tar"
        
        extraction_complete = self.is_extraction_complete()
        
        if extraction_complete:
            logger.info(f"Extraction already complete for {self.shard_id}, skipping download and extraction")
        else:
            # Only download if tar doesn't exist (it will be deleted after extraction)
            if not self.audio_tar_path.exists():
                logger.info(f"Downloading {filename} from {repo_id}")
                if not self.download_shard(repo_id, filename, self.audio_tar_path):
                    logger.error(f"Failed to download {filename}, aborting")
                    return
            else:
                logger.info(f"Audio tar already downloaded at {self.audio_tar_path}, reusing")

            # Step2: Extract the audio tar file
            if not self.extract_audio_tar():
                logger.error(f"Failed to extract {self.audio_tar_path}, aborting")
                return

        # ------ Step3: Process the audio files ------
        # get the list of audio files from:
        # "self.audio_extract_dir/{file_id}.json"
        # "self.audio_extract_dir/{file_id}.mp3"
        audio_files = list(self.audio_extract_dir.rglob("*.mp3"))
        audio_files = [file.name.replace(".mp3", "") for file in audio_files]
        # sort the audio files by the file_id
        audio_files.sort()

        # Load existing cache if available (for resumability)
        dict_of_audio_str = self.load_audio_cache()
        
        # Filter out already processed files
        remaining_files = [f for f in audio_files if f not in dict_of_audio_str]
        logger.info(f"Total audio files: {len(audio_files)}, Already processed: {len(dict_of_audio_str)}, Remaining: {len(remaining_files)}")
        
        if len(remaining_files) == 0:
            logger.info("All audio files already processed, skipping step 3")
        else:
            list_of_audio_arr, list_of_audio_id, list_of_transcript = [], [], []
            files_processed_since_last_cache = 0
            total_remaining = len(remaining_files)
            total_files = len(audio_files)
            already_processed = len(dict_of_audio_str)
            
            logger.info(f"Starting to process {total_remaining} remaining audio files")
            
            for idx, audio_file in enumerate(remaining_files, 1):
                audio_file_path = self.audio_extract_dir / f"{audio_file}.mp3"
                json_file_path = self.audio_extract_dir / f"{audio_file}.json"
                with open(json_file_path, 'r') as f:
                    metadata = json.load(f)
                transcript = metadata['text']
                audio_arr, sr = librosa.load(audio_file_path, sr=None)
                if sr != MIMI_SAMPLE_RATE:
                    audio_arr = resample_audio(audio_arr, sr, MIMI_SAMPLE_RATE)
                    logger.info(f"Resampled {audio_file} from {sr} to {MIMI_SAMPLE_RATE} Hz")
                    # print(f"Resampled {audio_file} from {sr} to {MIMI_SAMPLE_RATE} Hz", flush=True)
                    
                list_of_audio_arr.append(audio_arr)
                list_of_audio_id.append(audio_file) 
                list_of_transcript.append(transcript)
                
                # Log progress every 100 files (show global progress)
                if idx % 100 == 0:
                    global_processed = already_processed + idx
                    msg = f"Progress: {global_processed}/{total_files} files ({100*global_processed/total_files:.1f}%)"
                    print(msg, flush=True)
                    logger.info(msg)
                
                if len(list_of_audio_arr) < self.batch_size:
                    # continue accumulating the audio arrays until the batch size is reached
                    continue

                # encode the audio arrays
                list_of_audio_codes = self.mimi_encoder.encode_audio_batch(list_of_audio_arr, MIMI_SAMPLE_RATE)
                assert len(list_of_audio_codes) == len(list_of_audio_id)
                assert len(list_of_audio_codes) == len(list_of_transcript)
                for audio_codes, audio_file, transcript in zip(list_of_audio_codes, list_of_audio_id, list_of_transcript):
                    audio_codes = audio_codes[:NUM_CODEBOOKS, :]
                    audio_str = codes_to_chars(audio_codes, codebook_size=CODEBOOK_SIZE)
                    dict_of_audio_str[audio_file] = {
                        "audio_str": audio_str,
                        "transcript": transcript,
                    }
                    files_processed_since_last_cache += 1
                
                # clear the audio arrays
                list_of_audio_arr, list_of_audio_id, list_of_transcript = [], [], []
                
                # Save cache periodically for resumability
                if files_processed_since_last_cache >= self.cache_interval:
                    self.save_audio_cache(dict_of_audio_str)
                    files_processed_since_last_cache = 0

            # process the remaining audio arrays
            if len(list_of_audio_arr) > 0:
                # encode the audio arrays
                list_of_audio_codes = self.mimi_encoder.encode_audio_batch(list_of_audio_arr, MIMI_SAMPLE_RATE)
                assert len(list_of_audio_codes) == len(list_of_audio_id)
                assert len(list_of_audio_codes) == len(list_of_transcript)
                for audio_codes, audio_file, transcript in zip(list_of_audio_codes, list_of_audio_id, list_of_transcript):
                    audio_codes = audio_codes[:NUM_CODEBOOKS, :]
                    audio_str = codes_to_chars(audio_codes, codebook_size=CODEBOOK_SIZE)
                    dict_of_audio_str[audio_file] = {
                        "audio_str": audio_str,
                        "transcript": transcript,
                    }
            
            # Final cache save
            self.save_audio_cache(dict_of_audio_str)
            logger.info(f"Completed processing all {len(audio_files)} audio files")

        # dict_of_audio_str should be a dictionary of audio_file_id -> audio_str
        # end of step3: process the audio files ------


        # Step4: build document-level text for pretraining
        # group/merge the audio files into documents (i.e., same audio clip)
        # audio_file --> EN_B00000_S00040_W000004 --> "{shard_id}_{speaker_id}_{utterance_id}"
        documents = {}
        for audio_file in audio_files:
            items = audio_file.split("_")
            shard_speaker_id = "_".join(items[:-1])
            # utterance_id = items[-1]
            
            if shard_speaker_id not in documents:
                documents[shard_speaker_id] = []
            documents[shard_speaker_id].append(audio_file)

        pretraining_data = []
        for doc, audio_files_in_doc in documents.items():
            doc_text_for_type1 = ""
            doc_text_for_type2 = ""

            for audio_file in audio_files_in_doc:
                audio_str = dict_of_audio_str[audio_file]["audio_str"]
                transcript = dict_of_audio_str[audio_file]["transcript"]
                doc_text_for_type1 += f"<|text_start|>{transcript}<|text_end|><|audio_start|>{audio_str}<|audio_end|>"
                doc_text_for_type2 += f"<|audio_start|>{audio_str}<|audio_end|><|text_start|>{transcript}<|text_end|>"


            doc_text_for_type1 = f"<|begin_of_text|>{doc_text_for_type1}<|end_of_text|>"
            doc_text_for_type2 = f"<|begin_of_text|>{doc_text_for_type2}<|end_of_text|>"

            pretraining_data.append({
                "id": doc + "_type1",
                "split": f"{self.split}-{self.shard_id}",
                "text": doc_text_for_type1,
            })
            pretraining_data.append({
                "id": doc + "_type2",
                "split": f"{self.split}-{self.shard_id}",
                "text": doc_text_for_type2,
            })

        # Step5: Push to HuggingFace
        logger.info(f"Creating dataset with {len(pretraining_data)} samples")
        print(f"Creating dataset with {len(pretraining_data)} samples", flush=True)
        
        dataset = Dataset.from_dict({
            "id": [item["id"] for item in pretraining_data],
            "split": [item["split"] for item in pretraining_data],
            "text": [item["text"] for item in pretraining_data]
        })
        
        # Save as parquet locally
        parquet_filename = f"{self.shard_id}.parquet"
        local_parquet_path = self.output_dir / parquet_filename
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dataset to {local_parquet_path}")
        print(f"Saving dataset to {local_parquet_path}", flush=True)
        dataset.to_parquet(local_parquet_path)
        
        # Upload to HuggingFace with the correct folder structure
        repo_path = f"{self.split}/{self.lang}/{parquet_filename}"
        logger.info(f"Uploading to HuggingFace: {self.hf_repo_id}/{repo_path}")
        print(f"Uploading to HuggingFace: {self.hf_repo_id}/{repo_path}", flush=True)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(local_parquet_path),
            path_in_repo=repo_path,
            repo_id=self.hf_repo_id,
            repo_type="dataset",
        )
        
        logger.info(f"Successfully uploaded {len(pretraining_data)} samples to {self.hf_repo_id}/{repo_path}")
        print(f"Successfully uploaded {len(pretraining_data)} samples to {self.hf_repo_id}/{repo_path}", flush=True)
        
        # Verify upload and clean up local parquet file
        try:
            files_in_repo = api.list_repo_files(repo_id=self.hf_repo_id, repo_type="dataset")
            if repo_path in files_in_repo:
                logger.info(f"Verified: {repo_path} exists in HF repo, deleting local copy")
                print(f"Verified: {repo_path} exists in HF repo, deleting local copy", flush=True)
                if local_parquet_path.exists() and local_parquet_path.is_file():
                    parquet_size_mb = local_parquet_path.stat().st_size / (1024 * 1024)
                    os.remove(local_parquet_path)
                    logger.info(f"Deleted local parquet file, freed {parquet_size_mb:.2f} MB")
                    print(f"Deleted local parquet file, freed {parquet_size_mb:.2f} MB", flush=True)
            else:
                logger.warning(f"Could not verify {repo_path} in HF repo, keeping local copy")
                print(f"Could not verify {repo_path} in HF repo, keeping local copy", flush=True)
        except Exception as e:
            logger.warning(f"Error verifying upload: {e}, keeping local copy")
            print(f"Error verifying upload: {e}, keeping local copy", flush=True)
        
        # Step6: Write progress and clean up work directory after successful completion
        self.write_progress("completed", num_samples=len(pretraining_data))
        
        print("=" * 80, flush=True)
        print("All processing complete, cleaning up work directory", flush=True)
        print("=" * 80, flush=True)
        self.cleanup_work_directory()
        
        print("=" * 80, flush=True)
        print(f"Successfully completed processing shard: {self.shard_id}", flush=True)
        print("=" * 80, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Process an Emilia shard with Mimi encoding")
    parser.add_argument("--split", type=str, default="Emilia", help="Split to process")
    parser.add_argument("--shard-id", type=str, default="EN-B000000", help="Shard ID")
    parser.add_argument("--work-dir", type=str, default="./work", help="Working directory for temporary files")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory for processed files")
    parser.add_argument("--progress-dir", type=str, default="./progress", help="Directory for progress tracking")
    parser.add_argument("--device", type=str, default="cuda", help="Device for Mimi model (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding chunks (default: 16)")
    parser.add_argument("--cache-interval", type=int, default=2048, help="Number of audio files to process before saving cache (default: 2048)")
    parser.add_argument("--hf-repo-id", type=str, default="potsawee/emilia-mm-pretrain", help="HuggingFace repo ID for the dataset")
    args = parser.parse_args()
    processor = EmiliaShardProcessor(
        split=args.split,
        shard_id=args.shard_id,
        work_dir=Path(args.work_dir),
        output_dir=Path(args.output_dir),
        progress_dir=Path(args.progress_dir),
        batch_size=args.batch_size,
        hf_repo_id=args.hf_repo_id,
        device=args.device,
        cache_interval=args.cache_interval,
    )
    processor.process()
    
if __name__ == "__main__":
    main()
    # Emilia-EN
    # usage: python process_shard.py --split Emilia --shard-id EN-B000000 --hf-repo-id potsawee/emilia-mm-pretrain --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work --output-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/output --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress --device cuda --batch-size 64

    # Emilia-DE
    # usage: python process_shard.py --split Emilia --shard-id DE-B000000 --hf-repo-id potsawee/emilia-mm-pretrain --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work-de --output-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/output-de --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress-de --device cuda --batch-size 64

    # Emilia-ZH
    # usage: python process_shard.py --split Emilia --shard-id ZH-B000000 --hf-repo-id potsawee/emilia-mm-pretrain --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work-zh --output-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/output-zh --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress-zh --device cuda --batch-size 64

    # Emilia-YODAS-DE
    # usage: python process_shard.py --split Emilia-YODAS --shard-id DE-B000000 --hf-repo-id potsawee/emilia-mm-pretrain --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work-yodas --output-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/output-yodas --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress-yodas --device cuda --batch-size 64

