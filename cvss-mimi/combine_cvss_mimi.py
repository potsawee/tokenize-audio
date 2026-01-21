#!/usr/bin/env python3
"""
Script to download CVSS dataset from rma9248/converted_cvss, combine all languages,
and upload to a new HuggingFace repository.
"""
import argparse
import logging
import sys
from typing import List

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# All available languages in the dataset
LANGUAGES = [
    "ar", "ca","cy", "de", "es", "et", "fa", "fr", "id", "it", "ja",
    "lv", "mn", "nl", "pt", "ru", "sl", "sv-SE", "ta", "tr", "zh-CN"
]

SOURCE_REPO = "rma9248/converted_cvss"
SPLITS = ["train", "validation", "test"]


def load_language_split(lang: str, split: str) -> pd.DataFrame:
    """
    Load a specific language and split from the source repository.
    
    Args:
        lang: Language code (e.g., "de", "fr")
        split: Dataset split ("train", "validation", or "test")
        
    Returns:
        DataFrame with the loaded data
    """
    logger.info(f"Loading {lang}/{split}...")
    dataset = load_dataset(
        SOURCE_REPO, 
        data_dir=lang,
        split=split,
        download_mode="force_redownload"
    )
    df = dataset.to_pandas()
    # Add language column for reference
    df["lang"] = lang
    return df


def combine_text_fields(row: pd.Series, method: str = "method1") -> str:
    """
    Combine text fields into a single text string.
    Format: {original_text}{original_audio_str}{translated_text}{translated_audio_str}
    
    Available fields in row:
        - id, original_text, original_audio_str, translated_text, translated_audio_str, lang
    """
    lang = row['lang']
    text = "<|begin_of_text|>"
    if method == "method1":
        text += f"<|audio_start|>{row['original_audio_str']}<|audio_end|>"
        text += f"<|text_start|><language>{lang}</language>{row['original_text']}<|text_end|>"
        text += f"<|text_start|><language>en</language>{row['translated_text']}<|text_end|>"
        text += f"<|audio_start|>{row['translated_audio_str']}<|audio_end|>"
        text += "<|end_of_text|>"
    elif method == "method2":
        pass
    else:
        raise ValueError(f"Invalid method: {method}")
    
    return text


def process_split(split: str, languages: List[str], method: str = "method1") -> pd.DataFrame:
    """
    Process a single split across all languages.
    
    Args:
        split: Dataset split name
        languages: List of language codes to process
        
    Returns:
        Combined DataFrame with id and text columns
    """
    all_data = []
    
    for lang in tqdm(languages, desc=f"Processing {split}"):
        df = load_language_split(lang, split)
        all_data.append(df)
    
    # Combine all languages
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined {split}: {len(combined_df)} total examples from {len(languages)} languages")
    
    # Create new DataFrame with only id and combined text
    result_df = pd.DataFrame({
        "id": combined_df["id"],
        "lang": combined_df["lang"],
        "text": combined_df.apply(lambda row: combine_text_fields(row, method), axis=1)
    })
    
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Combine CVSS dataset from all languages and upload to HuggingFace"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="Target HuggingFace repo ID (e.g., 'username/cvss-combined')"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=LANGUAGES,
        help="Languages to process (default: all languages)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=SPLITS,
        help="Splits to process (default: train, validation, test)"
    )
    parser.add_argument(
        "--save-local",
        type=str,
        default=None,
        help="Optional: Save parquet files locally to this directory"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="method1",
        choices=["method1", "method2"],
        help="Method for combining text fields (default: method1)"
    )
    args = parser.parse_args()
    
    logger.info(f"Source repository: {SOURCE_REPO}")
    logger.info(f"Target repository: {args.hf_repo_id}")
    logger.info(f"Languages: {args.languages}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Method: {args.method}")
    
    for split in args.splits:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*50}")
        
        # Process and combine all languages for this split
        result_df = process_split(split, args.languages, args.method)
        
        # Optionally save locally
        if args.save_local:
            import os
            os.makedirs(args.save_local, exist_ok=True)
            local_path = os.path.join(args.save_local, f"{split}.parquet")
            result_df.to_parquet(local_path)
            logger.info(f"Saved locally to {local_path}")
        
        # Convert to HuggingFace Dataset and push
        dataset = Dataset.from_pandas(result_df)
        logger.info(f"Pushing {split} to {args.hf_repo_id} ({len(dataset)} examples)")
        dataset.push_to_hub(args.hf_repo_id, split=split)
        logger.info(f"Successfully pushed {split}")
    
    logger.info(f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
    # Example usage:
    # python combine_cvss_mimi.py --hf-repo-id potsawee/cvss-mm-method1 --method method1 
    # python combine_cvss_mimi.py --hf-repo-id potsawee/cvss-mm-method1.1 --method method1 # v1.1 just add missing 'ca'  
