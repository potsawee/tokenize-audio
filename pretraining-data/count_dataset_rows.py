#!/usr/bin/env python3
"""
Count rows in HuggingFace dataset by reading only parquet metadata (efficient).

This script uses HTTP range requests to read ONLY the metadata from parquet files,
not the full file content. This is very efficient and fast - it only downloads
a few KB of metadata per file instead of GB of actual data.

Usage:
    # Count all languages (plain text format)
    python count_dataset_rows.py --repo-id potsawee/yodas2-mm-pretrain
    
    # Count specific language with Markdown format (for README.md)
    python count_dataset_rows.py --repo-id potsawee/yodas2-mm-pretrain --language th --markdown
    
    # Count all languages with Markdown output
    python count_dataset_rows.py --repo-id potsawee/yodas2-mm-pretrain --markdown --output-file stats.md
"""

import argparse
import re
from collections import defaultdict
from typing import Dict, Optional

import fsspec
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url
from tqdm import tqdm


def get_parquet_files(repo_id: str, language: Optional[str] = None) -> Dict[str, list]:
    """
    Get list of parquet files from HuggingFace repo, grouped by language.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        language: Optional language filter (e.g., 'en', 'th')
    
    Returns:
        Dictionary mapping language codes to lists of parquet file paths
    """
    api = HfApi()
    
    print(f"Fetching file list from {repo_id}...")
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    
    # Filter parquet files
    parquet_files = [f for f in files if f.endswith('.parquet')]
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Group by language (extract from path like "en000/00000001.parquet" -> "en")
    language_files = defaultdict(list)
    
    for file_path in parquet_files:
        # Extract language from shard name (e.g., "en000" -> "en")
        match = re.match(r'([a-z]{2})\d+/', file_path)
        if match:
            lang = match.group(1)
            if language is None or lang == language:
                language_files[lang].append(file_path)
    
    return dict(language_files)


def count_rows_in_parquet(repo_id: str, file_path: str) -> int:
    """
    Count rows in a parquet file by reading only metadata using HTTP range requests.
    Does NOT download the full file - only reads the metadata at the end of the file.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        file_path: Path to parquet file in the repo
    
    Returns:
        Number of rows in the parquet file
    """
    # Construct HuggingFace URL for the file
    url = hf_hub_url(repo_id=repo_id, filename=file_path, repo_type="dataset")
    
    # Open file using fsspec with HTTP support
    # fsspec automatically uses HTTP range requests to seek to the end and read metadata
    with fsspec.open(url, 'rb') as f:
        # PyArrow reads only the metadata (uses HTTP range requests to seek to end of file)
        parquet_file = pq.ParquetFile(f)
        # Get total number of rows from metadata
        return parquet_file.metadata.num_rows


def count_dataset_rows(repo_id: str, language: Optional[str] = None) -> Dict[str, Dict]:
    """
    Count total rows for each language in the dataset using HTTP range requests.
    
    Args:
        repo_id: HuggingFace dataset repository ID
        language: Optional language filter
    
    Returns:
        Dictionary mapping language codes to statistics (rows, shards, subshards)
    """
    # Get parquet files grouped by language
    language_files = get_parquet_files(repo_id, language)
    
    if not language_files:
        print(f"No parquet files found for language: {language}")
        return {}
    
    # Count rows and track shards/subshards for each language
    language_stats = {}
    
    for lang, files in language_files.items():
        print(f"\nCounting rows for language: {lang}")
        print(f"  Files to process: {len(files)}")
        
        total_rows = 0
        shards = set()
        subshards = set()
        
        for file_path in tqdm(files, desc=f"  Processing {lang}"):
            try:
                # Extract shard and subshard from path (e.g., "en000/00000001.parquet")
                parts = file_path.split('/')
                if len(parts) == 2:
                    shard_id = parts[0]
                    subshard_id = parts[1].replace('.parquet', '')
                    shards.add(shard_id)
                    subshards.add(f"{shard_id}/{subshard_id}")
                
                # Count rows
                rows = count_rows_in_parquet(repo_id, file_path)
                total_rows += rows
            except Exception as e:
                print(f"  Warning: Failed to read {file_path}: {e}")
                continue
        
        language_stats[lang] = {
            'num_rows': total_rows,
            'num_shards': len(shards),
            'num_subshards': len(subshards)
        }
        
        print(f"  Total rows: {total_rows:,}, Shards: {len(shards)}, Subshards: {len(subshards)}")
    
    return language_stats


def format_results(language_stats: Dict[str, Dict], output_file: str = "dataset_summary.txt", markdown: bool = False):
    """Print formatted results and save to file."""
    # Sort by language code
    sorted_languages = sorted(language_stats.items())
    
    total_rows = 0
    total_shards = 0
    total_subshards = 0
    
    if markdown:
        # Markdown table format
        lines = []
        lines.append("## Dataset Statistics by Language")
        lines.append("")
        lines.append("| Language | Rows | Shards | Subshards |")
        lines.append("|----------|-----:|-------:|----------:|")
        
        for lang, stats in sorted_languages:
            num_rows = stats['num_rows']
            num_shards = stats['num_shards']
            num_subshards = stats['num_subshards']
            
            lines.append(f"| {lang} | {num_rows:,} | {num_shards:,} | {num_subshards:,} |")
            total_rows += num_rows
            total_shards += num_shards
            total_subshards += num_subshards
        
        lines.append(f"| **TOTAL** | **{total_rows:,}** | **{total_shards:,}** | **{total_subshards:,}** |")
    else:
        # Plain text format
        lines = []
        lines.append("=" * 80)
        lines.append("DATASET STATISTICS BY LANGUAGE")
        lines.append("=" * 80)
        lines.append(f"{'Language':<10} {'Rows':>15} {'Shards':>10} {'Subshards':>12}")
        lines.append("-" * 80)
        
        for lang, stats in sorted_languages:
            num_rows = stats['num_rows']
            num_shards = stats['num_shards']
            num_subshards = stats['num_subshards']
            
            lines.append(f"{lang:<10} {num_rows:>15,} {num_shards:>10,} {num_subshards:>12,}")
            total_rows += num_rows
            total_shards += num_shards
            total_subshards += num_subshards
        
        lines.append("-" * 80)
        lines.append(f"{'TOTAL':<10} {total_rows:>15,} {total_shards:>10,} {total_subshards:>12,}")
        lines.append("=" * 80)
    
    # Print to console
    output = "\n".join(lines)
    print("\n" + output)
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(output + "\n")
    
    print(f"\nSummary saved to: {output_file}")
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Count rows in HuggingFace dataset by reading parquet metadata"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace dataset repository ID (e.g., potsawee/yodas2-mm-pretrain)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional: Count only specific language (e.g., 'en', 'th')"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="dataset_summary.txt",
        help="Output file for summary (default: dataset_summary.txt)"
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Output in Markdown table format (default: plain text)"
    )
    
    args = parser.parse_args()
    
    print("Using HTTP range requests to read only metadata (efficient, no full downloads)")
    
    # Count rows
    language_stats = count_dataset_rows(args.repo_id, args.language)
    
    # Display results
    if language_stats:
        format_results(language_stats, args.output_file, args.markdown)
    else:
        print("No data found.")


if __name__ == "__main__":
    main()

