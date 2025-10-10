#!/usr/bin/env python3
"""
Estimate the number of tokens in the pretraining corpus.

Example usage:
# Basic sampling (tries to get total count from metadata)
python estimate_tokens.py \
    --repo-id potsawee/yodas2-mm-pretrain \
    --sample-size 1000

# If you know the total document count (faster, skips metadata lookup)
python estimate_tokens.py \
    --repo-id potsawee/yodas2-mm-pretrain \
    --sample-size 10000 \
    --total-docs 4258874 \
    --output-file token_stats.json

# Full analysis (slow, processes every document)
python estimate_tokens.py \
    --repo-id potsawee/yodas2-mm-pretrain \
    --full-analysis
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from datasets import load_dataset
from datasets import get_dataset_infos
from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


TOKENIZER_NAME = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"


@dataclass
class TokenStats:
    """Statistics for token counting."""
    num_documents: int = 0
    total_tokens: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def __add__(self, other: 'TokenStats') -> 'TokenStats':
        """Add two TokenStats together."""
        return TokenStats(
            num_documents=self.num_documents + other.num_documents,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class TokenEstimator:
    """Estimate tokens in the pretraining corpus."""
    
    def __init__(self):
        logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    def analyze_document(self, document: Dict) -> TokenStats:
        """Analyze a single document and return statistics."""
        text = document.get("text", "")
        token_count = len(self.tokenizer(text)["input_ids"])
        
        return TokenStats(
            num_documents=1,
            total_tokens=token_count
        )
    
    def estimate_from_sample(
        self, 
        repo_id: str, 
        sample_size: int = 1000,
        split: Optional[str] = None,
        total_docs: Optional[int] = None
    ) -> Tuple[TokenStats, TokenStats]:
        """
        Estimate statistics from a sample and extrapolate to full dataset.
        
        Returns:
            Tuple of (sample_stats, extrapolated_stats)
        """
        logger.info(f"Loading sample of {sample_size} documents from {repo_id}")
        
        # Load dataset in streaming mode
        dataset = load_dataset(
            repo_id,
            split=split or "train",
            streaming=True
        )
        
        # Process sample
        sample_stats = TokenStats()
        
        for i, doc in enumerate(tqdm(dataset, total=sample_size, desc="Analyzing sample")):
            if i >= sample_size:
                break
            doc_stats = self.analyze_document(doc)
            sample_stats = sample_stats + doc_stats
        
        if sample_stats.num_documents == 0:
            logger.error("No documents processed!")
            return sample_stats, sample_stats
        
        # If total_docs not provided, try to get from metadata
        if total_docs is None:
            logger.info("Getting total document count from metadata...")
            split_name = split or "train"
        else:
            logger.info(f"Using provided total document count: {total_docs}")
            # Skip metadata retrieval
            split_name = split or "train"
        
        # Only try metadata methods if total_docs wasn't provided
        if total_docs is None:
            # Try method 1: get_dataset_infos
            try:
                infos = get_dataset_infos(repo_id)
                # infos is a dict of config_name -> DatasetInfo
                for config_name, info in infos.items():
                    if hasattr(info, 'splits') and split_name in info.splits:
                        total_docs = info.splits[split_name].num_examples
                        logger.info(f"Total documents in dataset (config: {config_name}): {total_docs}")
                        break
            except Exception as e:
                logger.debug(f"Method 1 failed: {e}")
            
            # Try method 2: HfApi dataset_info
            if total_docs is None:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    dataset_info = api.dataset_info(repo_id)
                    # Try to parse from card_data or siblings
                    if hasattr(dataset_info, 'card_data') and dataset_info.card_data:
                        if 'dataset_info' in dataset_info.card_data:
                            info = dataset_info.card_data['dataset_info']
                            if 'splits' in info and split_name in info['splits']:
                                total_docs = info['splits'][split_name]['num_examples']
                                logger.info(f"Total documents in dataset: {total_docs}")
                except Exception as e:
                    logger.debug(f"Method 2 failed: {e}")
        
        if total_docs is None:
            logger.warning("Could not determine total document count")
            logger.warning("Returning sample stats only")
            return sample_stats, sample_stats
        
        # Extrapolate
        ratio = total_docs / sample_stats.num_documents
        extrapolated = TokenStats(
            num_documents=total_docs,
            total_tokens=int(sample_stats.total_tokens * ratio),
        )
        
        logger.info(f"Extrapolated from {sample_stats.num_documents} to {total_docs} documents (ratio: {ratio:.2f})")
        
        return sample_stats, extrapolated
    
    def full_analysis(
        self, 
        repo_id: str,
        split: Optional[str] = None,
        batch_size: int = 1000
    ) -> TokenStats:
        """
        Perform full analysis of entire dataset.
        
        This may take a long time for large datasets!
        """
        logger.info(f"Starting full analysis of {repo_id}")
        logger.warning("This may take a long time for large datasets!")
        
        # Load dataset in streaming mode
        dataset = load_dataset(
            repo_id,
            split=split or "train",
            streaming=True
        )
        
        total_stats = TokenStats()
        batch_stats = TokenStats()
        
        for i, doc in enumerate(tqdm(dataset, desc="Analyzing full dataset")):
            doc_stats = self.analyze_document(doc)
            batch_stats = batch_stats + doc_stats
            
            # Log progress every batch
            if (i + 1) % batch_size == 0:
                total_stats = total_stats + batch_stats
                logger.info(f"Processed {i + 1} documents")
                logger.info(f"  Total tokens so far: {total_stats.total_tokens:,}")
                batch_stats = TokenStats()
        
        # Add remaining batch
        total_stats = total_stats + batch_stats
        
        logger.info(f"Completed full analysis of {total_stats.num_documents} documents")
        
        return total_stats


def format_number(num: int) -> str:
    """Format large numbers with commas and in billions/millions."""
    if num >= 1e9:
        return f"{num:,} ({num/1e9:.2f}B)"
    elif num >= 1e6:
        return f"{num:,} ({num/1e6:.2f}M)"
    elif num >= 1e3:
        return f"{num:,} ({num/1e3:.2f}K)"
    else:
        return f"{num:,}"


def print_stats(stats: TokenStats, title: str = "Token Statistics"):
    """Pretty print statistics."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print(f"Documents:          {format_number(stats.num_documents)}")
    print(f"Total tokens:       {format_number(stats.total_tokens)}")
    print(f"Avg tokens per doc: {stats.total_tokens / max(stats.num_documents, 1):.1f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate tokens in pretraining corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (default: train)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of documents to sample (default: 1000)")
    parser.add_argument("--total-docs", type=int, default=None, help="Total number of documents (if known, skips metadata lookup)")
    parser.add_argument("--full-analysis", action="store_true", help="Perform full analysis instead of sampling")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for logging during full analysis")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSON file for statistics")
    
    args = parser.parse_args()
    
    # Initialize estimator
    estimator = TokenEstimator()
    
    if args.full_analysis:
        # Full analysis
        stats = estimator.full_analysis(
            repo_id=args.repo_id,
            split=args.split,
            batch_size=args.batch_size
        )
        print_stats(stats, "Full Dataset Statistics")
        
        if args.output_file:
            output = {
                "mode": "full_analysis",
                "repo_id": args.repo_id,
                "tokenizer": TOKENIZER_NAME,
                "statistics": stats.to_dict()
            }
    else:
        # Sample-based estimation
        sample_stats, extrapolated_stats = estimator.estimate_from_sample(
            repo_id=args.repo_id,
            sample_size=args.sample_size,
            split=args.split,
            total_docs=args.total_docs
        )
        
        print_stats(sample_stats, f"Sample Statistics ({args.sample_size} documents)")
        print_stats(extrapolated_stats, "Extrapolated Full Dataset Statistics")
        
        if args.output_file:
            output = {
                "mode": "sampling",
                "repo_id": args.repo_id,
                "tokenizer": TOKENIZER_NAME,
                "sample_size": args.sample_size,
                "sample_statistics": sample_stats.to_dict(),
                "extrapolated_statistics": extrapolated_stats.to_dict()
            }
    
    # Save to file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved statistics to {output_path}")


if __name__ == "__main__":
    main()

