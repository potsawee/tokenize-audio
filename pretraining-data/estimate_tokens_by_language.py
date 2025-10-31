#!/usr/bin/env python3
"""
Estimate the number of tokens in the pretraining corpus by language.

This script loads each language separately as a dataset config for efficiency.

Example usage:
# Basic sampling for a single language
python estimate_tokens_by_language.py \
    --repo-id potsawee/yodas2-mm-pretrain \
    --languages en \
    --sample-size 1000

# Multiple languages
python estimate_tokens_by_language.py \
    --repo-id potsawee/yodas2-mm-pretrain \
    --languages en es fr \
    --sample-size 10000

# Full analysis with output directory
python estimate_tokens_by_language.py \
    --repo-id potsawee/yodas2-mm-pretrain \
    --languages ar th zh hi de fr es en \
    --full-analysis \
    --output-path ./token_stats_by_language/
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

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


class LanguageTokenEstimator:
    """Estimate tokens in the pretraining corpus by language."""
    
    def __init__(self, languages: List[str]):
        logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        # Preserve order while removing duplicates
        self.languages_list = []
        seen = set()
        for lang in languages:
            if lang not in seen:
                self.languages_list.append(lang)
                seen.add(lang)
        logger.info(f"Languages to process: {', '.join(self.languages_list)}")
    
    def analyze_document(self, document: Dict) -> TokenStats:
        """Analyze a single document and return statistics."""
        text = document.get("text", "")
        token_count = len(self.tokenizer(text)["input_ids"])
        
        return TokenStats(
            num_documents=1,
            total_tokens=token_count
        )
    
    def _save_language_stats(
        self,
        output_path: Path,
        language: str,
        mode: str,
        repo_id: str,
        stats: Optional[TokenStats] = None,
        sample_size: Optional[int] = None,
        sample_stats: Optional[TokenStats] = None,
        extrapolated_stats: Optional[TokenStats] = None
    ):
        """Save statistics for a single language to a JSON file."""
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"{language}.json"
        
        if mode == "full_analysis":
            output = {
                "mode": "full_analysis",
                "repo_id": repo_id,
                "language": language,
                "tokenizer": TOKENIZER_NAME,
                "statistics": stats.to_dict()
            }
        else:  # sampling
            output = {
                "mode": "sampling",
                "repo_id": repo_id,
                "language": language,
                "tokenizer": TOKENIZER_NAME,
                "sample_size": sample_size,
                "sample_statistics": sample_stats.to_dict(),
                "extrapolated_statistics": extrapolated_stats.to_dict()
            }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved statistics for {language} to {output_file}")
    
    def estimate_language_from_sample(
        self, 
        repo_id: str,
        language: str,
        sample_size: int = 1000,
        split: Optional[str] = None,
        total_docs: Optional[int] = None,
        output_path: Optional[Path] = None
    ) -> tuple[TokenStats, TokenStats]:
        """
        Estimate statistics from a sample for a single language.
        
        Returns:
            Tuple of (sample_stats, extrapolated_stats)
        """
        logger.info(f"Processing language: {language}")
        logger.info(f"Loading sample of {sample_size} documents")
        
        # Load dataset for specific language in streaming mode
        dataset = load_dataset(
            repo_id,
            language,  # Language as config name
            split=split or "train",
            streaming=True
        )
        
        # Process sample
        sample_stats = TokenStats()
        
        for i, doc in enumerate(tqdm(dataset, total=sample_size, desc=f"Analyzing {language}")):
            if i >= sample_size:
                break
            doc_stats = self.analyze_document(doc)
            sample_stats = sample_stats + doc_stats
        
        if sample_stats.num_documents == 0:
            logger.error(f"No documents processed for language: {language}")
            return sample_stats, sample_stats
        
        # If total_docs not provided, try to get from metadata
        if total_docs is None:
            logger.info(f"Getting total document count for {language} from metadata...")
            split_name = split or "train"
            
            # Try method 1: get_dataset_infos
            try:
                infos = get_dataset_infos(repo_id)
                if language in infos:
                    info = infos[language]
                    if hasattr(info, 'splits') and split_name in info.splits:
                        total_docs = info.splits[split_name].num_examples
                        logger.info(f"Total documents for {language}: {total_docs}")
            except Exception as e:
                logger.debug(f"Method 1 failed: {e}")
            
            # Try method 2: HfApi dataset_info
            if total_docs is None:
                try:
                    from huggingface_hub import HfApi
                    api = HfApi()
                    dataset_info = api.dataset_info(repo_id)
                    if hasattr(dataset_info, 'card_data') and dataset_info.card_data:
                        if 'dataset_info' in dataset_info.card_data:
                            configs = dataset_info.card_data['dataset_info']
                            if isinstance(configs, list):
                                for config in configs:
                                    if config.get('config_name') == language:
                                        if 'splits' in config:
                                            for split_info in config['splits']:
                                                if split_info.get('name') == split_name:
                                                    total_docs = split_info.get('num_examples')
                                                    logger.info(f"Total documents for {language}: {total_docs}")
                                                    break
                except Exception as e:
                    logger.debug(f"Method 2 failed: {e}")
        
        if total_docs is None:
            logger.warning(f"Could not determine total document count for {language}")
            logger.warning(f"Returning sample stats only for {language}")
            return sample_stats, sample_stats
        
        # Extrapolate
        ratio = total_docs / sample_stats.num_documents
        extrapolated = TokenStats(
            num_documents=total_docs,
            total_tokens=int(sample_stats.total_tokens * ratio),
        )
        
        logger.info(f"Extrapolated from {sample_stats.num_documents} to {total_docs} documents (ratio: {ratio:.2f})")
        
        # Save to file if output path provided
        if output_path is not None:
            self._save_language_stats(
                output_path=output_path,
                language=language,
                mode="sampling",
                repo_id=repo_id,
                sample_size=sample_size,
                sample_stats=sample_stats,
                extrapolated_stats=extrapolated
            )
        
        return sample_stats, extrapolated
    
    def full_analysis_language(
        self, 
        repo_id: str,
        language: str,
        split: Optional[str] = None,
        batch_size: int = 1000,
        output_path: Optional[Path] = None
    ) -> TokenStats:
        """
        Perform full analysis for a single language.
        
        This may take a long time for large datasets!
        """
        logger.info(f"Starting full analysis for language: {language}")
        
        # Load dataset for specific language in streaming mode
        dataset = load_dataset(
            repo_id,
            language,  # Language as config name
            split=split or "train",
            streaming=True
        )
        
        total_stats = TokenStats()
        batch_stats = TokenStats()
        
        for i, doc in enumerate(tqdm(dataset, desc=f"Analyzing {language}")):
            doc_stats = self.analyze_document(doc)
            batch_stats = batch_stats + doc_stats
            
            # Log progress every batch
            if (i + 1) % batch_size == 0:
                total_stats = total_stats + batch_stats
                logger.info(f"Processed {i + 1} documents for {language}")
                logger.info(f"  Total tokens so far: {total_stats.total_tokens:,}")
                batch_stats = TokenStats()
        
        # Add remaining batch
        total_stats = total_stats + batch_stats
        
        logger.info(f"Completed analysis of {language}: {total_stats.num_documents} documents")
        
        # Save to file if output path provided
        if output_path is not None:
            self._save_language_stats(
                output_path=output_path,
                language=language,
                mode="full_analysis",
                repo_id=repo_id,
                stats=total_stats
            )
        
        return total_stats
    
    def estimate_from_sample(
        self, 
        repo_id: str, 
        sample_size: int = 1000,
        split: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> tuple[Dict[str, TokenStats], Dict[str, TokenStats]]:
        """
        Estimate statistics from samples for all languages.
        
        Returns:
            Tuple of (sample_stats_by_lang, extrapolated_stats_by_lang)
        """
        sample_stats_by_lang = {}
        extrapolated_by_lang = {}
        
        for language in self.languages_list:
            sample_stats, extrapolated_stats = self.estimate_language_from_sample(
                repo_id=repo_id,
                language=language,
                sample_size=sample_size,
                split=split,
                output_path=output_path
            )
            sample_stats_by_lang[language] = sample_stats
            extrapolated_by_lang[language] = extrapolated_stats
            logger.info("-" * 80)
        
        return sample_stats_by_lang, extrapolated_by_lang
    
    def full_analysis(
        self, 
        repo_id: str,
        split: Optional[str] = None,
        batch_size: int = 1000,
        output_path: Optional[Path] = None
    ) -> Dict[str, TokenStats]:
        """
        Perform full analysis for all languages.
        
        This may take a long time for large datasets!
        """
        logger.warning("Starting full analysis for all languages - this may take a long time!")
        
        stats_by_lang = {}
        
        for language in self.languages_list:
            stats = self.full_analysis_language(
                repo_id=repo_id,
                language=language,
                split=split,
                batch_size=batch_size,
                output_path=output_path
            )
            stats_by_lang[language] = stats
            logger.info("-" * 80)
        
        return stats_by_lang


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


def print_stats_by_language(stats_by_lang: Dict[str, TokenStats], languages_order: List[str], title: str = "Token Statistics by Language"):
    """Pretty print statistics by language."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    
    if not stats_by_lang:
        print("No data available")
        print("=" * 80 + "\n")
        return
    
    # Print per-language stats in the specified order
    for lang in languages_order:
        if lang not in stats_by_lang:
            continue
        stats = stats_by_lang[lang]
        avg_tokens = stats.total_tokens / max(stats.num_documents, 1)
        print(f"\nLanguage: {lang}")
        print(f"  Documents:          {format_number(stats.num_documents)}")
        print(f"  Total tokens:       {format_number(stats.total_tokens)}")
        print(f"  Avg tokens per doc: {avg_tokens:.1f}")
    
    # Print totals
    total_docs = sum(s.num_documents for s in stats_by_lang.values())
    total_tokens = sum(s.total_tokens for s in stats_by_lang.values())
    avg_tokens = total_tokens / max(total_docs, 1)
    
    print("\n" + "-" * 80)
    print("TOTAL (all languages)")
    print(f"  Documents:          {format_number(total_docs)}")
    print(f"  Total tokens:       {format_number(total_tokens)}")
    print(f"  Avg tokens per doc: {avg_tokens:.1f}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate tokens in pretraining corpus by language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--languages", type=str, nargs='+', required=True, help="Languages to process (e.g., en es fr)")
    parser.add_argument("--split", type=str, default=None, help="Dataset split (default: train)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of documents to sample per language (default: 1000)")
    parser.add_argument("--full-analysis", action="store_true", help="Perform full analysis instead of sampling")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for logging during full analysis")
    parser.add_argument("--output-path", type=str, default=None, help="Output directory path for per-language JSON files")
    
    args = parser.parse_args()
    
    # Parse output path if provided
    output_path = Path(args.output_path) if args.output_path else None
    
    # Initialize estimator
    estimator = LanguageTokenEstimator(languages=args.languages)
    
    if args.full_analysis:
        # Full analysis
        stats_by_lang = estimator.full_analysis(
            repo_id=args.repo_id,
            split=args.split,
            batch_size=args.batch_size,
            output_path=output_path
        )
        print_stats_by_language(stats_by_lang, estimator.languages_list, "Full Dataset Statistics by Language")
    else:
        # Sample-based estimation
        sample_stats_by_lang, extrapolated_by_lang = estimator.estimate_from_sample(
            repo_id=args.repo_id,
            sample_size=args.sample_size,
            split=args.split,
            output_path=output_path
        )
        
        print_stats_by_language(sample_stats_by_lang, estimator.languages_list, f"Sample Statistics ({args.sample_size} documents per language)")
        print_stats_by_language(extrapolated_by_lang, estimator.languages_list, "Extrapolated Full Dataset Statistics by Language")
    
    if output_path:
        logger.info(f"All language statistics saved to {output_path}/")


if __name__ == "__main__":
    main()
