#!/usr/bin/env python3
"""
Estimate tokens with stratified sampling to ensure representative sample.

Usage:
    python estimate_tokens_stratified.py \
        --repo-id potsawee/yodas2-mm-pretrain \
        --sample-size 2000 \
        --output-file token_stats_stratified.json
"""

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict

from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Language distribution from stats.md (rows per language)
LANGUAGE_DISTRIBUTION = {
    'en': 1791316,
    'es': 476532,
    'ru': 405758,
    'ko': 239366,
    'pt': 220746,
    'fr': 202596,
    'id': 153938,
    'vi': 133954,
    'de': 132832,
    'it': 115366,
    'ja': 65482,
    'tr': 64938,
    'nl': 51700,
    'hi': 36462,
    'ar': 20838,
    'uk': 20554,
    'pl': 19056,
    'th': 16372,
    'bn': 11060,
    'ur': 5328,
    'ta': 4954,
    'cs': 4746,
    'zh': 4330,
    'el': 3220,
    'fi': 3220,
    'ro': 3770,
    'ca': 2662,
    'sv': 2638,
    'mr': 2570,
    'hu': 2558,
    'te': 2510,
    'no': 2500,
    'be': 2126,
    'ky': 1824,
    'bg': 1668,
    'ml': 1496,
    'sk': 1426,
    'da': 1338,
    'uz': 1200,
    'gu': 1178,
    'ms': 1160,
    'eu': 954,
    'fa': 914,
    'km': 816,
    'sl': 748,
    'hr': 738,
    'ka': 694,
    'am': 664,
    'pa': 636,
    'si': 616,
    'kn': 574,
    'ne': 554,
    'kk': 540,
    'sr': 540,
    'or': 540,
    'so': 426,
    'gl': 432,
    'az': 416,
    'eo': 396,
    'sw': 350,
    'ha': 346,
    'is': 306,
    'sq': 302,
    'et': 282,
    'hy': 284,
    'cy': 260,
    'zu': 260,
    'ab': 264,
    'bs': 252,
    'as': 240,
    'ht': 228,
    'om': 206,
    'mk': 196,
    'af': 174,
    'my': 174,
    'ps': 172,
    'la': 162,
    'mi': 132,
    'mn': 120,
    'lv': 108,
    'jv': 106,
    'ku': 102,
    'ga': 102,
    'rw': 94,
    'tg': 90,
    'aa': 86,
    'yo': 28,
    'lo': 28,
    'fo': 28,
    'ie': 28,
    'dz': 18,
    'ks': 18,
    'ay': 48,
    'gn': 40,
    'ba': 40,
    'sd': 40,
    'sa': 42,
    'mg': 38,
    'su': 36,
    'ug': 36,
    'br': 36,
    'ak': 32,
    'ia': 32,
    'lb': 16,
    'fy': 16,
    'gd': 16,
    'ig': 16,
    'bo': 14,
    'fj': 14,
    'xh': 14,
    'yi': 14,
    'co': 12,
    'ff': 12,
    'iu': 12,
    'oc': 12,
    'bi': 10,
    'ee': 10,
    'ho': 10,
    'na': 10,
    'tn': 10,
    'tt': 10,
    'bm': 8,
    've': 8,
    'kl': 8,
    'cr': 6,
    'ik': 6,
    'nv': 6,
    'qu': 6,
    'rn': 6,
    'sc': 6,
    'sh': 6,
    'sn': 6,
    'st': 6,
    'bh': 4,
    'ki': 2,
    'lg': 4,
    'sm': 4,
    'ts': 4,
    'vo': 4,
    'wo': 54,
    'ln': 64,
    'nd': 2,
    'sg': 2,
    'to': 2,
    'rm': 20,
    'tk': 20,
    'ti': 24,
}

TOTAL_DOCS = sum(LANGUAGE_DISTRIBUTION.values())


def stratified_sample(repo_id: str, sample_size: int, tokenizer_name: str):
    """Sample documents proportionally from each language."""
    print(f"Total documents: {TOTAL_DOCS:,}")
    print(f"Target sample size: {sample_size:,}")
    
    # Calculate samples per language (proportional)
    samples_per_lang = {}
    for lang, count in LANGUAGE_DISTRIBUTION.items():
        proportion = count / TOTAL_DOCS
        n_samples = max(1, int(sample_size * proportion))  # At least 1 sample per language
        samples_per_lang[lang] = n_samples
    
    actual_sample_size = sum(samples_per_lang.values())
    print(f"Actual sample size (after rounding): {actual_sample_size:,}")
    print(f"\nSampling from {len(samples_per_lang)} languages...")
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Load dataset in streaming mode
    print(f"Loading dataset: {repo_id}")
    ds = load_dataset(repo_id, split='train', streaming=True)
    
    # Collect samples per language
    lang_samples = defaultdict(list)
    lang_targets = samples_per_lang.copy()
    
    total_tokens = 0
    total_docs_processed = 0
    
    print("\nCollecting stratified sample...")
    pbar = tqdm(total=actual_sample_size, desc="Sampling")
    
    for item in ds:
        # Extract language from split
        split = item['split']
        lang = split.split('/')[0][:2]
        
        # Check if we need more samples from this language
        if lang in lang_targets and lang_targets[lang] > 0:
            # Tokenize and count
            text = item['text']
            tokens = tokenizer.encode(text, add_special_tokens=False)
            n_tokens = len(tokens)
            
            lang_samples[lang].append(n_tokens)
            total_tokens += n_tokens
            total_docs_processed += 1
            lang_targets[lang] -= 1
            pbar.update(1)
        
        # Check if we're done
        if sum(lang_targets.values()) == 0:
            break
    
    pbar.close()
    
    # Calculate statistics
    avg_tokens_per_doc = total_tokens / total_docs_processed if total_docs_processed > 0 else 0
    extrapolated_total = avg_tokens_per_doc * TOTAL_DOCS
    
    print(f"\n{'=' * 80}")
    print(f"{'Sample Statistics':^80}")
    print(f"{'=' * 80}")
    print(f"Documents sampled:  {total_docs_processed:,} ({total_docs_processed / 1000:.2f}K)")
    print(f"Total tokens:       {total_tokens:,} ({total_tokens / 1e6:.2f}M)")
    print(f"Avg tokens/doc:     {avg_tokens_per_doc:,.1f}")
    print(f"{'=' * 80}")
    
    print(f"\n{'=' * 80}")
    print(f"{'Extrapolated Full Dataset':^80}")
    print(f"{'=' * 80}")
    print(f"Documents:          {TOTAL_DOCS:,} ({TOTAL_DOCS / 1e6:.2f}M)")
    print(f"Total tokens:       {extrapolated_total:,.0f} ({extrapolated_total / 1e9:.2f}B)")
    print(f"Avg tokens/doc:     {avg_tokens_per_doc:,.1f}")
    print(f"{'=' * 80}")
    
    # Language breakdown
    print(f"\nTop 10 languages by sample count:")
    lang_token_counts = {lang: sum(tokens) for lang, tokens in lang_samples.items()}
    for lang, total in sorted(lang_token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        n_docs = len(lang_samples[lang])
        avg = total / n_docs if n_docs > 0 else 0
        print(f"  {lang}: {n_docs:5d} docs, {total:10,} tokens, {avg:8,.1f} avg tokens/doc")
    
    return {
        'sample_size': total_docs_processed,
        'total_tokens_in_sample': total_tokens,
        'avg_tokens_per_doc': avg_tokens_per_doc,
        'extrapolated_total_tokens': extrapolated_total,
        'total_documents': TOTAL_DOCS,
        'language_samples': {lang: len(samples) for lang, samples in lang_samples.items()},
        'language_token_counts': lang_token_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Estimate tokens with stratified sampling")
    parser.add_argument('--repo-id', type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument('--sample-size', type=int, default=10000, help="Target sample size")
    parser.add_argument('--tokenizer', type=str, default="potsawee/marin-mimi-bpe-8cb-16k-tokenizer", 
                        help="Tokenizer to use")
    parser.add_argument('--output-file', type=str, default=None, help="Output JSON file")
    
    args = parser.parse_args()
    
    stats = stratified_sample(args.repo_id, args.sample_size, args.tokenizer)
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved statistics to: {args.output_file}")


if __name__ == "__main__":
    main()

