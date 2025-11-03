#!/usr/bin/env python3
"""
Check completion status for all languages and splits in one command.
Displays a comprehensive summary table of upload status.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from huggingface_hub import HfApi
from tqdm import tqdm


def read_file_list(file_list_path: Path) -> List[str]:
    """Read shard IDs from a file list."""
    if not file_list_path.exists():
        return []
    
    with open(file_list_path, 'r') as f:
        shard_ids = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    
    return shard_ids


def get_hf_repo_files(hf_repo_id: str) -> set:
    """Get all files in the HuggingFace repository."""
    api = HfApi()
    files = api.list_repo_files(repo_id=hf_repo_id, repo_type="dataset")
    return set(files)


def check_language_split(
    file_list_path: Path,
    split: str,
    repo_files: set
) -> Tuple[int, int, List[str]]:
    """
    Check completion for a specific language/split combination.
    
    Returns:
        Tuple of (total, existing, missing_shard_ids)
    """
    shard_ids = read_file_list(file_list_path)
    if not shard_ids:
        return 0, 0, []
    
    existing_count = 0
    missing_shards = []
    
    for shard_id in shard_ids:
        lang = shard_id.split("-")[0]
        repo_path = f"{split}/{lang}/{shard_id}.parquet"
        
        if repo_path in repo_files:
            existing_count += 1
        else:
            missing_shards.append(shard_id)
    
    return len(shard_ids), existing_count, missing_shards


def print_table_row(split: str, lang: str, total: int, existing: int, width_split: int = 15, width_lang: int = 8):
    """Print a single row of the summary table."""
    missing = total - existing
    pct = (existing / total * 100) if total > 0 else 0
    
    status = "✓" if missing == 0 else "✗"
    status_color = "\033[92m" if missing == 0 else "\033[91m"  # Green or Red
    reset_color = "\033[0m"
    
    print(f"│ {split:<{width_split}} │ {lang:<{width_lang}} │ {total:>7} │ {existing:>7} │ {missing:>7} │ {pct:>6.1f}% │ {status_color}{status:^8}{reset_color} │")


def print_separator(width_split: int = 15, width_lang: int = 8):
    """Print table separator line."""
    total_width = width_split + width_lang + 7 + 7 + 7 + 7 + 8 + 14  # columns + separators
    print("├" + "─" * (width_split + 2) + "┼" + "─" * (width_lang + 2) + "┼" + "─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 9 + "┼" + "─" * 8 + "┼" + "─" * 10 + "┤")


def main():
    parser = argparse.ArgumentParser(
        description="Check completion status for all languages and splits"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="potsawee/emilia-mm-pretrain",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--file-lists-dir",
        type=str,
        default="file_lists",
        help="Directory containing file lists"
    )
    parser.add_argument(
        "--languages",
        type=str,
        default="en,de,fr,ja,ko,zh",
        help="Comma-separated list of language codes to check"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="Emilia,Emilia-YODAS",
        help="Comma-separated list of splits to check"
    )
    parser.add_argument(
        "--save-missing",
        type=str,
        help="Directory to save missing shard lists (optional)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show missing shard IDs"
    )
    
    args = parser.parse_args()
    
    file_lists_dir = Path(args.file_lists_dir)
    languages = [lang.strip() for lang in args.languages.split(",")]
    splits = [split.strip() for split in args.splits.split(",")]
    
    # Fetch HuggingFace repo files once
    print(f"Fetching file list from HuggingFace repo: {args.hf_repo_id}...")
    repo_files = get_hf_repo_files(args.hf_repo_id)
    print(f"Found {len(repo_files)} files in repository\n")
    
    # Collect results
    results = {}
    all_missing = defaultdict(lambda: defaultdict(list))
    
    # Check each split and language combination
    for split in splits:
        for lang in languages:
            # Determine file list name
            if split == "Emilia":
                file_list_path = file_lists_dir / f"{lang}.txt"
            elif split == "Emilia-YODAS":
                file_list_path = file_lists_dir / f"{lang}_yodas.txt"
            else:
                file_list_path = file_lists_dir / f"{lang}_{split.lower()}.txt"
            
            total, existing, missing_shards = check_language_split(
                file_list_path, split, repo_files
            )
            
            if total > 0:  # Only include if file list exists and has shards
                results[(split, lang)] = (total, existing)
                all_missing[split][lang] = missing_shards
    
    # Print summary table
    width_split = 15
    width_lang = 8
    
    print("\n" + "┌" + "─" * (width_split + 2) + "┬" + "─" * (width_lang + 2) + "┬" + "─" * 9 + "┬" + "─" * 9 + "┬" + "─" * 9 + "┬" + "─" * 8 + "┬" + "─" * 10 + "┐")
    print(f"│ {'Split':<{width_split}} │ {'Language':<{width_lang}} │ {'Total':>7} │ {'Exists':>7} │ {'Missing':>7} │ {'Done %':>7} │ {'Status':^8} │")
    print("├" + "═" * (width_split + 2) + "╪" + "═" * (width_lang + 2) + "╪" + "═" * 9 + "╪" + "═" * 9 + "╪" + "═" * 9 + "╪" + "═" * 8 + "╪" + "═" * 10 + "┤")
    
    current_split = None
    for (split, lang), (total, existing) in sorted(results.items()):
        # Add separator between splits
        if current_split is not None and current_split != split:
            print_separator(width_split, width_lang)
        
        print_table_row(split, lang.upper(), total, existing, width_split, width_lang)
        current_split = split
    
    print("└" + "─" * (width_split + 2) + "┴" + "─" * (width_lang + 2) + "┴" + "─" * 9 + "┴" + "─" * 9 + "┴" + "─" * 9 + "┴" + "─" * 8 + "┴" + "─" * 10 + "┘")
    
    # Overall statistics
    total_all = sum(total for total, _ in results.values())
    existing_all = sum(existing for _, existing in results.values())
    missing_all = total_all - existing_all
    pct_all = (existing_all / total_all * 100) if total_all > 0 else 0
    
    print(f"\n{'OVERALL STATISTICS':^{width_split + width_lang + 52}}")
    print("─" * (width_split + width_lang + 52))
    print(f"Total shards across all languages/splits: {total_all}")
    print(f"Uploaded to HuggingFace: {existing_all} ({pct_all:.1f}%)")
    print(f"Missing from HuggingFace: {missing_all} ({100 - pct_all:.1f}%)")
    print()
    
    # Show missing shards if verbose
    if args.verbose:
        for split in splits:
            for lang in languages:
                missing_shards = all_missing[split][lang]
                if missing_shards:
                    print(f"\n{split}/{lang.upper()} - Missing {len(missing_shards)} shards:")
                    for shard_id in missing_shards[:10]:  # Show first 10
                        print(f"  ✗ {shard_id}")
                    if len(missing_shards) > 10:
                        print(f"  ... and {len(missing_shards) - 10} more")
    
    # Save missing lists if requested
    if args.save_missing:
        save_dir = Path(args.save_missing)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for split in splits:
            for lang in languages:
                missing_shards = all_missing[split][lang]
                if missing_shards:
                    if split == "Emilia":
                        filename = f"{lang}_missing.txt"
                    else:
                        filename = f"{lang}_{split.lower()}_missing.txt"
                    
                    output_path = save_dir / filename
                    with open(output_path, 'w') as f:
                        for shard_id in missing_shards:
                            f.write(f"{shard_id}\n")
                    print(f"Saved missing shards to: {output_path}")
    
    # Exit with non-zero if any missing
    if missing_all > 0:
        sys.exit(1)
    else:
        print("\n✓ All shards uploaded to HuggingFace!")
        sys.exit(0)


if __name__ == "__main__":
    main()

