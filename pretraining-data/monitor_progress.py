#!/usr/bin/env python3
"""
Monitor progress of pretraining data preparation across all shards.

This script reads progress files from ./progress/ directory and displays:
- Overall statistics across all shards
- Per-shard completion status
- Estimated remaining work

Usage:
    python monitor_progress.py [--progress-dir ./progress] [--watch] [--sort-by completion]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_subshard_counts(counts_file: Optional[Path] = None) -> Dict[str, int]:
    """Load actual subshard counts from JSON file."""
    if counts_file is None:
        # Try to find it in common locations
        possible_paths = [
            Path("../yodas2-mimi/subshard_counts.json"),
            Path("subshard_counts.json"),
            Path("./subshard_counts.json"),
        ]
        for path in possible_paths:
            if path.exists():
                counts_file = path
                break
    
    if counts_file and counts_file.exists():
        try:
            with open(counts_file, 'r') as f:
                counts = json.load(f)
                print(f"Loaded subshard counts from: {counts_file}")
                return counts
        except Exception as e:
            print(f"Warning: Failed to load subshard counts from {counts_file}: {e}")
    
    print("Warning: Could not find subshard_counts.json, using default of 1000 per shard")
    return {}


def load_all_progress(progress_dir: Path) -> Dict[str, Dict]:
    """Load all progress files from directory."""
    progress_files = sorted(progress_dir.glob("*_progress.json"))
    
    all_progress = {}
    for progress_file in progress_files:
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
                shard_id = data.get('shard_id', progress_file.stem.replace('_progress', ''))
                all_progress[shard_id] = data
        except Exception as e:
            print(f"Warning: Failed to load {progress_file}: {e}")
    
    return all_progress


def calculate_stats(progress_data: Dict, shard_id: str, subshard_counts: Dict[str, int]) -> Tuple[int, int, int, int, float]:
    """
    Calculate statistics for a shard.
    
    Returns:
        (completed, failed, remaining, total, completion_pct)
    """
    completed = len(progress_data.get('completed_subshards', []))
    failed = len(progress_data.get('failed_subshards', []))
    
    # Get actual total from subshard_counts, default to 1000 if not found
    total = subshard_counts.get(shard_id, 1000)
    
    remaining = max(0, total - completed - failed)
    completion_pct = (completed / total * 100) if total > 0 else 0
    
    return completed, failed, remaining, total, completion_pct


def create_progress_bar(percentage: float, width: int = 30) -> str:
    """Create a text-based progress bar."""
    filled = int(width * percentage / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}]"


def display_summary(all_progress: Dict[str, Dict], subshard_counts: Dict[str, int], sort_by: str = 'name'):
    """Display summary of all shards."""
    if not all_progress:
        print("No progress files found.")
        return
    
    # Calculate statistics for each shard
    shard_stats = []
    for shard_id, progress_data in all_progress.items():
        completed, failed, remaining, total, completion_pct = calculate_stats(progress_data, shard_id, subshard_counts)
        shard_stats.append({
            'shard_id': shard_id,
            'completed': completed,
            'failed': failed,
            'remaining': remaining,
            'total': total,
            'completion_pct': completion_pct
        })
    
    # Sort
    if sort_by == 'completion':
        shard_stats.sort(key=lambda x: x['completion_pct'], reverse=True)
    elif sort_by == 'remaining':
        shard_stats.sort(key=lambda x: x['remaining'], reverse=True)
    else:  # name
        shard_stats.sort(key=lambda x: x['shard_id'])
    
    # Display header
    print("\n" + "=" * 100)
    print("PRETRAINING DATA PREPARATION - PROGRESS MONITOR")
    print("=" * 100)
    
    # Overall statistics
    total_completed = sum(s['completed'] for s in shard_stats)
    total_failed = sum(s['failed'] for s in shard_stats)
    total_remaining = sum(s['remaining'] for s in shard_stats)
    total_subshards = sum(s['total'] for s in shard_stats)
    overall_completion = (total_completed / total_subshards * 100) if total_subshards > 0 else 0
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total Shards:        {len(shard_stats)}")
    print(f"  Total Subshards:     {total_subshards:,}")
    print(f"  Completed:           {total_completed:,} ({overall_completion:.1f}%)")
    print(f"  Failed:              {total_failed:,}")
    print(f"  Remaining:           {total_remaining:,}")
    print(f"  Progress:            {create_progress_bar(overall_completion)} {overall_completion:.1f}%")
    
    # Group by language if shard IDs follow pattern like en000, es000, etc.
    language_stats = defaultdict(lambda: {'completed': 0, 'failed': 0, 'remaining': 0, 'total': 0, 'count': 0})
    for stats in shard_stats:
        # Extract language code (first 2 chars)
        lang = stats['shard_id'][:2] if len(stats['shard_id']) >= 2 else 'unknown'
        language_stats[lang]['completed'] += stats['completed']
        language_stats[lang]['failed'] += stats['failed']
        language_stats[lang]['remaining'] += stats['remaining']
        language_stats[lang]['total'] += stats['total']
        language_stats[lang]['count'] += 1
    
    if len(language_stats) > 1:
        print(f"\nPROGRESS BY LANGUAGE:")
        for lang in sorted(language_stats.keys()):
            lang_data = language_stats[lang]
            lang_completion = (lang_data['completed'] / lang_data['total'] * 100) if lang_data['total'] > 0 else 0
            print(f"  {lang.upper():6s} ({lang_data['count']:3d} shards): "
                  f"{create_progress_bar(lang_completion, width=20)} "
                  f"{lang_completion:5.1f}% "
                  f"({lang_data['completed']:,}/{lang_data['total']:,} subshards)")
    
    # Per-shard details
    print(f"\nPER-SHARD PROGRESS:")
    print(f"{'Shard ID':<12} {'Progress':<35} {'Completed':>10} {'Failed':>8} {'Remaining':>10} {'%':>7}")
    print("-" * 100)
    
    for stats in shard_stats:
        progress_bar = create_progress_bar(stats['completion_pct'], width=30)
        
        # Color coding (using ANSI colors)
        if stats['completion_pct'] == 100:
            status = '✓'
        elif stats['failed'] > 0:
            status = '⚠'
        else:
            status = '→'
        
        print(f"{stats['shard_id']:<12} {progress_bar} {status} "
              f"{stats['completed']:>10,} {stats['failed']:>8,} {stats['remaining']:>10,} "
              f"{stats['completion_pct']:>6.1f}%")
    
    # Shards that need attention
    problematic_shards = [s for s in shard_stats if s['failed'] > 10 or (s['completed'] == 0 and s['failed'] > 0)]
    if problematic_shards:
        print(f"\nSHARDS NEEDING ATTENTION ({len(problematic_shards)}):")
        for stats in problematic_shards:
            print(f"  {stats['shard_id']}: {stats['failed']} failures")
    
    # Completed shards
    completed_shards = [s for s in shard_stats if s['completion_pct'] == 100]
    if completed_shards:
        print(f"\nCOMPLETED SHARDS ({len(completed_shards)}):")
        print(f"  {', '.join(s['shard_id'] for s in completed_shards)}")
    
    # In-progress shards
    in_progress_shards = [s for s in shard_stats if 0 < s['completion_pct'] < 100]
    if in_progress_shards:
        print(f"\nIN-PROGRESS SHARDS ({len(in_progress_shards)}):")
        for stats in sorted(in_progress_shards, key=lambda x: x['completion_pct'], reverse=True):
            print(f"  {stats['shard_id']}: {stats['completion_pct']:5.1f}% "
                  f"({stats['completed']:,}/{stats['total']:,} subshards)")
    
    print("\n" + "=" * 100)


def watch_progress(progress_dir: Path, subshard_counts: Dict[str, int], sort_by: str = 'name', interval: int = 10):
    """Continuously monitor progress."""
    print("Starting progress monitor (press Ctrl+C to stop)...")
    
    while True:
        # Clear screen (works on Unix/Linux)
        print("\033[2J\033[H", end="")
        
        all_progress = load_all_progress(progress_dir)
        display_summary(all_progress, subshard_counts, sort_by)
        
        print(f"\nRefreshing in {interval} seconds... (Ctrl+C to stop)")
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor progress of pretraining data preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--progress-dir",
        type=str,
        default="/sphinx/u/salt-checkpoints/yodas2-mm-pretrain/progress",
        help="Directory containing progress files (default: ./progress)"
    )
    parser.add_argument(
        "--subshard-counts",
        type=str,
        default="../yodas2-mimi/subshard_counts.json",
        help="Path to subshard_counts.json file (default: auto-detect)"
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch and update progress"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Refresh interval in seconds for watch mode (default: 10)"
    )
    parser.add_argument(
        "--sort-by",
        choices=['name', 'completion', 'remaining'],
        default='name',
        help="Sort shards by: name (default), completion (%), or remaining count"
    )
    
    args = parser.parse_args()
    
    progress_dir = Path(args.progress_dir)
    
    if not progress_dir.exists():
        print(f"Error: Progress directory not found: {progress_dir}")
        return 1
    
    # Load subshard counts
    counts_file = Path(args.subshard_counts) if args.subshard_counts else None
    subshard_counts = load_subshard_counts(counts_file)
    print()  # Blank line after loading message
    
    if args.watch:
        try:
            watch_progress(progress_dir, subshard_counts, args.sort_by, args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    else:
        all_progress = load_all_progress(progress_dir)
        display_summary(all_progress, subshard_counts, args.sort_by)
    
    return 0


if __name__ == "__main__":
    exit(main())

