#!/usr/bin/env python3
"""
Monitor progress of Yodas2 Mimi processing across all shards.

This script reads all progress files and provides a summary of:
- Completed sub-shards per shard
- Failed sub-shards per shard
- Overall progress statistics
- Estimated completion time
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys


def load_progress_files(progress_dir: Path) -> Dict[str, Dict]:
    """Load all progress files from the progress directory."""
    progress_files = list(progress_dir.glob("*_progress.json"))
    
    progress_data = {}
    for progress_file in sorted(progress_files):
        shard_id = progress_file.stem.replace("_progress", "")
        with open(progress_file, 'r') as f:
            progress_data[shard_id] = json.load(f)
    
    return progress_data


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def display_progress(progress_data: Dict[str, Dict], verbose: bool = False):
    """Display progress information."""
    
    print("=" * 80)
    print("Yodas2 Mimi Processing Progress Report")
    print("=" * 80)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    total_completed = 0
    total_failed = 0
    
    if not progress_data:
        print("No progress files found.")
        print("=" * 80)
        return
    
    # Per-shard summary
    print("\nPer-Shard Summary:")
    print("-" * 80)
    print(f"{'Shard ID':<15} {'Completed':<15} {'Failed':<15} {'Status':<20}")
    print("-" * 80)
    
    for shard_id in sorted(progress_data.keys()):
        data = progress_data[shard_id]
        completed = len(data.get("completed_subshards", []))
        failed = len(data.get("failed_subshards", []))
        
        total_completed += completed
        total_failed += failed
        
        if completed > 0 and failed == 0:
            status = "✓ Active"
        elif failed > 0:
            status = "⚠ Has failures"
        else:
            status = "Not started"
        
        print(f"{shard_id:<15} {completed:<15} {failed:<15} {status:<20}")
    
    print("-" * 80)
    print(f"{'TOTAL':<15} {total_completed:<15} {total_failed:<15}")
    print("-" * 80)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print("-" * 80)
    print(f"Active shards: {len(progress_data)}")
    print(f"Total completed sub-shards: {total_completed}")
    print(f"Total failed sub-shards: {total_failed}")
    
    if total_completed > 0:
        success_rate = (total_completed / (total_completed + total_failed)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    print("-" * 80)
    
    # Detailed view if verbose
    if verbose and total_failed > 0:
        print("\nFailed Sub-Shards (detailed):")
        print("-" * 80)
        for shard_id in sorted(progress_data.keys()):
            data = progress_data[shard_id]
            failed_list = data.get("failed_subshards", [])
            if failed_list:
                print(f"\n{shard_id}:")
                for subshard_id in failed_list:
                    print(f"  - {subshard_id}")
        print("-" * 80)
    
    print("\n" + "=" * 80)


def display_recently_completed(progress_data: Dict[str, Dict], limit: int = 10):
    """Display recently completed sub-shards."""
    print("\nRecently Completed Sub-Shards (last {}):".format(limit))
    print("-" * 80)
    
    all_completed = []
    for shard_id, data in progress_data.items():
        for subshard_id in data.get("completed_subshards", []):
            all_completed.append(f"{shard_id}/{subshard_id}")
    
    if all_completed:
        for item in all_completed[-limit:]:
            print(f"  ✓ {item}")
    else:
        print("  No completed sub-shards yet.")
    
    print("-" * 80)


def check_output_files(output_dir: Path, progress_data: Dict[str, Dict]):
    """Verify that output files exist for completed sub-shards."""
    print("\nOutput File Verification:")
    print("-" * 80)
    
    missing_files = []
    total_checked = 0
    
    for shard_id, data in progress_data.items():
        for subshard_id in data.get("completed_subshards", []):
            total_checked += 1
            expected_file = output_dir / shard_id / f"{subshard_id}.json"
            if not expected_file.exists():
                missing_files.append(f"{shard_id}/{subshard_id}")
    
    if missing_files:
        print(f"⚠ WARNING: {len(missing_files)} output files missing!")
        print("Missing files (first 10):")
        for item in missing_files[:10]:
            print(f"  - {item}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    else:
        print(f"✓ All {total_checked} output files verified successfully")
    
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monitor Yodas2 Mimi processing progress")
    parser.add_argument(
        "--progress-dir",
        type=str,
        default="./progress",
        help="Directory containing progress files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory containing output files"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information including failed sub-shards"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that output files exist for completed sub-shards"
    )
    parser.add_argument(
        "--watch",
        type=int,
        metavar="SECONDS",
        help="Continuously monitor progress every N seconds"
    )
    
    args = parser.parse_args()
    
    progress_dir = Path(args.progress_dir)
    output_dir = Path(args.output_dir)
    
    if not progress_dir.exists():
        print(f"Error: Progress directory not found: {progress_dir}", file=sys.stderr)
        sys.exit(1)
    
    if args.watch:
        import time
        try:
            while True:
                # Clear screen (works on most Unix terminals)
                print("\033[2J\033[H", end="")
                
                progress_data = load_progress_files(progress_dir)
                display_progress(progress_data, verbose=args.verbose)
                
                if args.verify:
                    check_output_files(output_dir, progress_data)
                
                print(f"\nRefreshing in {args.watch} seconds... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            sys.exit(0)
    else:
        progress_data = load_progress_files(progress_dir)
        display_progress(progress_data, verbose=args.verbose)
        display_recently_completed(progress_data)
        
        if args.verify:
            check_output_files(output_dir, progress_data)


if __name__ == "__main__":
    main()

