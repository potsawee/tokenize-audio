#!/usr/bin/env python3
"""
Monitor progress of Yodas2 Mimi processing across all shards.

This script reads all progress files and provides a summary of:
- Completed sub-shards per shard
- Failed sub-shards per shard
- Upload status (HuggingFace vs local)
- Overall progress statistics
- Estimated completion time
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
from huggingface_hub import HfApi


class HFStatusChecker:
    """Check file existence on HuggingFace."""
    
    def __init__(self, repo_id: Optional[str] = None):
        self.repo_id = repo_id
        self.enabled = repo_id is not None
        if self.enabled:
            self.api = HfApi()
            self._cache = {}  # Cache to avoid repeated API calls
    
    def file_exists_on_hf(self, path_in_repo: str) -> bool:
        """Check if a file exists on HuggingFace (with caching)."""
        if not self.enabled:
            return False
        
        if path_in_repo in self._cache:
            return self._cache[path_in_repo]
        
        try:
            exists = self.api.file_exists(
                repo_id=self.repo_id,
                filename=path_in_repo,
                repo_type="dataset",
            )
            self._cache[path_in_repo] = exists
            return exists
        except Exception:
            return False


def load_progress_files(progress_dir: Path) -> Dict[str, Dict]:
    """Load all progress files from the progress directory."""
    progress_files = list(progress_dir.glob("*_progress.json"))
    
    progress_data = {}
    for progress_file in sorted(progress_files):
        shard_id = progress_file.stem.replace("_progress", "")
        with open(progress_file, 'r') as f:
            progress_data[shard_id] = json.load(f)
    
    return progress_data


def load_subshard_counts(cache_file: Path) -> Optional[Dict[str, int]]:
    """Load cached sub-shard counts from file."""
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


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


def analyze_subshard_status(
    shard_id: str,
    subshard_id: str,
    output_dir: Path,
    hf_checker: HFStatusChecker
) -> str:
    """
    Analyze the status of a sub-shard.
    
    Returns:
        - "on_hf": File is on HuggingFace (may or may not be local)
        - "local_only": File is local but not on HF (pending upload)
        - "missing": File is neither on HF nor local (progress tracking issue)
    """
    hf_path = f"{shard_id}/{subshard_id}.json"
    local_path = output_dir / shard_id / f"{subshard_id}.json"
    
    on_hf = hf_checker.file_exists_on_hf(hf_path)
    on_local = local_path.exists()
    
    if on_hf:
        return "on_hf"
    elif on_local:
        return "local_only"
    else:
        return "missing"


def get_shard_statistics(
    shard_id: str,
    progress_data: Dict,
    output_dir: Path,
    hf_checker: HFStatusChecker
) -> Tuple[int, int, int, int]:
    """
    Get statistics for a shard.
    
    Returns:
        (on_hf_count, local_only_count, missing_count, failed_count)
    """
    completed_subshards = progress_data.get("completed_subshards", [])
    failed_subshards = progress_data.get("failed_subshards", [])
    
    on_hf = 0
    local_only = 0
    missing = 0
    
    for subshard_id in completed_subshards:
        status = analyze_subshard_status(shard_id, subshard_id, output_dir, hf_checker)
        if status == "on_hf":
            on_hf += 1
        elif status == "local_only":
            local_only += 1
        else:
            missing += 1
    
    return on_hf, local_only, missing, len(failed_subshards)


def display_progress(
    progress_data: Dict[str, Dict],
    output_dir: Path,
    hf_checker: HFStatusChecker,
    subshard_counts: Optional[Dict[str, int]] = None,
    verbose: bool = False
):
    """Display progress information with HF and local status."""
    
    print("=" * 80)
    print("Yodas2 Mimi Processing Progress Report")
    print("=" * 80)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if hf_checker.enabled:
        print(f"HuggingFace Repo: {hf_checker.repo_id}")
    else:
        print("HuggingFace: Not configured (local only)")
    print("=" * 80)
    
    total_on_hf = 0
    total_local_only = 0
    total_missing = 0
    total_failed = 0
    
    if not progress_data:
        print("No progress files found.")
        print("=" * 80)
        return
    
    # Per-shard summary
    print("\nPer-Shard Summary:")
    if subshard_counts:
        print("-" * 120)
        print(f"{'Shard ID':<12} {'On HF':<10} {'Local Only':<12} {'Missing':<10} {'Failed':<10} {'Target':<10} {'Status':<20}")
        print("-" * 120)
    else:
        print("-" * 100)
        print(f"{'Shard ID':<12} {'On HF':<10} {'Local Only':<12} {'Missing':<10} {'Failed':<10} {'Status':<20}")
        print("-" * 100)
    
    for shard_id in sorted(progress_data.keys()):
        data = progress_data[shard_id]
        on_hf, local_only, missing, failed = get_shard_statistics(
            shard_id, data, output_dir, hf_checker
        )
        
        total_on_hf += on_hf
        total_local_only += local_only
        total_missing += missing
        total_failed += failed
        
        # Determine status
        if on_hf > 0 or local_only > 0:
            if local_only > 0:
                status = "🔄 Pending upload"
            elif failed > 0:
                status = "⚠ Has failures"
            else:
                status = "✓ All uploaded"
        elif failed > 0:
            status = "⚠ Has failures"
        else:
            status = "Not started"
        
        # Show target if available
        if subshard_counts and shard_id in subshard_counts:
            target = subshard_counts[shard_id]
            print(f"{shard_id:<12} {on_hf:<10} {local_only:<12} {missing:<10} {failed:<10} {target:<10} {status:<20}")
        else:
            print(f"{shard_id:<12} {on_hf:<10} {local_only:<12} {missing:<10} {failed:<10} {status:<20}")
    
    if subshard_counts:
        print("-" * 120)
        print(f"{'TOTAL':<12} {total_on_hf:<10} {total_local_only:<12} {total_missing:<10} {total_failed:<10}")
        print("-" * 120)
    else:
        print("-" * 100)
        print(f"{'TOTAL':<12} {total_on_hf:<10} {total_local_only:<12} {total_missing:<10} {total_failed:<10}")
        print("-" * 100)
    
    # Overall statistics
    total_completed = total_on_hf + total_local_only + total_missing
    print("\nOverall Statistics:")
    print("-" * 80)
    print(f"Active shards: {len(progress_data)}")
    print(f"Total processed sub-shards: {total_completed}")
    print(f"  - Uploaded to HF: {total_on_hf}")
    print(f"  - Pending upload (local only): {total_local_only}")
    print(f"  - Missing (tracking issue): {total_missing}")
    print(f"Total failed sub-shards: {total_failed}")
    
    if total_completed > 0:
        success_rate = (total_completed / (total_completed + total_failed)) * 100
        print(f"Success rate: {success_rate:.1f}%")
    
    if total_on_hf > 0 and hf_checker.enabled:
        upload_rate = (total_on_hf / total_completed) * 100 if total_completed > 0 else 0
        print(f"Upload completion: {upload_rate:.1f}%")
    
    # Completion tracking (if subshard counts available)
    if subshard_counts:
        print("\n--- Completion Progress ---")
        total_target_subshards = sum(subshard_counts.values())
        
        if total_target_subshards > 0:
            completion_pct = (total_completed / total_target_subshards) * 100
            remaining = total_target_subshards - total_completed
            
            print(f"Total target sub-shards: {total_target_subshards:,}")
            print(f"Completed: {total_completed:,} / {total_target_subshards:,} ({completion_pct:.2f}%)")
            print(f"Remaining: {remaining:,}")
            
            # Progress bar
            bar_width = 50
            filled = int(bar_width * completion_pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"[{bar}] {completion_pct:.2f}%")
    else:
        print("\nℹ️  Run 'python get_total_subshards.py' to get completion estimates")
    
    print("-" * 80)
    
    # Detailed view if verbose
    if verbose:
        if total_failed > 0:
            print("\nFailed Sub-Shards:")
            print("-" * 80)
            for shard_id in sorted(progress_data.keys()):
                data = progress_data[shard_id]
                failed_list = data.get("failed_subshards", [])
                if failed_list:
                    print(f"\n{shard_id}:")
                    for subshard_id in failed_list:
                        print(f"  - {subshard_id}")
            print("-" * 80)
        
        if total_missing > 0:
            print("\nMissing Sub-Shards (marked complete but file not found):")
            print("-" * 80)
            for shard_id in sorted(progress_data.keys()):
                data = progress_data[shard_id]
                for subshard_id in data.get("completed_subshards", []):
                    status = analyze_subshard_status(shard_id, subshard_id, output_dir, hf_checker)
                    if status == "missing":
                        print(f"  - {shard_id}/{subshard_id}")
            print("-" * 80)
        
        if total_local_only > 0 and hf_checker.enabled:
            print("\nPending Upload (local files not yet on HF):")
            print("-" * 80)
            for shard_id in sorted(progress_data.keys()):
                data = progress_data[shard_id]
                pending_for_shard = []
                for subshard_id in data.get("completed_subshards", []):
                    status = analyze_subshard_status(shard_id, subshard_id, output_dir, hf_checker)
                    if status == "local_only":
                        pending_for_shard.append(subshard_id)
                if pending_for_shard:
                    print(f"\n{shard_id} ({len(pending_for_shard)} pending):")
                    for subshard_id in pending_for_shard[:10]:  # Show first 10
                        print(f"  - {subshard_id}")
                    if len(pending_for_shard) > 10:
                        print(f"  ... and {len(pending_for_shard) - 10} more")
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


def check_output_files(
    output_dir: Path,
    progress_data: Dict[str, Dict],
    hf_checker: HFStatusChecker
):
    """Verify that output files exist for completed sub-shards (on HF or locally)."""
    print("\nFile Verification:")
    print("-" * 80)
    
    missing_files = []
    total_checked = 0
    on_hf_count = 0
    local_only_count = 0
    
    for shard_id, data in progress_data.items():
        for subshard_id in data.get("completed_subshards", []):
            total_checked += 1
            status = analyze_subshard_status(shard_id, subshard_id, output_dir, hf_checker)
            
            if status == "on_hf":
                on_hf_count += 1
            elif status == "local_only":
                local_only_count += 1
            else:  # missing
                missing_files.append(f"{shard_id}/{subshard_id}")
    
    if hf_checker.enabled:
        print(f"✓ On HuggingFace: {on_hf_count}/{total_checked}")
        print(f"📁 Local only (pending upload): {local_only_count}/{total_checked}")
    else:
        print(f"📁 Local files: {local_only_count}/{total_checked}")
    
    if missing_files:
        print(f"⚠ WARNING: {len(missing_files)} files missing (neither on HF nor local)!")
        print("Missing files (first 10):")
        for item in missing_files[:10]:
            print(f"  - {item}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    else:
        print(f"✓ All {total_checked} files accounted for")
    
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Monitor Yodas2 Mimi processing progress")
    parser.add_argument(
        "--progress-dir",
        type=str,
        default="/sphinx/u/salt-checkpoints/yodas2-mm/progress",
        help="Directory containing progress files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/sphinx/u/salt-checkpoints/yodas2-mm/output",
        help="Directory containing output files"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo ID (e.g., potsawee/yodas2-mm). If not provided, only local files will be checked."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information including failed sub-shards and pending uploads"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that output files exist for completed sub-shards (on HF or locally)"
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
    
    # Initialize HF checker
    hf_checker = HFStatusChecker(repo_id=args.hf_repo_id)
    
    # Load subshard counts cache if available
    subshard_counts_file = Path("subshard_counts.json")
    subshard_counts = load_subshard_counts(subshard_counts_file)
    
    if args.hf_repo_id:
        print(f"Checking HuggingFace repo: {args.hf_repo_id}")
        print("This may take a moment to query the HF API...\n")
    else:
        print("HuggingFace repo not specified, checking local files only\n")
    
    if args.watch:
        import time
        try:
            while True:
                # Clear screen (works on most Unix terminals)
                print("\033[2J\033[H", end="")
                
                progress_data = load_progress_files(progress_dir)
                display_progress(progress_data, output_dir, hf_checker, subshard_counts, verbose=args.verbose)
                
                if args.verify:
                    check_output_files(output_dir, progress_data, hf_checker)
                
                print(f"\nRefreshing in {args.watch} seconds... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            sys.exit(0)
    else:
        progress_data = load_progress_files(progress_dir)
        display_progress(progress_data, output_dir, hf_checker, subshard_counts, verbose=args.verbose)
        display_recently_completed(progress_data)
        
        if args.verify:
            check_output_files(output_dir, progress_data, hf_checker)


if __name__ == "__main__":
    main()
    #  usage: python monitor_progress.py --hf-repo-id potsawee/yodas2-mm --verbose

    # usage: python monitor_progress.py --progress-dir /sphinx/u/salt-checkpoints/yodas2-mm/progress --hf-repo-id potsawee/yodas2-mm 
