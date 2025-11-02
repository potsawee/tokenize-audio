#!/usr/bin/env python3
"""
Monitor progress of Emilia shard processing.
Reads shard list from file_lists/ and checks:
1. progress_dir for completed shards
2. work_dir for in-progress shards
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import sys

def load_shard_list(file_path: Path) -> List[str]:
    """Load list of shard IDs from file."""
    with open(file_path, 'r') as f:
        shards = [line.strip() for line in f if line.strip()]
    return shards

def load_progress(progress_dir: Path, shard_id: str) -> Optional[Dict]:
    """Load progress for a single shard from progress file."""
    progress_file = progress_dir / f"{shard_id}.json"
    if not progress_file.exists():
        return None
    
    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading progress for {shard_id}: {e}", file=sys.stderr)
        return None

def check_work_dir_status(work_dir: Path, shard_id: str) -> Optional[Dict]:
    """Check the work directory to see if shard is currently being processed."""
    shard_work_dir = work_dir / shard_id
    
    if not shard_work_dir.exists():
        return None
    
    status = {
        "work_dir_exists": True,
        "tar_exists": False,
        "extraction_complete": False,
        "cache_exists": False,
        "cache_file_count": 0,
    }
    
    # Check for tar file
    tar_path = shard_work_dir / "shard.tar"
    if tar_path.exists():
        status["tar_exists"] = True
        status["tar_size_mb"] = tar_path.stat().st_size / (1024 * 1024)
    
    # Check for extraction marker
    extraction_marker = shard_work_dir / "extracted.complete"
    if extraction_marker.exists():
        status["extraction_complete"] = True
    
    # Check for extraction directory
    extraction_dir = shard_work_dir / "extracted"
    if extraction_dir.exists():
        status["extraction_dir_exists"] = True
        # Count mp3 files
        mp3_files = list(extraction_dir.rglob("*.mp3"))
        status["total_audio_files"] = len(mp3_files)
    
    # Check for audio cache
    cache_path = shard_work_dir / "audio_str_cache.json"
    if cache_path.exists():
        status["cache_exists"] = True
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
                status["cache_file_count"] = len(cache)
        except:
            pass
    
    return status

def get_progress_summary(shard_list: List[str], progress_dir: Path, work_dir: Optional[Path] = None) -> Tuple[Dict, Dict]:
    """Get progress summary for all shards."""
    status_counts = defaultdict(int)
    shard_details = {}
    
    for shard_id in shard_list:
        # First check progress file
        progress = load_progress(progress_dir, shard_id)
        work_status = None
        
        if progress is not None:
            # Completed
            status = progress.get("status", "unknown")
        elif work_dir is not None:
            # Check work directory
            work_status = check_work_dir_status(work_dir, shard_id)
            
            if work_status is None or not work_status["work_dir_exists"]:
                status = "not_started"
            elif work_status["cache_exists"]:
                # Currently processing audio files
                cache_count = work_status.get("cache_file_count", 0)
                total_files = work_status.get("total_audio_files", 0)
                if total_files > 0:
                    progress_pct = (cache_count / total_files) * 100
                    status = f"processing_audio ({cache_count}/{total_files} = {progress_pct:.1f}%)"
                else:
                    status = "processing_audio"
            elif work_status["extraction_complete"]:
                status = "extraction_complete"
            elif work_status["tar_exists"]:
                status = "tar_downloaded"
            else:
                status = "in_progress"
        else:
            status = "not_started"
        
        status_counts[status] += 1
        shard_details[shard_id] = {
            "status": status,
            "progress": progress,
            "work_status": work_status
        }
    
    return dict(status_counts), shard_details

def print_summary(shard_list: List[str], status_counts: Dict, shard_details: Dict, verbose: bool = False):
    """Print progress summary."""
    total_shards = len(shard_list)
    
    print("=" * 80)
    print("EMILIA PROCESSING PROGRESS SUMMARY")
    print("=" * 80)
    print(f"\nTotal shards: {total_shards}")
    print("\nStatus breakdown:")
    print("-" * 80)
    
    # Sort statuses for consistent display
    for status in sorted(status_counts.keys()):
        count = status_counts[status]
        percentage = (count / total_shards) * 100
        print(f"  {status:30s}: {count:5d} ({percentage:5.1f}%)")
    
    # Calculate completion percentage
    completed_statuses = {"completed", "completed_already_uploaded"}
    completed_count = sum(status_counts.get(status, 0) for status in completed_statuses)
    completion_percentage = (completed_count / total_shards) * 100
    
    print("-" * 80)
    print(f"  {'Overall completion':30s}: {completed_count:5d} ({completion_percentage:5.1f}%)")
    print("=" * 80)
    
    # Detailed view if requested
    if verbose:
        print("\nDetailed shard status:")
        print("-" * 80)
        
        # Group by status
        by_status = defaultdict(list)
        for shard_id, details in shard_details.items():
            by_status[details["status"]].append(shard_id)
        
        for status in sorted(by_status.keys()):
            print(f"\n{status} ({len(by_status[status])} shards):")
            for shard_id in sorted(by_status[status])[:10]:  # Show first 10
                details = shard_details[shard_id]
                if details["progress"]:
                    timestamp = details["progress"].get("timestamp", "N/A")
                    num_samples = details["progress"].get("num_samples", "N/A")
                    print(f"  {shard_id}: timestamp={timestamp}, samples={num_samples}")
                elif details.get("work_status"):
                    ws = details["work_status"]
                    info = []
                    if ws.get("cache_exists"):
                        info.append(f"cache={ws.get('cache_file_count', 0)}/{ws.get('total_audio_files', 0)}")
                    if ws.get("extraction_complete"):
                        info.append("extraction_done")
                    if ws.get("tar_exists"):
                        info.append(f"tar={ws.get('tar_size_mb', 0):.1f}MB")
                    print(f"  {shard_id}: {', '.join(info) if info else 'work_dir_exists'}")
                else:
                    print(f"  {shard_id}: No activity")
            
            if len(by_status[status]) > 10:
                print(f"  ... and {len(by_status[status]) - 10} more")
        
        print("=" * 80)

def list_incomplete_shards(shard_list: List[str], shard_details: Dict, output_file: Path = None):
    """List shards that are not completed."""
    completed_statuses = {"completed", "completed_already_uploaded"}
    
    incomplete_shards = [
        shard_id for shard_id in shard_list
        if shard_details[shard_id]["status"] not in completed_statuses
    ]
    
    if output_file:
        with open(output_file, 'w') as f:
            for shard_id in incomplete_shards:
                f.write(f"{shard_id}\n")
        print(f"\nWrote {len(incomplete_shards)} incomplete shards to {output_file}")
    else:
        print(f"\nIncomplete shards ({len(incomplete_shards)}):")
        for shard_id in incomplete_shards[:20]:  # Show first 20
            print(f"  {shard_id}")
        if len(incomplete_shards) > 20:
            print(f"  ... and {len(incomplete_shards) - 20} more")

def main():
    parser = argparse.ArgumentParser(description="Monitor Emilia shard processing progress")
    parser.add_argument("--shard-list", type=str, default="./file_lists/en.txt",
                       help="Path to file containing list of shard IDs")
    parser.add_argument("--progress-dir", type=str, required=True,
                       help="Directory containing progress JSON files")
    parser.add_argument("--work-dir", type=str,
                       help="Work directory to check for in-progress shards")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed shard information")
    parser.add_argument("--list-incomplete", type=str,
                       help="Output file to write incomplete shard IDs")
    
    args = parser.parse_args()
    
    shard_list_path = Path(args.shard_list)
    progress_dir = Path(args.progress_dir)
    work_dir = Path(args.work_dir) if args.work_dir else None
    
    if not shard_list_path.exists():
        print(f"Error: Shard list file not found: {shard_list_path}", file=sys.stderr)
        sys.exit(1)
    
    if not progress_dir.exists():
        print(f"Warning: Progress directory not found: {progress_dir}", file=sys.stderr)
        print("Creating progress directory...", file=sys.stderr)
        progress_dir.mkdir(parents=True, exist_ok=True)
    
    if work_dir and not work_dir.exists():
        print(f"Warning: Work directory not found: {work_dir}", file=sys.stderr)
        work_dir = None
    
    # Load shard list
    print(f"Loading shard list from: {shard_list_path}")
    shard_list = load_shard_list(shard_list_path)
    print(f"Found {len(shard_list)} shards")
    
    if work_dir:
        print(f"Checking work directory: {work_dir}")
    
    # Get progress summary
    status_counts, shard_details = get_progress_summary(shard_list, progress_dir, work_dir)
    
    # Print summary
    print_summary(shard_list, status_counts, shard_details, verbose=args.verbose)
    
    # List incomplete shards if requested
    if args.list_incomplete:
        list_incomplete_shards(shard_list, shard_details, Path(args.list_incomplete))

if __name__ == "__main__":
    main()
    # Example usage:
    # python monitor_progress.py --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work
    # python monitor_progress.py --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work --verbose
    # python monitor_progress.py --progress-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/progress --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain/work --list-incomplete incomplete_shards.txt

