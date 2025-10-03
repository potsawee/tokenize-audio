#!/usr/bin/env python3
"""
Live monitor for Yodas2 Mimi processing - shows both completed and in-progress work.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys


def check_work_in_progress(work_dir: Path, output_dir: Path):
    """Check for work in progress by examining work and output directories."""
    print("\nWork In Progress:")
    print("-" * 80)
    
    if not work_dir.exists():
        print("No work directory found")
        return
    
    # Find all shard directories
    shard_dirs = [d for d in work_dir.iterdir() if d.is_dir()]
    
    if not shard_dirs:
        print("No active work found")
        return
    
    for shard_dir in sorted(shard_dirs):
        shard_id = shard_dir.name
        print(f"\n{shard_id}:")
        
        # Find sub-shard work directories
        subshard_dirs = [d for d in shard_dir.iterdir() if d.is_dir() and d.name.endswith('_audio')]
        
        for subshard_dir in sorted(subshard_dirs):
            subshard_id = subshard_dir.name.replace('_audio', '')
            
            # Count audio files
            audio_files = list(subshard_dir.glob('*.wav'))
            num_audio_files = len(audio_files)
            
            # Check output file
            output_file = output_dir / shard_id / f"{subshard_id}.json"
            if output_file.exists():
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                    processed = len(data)
                    
                    # Count how many have codes
                    with_codes = sum(1 for entry in data if 'codes' in entry)
                    
                    status = f"📝 {with_codes}/{processed} audio files encoded"
                    if with_codes == processed:
                        status += " (may be completing...)"
                except:
                    status = "⚠️  Output file exists but couldn't read"
            else:
                status = f"🔄 Extracting/starting ({num_audio_files} audio files found)"
            
            print(f"  {subshard_id}: {status}")


def check_completed(progress_dir: Path):
    """Check completed sub-shards from progress files."""
    progress_files = list(progress_dir.glob("*_progress.json"))
    
    if not progress_files:
        return None
    
    results = {}
    for progress_file in sorted(progress_files):
        shard_id = progress_file.stem.replace("_progress", "")
        with open(progress_file, 'r') as f:
            data = json.load(f)
        results[shard_id] = data
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Live monitor for Yodas2 processing")
    parser.add_argument("--work-dir", type=str, default="./work", help="Work directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--progress-dir", type=str, default="./progress", help="Progress directory")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="Continuously monitor every N seconds")
    
    args = parser.parse_args()
    
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    progress_dir = Path(args.progress_dir)
    
    def display():
        # Clear screen if watching
        if args.watch:
            print("\033[2J\033[H", end="")
        
        print("=" * 80)
        print("Yodas2 Mimi Processing - Live Monitor")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Check work in progress
        check_work_in_progress(work_dir, output_dir)
        
        # Check completed
        completed = check_completed(progress_dir)
        if completed:
            print("\n" + "=" * 80)
            print("Completed Sub-Shards:")
            print("-" * 80)
            for shard_id, data in sorted(completed.items()):
                num_completed = len(data.get("completed_subshards", []))
                num_failed = len(data.get("failed_subshards", []))
                print(f"{shard_id}: {num_completed} completed, {num_failed} failed")
        else:
            print("\n" + "=" * 80)
            print("No sub-shards fully completed yet")
        
        print("=" * 80)
    
    if args.watch:
        import time
        try:
            while True:
                display()
                print(f"\nRefreshing in {args.watch} seconds... (Ctrl+C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            sys.exit(0)
    else:
        display()


if __name__ == "__main__":
    main()

