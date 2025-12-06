#!/usr/bin/env python3
"""
Script to fix whitespace issue in emilia-mm-pretrain dataset.
Replaces "<|text_start|> " with "<|text_start|>" and " <|text_end|>" with "<|text_end|>".

Three modes:
1. Fix mode (default): Download, fix, and save parquet files locally
2. Upload mode (--upload-dir): Upload all fixed parquet files from a directory to HuggingFace
3. Check status mode (--check-status + --shard-list): Check which shards are completed/pending
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, CommitOperationAdd


# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', 1)

SOURCE_REPO_ID = "potsawee/emilia-mm-pretrain"
TARGET_REPO_ID = "potsawee/emilia-mm-pretrain-fix"


def fix_text(text: str) -> str:
    """Fix whitespace issues in text column."""
    # Fix leading whitespace after <|text_start|>
    text = text.replace("<|text_start|> ", "<|text_start|>")
    # Fix trailing whitespace before <|text_end|>
    text = text.replace(" <|text_end|>", "<|text_end|>")
    return text


def is_shard_already_fixed_locally(fixed_parquet_path: Path) -> bool:
    """Check if the shard has already been fixed and saved locally."""
    return fixed_parquet_path.exists()


def is_shard_already_uploaded(repo_path: str) -> bool:
    """Check if the shard has already been uploaded to target HuggingFace repo."""
    api = HfApi()
    files = api.list_repo_files(repo_id=TARGET_REPO_ID, repo_type="dataset")
    return repo_path in files


def upload_directory(upload_dir: str, repo_path_prefix: str = "", batch_size: int = 100):
    """
    Upload all parquet files from a directory to HuggingFace.
    If more than batch_size files, uploads in batches of batch_size files per commit.
    
    Args:
        upload_dir: Local directory containing parquet files to upload
        repo_path_prefix: Path prefix in the repo (e.g., "Emilia/EN")
        batch_size: Maximum number of files per commit (default: 100)
    """
    upload_path = Path(upload_dir)
    
    if not upload_path.exists():
        print(f"Error: Upload directory does not exist: {upload_path}", flush=True)
        return
    
    # Find all parquet files
    parquet_files = sorted(list(upload_path.rglob("*.parquet")))
    
    if not parquet_files:
        print(f"No parquet files found in {upload_path}", flush=True)
        return
    
    print(f"Found {len(parquet_files)} parquet files to upload", flush=True)
    print(f"Will upload to: {TARGET_REPO_ID}/{repo_path_prefix}", flush=True)
    
    api = HfApi()
    
    if len(parquet_files) <= batch_size:
        # Use upload_folder to upload everything in a single commit
        print(f"Uploading folder {upload_path} to {TARGET_REPO_ID}/{repo_path_prefix}...", flush=True)
        api.upload_folder(
            folder_path=str(upload_path),
            path_in_repo=repo_path_prefix,
            repo_id=TARGET_REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload fixed parquet files to {repo_path_prefix}",
        )
        print(f"Upload complete. Uploaded {len(parquet_files)} files in a single commit.", flush=True)
    else:
        # Upload in batches
        num_batches = (len(parquet_files) + batch_size - 1) // batch_size
        print(f"Uploading in {num_batches} batches of up to {batch_size} files each...", flush=True)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(parquet_files))
            batch_files = parquet_files[start_idx:end_idx]
            
            print(f"\nBatch {batch_idx + 1}/{num_batches}: Uploading files {start_idx + 1}-{end_idx}...", flush=True)
            
            # Create commit operations for this batch
            operations = []
            for parquet_file in batch_files:
                relative_path = parquet_file.relative_to(upload_path)
                if repo_path_prefix:
                    path_in_repo = f"{repo_path_prefix}/{relative_path}"
                else:
                    path_in_repo = str(relative_path)
                
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=str(parquet_file),
                    )
                )
            
            # Create a single commit with all files in this batch
            api.create_commit(
                repo_id=TARGET_REPO_ID,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Upload fixed parquet files to {repo_path_prefix} (batch {batch_idx + 1}/{num_batches})",
            )
            
            print(f"Batch {batch_idx + 1}/{num_batches} complete. Uploaded {len(batch_files)} files.", flush=True)
        
        print(f"\nUpload complete. Uploaded {len(parquet_files)} files in {num_batches} batches.", flush=True)


def check_status(check_status_dir: str, shard_list_path: str, split: str):
    """
    Check the status of shards - which are completed locally and which are pending.
    
    Args:
        check_status_dir: Directory where fixed parquet files are saved (work-dir)
        shard_list_path: Path to file containing list of shard IDs (one per line)
        split: Split name (Emilia or Emilia-YODAS)
    """
    work_dir = Path(check_status_dir)
    shard_list_file = Path(shard_list_path)
    
    if not shard_list_file.exists():
        print(f"Error: Shard list file does not exist: {shard_list_file}", flush=True)
        return
    
    # Read shard IDs from file
    with open(shard_list_file, 'r') as f:
        shard_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Checking status for {len(shard_ids)} shards in split '{split}'", flush=True)
    print(f"Output directory: {work_dir}", flush=True)
    print("=" * 60, flush=True)
    
    completed = []
    pending = []
    
    for shard_id in shard_ids:
        lang = shard_id.split("-")[0]
        fixed_parquet_path = work_dir / split / lang / f"{shard_id}.parquet"
        
        if fixed_parquet_path.exists():
            completed.append(shard_id)
        else:
            pending.append(shard_id)
    
    if pending:
        print(f"\nPending shards ({len(pending)}):", flush=True)
        for shard_id in pending:
            print(f"  {shard_id}", flush=True)
    
    # Print summary at the end
    print("\n" + "=" * 60, flush=True)
    print(f"Checking status for {len(shard_ids)} shards in split '{split}'", flush=True)
    print(f"Output directory: {work_dir}", flush=True)
    print("=" * 60, flush=True)
    print(f"Completed: {len(completed)}/{len(shard_ids)} ({100*len(completed)/len(shard_ids):.1f}%)", flush=True)
    print(f"Pending:   {len(pending)}/{len(shard_ids)} ({100*len(pending)/len(shard_ids):.1f}%)", flush=True)
    print("=" * 60, flush=True)


def fix_shard(split: str, shard_id: str, cache_dir: Path, work_dir: Path):
    """Download, fix, and save a shard locally."""
    # Validate split
    assert split in ["Emilia", "Emilia-YODAS"], f"Invalid split: {split}"
    
    # Extract language from shard_id
    lang = shard_id.split("-")[0]
    assert lang in ["EN", "DE", "FR", "JA", "KO", "ZH"], f"Invalid language: {lang}"
    
    # Paths
    parquet_filename = f"{shard_id}.parquet"
    repo_path = f"{split}/{lang}/{parquet_filename}"
    fixed_parquet_path = work_dir / split / lang / f"{shard_id}.parquet"
    
    # Check if shard is already fixed locally
    if is_shard_already_fixed_locally(fixed_parquet_path):
        print(f"Shard already fixed locally: {fixed_parquet_path}, skipping", flush=True)
        return
    
    # Check if shard is already uploaded to HF
    print(f"Checking if shard {shard_id} is already uploaded...", flush=True)
    if is_shard_already_uploaded(repo_path):
        print(f"Shard already uploaded: {repo_path} exists in {TARGET_REPO_ID}, skipping", flush=True)
        return
    
    # Create directories
    cache_dir.mkdir(parents=True, exist_ok=True)
    fixed_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing shard: {shard_id} from {split}/{lang}", flush=True)
    
    # Step 1: Download the parquet file from source repo
    print(f"Downloading {repo_path} from {SOURCE_REPO_ID}...", flush=True)
    downloaded_path = hf_hub_download(
        repo_id=SOURCE_REPO_ID,
        filename=repo_path,
        repo_type="dataset",
        local_dir=cache_dir,
    )
    downloaded_path = Path(downloaded_path)
    print(f"Downloaded to: {downloaded_path}", flush=True)
    
    # Step 2: Load the parquet file
    print("Loading parquet file...", flush=True)
    df = pd.read_parquet(downloaded_path)
    print(f"Loaded {len(df)} rows", flush=True)
    
    # Step 3: Fix the text column
    print("Fixing text column...", flush=True)
    df["text"] = df["text"].apply(fix_text)
    
    # Step 4: Save as new parquet file
    print(f"Saving fixed parquet to {fixed_parquet_path}...", flush=True)
    df.to_parquet(fixed_parquet_path, index=False)
    
    # Step 5: Delete the downloaded original file to save space
    print("Cleaning up downloaded file...", flush=True)
    if downloaded_path.exists():
        os.remove(downloaded_path)
        
    print(f"Successfully fixed shard: {shard_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fix whitespace in emilia-mm-pretrain dataset")
    parser.add_argument("--split", type=str, help="Split to process (Emilia or Emilia-YODAS)")
    parser.add_argument("--shard-id", type=str, help="Shard ID (e.g., EN-B000000 or EN-000000)")
    parser.add_argument("--cache-dir", type=str, default="./cache_fix", help="Cache directory for downloaded files")
    parser.add_argument("--work-dir", type=str, default="./work_fix", help="Working directory for fixed parquet files")
    parser.add_argument("--upload-dir", type=str, help="Upload all parquet files from this directory to HuggingFace")
    parser.add_argument("--repo-path-prefix", type=str, default="", help="Path prefix in the repo (e.g., 'Emilia/EN')")
    parser.add_argument("--check-status", type=str, help="Check status of shards in this output directory")
    parser.add_argument("--shard-list", type=str, help="Path to file containing list of shard IDs (one per line)")
    args = parser.parse_args()

    # Check status mode
    if args.check_status:
        if not args.shard_list:
            parser.error("--shard-list is required when using --check-status")
        if not args.split:
            parser.error("--split is required when using --check-status")
        check_status(args.check_status, args.shard_list, args.split)
        return

    # Upload mode
    if args.upload_dir:
        upload_directory(args.upload_dir, args.repo_path_prefix)
        return
    
    # Fix mode - requires split and shard-id
    if not args.split or not args.shard_id:
        parser.error("--split and --shard-id are required when not using --upload-dir or --check-status")
    
    fix_shard(
        split=args.split,
        shard_id=args.shard_id,
        cache_dir=Path(args.cache_dir),
        work_dir=Path(args.work_dir),
    )


if __name__ == "__main__":
    main()
    # Usage examples:
    # 
    # Fix mode (download, fix, save locally):
    # python process_shard_fix.py --split Emilia --shard-id EN-B000001 --work-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir --cache-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/cache-dir
    #
    #
    # Check status mode (see which shards are completed/pending):
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/ --split Emilia --shard-list file_lists/en.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/ --split Emilia --shard-list file_lists/de.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/ --split Emilia --shard-list file_lists/zh.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/ --split Emilia --shard-list file_lists/ja.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/ --split Emilia --shard-list file_lists/ko.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/ --split Emilia --shard-list file_lists/fr.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/ --split Emilia-YODAS --shard-list file_lists/en_yodas.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/ --split Emilia-YODAS --shard-list file_lists/de_yodas.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/ --split Emilia-YODAS --shard-list file_lists/zh_yodas.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/ --split Emilia-YODAS --shard-list file_lists/ja_yodas.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/ --split Emilia-YODAS --shard-list file_lists/ko_yodas.txt
    # python process_shard_fix.py --check-status /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/ --split Emilia-YODAS --shard-list file_lists/fr_yodas.txt
    #
    #
    # Upload mode (upload entire folder in a single commit):
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/EN --repo-path-prefix Emilia/EN
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/DE --repo-path-prefix Emilia/DE
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/FR --repo-path-prefix Emilia/FR
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/JA --repo-path-prefix Emilia/JA
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/KO --repo-path-prefix Emilia/KO
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/ZH --repo-path-prefix Emilia/ZH
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/EN --repo-path-prefix Emilia-YODAS/EN
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/DE --repo-path-prefix Emilia-YODAS/DE
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/FR --repo-path-prefix Emilia-YODAS/FR
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/JA --repo-path-prefix Emilia-YODAS/JA
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/KO --repo-path-prefix Emilia-YODAS/KO
    # python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/ZH --repo-path-prefix Emilia-YODAS/ZH

    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n en -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/EN --repo-path-prefix Emilia/EN' -o emilia-mm-pretrain-fix/upload-en.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n de -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/DE --repo-path-prefix Emilia/DE' -o emilia-mm-pretrain-fix/upload-de.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n fr -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/FR --repo-path-prefix Emilia/FR' -o emilia-mm-pretrain-fix/upload-fr.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n ja -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/JA --repo-path-prefix Emilia/JA' -o emilia-mm-pretrain-fix/upload-ja.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n ko -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/KO --repo-path-prefix Emilia/KO' -o emilia-mm-pretrain-fix/upload-ko.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n zh -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir/Emilia/ZH --repo-path-prefix Emilia/ZH' -o emilia-mm-pretrain-fix/upload-zh.log

    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n en -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/EN --repo-path-prefix Emilia-YODAS/EN' -o emilia-mm-pretrain-fix/upload-en-yodas.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n de -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/DE --repo-path-prefix Emilia-YODAS/DE' -o emilia-mm-pretrain-fix/upload-de-yodas.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n fr -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/FR --repo-path-prefix Emilia-YODAS/FR' -o emilia-mm-pretrain-fix/upload-fr-yodas.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n ja -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/JA --repo-path-prefix Emilia-YODAS/JA' -o emilia-mm-pretrain-fix/upload-ja-yodas.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n ko -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/KO --repo-path-prefix Emilia-YODAS/KO' -o emilia-mm-pretrain-fix/upload-ko-yodas.log
    # nlprun -q jag -p standard -g 0 -r 50G -c 4 -n zh -x jagupard[19-20,26-27] 'python process_shard_fix.py --upload-dir /sphinx/u/salt-checkpoints/emilia-mm-pretrain-fix/work-dir-yodas/Emilia-YODAS/ZH --repo-path-prefix Emilia-YODAS/ZH' -o emilia-mm-pretrain-fix/upload-zh-yodas.log