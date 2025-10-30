# MLS-EN Mimi Pretrain Processing

This directory contains scripts for processing MLS-EN dataset shards with Mimi encoding in parallel using SLURM.

## Quick Start

### Submit Jobs (Limited to 100 concurrent)
```bash
./submit_all_shards.sh 100
```

### Check Progress
```bash
./check_progress.sh
```

### Monitor Jobs
```bash
squeue -u $USER
watch -n 10 'squeue -u $USER | grep mls'
```

### Cancel All Jobs
```bash
./cancel_all_jobs.sh
```

## Scripts Overview

### Main Processing Script
- **`process_shard.py`** - Processes a single shard of MLS-EN data
  - Downloads parquet file from HuggingFace
  - Encodes audio with Mimi model
  - Saves JSON with audio codes
  - Tracks progress for resumability

### Submission Scripts

#### `submit_all_shards.sh`
**Submits jobs with concurrency control**

```bash
./submit_all_shards.sh [max_concurrent_jobs]
```

- **Default**: 100 concurrent jobs
- **Features**:
  - Maintains maximum number of concurrent jobs
  - Waits when queue is full
  - Skips shards already in queue
  - Can be safely rerun

**Examples:**
```bash
./submit_all_shards.sh 50   # Limit to 50 jobs
./submit_all_shards.sh 200  # Limit to 200 jobs
./submit_all_shards.sh      # Default: 100 jobs
```

#### `submit_shard.sh`
**Submits a single shard**

```bash
./submit_shard.sh train-00000-of-01416
```

- For manual job submission
- Used internally by other scripts

### Monitoring Scripts

#### `check_progress.sh`
**Check overall processing progress**

```bash
./check_progress.sh
```

**Output:**
```
============================================
Shard Processing Progress Summary
============================================

Total Shards:        1416
Completed:           450
In Progress:         100
Not Started:         866

Completion Rate:     31.78%
============================================
```

#### `cancel_all_jobs.sh`
**Cancel all submitted jobs**

```bash
./cancel_all_jobs.sh
```

- Finds all jobs with names starting with `mls`
- Shows list of jobs to be cancelled
- Asks for confirmation before cancelling
- Reports cancellation status

## File Structure

```
mls-en-mimi-pretrain/
├── process_shard.py              # Main processing script
├── utils.py                      # Helper utilities
├── submit_all_shards.sh          # Submit with concurrency limit
├── submit_shard.sh               # Submit single shard
├── check_progress.sh             # Check progress
├── cancel_all_jobs.sh            # Cancel all jobs
├── file_lists/
│   └── train_files.txt          # List of all shards (1,416)
└── submit/
    └── job_template.sh          # SLURM job template
```

## Progress Tracking

Progress is automatically tracked in `/sphinx/u/salt-checkpoints/mls-mm-pretrain/progress/`

**Progress file format:**
```json
{
  "shard_id": "train-00000-of-01416",
  "processed_count": 1500,
  "total_count": 3000,
  "last_processed_index": 1499
}
```

- Jobs can be safely interrupted and resumed
- Progress is saved every 500 entries
- Resumption is fast (jumps to last processed index)

## Output

Processed files are saved to:
```
/sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train/
├── <speaker_id>/
│   └── <book_id>/
│       └── <entry_id>.json
```

**Output JSON format:**
```json
{
  "entry_id": "12345-67890-00001000-00002000-abc123",
  "original_path": "...",
  "speaker_id": "12345",
  "book_id": "67890",
  "transcript": "...",
  "begin_time": 10.0,
  "end_time": 20.0,
  "audio_duration": 10.0,
  "audio_str": "encoded_audio_tokens..."
}
```

## Logs

Logs are saved to:
```
/sphinx/u/salt-checkpoints/mls-mm-pretrain/logs/<shard_id>.log
```

**View logs:**
```bash
# View specific shard
tail -f /sphinx/u/salt-checkpoints/mls-mm-pretrain/logs/train-00000-of-01416.log

# View all recent logs
ls -lt /sphinx/u/salt-checkpoints/mls-mm-pretrain/logs/ | head -20
```

## Commenting Out Shards

You can skip specific shards by commenting them out in `file_lists/train_files.txt`:

```
# data/train-00000-of-01416.parquet  <- This will be skipped
data/train-00001-of-01416.parquet
data/train-00002-of-01416.parquet
```

## Job Configuration

Edit `submit/job_template.sh` to modify:
- Time limit: `#SBATCH --time=14-00:00:00`
- Memory: `#SBATCH --mem=20G`
- CPUs: `#SBATCH --cpus-per-task=8`
- GPU constraints: `#SBATCH --constraint=[40G|48G|80G]`

## Troubleshooting

### Jobs keep dying
1. Check logs: `tail -f /sphinx/u/salt-checkpoints/mls-mm-pretrain/logs/<shard_id>.log`
2. Check progress: `cat /sphinx/u/salt-checkpoints/mls-mm-pretrain/progress/progress_<shard_id>.json`
3. Resubmit: Jobs will automatically resume from last checkpoint

### Queue is full
Use `submit_all_shards.sh` with a lower limit:
```bash
./submit_all_shards.sh 50
```

### Want to cancel all jobs
```bash
./cancel_all_jobs.sh
```

Or directly (cancel all one by one):
```bash
squeue -u $USER -h -o "%i %j" | grep "mls" | awk '{print $1}' | xargs -n1 scancel
```

### Check specific shard status
```bash
squeue -u $USER -n mls00000
```

## Resource Usage

Per job:
- **GPU**: 1x (40GB/48GB/80GB)
- **CPU**: 8 cores
- **Memory**: 20GB
- **Time**: Up to 14 days
- **Storage**: ~50GB temporary (cleaned after processing)

## Notes

- Each parquet file contains ~300-500 audio entries
- Processing time: ~1-3 hours per shard (varies by size)
- Total dataset: 1,416 shards to process
- Progress tracking ensures no data loss on job failures

