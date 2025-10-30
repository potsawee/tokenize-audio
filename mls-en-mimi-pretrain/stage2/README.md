# Stage 2: Merge and Upload to HuggingFace

This stage merges audio_str from multiple JSON files (from the same original audio file) and creates interleaved text-audio documents for upload to HuggingFace.

## Overview

The script groups JSON files by their `original_path`, orders them by `begin_time`, and creates two types of interleaved documents:

1. **Text-first**: `<|begin_of_text|><|text_start|>text1<|text_end|><|audio_start|>audio1<|audio_end|>...<|end_of_text|>`
2. **Audio-first**: `<|begin_of_text|><|audio_start|>audio1<|audio_end|><|text_start|>text1<|text_end|>...<|end_of_text|>`

## Usage

```bash
python stage2/merge_and_upload.py \
    --list-dir stage2/list_of_batches \
    --output-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train \
    --hf-repo-id your-username/mls-mm-merged
```

The script will process all `.txt` files in the directory. Each list file will be uploaded as a separate parquet file with the same name (e.g., `temp000.txt` → `temp000.parquet`).

### Parameters

- `--list-dir`: Directory containing list files (one per batch, each with speaker_id-book_id pairs)
- `--output-dir`: Base directory containing the JSON output files from stage 1
- `--hf-repo-id`: HuggingFace repository ID where data will be uploaded

**Note**: The parquet filename is automatically extracted from each list file name. For example, `temp000.txt` → `temp000.parquet`, `temp001.txt` → `temp001.parquet`, etc.

## Input Format

### List File (`list_of_batches/000.txt`)
```
1472-1437
1472-1477
2300-3456
```

Each line contains `speaker_id-book_id` identifying a folder in the output directory.

### JSON Files Structure
```
output_dir/
├── 1472/
│   └── 1437/
│       ├── 1472-1437-00001000-00002000-hash1.json
│       └── 1472-1437-00002000-00003000-hash2.json
```

Each JSON file contains:
```json
{
  "entry_id": "1472-1437-00001000-00002000-hash",
  "original_path": "http://example.com/audio.mp3",
  "speaker_id": "1472",
  "book_id": "1437",
  "transcript": "the text content",
  "begin_time": 10.0,
  "end_time": 20.0,
  "audio_duration": 10.0,
  "audio_str": "encoded_audio_tokens"
}
```

## Output Format

The script creates a parquet file with the following fields:

- `original_path`: URL/path of the original audio file
- `text_first`: Interleaved document with text before audio
- `audio_first`: Interleaved document with audio before text
- `num_segments`: Number of segments merged
- `speaker_id`: Speaker ID
- `book_id`: Book ID

## HuggingFace Upload

The parquet file is uploaded to the specified HuggingFace repository under `data/{batch_name}.parquet`.

Example: `list_of_batches/temp000.txt` → `your-username/mls-mm-merged/data/temp000.parquet`

## Creating Batch Lists

Use the provided script to automatically create batch list files:

```bash
python stage2/create_batch_lists.py \
    --output-dir /sphinx/u/salt-checkpoints/mls-mm-pretrain/output_audio_str_train \
    --list-dir stage2/list_of_batches \
    --speakers-per-batch 10
```

This will:
1. Scan the output directory for all speaker_id/book_id folders
2. Group them into batches of 10 unique speakers each
3. Create files named `train-0000-of-XXXX.txt`, `train-0001-of-XXXX.txt`, etc.
4. Each file contains speaker_id-book_id pairs for that batch

### Parameters

- `--output-dir`: Base directory containing the JSON output files from stage 1
- `--list-dir`: Directory to write batch list files (default: `stage2/list_of_batches`)
- `--speakers-per-batch`: Number of unique speakers per batch (default: 10)

## Notes

- Files from the same `original_path` are automatically grouped together
- Within each group, files are ordered by `begin_time`
- Empty folders or folders without JSON files are skipped
- Progress is logged for each folder processed

