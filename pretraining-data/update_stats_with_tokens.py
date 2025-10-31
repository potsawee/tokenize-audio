#!/usr/bin/env python3
"""
Update stats/stats_with_num_tokens.md with token counts from JSON files.

This script:
1. Reads stats/stats.md as the base
2. Reads all *.json files in token_stats_by_language/
3. Updates the "Num Tokens" column with token counts
4. Shows "XXX" for languages without JSON files
5. Sums only existing token counts for TOTAL row

Usage:
    python update_stats_with_tokens.py
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional


def parse_stats_md(filepath: Path) -> list[dict]:
    """Parse the stats.md file and return list of language entries."""
    entries = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (first 4 lines)
    for line in lines[4:]:
        line = line.strip()
        if not line or line.startswith('|-------') or '**TOTAL**' in line:
            continue
        
        # Parse table row: | language | rows | shards | subshards |
        match = re.match(r'\|\s*(\w+)\s*\|\s*([\d,]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|', line)
        if match:
            language = match.group(1)
            rows = match.group(2)
            shards = match.group(3)
            subshards = match.group(4)
            
            entries.append({
                'language': language,
                'rows': rows,
                'shards': shards,
                'subshards': subshards
            })
    
    return entries


def get_token_count(json_dir: Path, language: str) -> Optional[int]:
    """Get token count for a language from its JSON file."""
    json_file = json_dir / f"{language}.json"
    
    if not json_file.exists():
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Try to get token count from different possible locations
    if 'statistics' in data and 'total_tokens' in data['statistics']:
        return data['statistics']['total_tokens']
    elif 'extrapolated_statistics' in data and 'total_tokens' in data['extrapolated_statistics']:
        return data['extrapolated_statistics']['total_tokens']
    
    return None


def format_number(num: int) -> str:
    """Format number with commas."""
    return f"{num:,}"


def write_stats_with_tokens(output_file: Path, entries: list[dict], token_counts: Dict[str, Optional[int]]):
    """Write the updated stats file with token counts."""
    
    with open(output_file, 'w') as f:
        # Write header
        f.write("## Dataset Statistics by Language\n")
        f.write("\n")
        f.write("| Language | Rows | Shards | Subshards | Num Tokens |\n")
        f.write("|----------|-----:|-------:|----------:|-----------:|\n")
        
        # Write data rows
        total_tokens = 0
        count_with_tokens = 0
        
        for entry in entries:
            language = entry['language']
            rows = entry['rows']
            shards = entry['shards']
            subshards = entry['subshards']
            
            token_count = token_counts.get(language)
            
            if token_count is not None:
                tokens_str = format_number(token_count)
                total_tokens += token_count
                count_with_tokens += 1
            else:
                tokens_str = "XXX"
            
            f.write(f"| {language} | {rows} | {shards} | {subshards} | {tokens_str} |\n")
        
        # Write TOTAL row
        # Parse total rows from the first entry's rows
        total_rows = sum(int(e['rows'].replace(',', '')) for e in entries)
        total_shards = sum(int(e['shards']) for e in entries)
        total_subshards = sum(int(e['subshards']) for e in entries)
        
        f.write(f"| **TOTAL** | **{format_number(total_rows)}** | **{total_shards}** | **{total_subshards}** | **{format_number(total_tokens)}** |\n")
        f.write("\n")
        
        # Write summary comment
        f.write(f"<!-- Token counts: {count_with_tokens}/{len(entries)} languages completed -->\n")


def main():
    # Paths
    base_stats_file = Path("stats/stats.md")
    token_json_dir = Path("token_stats_by_language")
    output_file = Path("stats/stats_with_num_tokens.md")
    
    # Check if base stats file exists
    if not base_stats_file.exists():
        print(f"Error: Base stats file not found: {base_stats_file}")
        return 1
    
    # Parse base stats
    print(f"Reading base stats from: {base_stats_file}")
    entries = parse_stats_md(base_stats_file)
    print(f"Found {len(entries)} language entries")
    
    # Get token counts
    print(f"\nReading token counts from: {token_json_dir}")
    token_counts = {}
    completed_count = 0
    
    for entry in entries:
        language = entry['language']
        token_count = get_token_count(token_json_dir, language)
        token_counts[language] = token_count
        
        if token_count is not None:
            completed_count += 1
            print(f"  ✓ {language}: {format_number(token_count)} tokens")
        else:
            print(f"  ✗ {language}: No data")
    
    print(f"\nCompleted: {completed_count}/{len(entries)} languages")
    
    # Calculate total tokens
    total_tokens = sum(count for count in token_counts.values() if count is not None)
    print(f"Total tokens (completed languages): {format_number(total_tokens)} ({total_tokens/1e9:.2f}B)")
    
    # Write output
    print(f"\nWriting updated stats to: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_stats_with_tokens(output_file, entries, token_counts)
    
    print(f"✓ Successfully updated {output_file}")
    print(f"\nSummary:")
    print(f"  Languages processed: {completed_count}/{len(entries)}")
    print(f"  Languages pending: {len(entries) - completed_count}")
    print(f"  Total tokens: {format_number(total_tokens)}")
    
    return 0


if __name__ == "__main__":
    exit(main())

