#!/usr/bin/env bash
# Filter all matches JSONL files using human evaluation results

set -e

for csv in matching_results/*-corrections-humaneval.csv; do
    # Extract date prefix (e.g., 20240701)
    date=$(basename "$csv" | sed 's/-corrections-humaneval.csv//')

    input_jsonl="matching_results/${date}-matches.jsonl"
    output_jsonl="matching_results/${date}-matches-filtered.jsonl"

    if [[ -f "$input_jsonl" ]]; then
        echo "Processing $date..."
        uv run python wikidump/filter_by_humaneval.py "$csv" "$input_jsonl" "$output_jsonl"
        echo
    else
        echo "Skipping $date: $input_jsonl not found"
    fi
done
