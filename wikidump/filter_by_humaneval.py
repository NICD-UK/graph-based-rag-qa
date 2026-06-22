#!/usr/bin/env python3
"""Filter JSONL by accepted fuzzy match corrections."""

import argparse
import csv
import json


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL by accepted fuzzy match corrections."
    )
    parser.add_argument("csv_file", help="Human evaluation CSV file (*-humaneval.csv)")
    parser.add_argument("input_jsonl", help="Input JSONL file to filter")
    parser.add_argument("output_jsonl", help="Output filtered JSONL file")
    args = parser.parse_args()

    # Load human evaluation decisions
    # Collect rejected matches (accept != True) - these should be filtered out
    rejected_matches: set[str] = set()

    with open(args.csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # accept column contains 'True' or 'False' as strings
            if row["accept"] != "True":
                rejected_matches.add(row["matched"])

    print(f"Loaded {len(rejected_matches)} rejected matches from {args.csv_file}")

    # Filter JSONL: remove sources with rejected titles
    total_questions = 0
    questions_written = 0
    total_sources = 0
    sources_removed = 0

    with (
        open(args.input_jsonl, encoding="utf-8") as infile,
        open(args.output_jsonl, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            record = json.loads(line)
            total_questions += 1

            # Filter sources array, removing rejected titles
            original_sources = record.get("sources", [])
            total_sources += len(original_sources)

            # Check if any source has a rejected title
            has_rejected = any(
                src.get("title") in rejected_matches
                for src in original_sources
            )

            if has_rejected:
                sources_removed += len(original_sources)
                continue

            outfile.write(line)
            questions_written += 1

    sources_remaining = total_sources - sources_removed
    questions_removed = total_questions - questions_written
    print(f"Questions: {questions_written}/{total_questions} ({questions_removed} removed)")
    print(f"Sources: {sources_remaining}/{total_sources} ({sources_removed} removed)")
    print(f"Wrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()
