#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="/Volumes/WDBLACK/enwiki-dumps"
OUTPUT_DIR="matching_results"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/enwiki-*-pages-articles-multistream.xml.bz2; do
    basename=$(basename "$file")
    SNAPSHOT=$(echo "$basename" | sed -n 's/enwiki-\([0-9]*\)-pages-articles.*/\1/p')

    if [[ -z "$SNAPSHOT" ]]; then
        echo "Could not extract snapshot from: $basename"
        continue
    fi

    OUTPUT_FILE="$OUTPUT_DIR/${SNAPSHOT}-matches.jsonl"
    if [[ -f "$OUTPUT_FILE" ]]; then
        echo "[$SNAPSHOT] Skipping, output file already exists: $OUTPUT_FILE"
        continue
    fi

    INDEX_FILE="${file%-multistream.xml.bz2}-multistream-index.txt.bz2"
    if [[ ! -f "$INDEX_FILE" ]]; then
        echo "[$SNAPSHOT] Building index: $INDEX_FILE"
        uv run python wikidump/build_index.py "$file"
    fi

    echo "Processing snapshot: $SNAPSHOT"

    uv run python wikidump/load_monaco.py \
        --input-file "$file" \
        --output-jsonl "$OUTPUT_DIR/${SNAPSHOT}-matches.jsonl" \
        --corrections-csv "$OUTPUT_DIR/${SNAPSHOT}-corrections.csv" \
        2>&1 | tee "$OUTPUT_DIR/${SNAPSHOT}-monaco.log" | sed "s/^/[$SNAPSHOT] /"
done

echo "Done processing all snapshots."
