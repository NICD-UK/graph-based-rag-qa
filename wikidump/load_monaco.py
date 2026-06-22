#!/usr/bin/env python3
"""Load MoNaCo execution traces and match sources to Wikipedia titles."""

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import tempfile
from contextlib import nullcontext
from urllib.parse import unquote

import duckdb
import pyarrow as pa
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files
from rapidfuzz import fuzz, process
from tqdm import tqdm

from wiki_backend import (
    discover_index_pairs,
    find_sibling_index,
    iter_titles_from_index,
)

load_dotenv()

REPO_ID = "allenai/MoNaCo_Benchmark"


def get_hf_token():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in environment")
    return token


def _load_single_trace(args: tuple[str, str]) -> tuple[int, dict]:
    """Worker function to load a single execution trace."""
    filename, token = args
    ex_num = int(filename.split("_ex_")[1].replace(".json", ""))
    file_path = hf_hub_download(
        repo_id=REPO_ID, filename=filename, repo_type="dataset", token=token
    )
    with open(file_path) as f:
        return ex_num, json.load(f)


def load_all_execution_traces(num_workers: int | None = None) -> dict[int, dict]:
    """Load all execution traces from the dataset in parallel."""
    token = get_hf_token()
    files = list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=token)
    trace_files = [f for f in files if f.startswith("execution_traces/dataset_ex_")]

    if num_workers is None:
        num_workers = mp.cpu_count()

    args = [(f, token) for f in trace_files]

    traces = {}
    with mp.Pool(num_workers) as pool:
        for ex_num, trace in tqdm(
            pool.imap_unordered(_load_single_trace, args),
            total=len(trace_files),
            desc=f"Loading traces ({num_workers} workers)",
        ):
            traces[ex_num] = trace
    return traces


def extract_article_titles(raw: str) -> list[str]:
    """Extract article titles from a string that may contain multiple Wikipedia URLs.

    Handles URLs concatenated with various delimiters (\\r, \\n, ',', ';', etc.)
    by splitting on 'http' to find individual URLs.

    Returns a list of cleaned article titles.
    """
    titles = []

    # Split on 'http' boundaries to separate concatenated URLs
    for part in re.split(r"(?=https?://)", raw):
        if "/wiki/" not in part:
            continue

        # Extract the path after /wiki/, stopping at anchors, query params, or whitespace
        match = re.search(r"/wiki/([^#?\s\r\n]+)", part)
        if not match:
            continue

        raw_title = match.group(1)
        title = unquote(raw_title).replace("_", " ")

        # Strip trailing garbage punctuation, but preserve closing parens/brackets
        # which are valid in titles like "Python (programming language)"
        title = re.sub(r"[,;:!.\s]+$", "", title)

        if title:
            titles.append(title)

    return titles


def create_titles_db(
    pairs: list[tuple[str, str]], db_path: str, batch_size: int = 500_000
) -> duckdb.DuckDBPyConnection:
    """Stream titles from index pairs into a DuckDB table with an FTS index.

    Uses batched pyarrow inserts to keep memory bounded while ingesting tens
    of millions of rows.
    """
    conn = duckdb.connect(db_path)
    conn.install_extension("fts")
    conn.load_extension("fts")

    conn.execute(
        """
        CREATE TABLE titles (
            page_id BIGINT,
            title VARCHAR,
            normalized VARCHAR
        )
        """
    )

    batch_page_ids: list[int] = []
    batch_titles: list[str] = []
    batch_norm: list[str] = []

    def flush() -> None:
        if not batch_page_ids:
            return
        arrow_table = pa.table(
            {
                "page_id": batch_page_ids,
                "title": batch_titles,
                "normalized": batch_norm,
            }
        )
        conn.register("batch_table", arrow_table)
        conn.execute("INSERT INTO titles SELECT * FROM batch_table")
        conn.unregister("batch_table")
        batch_page_ids.clear()
        batch_titles.clear()
        batch_norm.clear()

    for page_id, title in iter_titles_from_index(pairs):
        batch_page_ids.append(int(page_id))
        batch_titles.append(title)
        batch_norm.append(title.lower().replace("_", " "))
        if len(batch_page_ids) >= batch_size:
            flush()
    flush()

    conn.execute("CREATE INDEX idx_title ON titles(title)")
    conn.execute("CREATE INDEX idx_normalized ON titles(normalized)")
    conn.execute("PRAGMA create_fts_index('titles', 'page_id', 'title')")

    return conn


def _match_title(
    query: str, conn: duckdb.DuckDBPyConnection, fts_limit: int
) -> tuple[str | None, str | None, tuple[str, float] | None]:
    """
    Match a query to a title using DuckDB.

    Returns (page_id, matched_title, correction_or_none).
    correction_or_none is (matched_title, score) if fuzzy matched, else None.
    """
    # 1. Exact match
    result = conn.execute(
        "SELECT page_id, title FROM titles WHERE title = ?", [query]
    ).fetchone()
    if result:
        return str(result[0]), result[1], None

    # 2. Normalized match
    normalized = query.lower().replace("_", " ").strip()
    result = conn.execute(
        "SELECT page_id, title FROM titles WHERE normalized = ?", [normalized]
    ).fetchone()
    if result:
        return str(result[0]), result[1], None

    # 3. FTS + fuzzy scoring
    candidates = conn.execute(
        """
        SELECT page_id, title, fts_main_titles.match_bm25(page_id, ?) AS score
        FROM titles
        WHERE score IS NOT NULL
        ORDER BY score DESC
        LIMIT ?
    """,
        [query, fts_limit],
    ).fetchall()

    if not candidates:
        return None, None, None

    # Score with rapidfuzz
    titles = [c[1] for c in candidates]
    page_ids = {c[1]: str(c[0]) for c in candidates}

    results = process.extract(query, titles, scorer=fuzz.WRatio, limit=fts_limit)
    if not results:
        return None, None, None

    top_score = results[0][1]
    top_matches = [(title, score) for title, score, _ in results if score == top_score]

    # Among ties, prefer title closest in length to query
    query_len = len(query)
    matched_title, score = min(top_matches, key=lambda x: abs(len(x[0]) - query_len))

    return page_ids[matched_title], matched_title, (matched_title, score)


def process_traces(
    traces: dict[int, dict],
    conn: duckdb.DuckDBPyConnection,
    output_jsonl_path: str,
    corrections_csv_path: str,
    fts_limit: int,
    unmatched_csv_path: str | None = None,
) -> tuple[int, int, int, int]:
    """
    Process all traces sequentially and stream results to disk.

    Uses DuckDB FTS for efficient title matching.

    Returns (num_results, num_sources, num_corrections, num_unmatched).
    """
    num_results = 0
    total_sources = 0
    num_unmatched = 0
    seen_corrections: set[str] = set()

    with (
        open(output_jsonl_path, "w", encoding="utf-8") as jsonl_file,
        open(corrections_csv_path, "w", newline="", encoding="utf-8") as csv_file,
        (
            open(unmatched_csv_path, "w", newline="", encoding="utf-8")
            if unmatched_csv_path
            else nullcontext()
        ) as unmatched_file,
    ):
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["original", "matched", "score"])

        unmatched_writer = None
        if unmatched_file is not None:
            unmatched_writer = csv.writer(unmatched_file)
            unmatched_writer.writerow(["ex_num", "title"])

        for ex_num, trace in tqdm(traces.items(), desc="Processing traces"):
            corrections: dict[str, tuple[str, float]] = {}

            question = next(k for k in trace.keys() if k != "canary")
            provenance = trace[question].get("provenance", {})

            # Collect all source URLs
            source_titles = set()
            for decompositions in provenance.values():
                if isinstance(decompositions, list):
                    for decomp in decompositions:
                        for url in decomp.get("source_url") or []:
                            source_titles.update(extract_article_titles(url))

            # Match each source
            matched_sources = []
            for title in source_titles:
                page_id, matched_title, correction = _match_title(title, conn, fts_limit)
                if page_id is not None:
                    matched_sources.append({
                        "page_id": page_id,
                        "title": matched_title,
                    })
                    if correction is not None:
                        corrections[title] = correction
                else:
                    num_unmatched += 1
                    if unmatched_writer is not None:
                        unmatched_writer.writerow([ex_num, title])

            result = {
                "ex_num": ex_num,
                "question": question,
                "sources": matched_sources,
            }

            # Stream result to JSONL
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            num_results += 1
            total_sources += len(result["sources"])

            # Stream new corrections to CSV
            for original, (matched, score) in corrections.items():
                if original not in seen_corrections:
                    seen_corrections.add(original)
                    csv_writer.writerow([original, matched, score])
                    print(f"  {original!r} -> {matched!r} (score: {score:.1f})")

    return num_results, total_sources, len(seen_corrections), num_unmatched


def resolve_index_pairs(
    input_file: str | None, dump_dir: str | None
) -> list[tuple[str, str]]:
    """Resolve index/xml pairs from --input-file or --dump-dir, erroring if none."""
    if input_file:
        input_path = os.path.abspath(input_file)
        if not os.path.exists(input_path):
            raise SystemExit(f"Error: input file not found: {input_path}")
        sibling = find_sibling_index(input_path)
        if sibling is None:
            raise SystemExit(
                f"Error: no sibling index found next to {input_path}. "
                f"Run build_index.py to create one."
            )
        return [(sibling, input_path)]

    assert dump_dir is not None
    dump_path = os.path.abspath(dump_dir)
    if not os.path.isdir(dump_path):
        raise SystemExit(f"Error: dump directory not found: {dump_path}")
    pairs = discover_index_pairs(dump_path)
    if not pairs:
        raise SystemExit(
            f"Error: no *-multistream-index*.txt-p*.bz2 files found in {dump_path}"
        )
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match MoNaCo execution trace sources to Wikipedia titles."
    )
    parser.add_argument(
        "--input-file",
        help=(
            "Path to a single multistream .xml.bz2 file. Requires a sibling "
            "index file produced by build_index.py."
        ),
    )
    parser.add_argument(
        "--dump-dir",
        help="Path to a dump directory containing *-multistream-index*.txt-p*.bz2.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL file with matched results.",
    )
    parser.add_argument(
        "--corrections-csv",
        required=True,
        help="Output CSV file with fuzzy match corrections (original -> matched).",
    )
    parser.add_argument(
        "--unmatched-csv",
        help=(
            "Optional output CSV (columns: ex_num, title) listing every source "
            "title that could not be matched to any Wikipedia article."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers for loading traces. Default: CPU count.",
    )
    parser.add_argument(
        "--fts-limit",
        type=int,
        default=1000,
        help="Number of FTS candidates for fuzzy matching. Default: 1000.",
    )
    args = parser.parse_args()

    if not args.input_file and not args.dump_dir:
        parser.error("Either --input-file or --dump-dir is required")
    if args.input_file and args.dump_dir:
        parser.error("Cannot use both --input-file and --dump-dir")

    pairs = resolve_index_pairs(args.input_file, args.dump_dir)

    # Create a temporary file path for DuckDB (deleted on exit)
    db_path = tempfile.mktemp(suffix=".duckdb", prefix="titles_", dir="/tmp")

    try:
        print(f"Building titles DuckDB ({db_path}) from {len(pairs)} index file(s)...")
        conn = create_titles_db(pairs, db_path)
        title_count = conn.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
        print(f"Loaded {title_count:,} titles with FTS index")

        # Load execution traces (still uses multiprocessing for HuggingFace downloads)
        num_workers = args.workers if args.workers else mp.cpu_count()
        traces = load_all_execution_traces(num_workers)

        # Process traces sequentially using DuckDB FTS
        print(f"\nStreaming results to {args.output_jsonl}")
        print(f"Streaming corrections to {args.corrections_csv}")
        if args.unmatched_csv:
            print(f"Streaming unmatched sources to {args.unmatched_csv}")
        print("Corrections found:")

        num_results, total_sources, num_corrections, num_unmatched = process_traces(
            traces,
            conn,
            args.output_jsonl,
            args.corrections_csv,
            args.fts_limit,
            args.unmatched_csv,
        )

        conn.close()
    finally:
        # Clean up temp database files
        for suffix in ("", ".wal"):
            path = db_path + suffix
            if os.path.exists(path):
                os.remove(path)

    print(f"\nDone! Processed {num_results} traces with {total_sources:,} matched sources.")
    print(f"  Corrections: {num_corrections}")
    print(f"  Unmatched sources: {num_unmatched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
