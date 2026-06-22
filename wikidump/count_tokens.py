"""Count tokens in Parquet text columns using tiktoken.

This module scans Parquet files and counts the total number of tokens in a
specified text column. Useful for estimating embedding costs or understanding
dataset size.
"""

import argparse
import os
from typing import Iterable

import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm


def iter_text_batches(
    path: str, text_col: str, batch_size: int
) -> Iterable[tuple[list[str], int]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[text_col]):
        texts = batch.column(text_col).to_pylist()
        filtered = [t for t in texts if t is not None and t != ""]
        yield filtered, len(texts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count total tokens for embedding text in nodes parquet files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more nodes-*.parquet files to scan.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name for tiktoken encoding.",
    )
    parser.add_argument(
        "--text-col",
        default="text:string",
        help="Name of the text column to count tokens from.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for tokenization.",
    )
    args = parser.parse_args()

    try:
        encoding = tiktoken.encoding_for_model(args.model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    total_rows = 0
    for path in args.files:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        total_rows += pq.ParquetFile(path).metadata.num_rows

    progress = tqdm(total=total_rows, unit="row")
    total_tokens = 0
    total_texts = 0

    for path in args.files:
        for texts, raw_count in iter_text_batches(path, args.text_col, args.batch_size):
            progress.update(raw_count)
            if not texts:
                continue
            lengths = [len(encoding.encode(t)) for t in texts]
            total_tokens += sum(lengths)
            total_texts += len(lengths)
            if progress.n > 0 and progress.total:
                est_total = int(total_tokens * (progress.total / progress.n))
            else:
                est_total = total_tokens
            progress.set_postfix(
                tokens=format_count(total_tokens),
                est_total=format_count(est_total),
            )

    progress.close()
    print(f"Total texts: {total_texts}")
    print(f"Total tokens: {total_tokens}")
    return 0


def format_count(n: int) -> str:
    """Format a number with K/M/B suffix for human readability."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


if __name__ == "__main__":
    raise SystemExit(main())
