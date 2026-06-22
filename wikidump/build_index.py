#!/usr/bin/env python3
"""Build a multistream index from a single MediaWiki .xml.bz2 file.

The output format matches Wikipedia's -multistream-index.txt.bz2 files:

    <byte_offset>:<page_id>:<title>

where byte_offset is the start of the bz2 stream in the compressed file that
contains the page. Seeking to that offset and decompressing yields the ~100
pages in that stream.

Reproducing the index enables parallel-mode processing in wiki2parquet.py
without having the original -index.txt-p*.bz2 files from the dump mirror.
"""

import argparse
import bz2
import html
import mmap
import multiprocessing as mp
import os
import re
import sys
from typing import BinaryIO, Iterator

from tqdm import tqdm

PAGE_BLOCK_RE = re.compile(rb"<page>(.*?)</page>", re.DOTALL)
TITLE_RE = re.compile(rb"<title>([^<]*)</title>")
ID_RE = re.compile(rb"<id>(\d+)</id>")


def iter_streams(
    f: BinaryIO, chunk_size: int = 1 << 20
) -> Iterator[tuple[int, int, bytes]]:
    """Yield (start_offset, end_offset, decompressed_bytes) per bz2 stream.

    A multistream bz2 file is a concatenation of independent bz2 streams.
    We walk it with a fresh BZ2Decompressor per stream, carrying over the
    unused tail bytes (returned via decomp.unused_data after eof) as the
    head of the next stream.
    """
    pos = 0
    leftover = b""

    while True:
        stream_start = pos
        decomp = bz2.BZ2Decompressor()
        out_chunks: list[bytes] = []

        if leftover:
            data = leftover
            leftover = b""
        else:
            data = f.read(chunk_size)
            if not data:
                return

        while True:
            try:
                out = decomp.decompress(data)
            except OSError as e:
                raise RuntimeError(
                    f"bz2 decompression failed at offset {stream_start}: {e}"
                ) from e
            if out:
                out_chunks.append(out)
            if decomp.eof:
                consumed = len(data) - len(decomp.unused_data)
                pos += consumed
                leftover = decomp.unused_data
                break
            pos += len(data)
            data = f.read(chunk_size)
            if not data:
                raise RuntimeError(
                    f"Unexpected EOF mid-stream (stream started at offset "
                    f"{stream_start}). Is this a valid multistream bz2 file?"
                )

        yield stream_start, pos, b"".join(out_chunks)


def extract_pages(stream_bytes: bytes) -> Iterator[tuple[str, str]]:
    """Yield (page_id, title) for each <page> in the decompressed stream."""
    for match in PAGE_BLOCK_RE.finditer(stream_bytes):
        body = match.group(1)
        # The page's own <id> is always before <revision>; slicing at
        # <revision> avoids capturing revision/contributor ids by mistake.
        rev_idx = body.find(b"<revision>")
        head = body[:rev_idx] if rev_idx >= 0 else body

        title_match = TITLE_RE.search(head)
        id_match = ID_RE.search(head)
        if not (title_match and id_match):
            continue

        title = html.unescape(title_match.group(1).decode("utf-8", "replace"))
        page_id = id_match.group(1).decode("ascii")
        yield page_id, title


def build_index(xml_bz2_path: str, out_path: str) -> tuple[int, int]:
    """Build a multistream index and write it to out_path.

    Returns (num_streams, num_pages).
    """
    file_size = os.path.getsize(xml_bz2_path)
    open_out = bz2.open if out_path.endswith(".bz2") else open

    num_streams = 0
    num_pages = 0

    with (
        open(xml_bz2_path, "rb") as f_in,
        open_out(out_path, "wt", encoding="utf-8") as f_out,
        tqdm(total=file_size, unit="B", unit_scale=True, desc="Indexing") as pbar,
    ):
        for stream_start, stream_end, stream_bytes in iter_streams(f_in):
            num_streams += 1
            for page_id, title in extract_pages(stream_bytes):
                f_out.write(f"{stream_start}:{page_id}:{title}\n")
                num_pages += 1
            pbar.update(stream_end - pbar.n)

    return num_streams, num_pages


# bz2 stream-start signature: the 6-byte compressed-block magic (0x314159265359,
# the leading digits of pi) that follows "BZh<level>" at the start of every stream.
# Stream starts are byte-aligned in a multistream file, so a byte scan for
# "BZh" + level + this magic finds every stream without decompressing anything.
BZ2_BLOCK_MAGIC = b"\x31\x41\x59\x26\x53\x59"


def find_stream_offsets(mm: mmap.mmap, show_progress: bool = True) -> list[int]:
    """Return the byte offset of every bz2 stream start in a multistream file."""
    offsets: list[int] = []
    size = len(mm)
    pos = 0
    last = 0
    with tqdm(
        total=size,
        unit="B",
        unit_scale=True,
        desc="Scanning",
        disable=not show_progress,
    ) as pbar:
        while True:
            i = mm.find(b"BZh", pos)
            if i < 0:
                pbar.update(size - last)
                break
            if (
                i + 10 <= size
                and 0x31 <= mm[i + 3] <= 0x39  # level digit '1'..'9'
                and mm[i + 4 : i + 10] == BZ2_BLOCK_MAGIC
            ):
                offsets.append(i)
                pos = i + 10
            else:
                pos = i + 1
            pbar.update(pos - last)
            last = pos
    return offsets


_WORKER_MM = None


def _init_worker(path: str) -> None:
    """Pool initializer: memory-map the dump once per worker process."""
    global _WORKER_MM
    f = open(path, "rb")
    _WORKER_MM = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)


def _decode_batch(ranges: list[tuple[int, int]]) -> tuple[str, int, int, int]:
    """Decode a contiguous batch of streams.

    Returns (index_text, num_streams, num_pages, num_compressed_bytes). Each
    (start, end) range is one self-contained bz2 stream; decode is independent
    of every other range, which is what makes the build parallelizable.
    """
    assert _WORKER_MM is not None
    mm = _WORKER_MM
    rows: list[str] = []
    num_pages = 0
    for start, end in ranges:
        decomp = bz2.BZ2Decompressor()
        out = decomp.decompress(mm[start:end])
        if not decomp.eof:
            raise RuntimeError(
                f"Incomplete bz2 stream at offset {start}: the range did not end "
                f"on a stream boundary. Re-run with --no-parallel."
            )
        for page_id, title in extract_pages(out):
            rows.append(f"{start}:{page_id}:{title}")
            num_pages += 1
    text = "\n".join(rows)
    if text:
        text += "\n"
    consumed = ranges[-1][1] - ranges[0][0]
    return text, len(ranges), num_pages, consumed


def build_index_parallel(
    xml_bz2_path: str, out_path: str, workers: int, batch_size: int
) -> tuple[int, int]:
    """Build the index in parallel: scan stream boundaries, then decode in a pool.

    The output is byte-for-byte identical to build_index(): streams are decoded
    in offset order (pool.imap preserves order) and pages keep their in-stream
    order, so the merged `offset:page_id:title` lines match the sequential path.
    """
    file_size = os.path.getsize(xml_bz2_path)

    with open(xml_bz2_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            offsets = find_stream_offsets(mm)
        finally:
            mm.close()

    if not offsets:
        raise RuntimeError(
            "No bz2 streams found. Is this a valid multistream .xml.bz2 file?"
        )

    bounds = offsets + [file_size]
    ranges = [(bounds[i], bounds[i + 1]) for i in range(len(offsets))]
    batches = [ranges[i : i + batch_size] for i in range(0, len(ranges), batch_size)]
    print(f"Found {len(ranges):,} streams; decoding with {workers} workers.")

    open_out = bz2.open if out_path.endswith(".bz2") else open
    num_streams = 0
    num_pages = 0

    with (
        open_out(out_path, "wt", encoding="utf-8") as f_out,
        tqdm(total=file_size, unit="B", unit_scale=True, desc="Indexing") as pbar,
        mp.Pool(workers, initializer=_init_worker, initargs=(xml_bz2_path,)) as pool,
    ):
        for text, n_streams, n_pages, consumed in pool.imap(_decode_batch, batches):
            if text:
                f_out.write(text)
            num_streams += n_streams
            num_pages += n_pages
            pbar.update(consumed)

    return num_streams, num_pages


def default_output_path(input_path: str) -> str:
    base = os.path.basename(input_path)
    directory = os.path.dirname(input_path) or "."
    if "-multistream.xml.bz2" in base:
        out_base = base.replace(
            "-multistream.xml.bz2", "-multistream-index.txt.bz2"
        )
    elif base.endswith(".xml.bz2"):
        out_base = base[: -len(".xml.bz2")] + "-index.txt.bz2"
    else:
        out_base = base + ".index.txt.bz2"
    return os.path.join(directory, out_base)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a multistream index from a single MediaWiki .xml.bz2 file. "
            "The output format matches Wikipedia's -multistream-index.txt.bz2."
        ),
    )
    parser.add_argument(
        "input",
        help="Path to the multistream .xml.bz2 file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Output path. Defaults to the input filename with "
            "'-multistream.xml.bz2' replaced by '-multistream-index.txt.bz2'. "
            "Output is bz2-compressed if the path ends in .bz2."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Worker processes for parallel decoding. Default: CPU count.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of bz2 streams decoded per worker task. Default: 64.",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Use the single-threaded decoder (boundary discovery and decode in one pass).",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    out_path = (
        os.path.abspath(args.output)
        if args.output
        else default_output_path(input_path)
    )

    print(f"Input:  {input_path}")
    print(f"Output: {out_path}")

    if args.no_parallel or args.workers <= 1:
        num_streams, num_pages = build_index(input_path, out_path)
    else:
        num_streams, num_pages = build_index_parallel(
            input_path, out_path, workers=args.workers, batch_size=args.batch_size
        )
    print(f"Done. {num_streams:,} streams, {num_pages:,} pages.")

    if num_streams <= 1:
        print(
            "Warning: only one stream detected. This file is likely NOT "
            "multistream — the index will not enable parallel seeks.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
