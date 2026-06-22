"""List all namespace IDs found in a MediaWiki multistream dump.

This module scans a Wikipedia dump directory and reports all unique namespace
IDs encountered. Useful for determining which namespaces to include or exclude
when running wiki2parquet.py.
"""

import bz2
import glob
import io
import os
import sys
from typing import Iterator

import mwxml
from tqdm import tqdm


def iter_stream_chunks(dump_dir: str) -> Iterator[tuple[str, int, int | None]]:
    idx_glob = os.path.join(dump_dir, "*-multistream-index*.txt-p*.bz2")
    offsets_by_xml: dict[str, set[int]] = {}

    for idx_path in sorted(glob.glob(idx_glob)):
        xml_path = idx_path.replace("-multistream-index", "-multistream").replace(
            ".txt-", ".xml-"
        )
        offsets_set = offsets_by_xml.setdefault(xml_path, set())
        with bz2.open(idx_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(":", 2)
                if not parts:
                    continue
                off_str = parts[0]
                if off_str.isdigit():
                    offsets_set.add(int(off_str))

    for xml_path, offsets_set in sorted(offsets_by_xml.items()):
        offsets = sorted(offsets_set)
        for i, off in enumerate(offsets):
            next_off = offsets[i + 1] if i + 1 < len(offsets) else None
            yield (xml_path, off, next_off)


_HEADER_CACHE: dict[str, bytes] = {}


def _get_mediawiki_header(xml_bz2_path: str, read_limit: int = 2_000_000) -> bytes:
    hdr = _HEADER_CACHE.get(xml_bz2_path)
    if hdr is not None:
        return hdr

    with bz2.open(xml_bz2_path, "rb") as f:
        prefix = f.read(read_limit)

    i = prefix.find(b"<page>")
    if i == -1:
        raise RuntimeError(
            "Couldn't find <page> in initial prefix; increase read_limit"
        )

    hdr = prefix[:i]
    _HEADER_CACHE[xml_bz2_path] = hdr
    return hdr


def iter_wrapped_decompressed(
    xml_bz2_path: str, start: int, end: int | None
) -> Iterator[bytes]:
    if start != 0:
        yield _get_mediawiki_header(xml_bz2_path)

    decomp = bz2.BZ2Decompressor()
    tail = b""
    chunk_size = 1 << 20  # 1 MiB compressed input at a time

    with open(xml_bz2_path, "rb") as f:
        f.seek(start)
        remaining = None if end is None else max(0, end - start)
        while True:
            if remaining is None:
                chunk = f.read(chunk_size)
            else:
                if remaining <= 0:
                    break
                chunk = f.read(min(chunk_size, remaining))
                remaining -= len(chunk)
            if not chunk:
                break

            out = decomp.decompress(chunk)
            if out:
                tail = (tail + out)[-256:]
                yield out

            if decomp.eof:
                break

    if b"</mediawiki>" not in tail:
        yield b"\n</mediawiki>\n"


class _IterBytesIO(io.RawIOBase):
    """Turn an iterator-of-bytes into a readable RawIOBase for TextIOWrapper."""

    def __init__(self, it: Iterator[bytes]):
        self._it = it
        self._buf = b""
        self._done = False

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._done:
            return 0
        mv = memoryview(b)
        n = 0
        while n < len(mv):
            if not self._buf:
                try:
                    self._buf = next(self._it)
                except StopIteration:
                    self._done = True
                    break
            take = min(len(mv) - n, len(self._buf))
            mv[n : n + take] = self._buf[:take]
            self._buf = self._buf[take:]
            n += take
        return n


def _namespace_sort_key(ns: str) -> tuple[int, int | str]:
    try:
        return (0, int(ns))
    except ValueError:
        return (1, ns)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python list_namespaces.py /path/to/dump")
        return 2

    dump_dir = sys.argv[1]
    if not os.path.isdir(dump_dir):
        print(f"Not a directory: {dump_dir}")
        return 2

    namespaces: set[str] = set()
    total_chunks = sum(1 for _ in iter_stream_chunks(dump_dir))
    chunk_iter = iter_stream_chunks(dump_dir)
    chunk_count = 0
    for xml_path, start, end in tqdm(chunk_iter, total=total_chunks, unit="chunk"):
        chunk_count += 1
        byte_iter = iter_wrapped_decompressed(xml_path, start, end)
        raw = _IterBytesIO(byte_iter)
        text = io.TextIOWrapper(
            io.BufferedReader(raw), encoding="utf-8", errors="replace"
        )
        try:
            dump = mwxml.iteration.Dump.from_file(text)
            changed = False
            for page in dump.pages:
                if page.namespace is None:
                    continue
                ns = str(page.namespace)
                if ns not in namespaces:
                    namespaces.add(ns)
                    changed = True
            if changed:
                tqdm.write(
                    "Namespaces: "
                    + ", ".join(sorted(namespaces, key=_namespace_sort_key))
                )
        finally:
            text.close()

    print(f"Chunks scanned: {chunk_count}")
    print(f"Namespaces found: {len(namespaces)}")
    for ns in sorted(namespaces, key=_namespace_sort_key):
        print(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
