"""Backend utilities for processing MediaWiki XML dumps.

This module provides shared functionality for streaming and parsing Wikipedia
dump files, including bz2 decompression, index scanning, and wikitext parsing.
"""

import bz2
import glob
import io
import os
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping

import wikitextparser as wtp
from tqdm import tqdm

# ----------------------------
# Data types
# ----------------------------


@dataclass(frozen=True, slots=True)
class ArticleText:
    title: str
    redirect_title: str | None
    wikitext: str


# ----------------------------
# Helpers
# ----------------------------


def canon_title(t: str) -> str:
    """Canonicalize a title by normalizing whitespace."""
    return t.replace("_", " ").replace("\r", " ").replace("\n", " ").strip()


def article_id_from_page_id(page_id: str) -> str:
    return f"article:{page_id}"


def section_id(article_id: str, sec_i: int) -> str:
    return f"{article_id}:s{sec_i}"


def paragraph_id(article_id: str, sec_i: int, para_i: int) -> str:
    return f"{article_id}:s{sec_i}:p{para_i}"


def chunk_id(para_id: str, chunk_i: int) -> str:
    return f"{para_id}:c{chunk_i}"


def chunk_text_by_chars(text: str, size: int, overlap: int) -> Iterator[str]:
    """Yield overlapping character chunks of ``text``.

    Each chunk is at most ``size`` characters long. Consecutive chunks share
    ``overlap`` characters at their boundary, so the window advances by
    ``size - overlap`` each step. The final chunk is whatever remains.

    - ``chunk_text_by_chars("", 500, 40)`` yields nothing.
    - If ``len(text) <= size`` a single chunk equal to ``text`` is yielded.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be in [0, size)")
    if not text:
        return
    step = size - overlap
    start = 0
    n = len(text)
    while start < n:
        yield text[start : start + size]
        if start + size >= n:
            break
        start += step


def is_node(d: dict[str, Any]) -> bool:
    return "id" in d and "type" in d and "start_id" not in d


def is_edge(d: dict[str, Any]) -> bool:
    return "start_id" in d and "end_id" in d and "rel_type" in d


def batched(it: Iterable, n: int) -> Iterator[list]:
    """Yield successive n-sized batches from an iterable."""
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


# ----------------------------
# Index scanning (dump directory mode)
# ----------------------------


def discover_index_pairs(dump_dir: str) -> list[tuple[str, str]]:
    """Discover (idx_path, xml_path) pairs in a dump directory.

    Globs for Wikipedia's split-dump naming convention
    (``*-multistream-index*.txt-p*.bz2``) and derives the matching XML path by
    string substitution.
    """
    idx_glob = os.path.join(dump_dir, "*-multistream-index*.txt-p*.bz2")
    pairs: list[tuple[str, str]] = []
    for idx_path in sorted(glob.glob(idx_glob)):
        xml_path = idx_path.replace("-multistream-index", "-multistream").replace(
            ".txt-", ".xml-"
        )
        pairs.append((idx_path, xml_path))
    return pairs


def find_sibling_index(xml_bz2_path: str) -> str | None:
    """Locate an index file sitting next to a single multistream .xml.bz2.

    Tries naming conventions produced by ``build_index.py``:

    - ``foo-multistream.xml.bz2`` -> ``foo-multistream-index.txt.bz2``
    - ``foo.xml.bz2`` -> ``foo-index.txt.bz2``

    Returns the path to the first candidate that exists, or None.
    """
    directory = os.path.dirname(os.path.abspath(xml_bz2_path))
    base = os.path.basename(xml_bz2_path)

    candidates: list[str] = []
    if "-multistream.xml.bz2" in base:
        candidates.append(
            base.replace("-multistream.xml.bz2", "-multistream-index.txt.bz2")
        )
    if base.endswith(".xml.bz2"):
        candidates.append(base[: -len(".xml.bz2")] + "-index.txt.bz2")

    for cand in candidates:
        path = os.path.join(directory, cand)
        if os.path.exists(path):
            return path
    return None


def _iter_index_lines_with_progress(
    pair_list: list[tuple[str, str]], desc: str
) -> Iterator[tuple[str, str]]:
    """Yield (xml_path, line) from bz2-compressed index files with byte-wise tqdm.

    Works as well for one index file as for many: the progress bar totals the
    on-disk size of every index in ``pair_list`` and advances by the underlying
    compressed file position, so single-file dumps get a meaningful byte-wise bar.
    """
    total_bytes = sum(os.path.getsize(idx_path) for idx_path, _ in pair_list)
    update_every = 50_000  # lines between raw.tell() polls

    with tqdm(
        total=total_bytes, desc=desc, unit="B", unit_scale=True
    ) as pbar:
        for idx_path, xml_path in pair_list:
            file_size = os.path.getsize(idx_path)
            with open(idx_path, "rb") as raw:
                last = 0
                with bz2.open(
                    raw, "rt", encoding="utf-8", errors="replace"
                ) as f:
                    line_count = 0
                    for line in f:
                        line_count += 1
                        if line_count % update_every == 0:
                            cur = raw.tell()
                            pbar.update(cur - last)
                            last = cur
                        yield xml_path, line
                pbar.update(file_size - last)


def scan_index(
    pairs: Iterable[tuple[str, str]],
    target_titles: set[str] | None = None,
    target_page_ids: set[str] | None = None,
    desc: str = "Scanning index",
) -> tuple[dict[str, str], list[tuple[str, int, int | None, str, str]]]:
    """Single-pass scan of bz2 index files.

    Returns ``(title_to_id, matched_entries)``:

    - ``title_to_id`` is the full title -> page_id mapping for the dump,
      needed by ``parse_wikitext`` to resolve LINKS_TO and REDIRECTS_TO edges.
    - ``matched_entries`` is the list of
      ``(xml_path, start_offset, end_offset, page_id, title)`` tuples for
      pages matching the optional ``target_titles`` / ``target_page_ids``
      filters. With no filters, every entry in the index is returned.

    Both outputs are produced from a single byte-tracked walk over the index
    files; the underlying tqdm advances by compressed bytes consumed and
    works equally well for one or many index files.
    """
    pair_list = list(pairs)
    title_to_id: dict[str, str] = {}
    offsets_by_xml: dict[str, set[int]] = {}
    entries_by_xml: dict[str, list[tuple[int, str, str]]] = {}

    for xml_path, line in _iter_index_lines_with_progress(pair_list, desc):
        parts = line.rstrip("\n").split(":", 2)
        if len(parts) < 3:
            continue
        off_str = parts[0]
        if not off_str.isdigit():
            continue
        off = int(off_str)
        page_id = parts[1]
        title = canon_title(parts[2])

        if title and page_id and title not in title_to_id:
            title_to_id[title] = page_id

        offsets_by_xml.setdefault(xml_path, set()).add(off)

        title_match = target_titles is None or title in target_titles
        page_id_match = target_page_ids is None or page_id in target_page_ids
        if title_match and page_id_match:
            entries_by_xml.setdefault(xml_path, []).append((off, page_id, title))

    matched_entries: list[tuple[str, int, int | None, str, str]] = []
    for xml_path, offsets_set in sorted(offsets_by_xml.items()):
        offsets = sorted(offsets_set)
        if not offsets:
            continue
        next_map: dict[int, int | None] = {
            off: (offsets[i + 1] if i + 1 < len(offsets) else None)
            for i, off in enumerate(offsets)
        }
        for off, page_id, title in entries_by_xml.get(xml_path, []):
            matched_entries.append(
                (xml_path, off, next_map.get(off), page_id, title)
            )

    return title_to_id, matched_entries


def iter_titles_from_index(
    pairs: Iterable[tuple[str, str]],
) -> Iterator[tuple[str, str]]:
    """Yield (page_id, title) for each page by scanning index files."""
    pair_list = list(pairs)
    for _xml_path, line in _iter_index_lines_with_progress(
        pair_list, "Scanning index files"
    ):
        parts = line.rstrip("\n").split(":", 2)
        if len(parts) < 3:
            continue
        page_id = parts[1]
        title = canon_title(parts[2])
        if title and page_id:
            yield page_id, title


# ----------------------------
# Header cache (per worker process)
# ----------------------------

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


# ----------------------------
# Streaming decompression
# ----------------------------


def iter_wrapped_decompressed(
    xml_bz2_path: str, start: int, end: int | None
) -> Iterator[bytes]:
    """Stream decompressed bytes from a bz2 file, wrapping with XML header/footer."""
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


class IterBytesIO(io.RawIOBase):
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


# ----------------------------
# Wikitext parsing
# ----------------------------


def parse_wikitext(
    article: ArticleText,
    article_id: str,
    title_to_id: Mapping[str, str],
    chunk_size: int = 500,
    chunk_overlap: int = 40,
) -> Iterator[dict[str, Any]]:
    """Parse wikitext and yield node/edge records for the article's graph structure.

    Paragraph nodes carry the full plain-text of each paragraph. Each paragraph
    is additionally split into ``Chunk`` nodes of at most ``chunk_size``
    characters with ``chunk_overlap`` characters of overlap, connected to the
    parent paragraph via ``HAS_CHUNK`` and to each other via ``NEXT_CHUNK`` /
    ``PREVIOUS_CHUNK`` edges. Chunks are the intended embedding target.
    """
    art_title = canon_title(article.title)
    art_id = article_id

    yield {"id": art_id, "type": "Article", "title": art_title}

    if article.redirect_title:
        end_title = canon_title(article.redirect_title)
        if end_title:
            end_id = title_to_id.get(end_title)
            if end_id is not None:
                yield {
                    "start_id": art_id,
                    "end_id": article_id_from_page_id(end_id),
                    "rel_type": "REDIRECTS_TO",
                }

    parsed = wtp.parse(article.wikitext)
    sections = list(parsed.sections)

    for sec_i, section in enumerate(sections):
        sec_id = section_id(art_id, sec_i)
        yield {"id": sec_id, "type": "Section"}

        yield {
            "start_id": art_id,
            "end_id": sec_id,
            "rel_type": "HAS_SECTION",
        }

        if sec_i > 0:
            prev_sec_id = section_id(art_id, sec_i - 1)
            yield {
                "start_id": prev_sec_id,
                "end_id": sec_id,
                "rel_type": "NEXT_SECTION",
            }
            yield {
                "start_id": sec_id,
                "end_id": prev_sec_id,
                "rel_type": "PREVIOUS_SECTION",
            }

        all_paragraph_ids = []
        para_counter = 0

        for para_str in section.string.split("\n\n"):
            para_str = para_str.strip()
            if not para_str:
                continue

            para = wtp.parse(para_str)
            para_text = para.plain_text()

            if not para_text.strip():
                continue

            pid = paragraph_id(art_id, sec_i, para_counter)
            para_counter += 1
            all_paragraph_ids.append(pid)

            yield {"id": pid, "type": "Paragraph", "text": para_text}

            yield {
                "start_id": sec_id,
                "end_id": pid,
                "rel_type": "HAS_PARAGRAPH",
            }

            for link in para.wikilinks:
                link_target = getattr(link, "title", None) or getattr(
                    link, "target", ""
                )
                end_title = canon_title(str(link_target))
                if not end_title:
                    continue
                end_id = title_to_id.get(end_title)
                if end_id is None:
                    continue
                yield {
                    "start_id": pid,
                    "end_id": article_id_from_page_id(end_id),
                    "rel_type": "LINKS_TO",
                }

            prev_cid: str | None = None
            for chunk_i, chunk in enumerate(
                chunk_text_by_chars(para_text, chunk_size, chunk_overlap)
            ):
                cid = chunk_id(pid, chunk_i)
                yield {"id": cid, "type": "Chunk", "text": chunk}
                yield {
                    "start_id": pid,
                    "end_id": cid,
                    "rel_type": "HAS_CHUNK",
                }
                if prev_cid is not None:
                    yield {
                        "start_id": prev_cid,
                        "end_id": cid,
                        "rel_type": "NEXT_CHUNK",
                    }
                    yield {
                        "start_id": cid,
                        "end_id": prev_cid,
                        "rel_type": "PREVIOUS_CHUNK",
                    }
                prev_cid = cid

        for i in range(len(all_paragraph_ids) - 1):
            yield {
                "start_id": all_paragraph_ids[i],
                "end_id": all_paragraph_ids[i + 1],
                "rel_type": "NEXT_PARAGRAPH",
            }
            yield {
                "start_id": all_paragraph_ids[i + 1],
                "end_id": all_paragraph_ids[i],
                "rel_type": "PREVIOUS_PARAGRAPH",
            }
