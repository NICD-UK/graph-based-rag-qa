"""Convert MediaWiki multistream XML dumps to Parquet format for Neo4j import.

Produces sharded Parquet files containing nodes (Articles, Sections,
Paragraphs) and edges (relationships between them), compatible with Neo4j's
bulk import tool.

Requires index files:
- ``--dump-dir`` with the original split-dump ``*-multistream-index*.txt-p*.bz2``
- ``--input-file`` pointing at a single ``.xml.bz2`` that has a sibling index
  produced by ``build_index.py``
"""

import argparse
import gc
import io
import json
import multiprocessing as mp
import os
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from wiki_backend import (
    ArticleText,
    IterBytesIO,
    article_id_from_page_id,
    batched,
    canon_title,
    discover_index_pairs,
    find_sibling_index,
    is_edge,
    is_node,
    iter_wrapped_decompressed,
    parse_wikitext,
    scan_index,
)

# Shared globals for worker processes
TITLE_TO_ID: Mapping[str, str] = {}
NAMESPACE_IDS: set[str] | None = None
TARGET_PAGE_IDS: set[str] | None = None
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 40


def load_page_ids_from_matches(
    jsonl_path: str, question_nums: set[int] | None = None
) -> set[str]:
    """Load page_ids from a matches JSONL file, optionally filtered by question numbers.

    Args:
        jsonl_path: Path to the JSONL file with matching results.
        question_nums: Optional set of question numbers to include. If None, includes all.

    Returns:
        Set of page_id strings.
    """
    page_ids: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            ex_num = record.get("ex_num")
            if question_nums is not None and ex_num not in question_nums:
                continue
            for source in record.get("sources", []):
                page_id = source.get("page_id")
                if page_id:
                    page_ids.add(str(page_id))
    return page_ids


# ----------------------------
# Worker: batch -> tempfiles (parallel mode)
# ----------------------------


def process_batch_to_tempfiles(
    batch: list[tuple[str, int, int | None, str, str]],
    node_cols: Sequence[str],
    edge_cols: Sequence[str],
    node_field_map: Mapping[str, str],
    edge_field_map: Mapping[str, str],
) -> tuple[str, str]:
    """Process a batch of pages and write results to temporary Parquet files."""
    import mwxml

    title_to_id = TITLE_TO_ID
    namespace_ids = NAMESPACE_IDS
    chunk_size = CHUNK_SIZE
    chunk_overlap = CHUNK_OVERLAP
    node_schema = pa.schema([pa.field(c, pa.string()) for c in node_cols])
    edge_schema = pa.schema([pa.field(c, pa.string()) for c in edge_cols])
    nodes_tmp = tempfile.NamedTemporaryFile(
        delete=False, prefix="nodes_", suffix=".parquet"
    )
    edges_tmp = tempfile.NamedTemporaryFile(
        delete=False, prefix="edges_", suffix=".parquet"
    )
    nodes_path = nodes_tmp.name
    edges_path = edges_tmp.name
    nodes_tmp.close()
    edges_tmp.close()

    node_buf: dict[str, list[str]] = {c: [] for c in node_cols}
    edge_buf: dict[str, list[str]] = {c: [] for c in edge_cols}

    try:
        for path, start, end, page_id, title in batch:
            byte_iter = iter_wrapped_decompressed(path, start, end)
            raw = IterBytesIO(byte_iter)
            text = io.TextIOWrapper(
                io.BufferedReader(raw), encoding="utf-8", errors="replace"
            )
            try:
                dump = mwxml.iteration.Dump.from_file(text)

                found = False
                for page in dump.pages:
                    if str(page.id) != page_id:
                        continue
                    found = True
                    if namespace_ids is not None:
                        page_ns = str(page.namespace)
                        if page_ns not in namespace_ids:
                            break

                    latest = None
                    for rev in page:
                        if rev.text is None:
                            continue
                        if latest is None or rev.timestamp > latest.timestamp:
                            latest = rev
                    if latest is None:
                        break

                    redirect_title = None
                    if getattr(page, "redirect", None) is not None:
                        redirect_title = str(page.redirect)

                    art = ArticleText(title, redirect_title, str(latest.text))
                    art_id = article_id_from_page_id(page_id)

                    for rec in parse_wikitext(
                        art, art_id, title_to_id, chunk_size, chunk_overlap
                    ):
                        if is_node(rec):
                            for out_col in node_cols:
                                rec_key = node_field_map[out_col]
                                val = rec.get(rec_key, "")
                                node_buf[out_col].append(
                                    "" if val is None else str(val)
                                )
                        elif is_edge(rec):
                            for out_col in edge_cols:
                                rec_key = edge_field_map[out_col]
                                val = rec.get(rec_key, "")
                                edge_buf[out_col].append(
                                    "" if val is None else str(val)
                                )
                    break

                if not found:
                    raise RuntimeError(
                        f"Missing page id {page_id} at offset {start} in {path}"
                    )
            finally:
                text.close()
            del dump, text, raw, byte_iter
            gc.collect()

        pq.write_table(
            pa.table(node_buf, schema=node_schema),
            nodes_path,
            compression="NONE",
            use_dictionary=False,
            version="1.0",
        )
        pq.write_table(
            pa.table(edge_buf, schema=edge_schema),
            edges_path,
            compression="NONE",
            use_dictionary=False,
            version="1.0",
        )
        return nodes_path, edges_path

    except Exception:
        for path in (nodes_path, edges_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise


def _worker_loop(
    task_queue: "mp.Queue[tuple[list[tuple[str, int, int | None, str, str]], str] | None]",
    result_queue: "mp.Queue[tuple[str, str, str, str]]",
    shared_title_to_id: Mapping[str, str],
    namespace_ids: set[str] | None,
    node_cols: Sequence[str],
    edge_cols: Sequence[str],
    node_field_map: Mapping[str, str],
    edge_field_map: Mapping[str, str],
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    global TITLE_TO_ID
    global NAMESPACE_IDS
    global CHUNK_SIZE
    global CHUNK_OVERLAP
    TITLE_TO_ID = shared_title_to_id
    NAMESPACE_IDS = namespace_ids
    CHUNK_SIZE = chunk_size
    CHUNK_OVERLAP = chunk_overlap

    while True:
        item = task_queue.get()
        if item is None:
            break
        batch, label = item
        try:
            npath, epath = process_batch_to_tempfiles(
                batch, node_cols, edge_cols, node_field_map, edge_field_map
            )
            result_queue.put(("ok", label, npath, epath))
        except Exception:
            result_queue.put(("error", label, "", traceback.format_exc()))


# ----------------------------
# Parallel mode pipeline
# ----------------------------


def run_parallel_mode(
    pairs: list[tuple[str, str]],
    out_dir: str,
    target_titles: set[str] | None,
    target_page_ids: set[str] | None,
    namespace_ids: set[str] | None,
    node_cols: Sequence[str],
    edge_cols: Sequence[str],
    node_field_map: Mapping[str, str],
    edge_field_map: Mapping[str, str],
    batch_size: int,
    shard_size: int,
    n_workers: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Parallel pipeline: workers seek into indexed bz2 streams and emit temp
    Parquet files; the main process merges them into sharded Parquet output.
    """
    node_schema = pa.schema([pa.field(c, pa.string()) for c in node_cols])
    edge_schema = pa.schema([pa.field(c, pa.string()) for c in edge_cols])

    buffer_n = max(2, n_workers * 2)

    # True whenever we are producing a subset of the dump: any cross-article
    # edge (LINKS_TO / REDIRECTS_TO) may point to an Article that we aren't
    # writing a node for, so we need to emit stub Article nodes at the end.
    is_subset = (
        target_titles is not None
        or target_page_ids is not None
        or namespace_ids is not None
    )

    nodes_shard_idx = 0
    edges_shard_idx = 0
    nodes_row_count = 0
    edges_row_count = 0
    nodes_writer: pq.ParquetWriter | None = None
    edges_writer: pq.ParquetWriter | None = None

    article_seen: dict[str, tuple[str, str]] = {}
    ref_db: sqlite3.Connection | None = None
    if is_subset:
        ref_db_path = os.path.join(out_dir, "referenced_ids.sqlite")
        if os.path.exists(ref_db_path):
            os.unlink(ref_db_path)
        ref_db = sqlite3.connect(ref_db_path)
        ref_db.execute("PRAGMA journal_mode=WAL")
        ref_db.execute("PRAGMA synchronous=NORMAL")
        ref_db.execute("CREATE TABLE IF NOT EXISTS referenced (id TEXT PRIMARY KEY)")
        ref_db.execute("CREATE TABLE IF NOT EXISTS seen (id TEXT PRIMARY KEY)")
        ref_db.commit()

    manager = mp.Manager()
    workers: list[mp.Process] = []
    task_queue: "mp.Queue[tuple[list[tuple[str, int, int | None, str, str]], str] | None]" = mp.Queue(
        maxsize=buffer_n
    )
    result_queue: "mp.Queue[tuple[str, str, str, str]]" = mp.Queue()

    def get_nodes_writer() -> pq.ParquetWriter:
        nonlocal nodes_writer, nodes_shard_idx, nodes_row_count
        if nodes_writer is None or nodes_row_count >= shard_size:
            if nodes_writer is not None:
                nodes_writer.close()
            nodes_path = os.path.join(out_dir, f"nodes-{nodes_shard_idx:05d}.parquet")
            nodes_writer = pq.ParquetWriter(
                nodes_path, node_schema, compression="NONE",
                use_dictionary=False, version="1.0",
            )
            nodes_shard_idx += 1
            nodes_row_count = 0
        return nodes_writer

    def get_edges_writer() -> pq.ParquetWriter:
        nonlocal edges_writer, edges_shard_idx, edges_row_count
        if edges_writer is None or edges_row_count >= shard_size:
            if edges_writer is not None:
                edges_writer.close()
            edges_path = os.path.join(out_dir, f"edges-{edges_shard_idx:05d}.parquet")
            edges_writer = pq.ParquetWriter(
                edges_path, edge_schema, compression="NONE",
                use_dictionary=False, version="1.0",
            )
            edges_shard_idx += 1
            edges_row_count = 0
        return edges_writer

    def write_nodes_table(table: pa.Table) -> None:
        nonlocal nodes_row_count, nodes_writer
        remaining = table
        while remaining.num_rows > 0:
            writer = get_nodes_writer()
            capacity = shard_size - nodes_row_count
            if capacity <= 0:
                assert nodes_writer is not None
                nodes_writer.close()
                nodes_writer = None
                continue
            take = min(capacity, remaining.num_rows)
            writer.write_table(remaining.slice(0, take))
            nodes_row_count += take
            remaining = remaining.slice(take)

    def write_edges_table(table: pa.Table) -> None:
        nonlocal edges_row_count, edges_writer
        remaining = table
        while remaining.num_rows > 0:
            writer = get_edges_writer()
            capacity = shard_size - edges_row_count
            if capacity <= 0:
                assert edges_writer is not None
                edges_writer.close()
                edges_writer = None
                continue
            take = min(capacity, remaining.num_rows)
            writer.write_table(remaining.slice(0, take))
            edges_row_count += take
            remaining = remaining.slice(take)

    def write_stub_nodes(stub_ids: list[str]) -> None:
        if not stub_ids:
            return
        table = pa.table(
            {
                "nodeID:ID(wiki)": stub_ids,
                ":LABEL": ["Article"] * len(stub_ids),
                "title:string": [""] * len(stub_ids),
                "text:string": [""] * len(stub_ids),
            },
            schema=node_schema,
        )
        write_nodes_table(table)

    def add_seen_ids(ids: list[str]) -> None:
        if ref_db is None or not ids:
            return
        ref_db.executemany(
            "INSERT OR IGNORE INTO seen (id) VALUES (?)", [(i,) for i in ids]
        )
        ref_db.commit()

    def add_referenced_ids(ids: list[str]) -> None:
        if ref_db is None or not ids:
            return
        ref_db.executemany(
            "INSERT OR IGNORE INTO referenced (id) VALUES (?)", [(i,) for i in ids]
        )
        ref_db.commit()

    try:
        # Single-pass scan: builds the title->id map AND collects matched
        # entries from the same byte-tracked walk over the index files.
        title_to_id, matched_entries = scan_index(
            pairs, target_titles, target_page_ids, desc="Scanning index"
        )

        global TITLE_TO_ID
        TITLE_TO_ID = manager.dict(title_to_id)

        total_chunks = len(matched_entries)
        is_filtered = target_page_ids is not None or target_titles is not None

        if target_page_ids is not None:
            found_ids = {entry[3] for entry in matched_entries}
            missing = target_page_ids - found_ids
            if missing:
                print(
                    f"Error: {len(missing):,} of {len(target_page_ids):,} "
                    f"target page_ids are not present in the index for the "
                    f"given dump.",
                    file=sys.stderr,
                )
                print(
                    "This usually means --matches-jsonl was built against "
                    "a different dump snapshot. Re-run load_monaco.py "
                    "against the same dump you are passing here.",
                    file=sys.stderr,
                )
                for pid in sorted(missing)[:20]:
                    print(f"  - {pid}", file=sys.stderr)
                if len(missing) > 20:
                    print(
                        f"  ... and {len(missing) - 20:,} more",
                        file=sys.stderr,
                    )
                raise SystemExit(1)
            print(
                f"Verified all {len(target_page_ids):,} target page_ids "
                f"are in the index ({total_chunks:,} entries)."
            )

        if target_titles is not None:
            found_titles = {entry[4] for entry in matched_entries}
            missing_titles = target_titles - found_titles
            if missing_titles:
                print(
                    f"Error: {len(missing_titles):,} of "
                    f"{len(target_titles):,} target titles are not present "
                    f"in the index for the given dump.",
                    file=sys.stderr,
                )
                for t in sorted(missing_titles)[:20]:
                    print(f"  - {t}", file=sys.stderr)
                if len(missing_titles) > 20:
                    print(
                        f"  ... and {len(missing_titles) - 20:,} more",
                        file=sys.stderr,
                    )
                raise SystemExit(1)
            print(
                f"Verified all {len(target_titles):,} target titles "
                f"are in the index ({total_chunks:,} entries)."
            )

        if not is_filtered:
            print(f"Total pages: {total_chunks:,}")

        total_batches = (total_chunks + batch_size - 1) // batch_size
        print(f"Total batches: {total_batches:,}")

        workers = [
            mp.Process(
                target=_worker_loop,
                args=(
                    task_queue,
                    result_queue,
                    TITLE_TO_ID,
                    namespace_ids,
                    node_cols,
                    edge_cols,
                    node_field_map,
                    edge_field_map,
                    chunk_size,
                    chunk_overlap,
                ),
                daemon=True,
            )
            for _ in range(n_workers)
        ]
        for w in workers:
            w.start()

        batch_iter = batched(matched_entries, batch_size)

        def _producer() -> None:
            for batch in batch_iter:
                label = ",".join(
                    f"{os.path.basename(path)}@{start}:{page_id}"
                    for path, start, _, page_id, _ in batch
                )
                task_queue.put((batch, label))
            for _ in range(n_workers):
                task_queue.put(None)

        producer = threading.Thread(target=_producer, daemon=True)
        producer.start()

        completed = 0
        start_time = time.time()
        while completed < total_batches:
            status, label, npath, payload = result_queue.get()
            if status != "ok":
                for w in workers:
                    w.terminate()
                raise RuntimeError(f"Worker failed on batch {label}:\n{payload}")

            epath = payload

            nodes_pf = pq.ParquetFile(npath)
            if nodes_pf.metadata.num_rows:
                for batch in nodes_pf.iter_batches():
                    table = pa.Table.from_batches([batch], schema=node_schema)
                    ids = table.column("nodeID:ID(wiki)").to_pylist()
                    types = table.column(":LABEL").to_pylist()
                    seen_ids: list[str] = []
                    for node_id, node_type in zip(ids, types):
                        if node_type == "Article" and node_id:
                            prev = article_seen.get(node_id)
                            if prev is not None:
                                prev_path, prev_label = prev
                                raise RuntimeError(
                                    f"Duplicate article id {node_id!r} in {npath} "
                                    f"(batch {label}); first seen in {prev_path} "
                                    f"(batch {prev_label})"
                                )
                            article_seen[node_id] = (npath, label)
                            if is_subset:
                                seen_ids.append(node_id)
                    if seen_ids:
                        add_seen_ids(seen_ids)
                    write_nodes_table(table)

            edges_pf = pq.ParquetFile(epath)
            if edges_pf.metadata.num_rows:
                for batch in edges_pf.iter_batches():
                    table = pa.Table.from_batches([batch], schema=edge_schema)
                    if is_subset:
                        # Only LINKS_TO and REDIRECTS_TO cross article
                        # boundaries; every other edge type stays within one
                        # article's hierarchy and its END_ID is a
                        # Section/Paragraph/Chunk that we've already written.
                        # Using startswith("article:") would wrongly match
                        # those nested IDs because all node IDs share the
                        # 'article:<page_id>' prefix, causing duplicate
                        # stub-Article rows at real nested node IDs.
                        end_ids = table.column(":END_ID(wiki)").to_pylist()
                        rel_types = table.column(":TYPE").to_pylist()
                        ref_ids = [
                            end_id
                            for end_id, rel_type in zip(end_ids, rel_types)
                            if end_id
                            and rel_type in ("LINKS_TO", "REDIRECTS_TO")
                        ]
                        add_referenced_ids(ref_ids)
                    write_edges_table(table)

            os.unlink(npath)
            os.unlink(epath)
            completed += 1

            if completed % 100 == 0:
                progress_pct = (completed / total_batches) * 100 if total_batches else 0
                elapsed = time.time() - start_time
                batches_per_sec = completed / elapsed if elapsed > 0 else 0
                remaining_batches = total_batches - completed
                eta_seconds = (
                    remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                )
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                print(
                    f"Progress: {completed}/{total_batches} batches ({progress_pct:.1f}%) | "
                    f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                    f"Rate: {batches_per_sec:.1f} batches/s | "
                    f"Nodes: {nodes_row_count:,} (shard {nodes_shard_idx}), "
                    f"Edges: {edges_row_count:,} (shard {edges_shard_idx})"
                )

    finally:
        for w in workers:
            if w.is_alive():
                w.join(timeout=1)
        if is_subset and ref_db is not None:
            cursor = ref_db.execute(
                "SELECT r.id FROM referenced r "
                "LEFT JOIN seen s ON r.id = s.id "
                "WHERE s.id IS NULL"
            )
            while True:
                rows = cursor.fetchmany(100_000)
                if not rows:
                    break
                write_stub_nodes([r[0] for r in rows])
        if nodes_writer is not None:
            nodes_writer.close()
        if edges_writer is not None:
            edges_writer.close()
        if ref_db is not None:
            ref_db.close()
        manager.shutdown()


# ----------------------------
# Main
# ----------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a MediaWiki multistream dump to Parquet shards."
    )
    parser.add_argument(
        "--dump-dir",
        help="Path to directory with *-multistream*.xml-*.bz2 and index files.",
    )
    parser.add_argument(
        "--input-file",
        help=(
            "Path to a single .xml.bz2 file. Requires a sibling index file "
            "produced by build_index.py."
        ),
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for Parquet shard files.",
    )
    parser.add_argument(
        "--title",
        action="append",
        default=[],
        help="Restrict to a specific page title (repeatable).",
    )
    parser.add_argument(
        "--namespace-id",
        action="append",
        default=[],
        help="Restrict to a namespace id (repeatable).",
    )
    parser.add_argument(
        "--matches-jsonl",
        help="Path to JSONL file with matched sources (from load_monaco.py).",
    )
    parser.add_argument(
        "--question",
        type=int,
        action="append",
        default=[],
        help="Question number to include (repeatable). Requires --matches-jsonl.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of index entries per worker batch.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1_000_000,
        help="Maximum rows per output Parquet shard.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help=(
            "Chunk size in characters for Chunk nodes attached to each "
            "Paragraph. Default: 500."
        ),
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help=(
            "Overlap in characters between successive Chunk nodes. Must be "
            "in [0, chunk-size). Default: 40."
        ),
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error("--chunk-overlap must be in [0, --chunk-size)")

    if not args.dump_dir and not args.input_file:
        parser.error("Either --dump-dir or --input-file is required")
    if args.dump_dir and args.input_file:
        parser.error("Cannot use both --dump-dir and --input-file")

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    target_titles = {canon_title(t) for t in args.title} or None
    namespace_ids = {str(ns) for ns in args.namespace_id} or None

    # Load page_ids from matches JSONL if specified
    target_page_ids: set[str] | None = None
    if args.matches_jsonl:
        matches_path = os.path.abspath(args.matches_jsonl)
        if not os.path.exists(matches_path):
            print(f"Error: Matches file not found: {matches_path}")
            return 1
        question_nums = set(args.question) if args.question else None
        target_page_ids = load_page_ids_from_matches(matches_path, question_nums)
        if question_nums:
            print(f"Loaded {len(target_page_ids):,} page_ids for questions: {sorted(question_nums)}")
        else:
            print(f"Loaded {len(target_page_ids):,} page_ids from all questions")
    elif args.question:
        parser.error("--question requires --matches-jsonl")

    node_cols = ["nodeID:ID(wiki)", ":LABEL", "title:string", "text:string"]
    edge_cols = [":START_ID(wiki)", ":END_ID(wiki)", ":TYPE"]
    node_field_map = {
        "nodeID:ID(wiki)": "id",
        ":LABEL": "type",
        "title:string": "title",
        "text:string": "text",
    }
    edge_field_map = {
        ":START_ID(wiki)": "start_id",
        ":END_ID(wiki)": "end_id",
        ":TYPE": "rel_type",
    }

    # Single file mode: requires a sibling index produced by build_index.py
    if args.input_file:
        input_file = os.path.abspath(args.input_file)
        if not os.path.exists(input_file):
            print(f"Error: input file not found: {input_file}", file=sys.stderr)
            return 1

        sibling_index = find_sibling_index(input_file)
        if sibling_index is None:
            print(
                f"Error: no sibling index found next to {input_file}. "
                f"Run build_index.py to create one.",
                file=sys.stderr,
            )
            return 1

        print(f"Processing {input_file}")
        print(f"Using sibling index: {sibling_index}")
        run_parallel_mode(
            [(sibling_index, input_file)],
            out_dir,
            target_titles,
            target_page_ids,
            namespace_ids,
            node_cols,
            edge_cols,
            node_field_map,
            edge_field_map,
            args.batch_size,
            args.shard_size,
            args.workers,
            args.chunk_size,
            args.chunk_overlap,
        )
        print("Done.")
        return 0

    # Dump directory mode: split index files
    dump_dir = os.path.abspath(args.dump_dir)
    pairs = discover_index_pairs(dump_dir)
    if not pairs:
        print(
            f"Error: no *-multistream-index*.txt-p*.bz2 files found in {dump_dir}",
            file=sys.stderr,
        )
        return 1

    run_parallel_mode(
        pairs,
        out_dir,
        target_titles,
        target_page_ids,
        namespace_ids,
        node_cols,
        edge_cols,
        node_field_map,
        edge_field_map,
        args.batch_size,
        args.shard_size,
        args.workers,
        args.chunk_size,
        args.chunk_overlap,
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
