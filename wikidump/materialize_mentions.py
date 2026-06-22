"""Copy edge shards and append Article-[:MENTIONS]->Article edges.

Produces a drop-in replacement for the ``edges-*.parquet`` shards written by
``wiki2parquet.py``: every original edge file is copied into ``OUT_DIR``, and
additional ``edges-NNNNN.parquet`` shards are appended containing
``MENTIONS`` edges derived from the paragraph-level ``LINKS_TO`` edges. The
source article is taken from the paragraph node ID
(``article:<page_id>:s<i>:p<j>`` -> ``article:<page_id>``); the target is
already an ``Article`` ID.

Usage:
    python materialize_mentions.py IN_DIR OUT_DIR
"""

import argparse
import glob
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

EDGE_COLS = [":START_ID(wiki)", ":END_ID(wiki)", ":TYPE"]
ARTICLE_ID_PATTERN = r"^(?P<src>article:[^:]+)"
BATCH_SIZE = 100_000
SHARD_SIZE = 1_000_000


def collect_mentions(paths: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in paths:
        pf = pq.ParquetFile(path)
        missing = [c for c in EDGE_COLS if c not in pf.schema_arrow.names]
        if missing:
            raise ValueError(f"{path} missing columns {missing!r}")

        with tqdm(
            total=pf.metadata.num_rows,
            desc=os.path.basename(path),
            unit="rows",
        ) as pbar:
            for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=EDGE_COLS):
                mask = pc.equal(batch.column(":TYPE"), "LINKS_TO")
                filtered = batch.filter(mask)
                if filtered.num_rows:
                    extracted = pc.extract_regex(
                        filtered.column(":START_ID(wiki)"),
                        pattern=ARTICLE_ID_PATTERN,
                    )
                    src_list = pc.struct_field(extracted, "src").to_pylist()
                    end_list = filtered.column(":END_ID(wiki)").to_pylist()
                    for s, e in zip(src_list, end_list):
                        if s is not None and e is not None:
                            pairs.add((s, e))
                pbar.update(batch.num_rows)
    return pairs


def parse_shard_idx(path: str) -> int | None:
    name = os.path.basename(path)
    stem = name[len("edges-") : -len(".parquet")]
    try:
        return int(stem)
    except ValueError:
        return None


def write_mentions(
    pairs: set[tuple[str, str]], out_dir: str, start_idx: int
) -> int:
    ordered = sorted(pairs)
    schema = pa.schema([pa.field(c, pa.string()) for c in EDGE_COLS])

    n_shards = 0
    for i in range(0, len(ordered), SHARD_SIZE):
        shard = ordered[i : i + SHARD_SIZE]
        starts = [s for s, _ in shard]
        ends = [e for _, e in shard]
        types = ["MENTIONS"] * len(shard)
        table = pa.table([starts, ends, types], schema=schema)
        out_path = os.path.join(
            out_dir, f"edges-{start_idx + n_shards:05d}.parquet"
        )
        pq.write_table(table, out_path, compression=None, use_dictionary=False)
        n_shards += 1
    return n_shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_dir", help="directory containing edges-*.parquet from wiki2parquet.py")
    parser.add_argument("out_dir", help="directory to write the full edges-*.parquet set")
    args = parser.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(in_dir, "edges-*.parquet")))
    if not paths:
        print(f"Error: no edges-*.parquet files found in {in_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {len(paths)} edge file(s) in {in_dir}")
    pairs = collect_mentions(paths)
    print(f"Collected {len(pairs):,} unique Article->Article MENTIONS")

    same_dir = in_dir == out_dir
    max_idx = -1
    for p in paths:
        if not same_dir:
            shutil.copy2(p, os.path.join(out_dir, os.path.basename(p)))
        idx = parse_shard_idx(p)
        if idx is not None:
            max_idx = max(max_idx, idx)
    if not same_dir:
        print(f"Copied {len(paths)} shard(s) to {out_dir}")

    start_idx = max_idx + 1
    n_new = write_mentions(pairs, out_dir, start_idx)
    print(
        f"Wrote {n_new} MENTIONS shard(s) starting at "
        f"edges-{start_idx:05d}.parquet"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
