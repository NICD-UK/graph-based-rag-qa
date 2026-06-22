"""Add sentence-transformer embeddings to Parquet node files.

Embeds ``text:string`` for ``:LABEL=Chunk`` rows and ``title:string`` for
``:LABEL=Article`` rows into a single ``embedding:float[]`` column. Other
node types are written through with a null embedding. Supports multi-process
encoding for improved performance.
"""

import argparse
import os
import tempfile
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Type alias for sentence-transformers multi-process pool
Pool = dict[Literal["input", "output", "processes"], Any] | None


LABEL_COL = ":LABEL"
EMBED_COL = "embedding:float[]"
LABEL_TO_COL = {"Chunk": "text:string", "Article": "title:string"}


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    pool: Pool,
) -> list[list[float]]:
    if not texts:
        return []
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False, pool=pool
    )
    return embeddings.tolist()


def embed_file(
    path: str,
    out_path: str,
    model: SentenceTransformer,
    batch_size: int,
    progress: tqdm | None,
    pool: Pool,
    prefix: str,
) -> None:
    parquet = pq.ParquetFile(path)
    if EMBED_COL in parquet.schema.names:
        raise ValueError(f"Column {EMBED_COL!r} already exists in {path}")

    base_schema = parquet.schema_arrow
    required = [LABEL_COL, *set(LABEL_TO_COL.values())]
    missing = [c for c in required if c not in base_schema.names]
    if missing:
        raise ValueError(f"Missing required column(s) {missing!r} in {path}")

    out_schema = base_schema.append(pa.field(EMBED_COL, pa.string()))
    writer = pq.ParquetWriter(
        out_path,
        out_schema,
        compression=None,
        version="1.0",
        use_dictionary=False,
    )

    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            n_rows = batch.num_rows
            embed_strings: list[str | None] = [None] * n_rows
            to_encode: list[str] = []
            encode_positions: list[int] = []

            labels = batch.column(LABEL_COL).to_pylist()
            cols = {
                col: batch.column(col).to_pylist()
                for col in set(LABEL_TO_COL.values())
            }
            for i in range(n_rows):
                col = LABEL_TO_COL.get(labels[i])
                if col is None:
                    continue
                text = cols[col][i]
                if text is None or text == "":
                    continue
                to_encode.append(f"{prefix}{text}")
                encode_positions.append(i)

            vectors = encode_texts(model, to_encode, batch_size=batch_size, pool=pool)
            for pos, vec in zip(encode_positions, vectors):
                embed_strings[pos] = ";".join(str(x) for x in vec)

            embed_array = pa.array(embed_strings, type=pa.string())
            table = pa.Table.from_batches([batch], schema=base_schema)
            table = table.append_column(EMBED_COL, embed_array)
            writer.write_table(table)
            if progress is not None:
                progress.update(len(to_encode))
    finally:
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add sentence-transformer embeddings to nodes parquet files. "
            "Embeds Chunk text and Article titles into 'embedding:float[]'."
        )
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more nodes-*.parquet files to update.",
    )
    parser.add_argument(
        "--model",
        default="microsoft/harrier-oss-v1-0.6b",
        help="Hugging Face sentence-transformer model name or path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 4,
        help="Number of embedding worker processes.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated device list for embedding (e.g. 'cpu' or 'cuda:0,cuda:1').",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (defaults to in-place update).",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Text to prepend to each non-empty input before embedding.",
    )
    args = parser.parse_args()

    pending: list[str] = []
    skipped: list[str] = []
    for path in args.files:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if args.out_dir:
            out_path = os.path.join(args.out_dir, os.path.basename(path))
            if os.path.isfile(out_path):
                skipped.append(path)
                continue
        else:
            pf_check = pq.ParquetFile(path)
            if EMBED_COL in pf_check.schema_arrow.names:
                skipped.append(path)
                continue
        pending.append(path)

    if skipped:
        print(f"Skipping {len(skipped)} file(s) with existing embeddings.")
    if not pending:
        print("Nothing to do.")
        return 0

    model = SentenceTransformer(args.model)
    pool = None
    if args.workers > 1:
        if args.devices:
            base_devices = [d.strip() for d in args.devices.split(",") if d.strip()]
            if not base_devices:
                base_devices = ["cpu"]
            devices = [base_devices[i % len(base_devices)] for i in range(args.workers)]
        else:
            devices = ["cpu"] * args.workers
        pool = model.start_multi_process_pool(target_devices=devices)

    parquet_files: list[tuple[str, pq.ParquetFile]] = []
    scan_total = 0
    for path in pending:
        pf = pq.ParquetFile(path)
        if LABEL_COL not in pf.schema_arrow.names:
            raise ValueError(
                f"Cannot filter by label: column {LABEL_COL!r} not in {path}"
            )
        parquet_files.append((path, pf))
        scan_total += pf.metadata.num_rows

    total_rows = 0
    scan_cols = [LABEL_COL, *set(LABEL_TO_COL.values())]
    with tqdm(total=scan_total, unit="row", desc="Prescan") as scan_bar:
        for _, pf in parquet_files:
            for batch in pf.iter_batches(columns=scan_cols):
                labels = batch.column(LABEL_COL).to_pylist()
                cols = {
                    c: batch.column(c).to_pylist()
                    for c in set(LABEL_TO_COL.values())
                }
                for i, lab in enumerate(labels):
                    col = LABEL_TO_COL.get(lab)
                    if col is None:
                        continue
                    val = cols[col][i]
                    if val is None or val == "":
                        continue
                    total_rows += 1
                scan_bar.update(batch.num_rows)

    progress = tqdm(total=total_rows, unit="node")

    for path in pending:
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            out_path = os.path.join(args.out_dir, os.path.basename(path))
            embed_file(
                path,
                out_path,
                model,
                args.batch_size,
                progress,
                pool,
                args.prefix,
            )
        else:
            dir_name = os.path.dirname(path) or "."
            with tempfile.NamedTemporaryFile(
                dir=dir_name, prefix=".embed_", suffix=".parquet", delete=False
            ) as tmp:
                tmp_path = tmp.name
            embed_file(
                path,
                tmp_path,
                model,
                args.batch_size,
                progress,
                pool,
                args.prefix,
            )
            os.replace(tmp_path, path)

    if pool is not None:
        model.stop_multi_process_pool(pool)
    progress.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
