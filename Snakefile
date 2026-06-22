# Snakefile — reproducible WikiDump graph-construction pipeline.
#
# Default (reproducible) DAG — consumes the pinned matches file shipped with the repo:
#
#   build_index -> wiki2parquet -> embed_parquet -> (optional) neo4j_import
#                       ^
#                       └── 20250801-matches-filtered.jsonl  (pinned MoNaCo source pages)
#
# By default the workflow selects exactly the same MoNaCo source pages as the paper
# by reading the pinned `20250801-matches-filtered.jsonl` as a SOURCE input. The
# fuzzy-matching / human-review stages are skipped, so every run is reproducible —
# provided the dump is the matching snapshot (wiki2parquet errors if a pinned
# page_id is absent from the dump's index).
#
# To regenerate matches from scratch for a different snapshot, pass
# `--config rebuild_matches=true`, which re-enables:
#
#   load_monaco -> humaneval_corrections -> filter_by_humaneval -> wiki2parquet -> ...
#
# Run from the repository root inside the uv environment:
#
#   uv run snakemake -n --cores 8                # dry run: show the plan
#   uv run snakemake --cores 8                   # build the embedded Parquet graph
#   uv run snakemake --cores 8 neo4j_import      # also bulk-import into Neo4j
#
# Override any config value on the command line, e.g.:
#
#   uv run snakemake --cores 8 \
#       --config dump=/data/enwiki-20250801-pages-articles-multistream.xml.bz2 \
#                model=Qwen/Qwen3-Embedding-8B

import os
from dotenv import load_dotenv

# Load .env from the repo root (next to this Snakefile) so settings like the
# embedding model can live alongside HF_TOKEN.
load_dotenv(os.path.join(workflow.basedir, ".env"))

# ----- Configuration (override with --config key=value or --configfile) -------
DUMP     = config.get("dump", "data/enwiki-20250801-pages-articles-multistream.xml.bz2")
OUTDIR   = config.get("outdir", "output")
WORKERS  = int(config.get("workers", os.cpu_count() or 8))
NEO4J_DB = config.get("neo4j_db", "neo4j")

# Embedding model precedence: --config model=  >  AGENT_EMBEDDING_MODEL in .env  >  default.
# Shares the agent's var so the graph and query embeddings always use the same model.
MODEL = config.get("model") or os.environ.get("AGENT_EMBEDDING_MODEL") or "microsoft/harrier-oss-v1-0.6b"

# Reproducibility switch: default False -> consume the pinned matches file.
REBUILD_MATCHES = str(config.get("rebuild_matches", "false")).lower() in ("true", "1", "yes")

# Sibling index path produced by build_index.py (mirrors its default naming).
INDEX = DUMP.replace("-multistream.xml.bz2", "-multistream-index.txt.bz2")

# Intermediate / output artifacts.
MATCHES           = f"{OUTDIR}/matches.jsonl"
CORRECTIONS       = f"{OUTDIR}/corrections.csv"
CORRECTIONS_HUMAN = f"{OUTDIR}/corrections-humaneval.csv"
PARQUET_DIR       = f"{OUTDIR}/parquet"
EMBED_DIR         = f"{OUTDIR}/parquet-embed"
IMPORT_SENTINEL   = f"{OUTDIR}/.neo4j-import.done"

# The pre-filtered MoNaCo matches that wiki2parquet selects source pages from.
# Reproducible default: the pinned file at the repo root. When rebuilding, a fresh
# one is generated under OUTDIR instead.
if REBUILD_MATCHES:
    MATCHES_FILTERED = config.get("matches_filtered", f"{OUTDIR}/matches-filtered.jsonl")
else:
    MATCHES_FILTERED = config.get("matches_filtered", "20250801-matches-filtered.jsonl")


rule all:
    """Default target: the embedded Parquet graph, ready for Neo4j import."""
    input:
        EMBED_DIR


rule build_index:
    """Build a sibling multistream index next to the dump (if not already present)."""
    input:
        dump=DUMP,
    output:
        index=INDEX,
    threads: WORKERS
    shell:
        "python wikidump/build_index.py {input.dump} --output {output.index} "
        "--workers {threads}"


# ----- Matching stages: only defined when rebuilding matches from scratch --------
# In the reproducible default, MATCHES_FILTERED is the pinned source file and these
# rules are not part of the workflow at all.
if REBUILD_MATCHES:

    rule load_monaco:
        """Match MoNaCo source URLs to Wikipedia titles. Requires HF_TOKEN in .env.

        --input-file auto-discovers the sibling index, so `index` is listed only to
        order the DAG; it is not passed on the command line.
        """
        input:
            dump=DUMP,
            index=INDEX,
        output:
            matches=MATCHES,
            corrections=CORRECTIONS,
        threads: WORKERS
        shell:
            "python wikidump/load_monaco.py "
            "--input-file {input.dump} "
            "--output-jsonl {output.matches} "
            "--corrections-csv {output.corrections} "
            "--workers {threads}"

    rule humaneval_corrections:
        """INTERACTIVE: manually accept/reject each fuzzy match.

        Prompts in the terminal and writes <stem>-humaneval.csv next to the input.
        """
        input:
            corrections=CORRECTIONS,
        output:
            human=CORRECTIONS_HUMAN,
        shell:
            "python wikidump/humaneval_corrections.py {input.corrections}"

    rule filter_by_humaneval:
        """Drop questions whose sources include rejected fuzzy matches."""
        input:
            human=CORRECTIONS_HUMAN,
            matches=MATCHES,
        output:
            filtered=MATCHES_FILTERED,
        shell:
            "python wikidump/filter_by_humaneval.py "
            "{input.human} {input.matches} {output.filtered}"


rule wiki2parquet:
    """Convert the filtered MoNaCo source articles to Parquet nodes + edges.

    --input-file auto-discovers the sibling index (listed as an input for ordering).
    Errors if any pinned page_id is missing from the dump's index — the guard that
    enforces dump/matches snapshot consistency.
    """
    input:
        dump=DUMP,
        index=INDEX,
        matches=MATCHES_FILTERED,
    output:
        parquet=directory(PARQUET_DIR),
    threads: WORKERS
    shell:
        "python wikidump/wiki2parquet.py "
        "--input-file {input.dump} "
        "--matches-jsonl {input.matches} "
        "--out-dir {output.parquet} "
        "--workers {threads}"


rule embed_parquet:
    """Embed Chunk text + Article titles into the node shards."""
    input:
        parquet=PARQUET_DIR,
    output:
        embedded=directory(EMBED_DIR),
    params:
        model=MODEL,
    threads: WORKERS
    shell:
        "python wikidump/embed_parquet.py {input.parquet}/nodes-*.parquet "
        "--model {params.model} "
        "--workers 1 "
        "--devices mps "
        "--batch-size 16 "
        "--out-dir {output.embedded}"


rule neo4j_import:
    """Bulk-import the embedded nodes + edges into Neo4j (explicit opt-in target).

    Side-effecting: stops Neo4j, overwrites the target database, restarts it.
    Run as `snakemake neo4j_import`. Requires neo4j / neo4j-admin on PATH and
    permission to manage the server.
    """
    input:
        nodes=EMBED_DIR,
        edges=PARQUET_DIR,
    output:
        sentinel=touch(IMPORT_SENTINEL),
    params:
        db=NEO4J_DB,
    shell:
        r"""
        neo4j stop
        neo4j-admin database import full \
            --nodes='{input.nodes}/nodes-.*\.parquet' \
            --relationships='{input.edges}/edges-.*\.parquet' \
            --id-type=STRING \
            --array-delimiter=';' \
            --overwrite-destination \
            {params.db}
        neo4j start
        """
