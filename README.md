# Graph-based RAG QA

Code accompanying the paper:

> **Reducing Hallucinations in Complex Question Answering using Simple Graph-based
> Retrieval-Augmented Generation**
> Christopher J. Wedge, Joshua Stutter, Danny Dixon, and Jacek Cała.
> [arXiv:2606.05901](https://arxiv.org/abs/2606.05901)

## Abstract

> Large language models (LLMs) have fundamentally transformed the landscape of
> Natural Language Processing. Despite these advances, LLMs and LLM-based
> systems remain prone to a variety of failure modes. Retrieval-augmented
> generation (RAG) systems have emerged as a common deployment scenario seeking
> to both avoid the well known risk of the LLM "hallucinating" information, and
> to enable reasoning and question answering over proprietary information that
> the LLM did not have access to during training without resorting to expensive
> model fine-tuning. In this work, we explore the idea of using a lightweight
> graph structure with a relatively simple graph schema, to support the RAG
> subsystem via a dedicated toolset. We design an agentic system with a variety
> of vector search and graph query tools operating over a structured dataset
> based on a curated subset of English Wikipedia articles, and evaluate its
> performance on questions from MoNaCo, a challenging Wikipedia QA benchmark of
> complex query answering tasks. Our results show that the introduction of
> graph-based tools can significantly increase the precision and recall of
> factual correctness, can halve the number of hallucinated answers, and
> achieves the highest fine-grained truthfulness score among the three
> evaluated scenarios. All this with a modest increase in token usage.

This repository holds the full pipeline from the paper — graph construction,
agentic retrieval, and evaluation against the [MoNaCo
Benchmark](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark) (a
multi-hop QA dataset from Allen AI). The first stage, **WikiDump** (in
[`wikidump/`](wikidump/)), builds the Neo4j-compatible knowledge graph that the
retrieval tools run against, and is documented in full below.

## Repository layout

```
.
├── pyproject.toml        # Project + dependencies (managed with uv)
├── uv.lock               # Pinned, reproducible dependency lockfile
├── .python-version       # Python 3.13
├── .env.example          # Template for required secrets (HF_TOKEN, ...)
├── Snakefile             # Reproducible end-to-end pipeline (Snakemake)
├── CITATION.cff          # Machine-readable citation metadata
├── LICENSE               # MIT
├── 20250801-matches-filtered.jsonl   # Pinned MoNaCo source pages (the paper's exact subset)
└── wikidump/             # Stage 1 — graph construction scripts and batch helpers
```

`wikidump/` is the graph-construction stage; the agentic retrieval and
evaluation stages of the pipeline live alongside it at the repository root.

## Requirements

- Python >= 3.12, < 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management
- A Wikipedia multistream dump — the reproducible **2025-08-01** snapshot is on
  [Academic Torrents](https://academictorrents.com/details/19f6e3d1c44d4bc997cce8d2325964c28895c9cb)
  (it has rotated off the live [dumps.wikimedia.org](https://dumps.wikimedia.org/) mirror, which keeps
  only recent snapshots)
- A Hugging Face token (for MoNaCo benchmark access)
- Neo4j 5.x (for the final bulk import)

## Setup

```bash
uv sync                  # create the environment from the lockfile
cp .env.example .env      # then add your HF_TOKEN (and optionally AGENT_EMBEDDING_MODEL)
```

The Snakemake workflow reads `.env` at the repo root. `AGENT_EMBEDDING_MODEL` there sets the embedding
model (defaults to the paper's `microsoft/harrier-oss-v1-0.6b`); `HF_TOKEN` is needed only when
rebuilding matches from scratch or extracting the MoNaCo questions and gold answers
(`filter_monaco_questions`).

Or with pip:

```bash
pip install -e .
```

## Reproducible workflow

The graph-construction pipeline is encoded as a
[Snakemake](https://snakemake.github.io/) workflow in [`Snakefile`](Snakefile).
By default it is fully reproducible: it selects exactly the MoNaCo source pages
used in the paper by reading the pinned
[`20250801-matches-filtered.jsonl`](20250801-matches-filtered.jsonl) (1,207
questions), skipping the fuzzy-matching and human-review stages entirely.

Place the matching Wikipedia snapshot at
`data/enwiki-20250801-pages-articles-multistream.xml.bz2` (or point at it with
`--config dump=…`) and run from the repository root:

```bash
uv run snakemake -n --cores 8                # dry run: show the plan
uv run snakemake --cores 8                   # build the embedded Parquet graph
uv run snakemake --cores 8 neo4j_import      # also bulk-import into Neo4j
uv run snakemake --cores 8 filter_monaco_questions   # MoNaCo questions + gold answers
```

The evaluation stage needs the benchmark's questions and gold answers for the
same 1,207 questions. `filter_monaco_questions` joins the pinned matches file
with MoNaCo's `monaco_version_1_release.jsonl` (downloaded from Hugging Face,
so it needs `HF_TOKEN`) and writes `output/monaco_filtered.json`. It is an
explicit target, not part of the default graph build.

The dump **must be the 2025-08-01 snapshot** the pinned matches were built
against — page IDs are snapshot-specific, and `wiki2parquet` errors out listing
any pinned page ID missing from the dump's index. Override other settings via
`--config`, e.g. a different embedding model:

```bash
uv run snakemake --cores 8 --config model=Qwen/Qwen3-Embedding-8B
```

The embedding model resolves in this order: `--config model=…` (highest) → `AGENT_EMBEDDING_MODEL` in
`.env` → the built-in `microsoft/harrier-oss-v1-0.6b` default.

**Rebuilding matches from scratch** (for a different snapshot — *not*
reproducible against the paper): pass `--config rebuild_matches=true` to
re-enable `load_monaco → humaneval_corrections → filter_by_humaneval`. The
`humaneval_corrections` stage is interactive — Snakemake runs it once in your
terminal, then caches the result. The individual stages and their options are
documented below.

---

# WikiDump — building the graph

Tools for converting Wikipedia XML dumps into a Neo4j-compatible graph, with
support for filtering to articles referenced by the MoNaCo Benchmark.

This stage processes MediaWiki multistream dump files and converts them into a
graph structure with:
- **Nodes**: `Article`, `Section`, `Paragraph`, and `Chunk`
- **Edges**: `HAS_SECTION`, `HAS_PARAGRAPH`, `HAS_CHUNK`, `LINKS_TO`,
`REDIRECTS_TO`, and the sequential `NEXT_*` / `PREVIOUS_*` pairs.

The output is in Parquet format with column headers compatible with Neo4j's
bulk import tool.

`Paragraph` nodes hold the full plain-text of each paragraph (used for
retrieval and traversal). `Chunk` nodes are fixed-size character slices of each
paragraph with configurable overlap — they're the intended **embedding
target**, because embedding a whole long paragraph can overflow a model's
attention memory on consumer GPUs. The default is 500-character chunks with
40-character overlap.

> Prefix each command with `uv run` (e.g. `uv run python
> wikidump/build_index.py ...`), or activate the environment first with `source
> .venv/bin/activate`.

## Pipeline

The full workflow for building a filtered Neo4j graph from MoNaCo source
articles:

```
Wikipedia multistream dump (.xml.bz2)
    |
    v
build_index.py            Build a sibling index (if not already present)
    |
    v
load_monaco.py            Match MoNaCo source URLs to Wikipedia titles
    |
    v
humaneval_corrections.py  Manually accept/reject fuzzy matches
    |
    v
filter_by_humaneval.py    Remove rejected matches from JSONL
    |
    +-------------------> filter_monaco_questions.py
    |                     Questions + gold answers for the filtered subset
    v                     (evaluation input)
wiki2parquet.py           Convert filtered articles to Parquet
                          (Article, Section, Paragraph, Chunk nodes + edges)
    |
    v
embed_parquet.py          Embed Chunk text + Article titles (default)
    |
    v
neo4j-admin database      Bulk import Parquet shards into Neo4j
  import full
```

All stages that read pages require a multistream index. Either point at a dump
directory containing the original `*-multistream-index*.txt-p*.bz2` files, or
at a single `.xml.bz2` with a sibling index produced by `build_index.py`.

## Usage

### 1. Build an index (if needed)

If you have a single `.xml.bz2` dump without the matching index file, create
one next to it:

```bash
python wikidump/build_index.py enwiki-YYYYMMDD-pages-articles-multistream.xml.bz2
```

This writes `enwiki-YYYYMMDD-pages-articles-multistream-index.txt.bz2`
alongside the input. Skip this step if you already have a dump directory
with the original index files.

It decodes the dump's bz2 streams in parallel across all CPU cores by default. Use `--workers N` to
limit parallelism, or `--no-parallel` for the single-threaded path (the parallel output is identical).

### 2. Match MoNaCo sources to Wikipedia titles

Single-file input (auto-discovers the sibling index):

```bash
python wikidump/load_monaco.py \
    --input-file enwiki-YYYYMMDD-pages-articles-multistream.xml.bz2 \
    --output-jsonl matches.jsonl \
    --corrections-csv corrections.csv
```

Or a dump directory with the original split indices:

```bash
python wikidump/load_monaco.py \
    --dump-dir /path/to/dump \
    --output-jsonl matches.jsonl \
    --corrections-csv corrections.csv
```

This downloads MoNaCo execution traces from Hugging Face, extracts Wikipedia
source URLs, and fuzzy-matches them to titles. Requires `HF_TOKEN` in `.env`.

### 3. Review fuzzy match corrections

```bash
python wikidump/humaneval_corrections.py corrections.csv
```

Interactive prompt to accept/reject each fuzzy match. Writes a
`corrections-humaneval.csv` with your decisions.

### 4. Filter matches by human evaluation

```bash
python wikidump/filter_by_humaneval.py corrections-humaneval.csv matches.jsonl matches-filtered.jsonl
```

Removes questions whose sources include rejected fuzzy matches.

### 5. Extract the MoNaCo questions and gold answers

```bash
python wikidump/filter_monaco_questions.py \
    --matches-jsonl matches-filtered.jsonl \
    --output-json monaco_filtered.json
```

Downloads `monaco_version_1_release.jsonl` from Hugging Face (requires `HF_TOKEN`
in `.env`; pass `--release-jsonl` to use a local copy instead) and writes a JSON
list of `{ex_num, question, decomposition, validated_answer}` for every question
in the matches file, in the same order. It errors out if any `ex_num` is missing
or its question text differs from the release file — the guard against joining a
matches file with a different MoNaCo revision. This is the question set the
evaluation stage runs the agent over; with the pinned
`20250801-matches-filtered.jsonl` it yields the paper's 1,207 questions.

### 6. Convert to Parquet

Filter to only MoNaCo source articles (single-file input):

```bash
python wikidump/wiki2parquet.py \
    --input-file enwiki-YYYYMMDD-pages-articles-multistream.xml.bz2 \
    --matches-jsonl matches-filtered.jsonl \
    --out-dir output/ --workers 8
```

Or filter to specific MoNaCo questions:

```bash
python wikidump/wiki2parquet.py --input-file dump.xml.bz2 \
    --matches-jsonl matches-filtered.jsonl \
    --question 42 --question 87 \
    --out-dir output/
```

Or filter by title (no MoNaCo dependency):

```bash
python wikidump/wiki2parquet.py --input-file dump.xml.bz2 \
    --out-dir output/ \
    --title "Wheat" --title "Maize"
```

Using a dump directory with split index files:

```bash
python wikidump/wiki2parquet.py --dump-dir /path/to/dump --out-dir output/ --workers 8
```

Key options:
- `--input-file`: single `.xml.bz2` (requires a sibling index from `build_index.py`).
- `--dump-dir`: directory with the original split `*-multistream-index*.txt-p*.bz2` files.
- `--matches-jsonl`: filter nodes to MoNaCo source page_ids from `load_monaco.py`.
- `--question N`: restrict to specific MoNaCo questions (repeatable; requires `--matches-jsonl`).
- `--title T`: restrict to a page title (repeatable).
- `--namespace-id N`: restrict to a namespace id (repeatable).
- `--chunk-size`: character length of each `Chunk` node (default: `500`).
- `--chunk-overlap`: character overlap between consecutive chunks (default: `40`; must be `< chunk-size`).
- `--batch-size`: index entries per worker batch (default: `10`).
- `--shard-size`: maximum rows per output Parquet shard (default: `1_000_000`).
- `--workers`: number of worker processes (default: CPU count).

When any filter is active (`--matches-jsonl`, `--title`, or `--namespace-id`), `wiki2parquet.py`
verifies that every target page_id/title is present in the dump's index — it errors out with a list
of the missing ones if the matches file was built against a different dump snapshot. It also writes
empty-stub `Article` nodes for any cross-article `LINKS_TO` / `REDIRECTS_TO` targets that aren't in
the subset, so the resulting graph has no dangling edges.

### 7. Add embeddings

`Chunk` text and `Article` titles are embedded into the same `embedding:float[]` column; other node
types are written through with an empty `embedding` value. The progress bar reflects the number of
embedded rows, not total rows.

```bash
python wikidump/embed_parquet.py output/nodes-*.parquet \
    --out-dir output-embed/
```

This uses the default model, **`microsoft/harrier-oss-v1-0.6b`** — the model the paper uses. Keep this
default to reproduce the paper's graph; scale `--batch-size` down from `256` to fit your GPU.

For a different (heavier) model, e.g. on Apple Silicon:

```bash
python wikidump/embed_parquet.py output/nodes-*.parquet \
    --model Qwen/Qwen3-Embedding-8B \
    --workers 1 --devices mps \
    --batch-size 16 \
    --out-dir output-embed/
```

Options:
- `--model`: Hugging Face sentence-transformer model (default: `microsoft/harrier-oss-v1-0.6b`, the paper's model).
- `--batch-size`: batch size for inference (default: `256`). Scale this down for large models — the
  correct ceiling depends on the model's hidden size and your chunk length.
- `--workers`: number of embedding worker processes (default: CPU count). Use `1` when pinning to a
  single MPS/CUDA device.
- `--devices`: comma-separated device list (e.g. `cpu`, `mps`, `cuda:0,cuda:1`). Only honored when
  `--workers > 1`.
- `--out-dir`: output directory (defaults to in-place update).
- `--prefix`: text to prepend to each input before embedding (for models that want a task-specific
  prefix).

### 8. Import into Neo4j

Neo4j 5.x's bulk importer reads Parquet directly, detected by file extension. Stop Neo4j first, then
run the import against the embedded nodes and the edges parquet:

```bash
neo4j stop

neo4j-admin database import full \
    --nodes='output-embed/nodes-.*\.parquet' \
    --relationships='output/edges-.*\.parquet' \
    --id-type=STRING \
    --array-delimiter=';' \
    --overwrite-destination \
    neo4j

neo4j start
```

Flags that matter for this schema:
- `--id-type=STRING`: node IDs look like `article:12345:s2:p7:c0`, not integers.
- `--array-delimiter=';'`: the `embedding:float[]` column is stored as a string of `;`-joined floats;
  this tells the importer how to parse it. Also applies to any multi-label `:LABEL` values, but all
  nodes in this pipeline are single-labeled.
- `--overwrite-destination`: required if the `neo4j` database already has a store.

The nodes shards must come from the embed output (they carry the `embedding` column); the edges
shards come from the `wiki2parquet.py` output directory. Quick post-import sanity check:

```bash
cypher-shell -u neo4j -p <password> \
    "MATCH (n) RETURN labels(n)[0] AS label, count(*) ORDER BY count(*) DESC"
```

You should see `Chunk`, `Article`, `Paragraph`, `Section` buckets.

## Utilities

**List namespaces in a dump:**

```bash
python wikidump/list_namespaces.py /path/to/dump
```

**Count tokens in Parquet files:**

```bash
python wikidump/count_tokens.py nodes-*.parquet
```

**Plot humaneval summary:**

```bash
python wikidump/plot_humaneval_summary.py
```

Reads `humaneval_summary.csv` and generates `humaneval_summary.png` showing answerable questions per
dump snapshot.

## Batch scripts

Shell scripts for processing multiple dump snapshots (also invoked from the repo root):

- `wikidump/correct_all_monaco.sh` — Build indices (if missing) and run MoNaCo matching for every dump in `INPUT_DIR`
- `wikidump/filter_all_humaneval.sh` — Apply human evaluation filters to all matches in `matching_results/`

```bash
bash wikidump/correct_all_monaco.sh
bash wikidump/filter_all_humaneval.sh
```

## Output format

**Nodes** (`nodes-*.parquet`):

| Column | Description |
|--------|-------------|
| `nodeID:ID(wiki)` | Unique node identifier, namespaced as `article:<page_id>[:s<section_i>[:p<para_i>[:c<chunk_i>]]]`. |
| `:LABEL` | Node type: `Article`, `Section`, `Paragraph`, or `Chunk`. |
| `title:string` | Article title (populated for `Article` nodes; empty otherwise). |
| `text:string` | Text content (populated for `Paragraph` nodes with the full paragraph, and for `Chunk` nodes with a `--chunk-size`-character slice of the parent paragraph; empty for `Article` and `Section`). |
| `embedding:float[]` | Appended by `embed_parquet.py`. Stored as a `;`-delimited string of floats; by default populated from `text:string` for `Chunk` rows and from `title:string` for `Article` rows. |

**Edges** (`edges-*.parquet`):

| Column | Description |
|--------|-------------|
| `:START_ID(wiki)` | Source node ID |
| `:END_ID(wiki)` | Target node ID |
| `:TYPE` | Relationship type |

Relationship types:
- `HAS_SECTION` — `Article` → `Section`
- `HAS_PARAGRAPH` — `Section` → `Paragraph`
- `HAS_CHUNK` — `Paragraph` → `Chunk`
- `NEXT_SECTION` / `PREVIOUS_SECTION` — sequential links between sibling sections
- `NEXT_PARAGRAPH` / `PREVIOUS_PARAGRAPH` — sequential links between sibling paragraphs
- `NEXT_CHUNK` / `PREVIOUS_CHUNK` — sequential links between sibling chunks of the same paragraph
- `LINKS_TO` — wikilink from a `Paragraph` to an `Article`
- `REDIRECTS_TO` — redirect from one `Article` to another

---

## Data availability

The pinned MoNaCo source-page subset
([`20250801-matches-filtered.jsonl`](20250801-matches-filtered.jsonl)) is
included in this repository for exact reproducibility. The larger artifacts are
archived on Zenodo:

- the Wikipedia multistream dump (2025-08-01 snapshot)
- the **constructed graph as Parquet files** — the embedded node shards
  (`microsoft/harrier-oss-v1-0.6b`) — so you can bulk-import the graph into
  Neo4j directly, without re-running the multi-hour build;

> **Dataset:** _Graph-based RAG QA — Wikipedia knowledge graph and MoNaCo
> subset_ Zenodo, 2026. DOI:
> [`10.5281/zenodo.XXXXXXX`](https://doi.org/10.5281/zenodo.XXXXXXX)
> _(placeholder — to be minted on deposit)_

The 2025-08-01 Wikipedia snapshot has rotated off the live
[dumps.wikimedia.org](https://dumps.wikimedia.org/) mirror; download the
original dump from [Academic
Torrents](https://academictorrents.com/details/19f6e3d1c44d4bc997cce8d2325964c28895c9cb).
The snapshot is also mirrored in the Zenodo deposit above. English Wikipedia
text is licensed [CC BY-SA
4.0](https://creativecommons.org/licenses/by-sa/4.0/) (with code/data
exceptions); see **Licensing of redistributed data** below.

## Citation

If you use this code, please cite:

```bibtex
@article{wedge2026reducing,
  title   = {Reducing Hallucinations in Complex Question Answering using Simple Graph-based Retrieval-Augmented Generation},
  author  = {Wedge, Christopher J. and Stutter, Joshua and Dixon, Danny and Ca{\l}a, Jacek},
  journal = {arXiv preprint arXiv:2606.05901},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.05901}
}
```

## License

**Code** in this repository is [MIT](LICENSE) © 2026 National Innovation Centre
for Data.

### Licensing of redistributed data

The MIT license covers the **code only**. Data derived from English Wikipedia
is licensed separately:

- The **Wikipedia snapshot** and the artifacts derived from its text — the
  `text:string` content of the `Paragraph` / `Chunk` nodes, and (to be safe) the
  embeddings computed from it — are adaptations of Wikipedia content and so are
  licensed **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**,
  © Wikipedia contributors. Redistribution (on Academic Torrents or in the
  Zenodo deposit) is permitted by CC BY-SA 4.0 as long as we **attribute** the
  source, **share alike** (keep the same license on the data), and **indicate
  changes**. The Zenodo data record is therefore released under CC BY-SA 4.0
  with an attribution note crediting English Wikipedia (`enwiki`, 2025-08-01
  snapshot) and linking back to the source.
- The **MoNaCo benchmark** questions referenced by
  [`20250801-matches-filtered.jsonl`](20250801-matches-filtered.jsonl) are ©
  Allen Institute for AI under the MoNaCo dataset's own license — see its
  [dataset card](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark).
