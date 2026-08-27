"""Semantic search over Chunk embeddings, resolved up to whole Paragraphs."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .embeddings import embed
from .neo4j_client import run_read
from .vector_index import ensure_chunk_index

CYPHER = """
CALL db.index.vector.queryNodes($index_name, toInteger($k * 4), $query_vector)
YIELD node AS chunk, score

MATCH (para:Paragraph)-[:HAS_CHUNK]->(chunk)
WITH para, chunk, score
ORDER BY score DESC

WITH para,
     head(collect({chunk_id: chunk.nodeID, score: score})) AS best_chunk
ORDER BY best_chunk.score DESC
LIMIT toInteger($k)

RETURN para.nodeID          AS paragraph_nodeID,
       para.text             AS text,
       best_chunk.chunk_id   AS matched_chunk_id,
       best_chunk.score      AS score
"""


@tool(parse_docstring=True)
def vector_search_paragraph(query_text: str, k: int = 5) -> str:
    """Semantic search over the Wikipedia graph at Paragraph granularity. Embeds
    ``query_text``, finds the top Chunks by cosine similarity, and returns the full
    Paragraph containing each winning chunk (deduplicated). Use this for
    entry-point discovery when you need relevant prose, not a bare chunk.

    Args:
        query_text: Text to embed and search for.
        k: Number of distinct paragraphs to return (default 5, capped at 50).
    """
    capped_k = max(1, min(int(k), 50))
    try:
        vector = embed(query_text)
        index_name = ensure_chunk_index(len(vector))
        rows = run_read(
            CYPHER,
            params={"index_name": index_name, "k": capped_k, "query_vector": vector},
            limit=capped_k,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"query": query_text, "k": capped_k, "results": rows},
        default=str,
    )
