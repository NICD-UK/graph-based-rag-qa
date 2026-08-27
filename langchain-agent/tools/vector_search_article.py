"""Semantic search over Article title embeddings."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .embeddings import embed
from .neo4j_client import run_read
from .vector_index import ensure_article_index

CYPHER = """
CALL db.index.vector.queryNodes($index_name, toInteger($k), $query_vector)
YIELD node AS article, score
RETURN article.nodeID  AS article_nodeID,
       article.title   AS title,
       score
ORDER BY score DESC
"""


@tool(parse_docstring=True)
def vector_search_article(query_text: str, k: int = 5) -> str:
    """Semantic search over Article title embeddings. Embeds ``query_text`` and
    returns the top Articles by cosine similarity. Use this to find candidate
    Articles by topic/name when exact title lookup is not sufficient.

    Args:
        query_text: Text to embed and search for.
        k: Number of articles to return (default 5, capped at 50).
    """
    capped_k = max(1, min(int(k), 50))
    try:
        vector = embed(query_text)
        index_name = ensure_article_index(len(vector))
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
