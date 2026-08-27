"""Return Paragraphs that link to (mention) a given Article."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read

CYPHER = """
MATCH (start:Article {title: $title})
OPTIONAL MATCH (start)-[:REDIRECTS_TO*1..10]->(t:Article)
WHERE NOT (t)-[:REDIRECTS_TO]->()
WITH coalesce(t, start) AS target

MATCH (para:Paragraph)-[:LINKS_TO]->(target)
MATCH (src:Article)-[:HAS_SECTION]->(section:Section)-[:HAS_PARAGRAPH]->(para)
WHERE src <> target

RETURN src.title    AS source_article,
       para.nodeID  AS paragraph_nodeID,
       para.text    AS text
ORDER BY source_article
LIMIT toInteger($limit)
"""


@tool(parse_docstring=True)
def get_backlinks(title: str, limit: int = 50) -> str:
    """Given an Article title (resolved via REDIRECTS_TO), return every Paragraph in
    any other Article that LINKS_TO the target. Ordered by source article title.
    Use this to find where a topic is discussed elsewhere in the corpus.

    Args:
        title: Target Article title.
        limit: Max paragraphs to return (default 50, capped at 500).
    """
    if not isinstance(title, str):
        return json.dumps({"error": "title must be a string"})
    capped = max(1, min(int(limit), 500))
    try:
        rows = run_read(CYPHER, params={"title": title, "limit": capped}, limit=capped)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"target": title, "count": len(rows), "backlinks": rows},
        default=str,
    )
