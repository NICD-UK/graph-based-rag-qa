"""Shortest MENTIONS-path between two Articles, with paragraph evidence per hop."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read

CYPHER = """
MATCH (start_a:Article {title: $title_a})
OPTIONAL MATCH (start_a)-[:REDIRECTS_TO*1..10]->(ta:Article)
WHERE NOT (ta)-[:REDIRECTS_TO]->()
WITH coalesce(ta, start_a) AS a

MATCH (start_b:Article {title: $title_b})
OPTIONAL MATCH (start_b)-[:REDIRECTS_TO*1..10]->(tb:Article)
WHERE NOT (tb)-[:REDIRECTS_TO]->()
WITH a, coalesce(tb, start_b) AS b

MATCH path = shortestPath((a)-[:MENTIONS*..6]->(b))

WITH path, nodes(path) AS articles, range(0, length(path) - 1) AS idx
UNWIND idx AS i
WITH articles[i] AS from_article, articles[i+1] AS to_article, i

MATCH (from_article)-[:HAS_SECTION]->(section:Section)
      -[:HAS_PARAGRAPH]->(para:Paragraph)-[:LINKS_TO]->(to_target:Article)
WHERE to_target = to_article
   OR EXISTS {
        MATCH (to_target)-[:REDIRECTS_TO*1..10]->(to_article)
      }
WITH i, from_article, to_article, section, para
ORDER BY i, size(para.text)
WITH i, from_article, to_article,
     head(collect({section_id: section.nodeID,
                   paragraph_id: para.nodeID,
                   text: para.text})) AS evidence

RETURN i                          AS hop,
       from_article.title         AS from_article,
       to_article.title           AS to_article,
       evidence.paragraph_id      AS paragraph_nodeID,
       evidence.text              AS paragraph_text
ORDER BY hop
"""


@tool(parse_docstring=True)
def shortest_path(title_a: str, title_b: str) -> str:
    """Find the shortest path (up to 6 hops) between two Articles along the projected
    MENTIONS relationship, returning a Paragraph of evidence for each hop. Both
    endpoints are resolved via REDIRECTS_TO to their canonical form.

    Args:
        title_a: Start Article title.
        title_b: End Article title.
    """
    if not isinstance(title_a, str) or not isinstance(title_b, str):
        return json.dumps({"error": "title_a and title_b must both be strings"})
    try:
        rows = run_read(
            CYPHER,
            params={"title_a": title_a, "title_b": title_b},
            limit=100,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"from": title_a, "to": title_b, "hop_count": len(rows), "hops": rows},
        default=str,
    )
