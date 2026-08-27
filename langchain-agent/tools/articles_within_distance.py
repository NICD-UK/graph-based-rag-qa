"""Return all Articles reachable within MENTIONS distance n from a starting Article."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read


def _build_cypher(n: int) -> str:
    return f"""
MATCH (start:Article {{title: $title}})
OPTIONAL MATCH (start)-[:REDIRECTS_TO*1..10]->(t:Article)
WHERE NOT (t)-[:REDIRECTS_TO]->()
WITH coalesce(t, start) AS canonical

MATCH path = (canonical)-[:MENTIONS*1..{n}]->(neighbour:Article)
WHERE neighbour <> canonical
WITH canonical, neighbour, min(length(path)) AS distance

OPTIONAL MATCH (neighbour)-[:REDIRECTS_TO*1..10]->(nt:Article)
WHERE NOT (nt)-[:REDIRECTS_TO]->()
WITH canonical.title AS source,
     coalesce(nt, neighbour) AS resolved_node,
     distance
WHERE resolved_node.title <> source

RETURN resolved_node.title AS title,
       min(distance)       AS distance
ORDER BY distance, title
"""


@tool(parse_docstring=True)
def articles_within_distance(title: str, n: int = 2) -> str:
    """Starting from an Article (resolved through REDIRECTS_TO to its canonical form),
    return every Article reachable via 1..n MENTIONS hops together with its minimum
    distance. Useful for exploring a topic's neighborhood. ``n`` is capped at 3 to
    avoid blowup.

    Args:
        title: Starting Article title.
        n: Max MENTIONS distance, 1-3. Default 2.
    """
    if not isinstance(title, str):
        return json.dumps({"error": "title must be a string"})
    capped_n = max(1, min(int(n), 3))
    try:
        rows = run_read(_build_cypher(capped_n), params={"title": title}, limit=2000)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"source": title, "max_distance": capped_n, "count": len(rows), "neighbours": rows},
        default=str,
    )
