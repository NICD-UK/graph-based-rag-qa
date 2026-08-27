"""Return a window of Paragraphs centred on a given Paragraph (+/- n)."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read

CYPHER = """
MATCH (hit:Paragraph {nodeID: $paragraph_id})

OPTIONAL MATCH back_path = (hit)<-[:NEXT_PARAGRAPH*1..]-(prev:Paragraph)
WHERE length(back_path) <= $n
WITH hit, collect({para: prev, offset: -length(back_path)}) AS backward

OPTIONAL MATCH fwd_path = (hit)-[:NEXT_PARAGRAPH*1..]->(next:Paragraph)
WHERE length(fwd_path) <= $n
WITH hit, backward, collect({para: next, offset: length(fwd_path)}) AS forward

WITH [{para: hit, offset: 0}] + backward + forward AS all_entries
UNWIND all_entries AS e
WITH e.para AS para, e.offset AS offset
WHERE para IS NOT NULL

RETURN para.nodeID AS paragraph_nodeID,
       offset,
       para.text    AS text
ORDER BY offset
"""


@tool(parse_docstring=True)
def window_paragraphs(paragraph_id: str, n: int = 2) -> str:
    """Given a Paragraph nodeID, return it plus up to n previous and n next
    Paragraphs along the NEXT_PARAGRAPH chain. Offset 0 is the hit; negative
    offsets are preceding, positive offsets are following.

    Args:
        paragraph_id: Paragraph nodeID.
        n: Window half-width (default 2, capped at 20).
    """
    if not isinstance(paragraph_id, str):
        return json.dumps({"error": "paragraph_id must be a string"})
    capped_n = max(0, min(int(n), 20))
    try:
        rows = run_read(
            CYPHER,
            params={"paragraph_id": paragraph_id, "n": capped_n},
            limit=2 * capped_n + 1,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"centre_paragraph": paragraph_id, "n": capped_n, "window": rows},
        default=str,
    )
