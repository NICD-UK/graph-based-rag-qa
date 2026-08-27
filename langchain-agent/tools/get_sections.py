"""Fetch the paragraphs of one or more Sections by nodeID, in order."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read

CYPHER = """
WITH $section_ids AS ids
UNWIND range(0, size(ids) - 1) AS i
WITH i, ids[i] AS sid

MATCH (section:Section {nodeID: sid})
MATCH (section)-[:HAS_PARAGRAPH]->(first_para:Paragraph)
WHERE NOT EXISTS {
  (section)-[:HAS_PARAGRAPH]->(:Paragraph)-[:NEXT_PARAGRAPH]->(first_para)
}

MATCH para_path = (first_para)-[:NEXT_PARAGRAPH*0..]->(para:Paragraph)
WHERE (section)-[:HAS_PARAGRAPH]->(para)

RETURN section.nodeID   AS section_nodeID,
       para.nodeID       AS paragraph_nodeID,
       para.text         AS text
ORDER BY i, length(para_path)
"""


@tool(parse_docstring=True)
def get_sections(section_ids: list[str]) -> str:
    """Given a list of Section nodeIDs, return all their Paragraphs in order.
    Preserves the order of the input list (earlier IDs appear first).

    Args:
        section_ids: Section nodeIDs to fetch.
    """
    if not isinstance(section_ids, list) or not all(isinstance(s, str) for s in section_ids):
        return json.dumps({"error": "section_ids must be a list of strings"})
    try:
        rows = run_read(CYPHER, params={"section_ids": section_ids}, limit=5000)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {
            "section_count": len(section_ids),
            "paragraph_count": len(rows),
            "paragraphs": rows,
        },
        default=str,
    )
