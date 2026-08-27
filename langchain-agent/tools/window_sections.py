"""Return a window of Sections centred on a given Section (+/- n) with first paragraph."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read

CYPHER = """
MATCH (hit:Section {nodeID: $section_id})

OPTIONAL MATCH back_path = (hit)<-[:NEXT_SECTION*1..]-(prev:Section)
WHERE length(back_path) <= $n
WITH hit, collect({section: prev, offset: -length(back_path)}) AS backward

OPTIONAL MATCH fwd_path = (hit)-[:NEXT_SECTION*1..]->(next:Section)
WHERE length(fwd_path) <= $n
WITH hit, backward, collect({section: next, offset: length(fwd_path)}) AS forward

WITH [{section: hit, offset: 0}] + backward + forward AS all_entries
UNWIND all_entries AS e
WITH e.section AS section, e.offset AS offset
WHERE section IS NOT NULL

OPTIONAL MATCH (section)-[:HAS_PARAGRAPH]->(first_para:Paragraph)
WHERE NOT EXISTS {
  (section)-[:HAS_PARAGRAPH]->(:Paragraph)-[:NEXT_PARAGRAPH]->(first_para)
}

OPTIONAL MATCH (section)-[:HAS_PARAGRAPH]->(p:Paragraph)

RETURN section.nodeID     AS section_nodeID,
       offset,
       first_para.nodeID  AS first_paragraph_nodeID,
       first_para.text    AS first_paragraph_text,
       count(p)           AS paragraph_count
ORDER BY offset
"""


@tool(parse_docstring=True)
def window_sections(section_id: str, n: int = 2) -> str:
    """Given a Section nodeID, return it plus up to n previous and n next Sections
    along the NEXT_SECTION chain. Each row includes the section's first paragraph
    (typically the heading) and paragraph count. Offsets work like window_paragraphs.

    Args:
        section_id: Section nodeID.
        n: Window half-width (default 2, capped at 20).
    """
    if not isinstance(section_id, str):
        return json.dumps({"error": "section_id must be a string"})
    capped_n = max(0, min(int(n), 20))
    try:
        rows = run_read(
            CYPHER,
            params={"section_id": section_id, "n": capped_n},
            limit=2 * capped_n + 1,
        )
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"centre_section": section_id, "n": capped_n, "window": rows},
        default=str,
    )
