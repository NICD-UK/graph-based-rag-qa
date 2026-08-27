"""Return the Section titles (and Section nodeIDs) of an Article in order."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read

CYPHER = """
MATCH (start:Article {title: $title})
OPTIONAL MATCH (start)-[:REDIRECTS_TO*1..10]->(t:Article)
WHERE NOT (t)-[:REDIRECTS_TO]->()
WITH coalesce(t, start) AS canonical
MATCH (canonical)-[:HAS_SECTION]->(first_section:Section)
WHERE NOT (:Section)-[:NEXT_SECTION]->(first_section)
MATCH section_path = (first_section)-[:NEXT_SECTION*0..]->(section:Section)
WITH section, length(section_path) AS section_idx
MATCH (section)-[:HAS_PARAGRAPH]->(first_para:Paragraph)
WHERE NOT EXISTS {
  (section)-[:HAS_PARAGRAPH]->(:Paragraph)-[:NEXT_PARAGRAPH]->(first_para)
}
OPTIONAL MATCH (first_para)-[:HAS_CHUNK]->(fallback_chunk:Chunk)
  WHERE NOT (fallback_chunk)-[:PREVIOUS_CHUNK]->(:Chunk)
WITH section, section_idx, fallback_chunk,
     [line IN split(first_para.text, '\n')
      WHERE line =~ '\\s*=+[^=]+=+\\s*'
      | trim(replace(line, '=', ''))] AS heading_titles
WITH section, section_idx,
     CASE
       WHEN size(heading_titles) > 0 THEN heading_titles
       ELSE [fallback_chunk.text]
     END AS titles
UNWIND range(0, size(titles) - 1) AS title_idx
RETURN section.nodeID AS nodeID,
       titles[title_idx] AS title
ORDER BY section_idx, title_idx
"""


@tool(parse_docstring=True)
def get_section_titles(title: str) -> str:
    """List all the Section titles of an Article in order.
    For any sections that don't have a title, you will get the first text chunk of
    the section instead. You'll also get the text of the first infobox (if any)
    Returns section nodeID and parsed title/initial chunk. Use this to decide which Section(s)
    to fetch in detail via get_sections.

    Args:
        title: Article title.
    """
    if not isinstance(title, str):
        return json.dumps({"error": "title must be a string"})
    try:
        rows = run_read(CYPHER, params={"title": title}, limit=500)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"article": title, "section_count": len(rows), "sections": rows},
        default=str,
    )
