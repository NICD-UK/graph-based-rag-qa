"""Return every Paragraph of an Article in reading order."""
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
WITH canonical, section, length(section_path) AS section_idx

MATCH (section)-[:HAS_PARAGRAPH]->(first_para:Paragraph)
WHERE NOT EXISTS {
  (section)-[:HAS_PARAGRAPH]->(:Paragraph)-[:NEXT_PARAGRAPH]->(first_para)
}

MATCH para_path = (first_para)-[:NEXT_PARAGRAPH*0..]->(para:Paragraph)
WHERE (section)-[:HAS_PARAGRAPH]->(para)

RETURN para.nodeID AS nodeID,
       para.text   AS text
ORDER BY section_idx, length(para_path)
"""


@tool(parse_docstring=True)
def get_article_text(title: str) -> str:
    """Return all Paragraphs of an Article (resolved via REDIRECTS_TO) in reading
    order (section-by-section, paragraph-by-paragraph). Returns paragraph_nodeID
    and text.

    Args:
        title: Article title.
    """
    if not isinstance(title, str):
        return json.dumps({"error": "title must be a string"})
    try:
        rows = run_read(CYPHER, params={"title": title}, limit=5000)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"title": title, "paragraph_count": len(rows), "paragraphs": rows},
        default=str,
    )
