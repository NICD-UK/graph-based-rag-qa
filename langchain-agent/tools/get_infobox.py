"""Fetch all the infoboxes of an Article by title, in order."""
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
WITH section_idx, length(para_path) AS para_idx, para
ORDER BY section_idx, para_idx

WITH collect(DISTINCT para.text) AS texts
WITH reduce(s = "", t IN texts |
       CASE WHEN s = "" THEN t ELSE s + "\n" + t END) AS doc

// Locate each "{{Infobox" via split — k-th occurrence is at
// sum(sizes of preceding parts) + 9 * (k-1)
WITH doc, split(doc, "{{Infobox") AS parts
UNWIND range(1, size(parts) - 1) AS k
WITH doc, parts, k,
     reduce(off = 0, j IN range(0, k-1) | off + size(parts[j])) + 9 * (k - 1) AS start

// Walk forward by "}}" segments, counting "{{" opens, until depth returns to 0
WITH doc, start, split(substring(doc, start), "}}") AS segs
WITH doc, start, segs,
     reduce(st = {idx:-1, opens:0, found:false}, k IN range(0, size(segs) - 2) |
       CASE
         WHEN st.found
              THEN st
         WHEN st.opens + size(split(segs[k], "{{")) - 1 = k + 1
              THEN {idx:k, opens: st.opens + size(split(segs[k], "{{")) - 1, found:true}
         ELSE {idx:-1, opens: st.opens + size(split(segs[k], "{{")) - 1, found:false}
       END) AS r
WHERE r.found
WITH doc, start, segs, r,
     reduce(s = 0, k IN range(0, r.idx) | s + size(segs[k])) + 2 * (r.idx + 1) AS endOffset
RETURN substring(doc, start, endOffset) AS infobox
ORDER BY start
"""


@tool(parse_docstring=True)
def get_infoboxes(title: str) -> str:
    """Given an Article title, return all its Infoboxes in order.

    Args:
        title: Article title to fetch.
    """
    if not isinstance(title, str):
        return json.dumps({"error": "title must be a string"})
    try:
        rows = run_read(CYPHER, params={"title": title}, limit=5000)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(
        {"article": title, "infobox_count": len(rows), "infoboxes": rows},
        default=str,
    )
