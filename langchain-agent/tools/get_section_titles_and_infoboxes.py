"""Return the Section titles (and Section nodeIDs) of an Article in order.
Along with the full text of any infoboxes in the article."""
from __future__ import annotations

import json

from langchain_core.tools import tool

from .neo4j_client import run_read

CYPHER = """
MATCH (start:Article {title: $title})
OPTIONAL MATCH (start)-[:REDIRECTS_TO*1..10]->(t:Article)
WHERE NOT (t)-[:REDIRECTS_TO]->()
WITH coalesce(t, start) AS canonical

CALL (canonical) {
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
      AND NOT first_para.text CONTAINS '{{Infobox'
  WITH section, section_idx, fallback_chunk,
       [line IN split(first_para.text, '\n')
        WHERE line =~ '\\s*=+[^=]+=+\\s*'
        | trim(replace(line, '=', ''))] AS heading_titles
  WITH section, section_idx,
       CASE
         WHEN size(heading_titles) > 0 THEN heading_titles
         WHEN fallback_chunk IS NOT NULL THEN [fallback_chunk.text]
         ELSE []
       END AS titles
  UNWIND range(0, size(titles) - 1) AS title_idx
  WITH section_idx, title_idx,
       {nodeID: section.nodeID, title: titles[title_idx]} AS row
  ORDER BY section_idx, title_idx
  RETURN collect(row) AS sections
}

CALL (canonical) {
  MATCH (canonical)-[:HAS_SECTION]->(first_section:Section)
  WHERE NOT (:Section)-[:NEXT_SECTION]->(first_section)
  MATCH section_path = (first_section)-[:NEXT_SECTION*0..]->(section:Section)
  WITH section, length(section_path) AS section_idx
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
  WITH doc, split(doc, "{{Infobox") AS parts
  UNWIND range(1, size(parts) - 1) AS k
  WITH doc, parts, k,
       reduce(off = 0, j IN range(0, k-1) | off + size(parts[j])) + 9 * (k - 1) AS start
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
  WITH start, substring(doc, start, endOffset) AS infobox
  ORDER BY start
  RETURN collect(infobox) AS infoboxes
}

UNWIND (
  [ib IN infoboxes | {kind: 'infobox',       nodeID: NULL,      text: ib}] +
  [s  IN sections  | {kind: 'section title', nodeID: s.nodeID,  text: s.title}]
) AS row
RETURN row.kind AS kind, row.nodeID AS nodeID, row.text AS text
"""


@tool(parse_docstring=True)
def get_section_titles_and_infoboxes(title: str) -> str:
    """Return infoboxes and list all the Section titles of an Article in order.
    You'll get the text of all infoboxes in order, followed by a list of section titles
    with nodeID. For any sections that don't have a title, you will get the first text chunk of
    the section instead.
    Returns kind (infobox or section title), nodeID (for sections), and text
    (full infobox or section title/initial chunk). Use this to get key facts and
    to decide which if any Section(s) to fetch in detail via get_sections.

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
        {"article": title,
         "output_count": len(rows),
         "infoboxes_and_section_titles": rows},
        default=str,
    )
