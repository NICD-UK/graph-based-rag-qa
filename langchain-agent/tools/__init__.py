"""LangChain tools for querying the Neo4j Wikipedia graph.

Ported from the ``lm_studio/mcp_tools`` MCP server so the same read-only Cypher
queries can be wired into a ``langchain.agents.create_agent`` agent.

Typical use::

    from tools import all_tools, vector_search_tools, calculate_tools
    from langchain.agents import create_agent

    agent = create_agent(llm, all_tools, system_prompt=...)

If the caller already owns a configured ``neo4j.Driver`` or a query-embedding
function, pass them in via :func:`tools.neo4j_client.set_driver` and
:func:`tools.embeddings.set_embedder` before the first tool call to avoid
duplicate connections / model loads.
"""
from __future__ import annotations

from .articles_within_distance import articles_within_distance
from .calculate import calculate
from .get_article_text import get_article_text
from .get_backlinks import get_backlinks
from .get_infobox import get_infoboxes
from .get_section_titles import get_section_titles
from .get_section_titles_and_infoboxes import get_section_titles_and_infoboxes
from .get_sections import get_sections
from .shortest_path import shortest_path
from .vector_search_article import vector_search_article
from .vector_search_paragraph import vector_search_paragraph
from .window_paragraphs import window_paragraphs
from .window_sections import window_sections

all_tools = [
    vector_search_paragraph,
    vector_search_article,
    articles_within_distance,
    get_article_text,
    get_infoboxes,
    get_section_titles,
    get_section_titles_and_infoboxes,
    get_sections,
    window_paragraphs,
    window_sections,
    get_backlinks,
    shortest_path,
    calculate,
]

vector_search_tools = [
    vector_search_paragraph,
    vector_search_article,
]

calculate_tools = [
    calculate,
]

__all__ = [
    "all_tools",
    "vector_search_tools",
    "calculate_tools",
    "articles_within_distance",
    "calculate",
    "get_article_text",
    "get_backlinks",
    "get_infoboxes",
    "get_section_titles",
    "get_section_titles_and_infoboxes",
    "get_sections",
    "shortest_path",
    "vector_search_article",
    "vector_search_paragraph",
    "window_paragraphs",
    "window_sections",
]
