"""Ensure vector indexes on Chunk.embedding and Article.embedding exist."""
from __future__ import annotations

from neo4j import Driver

from .neo4j_client import get_database, get_driver

CHUNK_INDEX_NAME = "chunk_embedding_index"
ARTICLE_INDEX_NAME = "article_embedding_index"

_ensured: dict[str, bool] = {}


def _index_exists(driver: Driver, name: str) -> bool:
    with driver.session(database=get_database(), default_access_mode="READ") as s:
        result = s.run("SHOW INDEXES YIELD name WHERE name = $name RETURN name", name=name)
        return result.single() is not None


def _create_vector_index(driver: Driver, name: str, label: str, dim: int) -> None:
    cypher = (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.embedding) "
        "OPTIONS { indexConfig: { "
        "`vector.dimensions`: $dim, "
        "`vector.similarity_function`: 'cosine' "
        "} }"
    )
    with driver.session(database=get_database()) as s:
        s.run(cypher, dim=int(dim))


def ensure_chunk_index(dim: int) -> str:
    if _ensured.get(CHUNK_INDEX_NAME):
        return CHUNK_INDEX_NAME
    driver = get_driver()
    if not _index_exists(driver, CHUNK_INDEX_NAME):
        _create_vector_index(driver, CHUNK_INDEX_NAME, "Chunk", dim)
    _ensured[CHUNK_INDEX_NAME] = True
    return CHUNK_INDEX_NAME


def ensure_article_index(dim: int) -> str:
    if _ensured.get(ARTICLE_INDEX_NAME):
        return ARTICLE_INDEX_NAME
    driver = get_driver()
    if not _index_exists(driver, ARTICLE_INDEX_NAME):
        _create_vector_index(driver, ARTICLE_INDEX_NAME, "Article", dim)
    _ensured[ARTICLE_INDEX_NAME] = True
    return ARTICLE_INDEX_NAME
