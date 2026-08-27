"""Singleton Neo4j driver + read-only session helpers."""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase

_driver: Driver | None = None


def set_driver(driver: Driver) -> None:
    """Inject an externally-managed driver (e.g. the one the agent already owns)."""
    global _driver
    _driver = driver


def get_driver() -> Driver:
    global _driver
    if _driver is not None:
        return _driver
    load_dotenv()
    uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    _driver = GraphDatabase.driver(uri, auth=(user, password))
    _driver.verify_connectivity()
    return _driver


def get_database() -> str:
    load_dotenv()
    return os.getenv("NEO4J_DATABASE", "neo4j")


def run_read(
    cypher: str,
    params: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    driver = get_driver()
    with driver.session(database=get_database(), default_access_mode="READ") as session:
        result = session.run(cypher, parameters=params or {})
        rows: list[dict[str, Any]] = []
        for i, record in enumerate(result):
            if i >= limit:
                break
            rows.append(_serialize_record(record.data()))
        return rows


def _serialize_record(data: dict[str, Any]) -> dict[str, Any]:
    return {k: _serialize(v) for k, v in data.items()}


def _serialize(value: Any) -> Any:
    from neo4j.graph import Node, Path, Relationship

    if isinstance(value, Node):
        return {
            "_type": "node",
            "id": value.element_id,
            "labels": list(value.labels),
            "properties": dict(value),
        }
    if isinstance(value, Relationship):
        return {
            "_type": "relationship",
            "id": value.element_id,
            "type": value.type,
            "start": value.start_node.element_id if value.start_node else None,
            "end": value.end_node.element_id if value.end_node else None,
            "properties": dict(value),
        }
    if isinstance(value, Path):
        return {
            "_type": "path",
            "nodes": [_serialize(n) for n in value.nodes],
            "relationships": [_serialize(r) for r in value.relationships],
        }
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


def close() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None
