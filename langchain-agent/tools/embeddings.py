"""Embed query text locally with sentence-transformers.

The model and prompt-name used to embed *queries* are configurable via the
``AGENT_EMBEDDING_MODEL`` and ``AGENT_EMBEDDING_QUERY_PROMPT`` env vars. Both
must match the model / prompt used when the graph was embedded, otherwise
cosine similarity across the vector index will be meaningless.

Callers may also plug in their own embedder with :func:`set_embedder` (e.g. if
the agent already loaded a :class:`SentenceTransformer` and wants to reuse
the model weights).
"""
from __future__ import annotations

import os
from typing import Callable

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "microsoft/harrier-oss-v1-0.6b"
DEFAULT_QUERY_PROMPT = "sts_query"

_model: SentenceTransformer | None = None
_embedder: Callable[[str], list[float]] | None = None


def set_embedder(fn: Callable[[str], list[float]]) -> None:
    """Inject a custom single-text embedding function (returns a list of floats)."""
    global _embedder
    _embedder = fn


def _get_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model
    load_dotenv()
    name = os.getenv("AGENT_EMBEDDING_MODEL", DEFAULT_MODEL)
    _model = SentenceTransformer(name, trust_remote_code=True, device="cpu")
    return _model


def embed(text: str) -> list[float]:
    if _embedder is not None:
        return list(_embedder(text))
    load_dotenv()
    prompt_name = os.getenv("AGENT_EMBEDDING_QUERY_PROMPT", DEFAULT_QUERY_PROMPT) or None
    model = _get_model()
    kwargs = {"prompt_name": prompt_name} if prompt_name else {}
    vec = model.encode(text, **kwargs)
    return vec.tolist()
