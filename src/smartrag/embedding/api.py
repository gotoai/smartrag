"""What the rest of Smart RAG calls: one provider-neutral API.

Sync, search and the MCP server see only `Embedder` and `open_embedder`; which provider
answers, and how it wants queries and documents marked, stays behind them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from smartrag.chunking import estimate_tokens
from smartrag.embedding.engine import HttpEmbedder, Tokens
from smartrag.embedding.marking import Side
from smartrag.embedding.resolve import EmbeddingSpec, VectorSpace, resolve
from smartrag.kb.config import EmbeddingConfig

__all__ = ["Embedder", "EmbeddingSpec", "Side", "Tokens", "VectorSpace", "embed_counted",
           "open_embedder", "resolve"]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], side: Side) -> list[list[float]]: ...


def embed_counted(embedder: Embedder, texts: Sequence[str],
                  side: Side) -> tuple[list[list[float]], Tokens]:
    """Vectors and the tokens they used. An embedder that cannot count (one made for a
    test) gets Smart RAG's estimate of the texts."""
    counting = getattr(embedder, "embed_counted", None)
    if counting is not None:
        vectors, tokens = counting(texts, side)
        return vectors, tokens
    return (embedder.embed(texts, side),
            Tokens(estimated=sum(estimate_tokens(text) for text in texts)))


def open_embedder(config: EmbeddingConfig, *, environ: dict[str, str] | None = None,
                  timeout: float = 120.0, attempts: int = 3,
                  adaptive: bool = False) -> HttpEmbedder:
    """`adaptive` waits out rate limits (sync); without it a request fails fast (search)."""
    return HttpEmbedder(resolve(config), environ=environ, timeout=timeout, attempts=attempts,
                        adaptive=adaptive)
