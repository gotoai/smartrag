"""The retrieval-engine seam: what the query-plane API needs from an index.

The Web API depends only on this protocol. The v1 engine (PostgreSQL + pgvector +
PGroonga, query embeddings from OumiGo's `/v1/embeddings`) and the in-memory demo engine
both implement it, so the API never changes when the engine does.

Every call receives a `RequestContext`: who is calling (the API key) and, optionally, on
whose behalf (the end user, CommonRAG P2). v1 engines only log `on_behalf_of`; document
permissions will use it later without an interface change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from smartrag.models import (
    DocumentResponse,
    FeedbackRequest,
    ReadinessResponse,
    ReadResponse,
    ReadScope,
    SearchRequest,
    SearchResponse,
    SourcesResponse,
)


@dataclass(frozen=True)
class RequestContext:
    key_id: str
    on_behalf_of: str | None = None


class RetrievalEngine(Protocol):
    async def describe_sources(self, ctx: RequestContext) -> SourcesResponse:
        """What is indexed, which fields can be filtered, how fresh it is."""
        ...

    async def search(
        self, request: SearchRequest, ctx: RequestContext, query_id: str
    ) -> SearchResponse:
        """Ranked evidence for a natural-language query. `query_id` is issued by the
        API; the engine records it so feedback can refer to this search."""
        ...

    async def read_chunk(
        self,
        chunk_id: str,
        scope: ReadScope,
        window: int,
        max_chars: int,
        cursor: str | None,
        ctx: RequestContext,
    ) -> ReadResponse:
        """A chunk widened to `scope`. Raises `NotFound`, or `ChunkSuperseded` for a
        chunk of an older document version."""
        ...

    async def get_document(
        self,
        doc_id: str,
        include_text: bool,
        section: str | None,
        max_chars: int,
        cursor: str | None,
        ctx: RequestContext,
    ) -> DocumentResponse:
        """Document metadata and outline; text (paged) with `include_text`."""
        ...

    async def record_feedback(self, feedback: FeedbackRequest, ctx: RequestContext) -> None:
        """Store a relevance signal. Raises `NotFound` for an unknown `query_id`."""
        ...

    async def readiness(self) -> ReadinessResponse:
        """Whether the index and the embedder can serve queries."""
        ...
