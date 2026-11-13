"""The agent's four operations over one home: what the MCP server exposes (memo §3.9).

    list_kbs()                     usable kbs: id and description only
    describe_sources(kb_id)        sources, file types, dates, freshness
    search(kb_id, query, ...)      hybrid, chunks, at most 3 per document
    read(kb_id, chunk_id, ...)     a passage widened to its neighbours, section or document

Read-only. Results are the library's `kb search` and `kb read`, projected to what an
agent needs and stripped of every path (`smartrag.agent.models`). Errors are the
library's `SmartRAGError`s; `smartrag.agent.errors` turns them into what an agent sees.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from datetime import datetime

from smartrag.agent.catalog import AgentCatalog
from smartrag.agent.embedders import QueryEmbedders
from smartrag.agent.models import (
    AgentCitation,
    AgentHit,
    AgentKbList,
    AgentReadChunk,
    AgentReadResponse,
    AgentSearchResponse,
    KbDescription,
    ReadScope,
)
from smartrag.errors import InvalidRequest
from smartrag.kb.home import Home
from smartrag.search import KbCitation, SearchFilter, read_chunk, search_kb

DEGRADED = "The embedding service is unavailable; these are keyword results only."
READ_WINDOW = 1
"""Chunks on each side for `scope=neighbors`: the library's default."""
READ_MAX_CHARS = 8000
"""Characters per page of a read: the Web API's default."""

_ABSOLUTE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]|~[\\/])")


def safe_title(title: str) -> str:
    """A document title without a path. Office and PDF files often carry the full path
    they were saved under as their title (`C:\\Users\\tanaka\\見積.docx`): keep its last
    part, the file name, as the index already does for untitled files."""
    if _ABSOLUTE.match(title) or "\\" in title:
        return re.split(r"[\\/]", title.rstrip("\\/"))[-1] or title
    return title


def _citation(citation: KbCitation) -> AgentCitation:
    return AgentCitation(source_id=citation.source_id, page=citation.page,
                         page_end=citation.page_end, location=citation.location,
                         heading_path=citation.heading_path)


def _moment(value: str | None, name: str) -> datetime | None:
    if value is None or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise InvalidRequest(f"{name} must be a date (YYYY-MM-DD) or an ISO datetime, "
                             f"not {value!r}") from None


class AgentService:
    def __init__(self, home: Home, allow: Collection[str] | None = None,
                 embedders: QueryEmbedders | None = None) -> None:
        self.home = home
        self.catalog = AgentCatalog(home, allow)
        self.embedders = embedders or QueryEmbedders()

    @property
    def visible(self) -> tuple[str, ...]:
        """The kbs offered since startup: the server instructions and the `kb_id` enum."""
        return self.catalog.visible

    def list_kbs(self) -> AgentKbList:
        return AgentKbList(kbs=self.catalog.kbs())

    def describe_sources(self, kb_id: str) -> KbDescription:
        return self.catalog.describe(kb_id)

    def search(self, kb_id: str, query: str, *, k: int = 10, cursor: str | None = None,
               source: Sequence[str] = (), file_type: Sequence[str] = (),
               modified_after: str | None = None,
               modified_before: str | None = None) -> AgentSearchResponse:
        self.catalog.require(kb_id)
        config = self.home.load(kb_id)
        assert config.embedding is not None  # a usable kb is local
        filters = SearchFilter(source=list(source), file_type=list(file_type),
                               modified_after=_moment(modified_after, "modified_after"),
                               modified_before=_moment(modified_before, "modified_before"))
        r = search_kb(self.home, kb_id, query, limit=k, cursor=cursor, filters=filters,
                      embedder=self.embedders.get(kb_id, config.embedding))
        return AgentSearchResponse(
            kb_id=kb_id, query_id=r.query_id, query=r.query,
            degraded=DEGRADED if r.degraded else None,
            results=[AgentHit(rank=h.rank, chunk_id=h.chunk_id, doc_id=h.doc_id,
                              title=safe_title(h.title), snippet=h.snippet,
                              citation=_citation(h.citation), indexed_at=h.indexed_at,
                              file_type=h.metadata.get("file_type", ""))
                     for h in r.results],
            next_cursor=r.next_cursor)

    def read(self, kb_id: str, chunk_id: str, *, scope: ReadScope = "chunk",
             cursor: str | None = None) -> AgentReadResponse:
        self.catalog.require(kb_id)
        r = read_chunk(self.home, kb_id, chunk_id, scope=scope, window=READ_WINDOW,
                       max_chars=READ_MAX_CHARS, cursor=cursor)
        return AgentReadResponse(
            kb_id=kb_id, chunk_id=r.chunk_id, doc_id=r.doc_id, scope=r.scope,
            title=safe_title(r.title), citation=_citation(r.citation), indexed_at=r.indexed_at,
            chunks=[AgentReadChunk(chunk_id=c.chunk_id, ordinal=c.ordinal, text=c.text,
                                   prefix=c.prefix, heading_path=c.heading_path, page=c.page,
                                   page_end=c.page_end, location=c.location)
                    for c in r.chunks],
            total_chunks=r.total_chunks, truncated=r.truncated, page=r.page,
            next_cursor=r.next_cursor)
