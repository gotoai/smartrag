"""Request/response schemas of the query plane — the contract agents, the UI and MCP share.

These models are both the Web API's wire format and what a `RetrievalEngine` returns,
so the contract is defined once. Request models forbid unknown fields: a typo such as
`filter` for `filters` must fail loudly, not silently search without the filter.

Ids are stable (CommonRAG P3):

* `doc_id` — never changes for one file in one source.
* `chunk_id` — `<doc_id>:v<version>:c<ordinal>`; tied to one version of the document,
  so a chunk from before a re-ingestion is answered with `410 chunk_superseded`.

There is no `answer` field anywhere: Smart RAG returns evidence, never the answer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --- shared -----------------------------------------------------------------------


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Citation(BaseModel):
    """Where a passage came from — enough for any agent to cite traceably.

    `path` is relative to the source's root: it shows folders, never hosts or buckets
    (the network layout stays private). `link` points at the document through this API.
    """

    source_id: str
    path: str
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)
    link: str | None = Field(default=None, description="Filled in by the API layer.")


class IndexInfo(BaseModel):
    """Which embedding model produced the vectors (an OumiGo profile id, e.g.
    `ruri-v3-310m@8k`). Results from a different model are a different vector space."""

    embedding_model: str | None = None
    languages: list[str] = Field(default_factory=list)


# --- 1. GET /v1/sources -----------------------------------------------------------


class SourceInfo(BaseModel):
    """One indexed source. Deliberately no URI or credential reference."""

    source_id: str
    title: str
    type: str = Field(description="Connector type: local | s3 | gcs | smb | ...")
    documents: int
    chunks: int
    last_synced_at: datetime | None = None
    status: Literal["ready", "syncing", "error", "empty"]


class FieldInfo(BaseModel):
    """A field `search` can filter on (schema introspection for agents)."""

    name: str
    type: Literal["keyword", "text", "path", "datetime", "number"]
    values: list[str] | None = Field(
        default=None, description="Known values, for keyword fields with few of them.")


class SourcesResponse(BaseModel):
    sources: list[SourceInfo]
    fields: list[FieldInfo]
    index: IndexInfo


# --- 2/3. search ------------------------------------------------------------------


class DateRange(_Request):
    gte: datetime | None = None
    lte: datetime | None = None


class SearchFilters(_Request):
    """v1 filters: lists mean OR, `modified_at` is a range, `path_prefix` a folder.

    No query language in v1. Custom metadata fields will extend this model once
    ingestion records them (they will then also appear in `GET /v1/sources`).
    """

    source_id: list[str] | None = None
    file_type: list[str] | None = None
    modified_at: DateRange | None = None
    path_prefix: str | None = None


SearchMode = Literal["hybrid", "vector", "lexical"]
GroupBy = Literal["chunk", "document"]
SearchInclude = Literal["highlights", "facets", "text"]


class SearchRequest(_Request):
    query: str = Field(min_length=1, max_length=2000,
                       description="Natural language. The server adds the embedding "
                                   "model's query prefix; clients never send prefixes.")
    filters: SearchFilters | None = None
    limit: int = Field(default=10, ge=1, le=50)
    cursor: str | None = Field(default=None, description="`next_cursor` of a previous page.")
    mode: SearchMode = "hybrid"
    group_by: GroupBy = Field(
        default="chunk",
        description="`chunk`: passage-level evidence (agents). `document`: one result "
                    "per file (the search box).")
    include: list[SearchInclude] = Field(default_factory=list)
    snippet_chars: int = Field(default=300, ge=50, le=2000)


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    title: str
    snippet: str
    highlights: list[tuple[int, int]] | None = Field(
        default=None, description="[start, end) character offsets into `snippet`.")
    text: str | None = Field(default=None, description="Full chunk text, with include=text.")
    citation: Citation
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query_id: str = Field(description="Ties feedback and the server-side query log to "
                                      "this search.")
    results: list[SearchResult]
    facets: dict[str, dict[str, int]] | None = None
    next_cursor: str | None = None
    index: IndexInfo
    took_ms: int


# --- 4. GET /v1/chunks/{chunk_id} -------------------------------------------------

ReadScope = Literal["chunk", "neighbors", "section", "document"]


class ChunkView(BaseModel):
    chunk_id: str
    text: str
    page: int | None = None
    heading_path: list[str] = Field(default_factory=list)


class ReadResponse(BaseModel):
    """The anchor chunk widened to `scope`, paged at chunk boundaries by `max_chars`."""

    chunk_id: str
    doc_id: str
    version: int
    scope: ReadScope
    title: str
    citation: Citation
    chunks: list[ChunkView]
    truncated: bool = Field(
        default=False, description="A single chunk was longer than max_chars and was cut.")
    next_cursor: str | None = None


# --- 5. GET /v1/documents/{doc_id} ------------------------------------------------


class OutlineNode(BaseModel):
    section_id: str
    heading: str
    level: int = Field(ge=1)
    page: int | None = None


class DocumentResponse(BaseModel):
    doc_id: str
    version: int
    source_id: str
    title: str
    path: str
    file_type: str
    modified_at: datetime | None = None
    link: str | None = Field(default=None, description="Filled in by the API layer.")
    outline: list[OutlineNode]
    chunks: list[ChunkView] | None = Field(default=None, description="With include=text.")
    truncated: bool = False
    next_cursor: str | None = None


# --- 6. POST /v1/feedback ---------------------------------------------------------


class FeedbackRequest(_Request):
    query_id: str
    chunk_id: str | None = None
    signal: Literal["click", "relevant", "irrelevant"]
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    accepted: bool = True


# --- 13. ops ----------------------------------------------------------------------


class CheckResult(BaseModel):
    ok: bool
    detail: str = ""


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, CheckResult]


class VersionResponse(BaseModel):
    name: str = "smartrag"
    version: str
    api: str = "v1"
