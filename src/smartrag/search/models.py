"""What `kb search` and `kb read` return (design memo §3.7).

These follow the Web API's search contract (`smartrag.models`) and add what a kb-aware
search needs: the kb, the search mode that actually ran, why it fell back if it did, each
result's per-side ranks and freshness, and page ranges and cell locations in citations.
When `serve` is rebuilt for kbs, it returns these same models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchMode = Literal["hybrid", "keyword", "vector"]
Attribute = str | int | float | bool | None
GroupBy = Literal["chunk", "document"]
ReadScope = Literal["chunk", "neighbors", "section", "document"]


class SearchFilter(BaseModel):
    """Lists mean OR; different fields combine with AND."""

    model_config = ConfigDict(extra="forbid")

    source: list[str] = Field(default_factory=list)
    file_type: list[str] = Field(default_factory=list)
    path_prefix: str | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None

    def is_empty(self) -> bool:
        return not (self.source or self.file_type or self.path_prefix
                    or self.modified_after or self.modified_before)


class KbCitation(BaseModel):
    """Where a passage came from. `path` is relative to the source root; `file_path` is
    the whole location, for opening the file."""

    source_id: str
    path: str
    file_path: str = Field(description="The file's absolute local path, or its URI for a "
                                       "source that is not a local folder.")
    page: int | None = None
    page_end: int | None = None
    location: str | None = Field(default=None,
                                 description="A cell range or slide part, e.g. 見積!B4:E40.")
    heading_path: list[str] = Field(default_factory=list)
    origin_path: str | None = Field(
        default=None, description="For a file ingested into the sink: where it was copied from.")
    link: str | None = Field(default=None, description="The document's original URL, from its "
                                                     "source's manifest (design memo §3.10).")


class SideRanks(BaseModel):
    keyword: int | None = None
    vector: int | None = None


class SearchHit(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    score: float = Field(description="Fused rank score; orders this list only.")
    ranks: SideRanks
    title: str
    snippet: str
    highlights: list[tuple[int, int]] | None = None
    text: str | None = None
    citation: KbCitation
    indexed_at: str
    metadata: dict[str, str] = Field(default_factory=dict)
    attributes: dict[str, Attribute] = Field(
        default_factory=dict, description="The document's manifest attributes, other than "
                                          "`title` and `link` (design memo §3.10).")


class IndexDescription(BaseModel):
    embedding_model: str | None = None
    languages: list[str] = Field(default_factory=list)


class KbSearchResponse(BaseModel):
    kb: str
    query_id: str
    query: str
    search_mode: SearchMode = Field(
        description="What actually ran: `keyword` when a hybrid search fell back.")
    fallback_reason: str | None = Field(
        default=None, description="Why a hybrid search fell back to keyword; only then set.")
    group_by: GroupBy
    results: list[SearchHit]
    page: int
    next_cursor: str | None = None
    index: IndexDescription
    took_ms: int


class ReadChunk(BaseModel):
    chunk_id: str
    ordinal: int
    text: str
    prefix: str = Field(default="", description="A table's header row, for row chunks.")
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    page_end: int | None = None
    location: str | None = None


class KbReadResponse(BaseModel):
    """The anchor chunk widened to `scope`, cut into pages at chunk boundaries."""

    kb: str
    chunk_id: str
    doc_id: str
    scope: ReadScope
    title: str
    citation: KbCitation
    attributes: dict[str, Attribute] = Field(default_factory=dict)
    indexed_at: str
    chunks: list[ReadChunk]
    total_chunks: int = Field(description="Chunks in the whole scope, across all pages.")
    truncated: bool = Field(default=False,
                            description="A single chunk was longer than max_chars and was cut.")
    page: int
    next_cursor: str | None = None
