"""What an agent receives (design memo §3.9).

These are projections of the library's models, not new contracts: the same search and
read, with everything an agent does not need, or must not see, left out. In particular
no path of any kind: a citation names the source, the title and the place in the
document, and Kari resolves `(kb_id, doc_id)` to the file when the user opens it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ReadScope = Literal["chunk", "neighbors", "section", "document"]
AgentSearchMode = Literal["hybrid", "keyword"]


class AgentKb(BaseModel):
    kb_id: str = Field(description="The knowledge base's id: pass it to the other tools.")
    description: str = Field(description="What the knowledge base holds, and what it does not.")


class AgentKbList(BaseModel):
    kbs: list[AgentKb]


Attribute = str | int | float | bool | None


class AttributeInfo(BaseModel):
    name: str
    type: str | None = Field(default=None, description="string, number, integer or boolean.")
    format: str | None = Field(default=None, description="date (YYYY-MM-DD), date-time or uri.")
    description: str | None = Field(default=None,
                                    description="What it means, from the documents' publisher.")


class SourceSummary(BaseModel):
    source_id: str
    documents: int
    file_types: dict[str, int] = Field(description="Documents per file type.")
    modified_from: str | None = Field(default=None, description="Oldest file date, YYYY-MM-DD.")
    modified_to: str | None = Field(default=None, description="Newest file date, YYYY-MM-DD.")
    attributes: list[AttributeInfo] = Field(
        default_factory=list, description="Document attributes search and read results carry "
                                          "for this source, and what they mean.")


class FilterInfo(BaseModel):
    name: str
    values: list[str] | None = Field(default=None, description="The values this kb holds.")
    format: str | None = None


class KbDescription(BaseModel):
    kb_id: str
    description: str
    as_of: str | None = Field(default=None, description="When the kb was last synced.")
    documents: int
    sources: list[SourceSummary]
    filters: list[FilterInfo] = Field(description="The filters `search` accepts.")


class AgentCitation(BaseModel):
    source_id: str
    file_path: str | None = Field(
        default=None, description="The file's full path, for the user to open or find it. "
                                  "Not given when this knowledge base hides paths.")
    page: int | None = None
    page_end: int | None = None
    location: str | None = Field(default=None,
                                 description="A cell range or slide part, e.g. 見積!B4:E40.")
    heading_path: list[str] = Field(default_factory=list,
                                    description="The section the passage sits under.")
    link: str | None = Field(default=None, description="The document's original web page, "
                                                       "from its publisher. Not always given.")


class AgentHit(BaseModel):
    rank: int
    chunk_id: str = Field(description="Pass to `read` to widen this passage.")
    doc_id: str
    title: str
    snippet: str
    citation: AgentCitation
    indexed_at: str = Field(description="When the document was last synced.")
    file_type: str
    attributes: dict[str, Attribute] = Field(
        default_factory=dict, description="Facts about the document from its publisher, e.g. "
                                          "a court or a date; describe_sources says what each "
                                          "means.")


class AgentSearchResponse(BaseModel):
    kb_id: str
    query_id: str
    query: str
    search_mode: AgentSearchMode = Field(
        description="`hybrid`: keyword and vector search. `keyword`: vector search could not "
                    "be used (the embedding service was unavailable), keyword search only.")
    results: list[AgentHit]
    next_cursor: str | None = None


class AgentReadChunk(BaseModel):
    chunk_id: str
    ordinal: int
    text: str
    prefix: str = Field(default="", description="A table's header row, for row chunks.")
    heading_path: list[str] = Field(default_factory=list)
    page: int | None = None
    page_end: int | None = None
    location: str | None = None


class AgentReadResponse(BaseModel):
    kb_id: str
    chunk_id: str
    doc_id: str
    scope: ReadScope
    title: str
    citation: AgentCitation
    attributes: dict[str, Attribute] = Field(default_factory=dict)
    indexed_at: str
    chunks: list[AgentReadChunk]
    total_chunks: int = Field(description="Chunks in the whole scope, across all pages.")
    truncated: bool = False
    page: int
    next_cursor: str | None = None


class AgentError(BaseModel):
    code: str
    message: str
    hint: str
    details: dict[str, str] = Field(default_factory=dict)
