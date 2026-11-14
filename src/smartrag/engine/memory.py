"""An in-memory `RetrievalEngine` — for development, demos and API tests.

It implements the whole query-plane contract (filters, grouping, facets, highlights,
cursors, read scopes, stable/superseded ids, feedback) over a few documents held in
memory, so the API and a UI can be built before the real index exists.

Ranking is deliberately simple: character-bigram overlap after NFKC normalisation, which
works for Japanese without a word splitter. Every `search_mode` uses it (there are no vectors
here); the response's `index.embedding_model` is None to say so.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from smartrag.engine.base import RequestContext
from smartrag.errors import ChunkSuperseded, NotFound
from smartrag.models import (
    CheckResult,
    ChunkView,
    Citation,
    DocumentResponse,
    FeedbackRequest,
    FieldInfo,
    IndexInfo,
    OutlineNode,
    ReadinessResponse,
    ReadResponse,
    ReadScope,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SourceInfo,
    SourcesResponse,
)
from smartrag.paging import binding, decode_cursor, encode_cursor, search_binding

_CHUNK_ID_RE = re.compile(r"^(d_[0-9a-f]+):v(\d+):c(\d+)$")


@dataclass(frozen=True)
class Chunk:
    text: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None


@dataclass
class Document:
    source_id: str
    path: str  # relative to the source root
    title: str
    file_type: str
    version: int
    chunks: list[Chunk]
    modified_at: datetime | None = None

    @property
    def doc_id(self) -> str:
        """Stable for one file in one source, across versions."""
        digest = hashlib.sha256(f"{self.source_id}\0{self.path}".encode()).hexdigest()
        return f"d_{digest[:16]}"

    def chunk_id(self, ordinal: int) -> str:
        return f"{self.doc_id}:v{self.version}:c{ordinal:04d}"


@dataclass
class Source:
    source_id: str
    title: str
    type: str
    last_synced_at: datetime | None = None


@dataclass
class _Hit:
    doc: Document
    ordinal: int
    score: float


@dataclass
class InMemoryEngine:
    sources: list[Source]
    documents: list[Document]
    languages: list[str] = field(default_factory=lambda: ["ja", "en"])
    feedback: list[tuple[RequestContext, FeedbackRequest]] = field(default_factory=list)
    _query_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._by_id = {doc.doc_id: doc for doc in self.documents}
        known = {s.source_id for s in self.sources}
        unknown = {d.source_id for d in self.documents} - known
        if unknown:
            raise ValueError(f"documents reference unknown sources: {sorted(unknown)}")

    # --- describe_sources -----------------------------------------------------------

    async def describe_sources(self, ctx: RequestContext) -> SourcesResponse:
        infos = []
        for source in self.sources:
            docs = [d for d in self.documents if d.source_id == source.source_id]
            infos.append(SourceInfo(
                source_id=source.source_id,
                title=source.title,
                type=source.type,
                documents=len(docs),
                chunks=sum(len(d.chunks) for d in docs),
                last_synced_at=source.last_synced_at,
                status="ready" if docs else "empty",
            ))
        fields = [
            FieldInfo(name="source_id", type="keyword",
                      values=[s.source_id for s in self.sources]),
            FieldInfo(name="file_type", type="keyword",
                      values=sorted({d.file_type for d in self.documents})),
            FieldInfo(name="path", type="path"),
            FieldInfo(name="modified_at", type="datetime"),
            FieldInfo(name="title", type="text"),
        ]
        return SourcesResponse(sources=infos, fields=fields, index=self._index_info())

    # --- search -----------------------------------------------------------------

    async def search(
        self, request: SearchRequest, ctx: RequestContext, query_id: str
    ) -> SearchResponse:
        grams = _grams(request.query)
        hits: list[_Hit] = []
        for doc in self.documents:
            if not _matches(doc, request.filters):
                continue
            for ordinal, chunk in enumerate(doc.chunks):
                score = _score(grams, chunk.text + " " + " ".join(chunk.heading_path))
                if score > 0:
                    hits.append(_Hit(doc, ordinal, score))
        order = {doc.doc_id: i for i, doc in enumerate(self.documents)}
        hits.sort(key=lambda h: (-h.score, order[h.doc.doc_id], h.ordinal))
        if request.group_by == "document":  # keep each document's best chunk
            best: dict[str, _Hit] = {}
            for hit in hits:
                best.setdefault(hit.doc.doc_id, hit)
            hits = list(best.values())

        bound = search_binding(request)
        offset = decode_cursor(request.cursor, bound)
        page = hits[offset:offset + request.limit]
        more = offset + request.limit < len(hits)
        self._query_ids.add(query_id)

        include = set(request.include)
        results = [self._result(h, grams, request.snippet_chars, include) for h in page]
        facets = None
        if "facets" in include:
            facets = {"source_id": {}, "file_type": {}}
            for h in hits:
                for name, value in (("source_id", h.doc.source_id),
                                    ("file_type", h.doc.file_type)):
                    facets[name][value] = facets[name].get(value, 0) + 1
        return SearchResponse(
            query_id=query_id,
            results=results,
            facets=facets,
            next_cursor=encode_cursor(offset + request.limit, bound) if more else None,
            index=self._index_info(),
            took_ms=0,  # measured by the API layer
        )

    def _result(self, hit: _Hit, grams: set[str], snippet_chars: int,
                include: set[str]) -> SearchResult:
        doc, chunk = hit.doc, hit.doc.chunks[hit.ordinal]
        snippet = _snippet(chunk.text, grams, snippet_chars)
        return SearchResult(
            chunk_id=doc.chunk_id(hit.ordinal),
            doc_id=doc.doc_id,
            score=round(hit.score, 4),
            title=doc.title,
            snippet=snippet,
            highlights=_highlights(snippet, grams) if "highlights" in include else None,
            text=chunk.text if "text" in include else None,
            citation=_citation(doc, chunk),
            metadata={
                "file_type": doc.file_type,
                **({"modified_at": doc.modified_at.isoformat()} if doc.modified_at else {}),
            },
        )

    # --- read -------------------------------------------------------------------

    async def read_chunk(
        self,
        chunk_id: str,
        scope: ReadScope,
        window: int,
        max_chars: int,
        cursor: str | None,
        ctx: RequestContext,
    ) -> ReadResponse:
        doc, ordinal = self._resolve_chunk(chunk_id)
        anchor = doc.chunks[ordinal]
        if scope == "chunk":
            selected = [ordinal]
        elif scope == "neighbors":
            selected = list(range(max(0, ordinal - window),
                                  min(len(doc.chunks), ordinal + window + 1)))
        elif scope == "section":
            selected = [i for i, c in enumerate(doc.chunks)
                        if c.heading_path == anchor.heading_path]
        else:
            selected = list(range(len(doc.chunks)))

        bound = binding("read", chunk_id, scope, window)
        views, truncated, next_offset = _page(
            doc, selected, decode_cursor(cursor, bound), max_chars)
        return ReadResponse(
            chunk_id=chunk_id,
            doc_id=doc.doc_id,
            version=doc.version,
            scope=scope,
            title=doc.title,
            citation=_citation(doc, anchor),
            chunks=views,
            truncated=truncated,
            next_cursor=encode_cursor(next_offset, bound) if next_offset is not None else None,
        )

    async def get_document(
        self,
        doc_id: str,
        include_text: bool,
        section: str | None,
        max_chars: int,
        cursor: str | None,
        ctx: RequestContext,
    ) -> DocumentResponse:
        doc = self._by_id.get(doc_id)
        if doc is None:
            raise NotFound(f"document {doc_id!r} not found")
        outline, sections = _outline(doc)
        if section is not None and section not in sections:
            raise NotFound(f"section {section!r} not found in document {doc_id!r}",
                           {"doc_id": doc_id})

        views, truncated, next_cursor = None, False, None
        if include_text:
            prefix = sections.get(section, ()) if section else ()
            selected = [i for i, c in enumerate(doc.chunks)
                        if c.heading_path[:len(prefix)] == prefix]
            bound = binding("document", doc_id, doc.version, section)
            views, truncated, next_offset = _page(
                doc, selected, decode_cursor(cursor, bound), max_chars)
            if next_offset is not None:
                next_cursor = encode_cursor(next_offset, bound)
        return DocumentResponse(
            doc_id=doc.doc_id,
            version=doc.version,
            source_id=doc.source_id,
            title=doc.title,
            path=doc.path,
            file_type=doc.file_type,
            modified_at=doc.modified_at,
            outline=outline,
            chunks=views,
            truncated=truncated,
            next_cursor=next_cursor,
        )

    def _resolve_chunk(self, chunk_id: str) -> tuple[Document, int]:
        match = _CHUNK_ID_RE.match(chunk_id)
        doc = self._by_id.get(match.group(1)) if match else None
        if match is None or doc is None:
            raise NotFound(f"chunk {chunk_id!r} not found")
        version, ordinal = int(match.group(2)), int(match.group(3))
        if version != doc.version:
            raise ChunkSuperseded(
                f"chunk {chunk_id!r} is from version {version} of the document; "
                f"the current version is {doc.version}",
                {"doc_id": doc.doc_id, "current_version": doc.version})
        if ordinal >= len(doc.chunks):
            raise NotFound(f"chunk {chunk_id!r} not found")
        return doc, ordinal

    # --- feedback / readiness ---------------------------------------------------------

    async def record_feedback(self, feedback: FeedbackRequest, ctx: RequestContext) -> None:
        if feedback.query_id not in self._query_ids:
            raise NotFound(f"query {feedback.query_id!r} not found")
        self.feedback.append((ctx, feedback))

    async def readiness(self) -> ReadinessResponse:
        chunks = sum(len(d.chunks) for d in self.documents)
        checks = {
            "index": CheckResult(ok=True, detail=f"in-memory index, {chunks} chunks"),
            "embedder": CheckResult(ok=True, detail="not used by the in-memory engine"),
        }
        return ReadinessResponse(ready=all(c.ok for c in checks.values()), checks=checks)

    def _index_info(self) -> IndexInfo:
        return IndexInfo(embedding_model=None, languages=list(self.languages))


# --- helpers --------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).lower()


def _grams(text: str) -> set[str]:
    """Character bigrams of each alphanumeric run (CJK counts as alphanumeric);
    a one-character run contributes itself."""
    grams: set[str] = set()
    for run in re.findall(r"\w+", _normalize(text)):
        if len(run) == 1:
            grams.add(run)
        grams.update(run[i:i + 2] for i in range(len(run) - 1))
    return grams


def _score(query_grams: set[str], text: str) -> float:
    if not query_grams:
        return 0.0
    return len(query_grams & _grams(text)) / len(query_grams)


def _match_positions(text: str, grams: set[str]) -> list[int]:
    """Start offsets in `text` where a query bigram occurs (original-text offsets)."""
    positions = []
    for i in range(len(text) - 1):
        if _normalize(text[i]) + _normalize(text[i + 1]) in grams:
            positions.append(i)
    return positions


def _snippet(text: str, grams: set[str], chars: int) -> str:
    if len(text) <= chars:
        return text
    positions = _match_positions(text, grams)
    first = positions[0] if positions else 0
    start = max(0, min(first - chars // 3, len(text) - chars))
    body = text[start:start + chars]
    return ("…" if start > 0 else "") + body + ("…" if start + chars < len(text) else "")


def _highlights(snippet: str, grams: set[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for i in _match_positions(snippet, grams):
        if spans and i <= spans[-1][1]:
            spans[-1] = (spans[-1][0], i + 2)
        else:
            spans.append((i, i + 2))
    return spans


def _matches(doc: Document, filters: SearchFilters | None) -> bool:
    if filters is None:
        return True
    if filters.source_id and doc.source_id not in filters.source_id:
        return False
    if filters.file_type and doc.file_type not in filters.file_type:
        return False
    if filters.path_prefix and not doc.path.startswith(filters.path_prefix.lstrip("/")):
        return False
    if filters.modified_at:
        if doc.modified_at is None:
            return False
        if filters.modified_at.gte and doc.modified_at < filters.modified_at.gte:
            return False
        if filters.modified_at.lte and doc.modified_at > filters.modified_at.lte:
            return False
    return True


def _citation(doc: Document, chunk: Chunk) -> Citation:
    return Citation(source_id=doc.source_id, path=doc.path, page=chunk.page,
                    heading_path=list(chunk.heading_path))


def _page(doc: Document, selected: list[int], offset: int,
          max_chars: int) -> tuple[list[ChunkView], bool, int | None]:
    """Chunks from `selected[offset:]` up to `max_chars`, cut at chunk boundaries.

    Always returns at least one chunk; a single chunk longer than `max_chars` is cut and
    reported as truncated. Returns (views, truncated, next offset or None)."""
    views: list[ChunkView] = []
    used, truncated = 0, False
    position = offset
    while position < len(selected):
        ordinal = selected[position]
        chunk = doc.chunks[ordinal]
        text = chunk.text
        if views and used + len(text) > max_chars:
            break
        if len(text) > max_chars:
            text, truncated = text[:max_chars], True
        views.append(ChunkView(chunk_id=doc.chunk_id(ordinal), text=text, page=chunk.page,
                               heading_path=list(chunk.heading_path)))
        used += len(text)
        position += 1
    return views, truncated, position if position < len(selected) else None


def _outline(doc: Document) -> tuple[list[OutlineNode], dict[str, tuple[str, ...]]]:
    """The heading tree as a flat, ordered list, and section id -> heading path."""
    nodes: list[OutlineNode] = []
    sections: dict[str, tuple[str, ...]] = {}
    seen: set[tuple[str, ...]] = set()
    for chunk in doc.chunks:
        for depth in range(1, len(chunk.heading_path) + 1):
            prefix = chunk.heading_path[:depth]
            if prefix in seen:
                continue
            seen.add(prefix)
            section_id = f"s{len(nodes) + 1:03d}"
            sections[section_id] = prefix
            nodes.append(OutlineNode(section_id=section_id, heading=prefix[-1],
                                     level=depth, page=chunk.page))
    return nodes, sections
