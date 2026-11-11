"""`kb read`: a chunk widened to its neighbours, its section or its document.

`section` is the run of chunks around the anchor that sit under the same headings; for a
document with no headings, it is the whole document. Pages are cut at chunk boundaries
by `max_chars`; a single chunk longer than that is cut and marked `truncated`.

A chunk id that no longer exists while its document still does means the document was
re-synced and that passage changed: `chunk_superseded`, naming the document, so the
caller can search again rather than give up.
"""

from __future__ import annotations

import json
import sqlite3

from smartrag.errors import ChunkSuperseded, InvalidRequest, NotFound
from smartrag.kb.home import Home
from smartrag.paging import binding as request_binding
from smartrag.search.common import decode_cursor, encode_cursor, open_kb
from smartrag.search.models import KbCitation, KbReadResponse, ReadChunk, ReadScope

_CHUNK_SQL = (
    "SELECT c.id, c.chunk_id, c.doc_id, c.ordinal, c.prefix, c.heading_path, c.page, "
    "c.page_end, c.location, d.source_id, d.rel_path, d.title, d.indexed_at, "
    "{origin} AS origin_path, substr(ct.text, c.start + 1, c.end - c.start) AS text "
    "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
    "JOIN contents ct ON ct.content_hash = d.content_hash ")


def read_chunk(home: Home, kb_id: str, chunk_id: str, *, scope: ReadScope = "chunk",
               window: int = 1, max_chars: int = 4000, page: int | None = None,
               cursor: str | None = None) -> KbReadResponse:
    if not 0 <= window <= 20:
        raise InvalidRequest("window must be between 0 and 20")
    if not 200 <= max_chars <= 100_000:
        raise InvalidRequest("max_chars must be between 200 and 100000")
    if page is not None and cursor:
        raise InvalidRequest("give a page or a cursor, not both")
    if page is not None and page < 1:
        raise InvalidRequest("page must be 1 or more")
    kb = open_kb(home, kb_id, check_profile=False)
    try:
        select = _CHUNK_SQL.format(origin=kb.origin_column)
        anchor = kb.store.db.execute(select + "WHERE c.chunk_id = ?", (chunk_id,)).fetchone()
        if anchor is None:
            doc_id = chunk_id.split(":", 1)[0]
            if kb.store.db.execute("SELECT 1 FROM documents WHERE doc_id = ? AND status = 'ok'",
                                   (doc_id,)).fetchone():
                raise ChunkSuperseded(
                    "this chunk no longer exists: its document was re-synced and the passage "
                    "changed; search again", {"doc_id": doc_id})
            raise NotFound(f"kb {kb_id!r} has no chunk {chunk_id!r}", {"chunk_id": chunk_id})
        chunks = kb.store.db.execute(select + "WHERE c.doc_id = ? ORDER BY c.ordinal",
                                     (anchor["doc_id"],)).fetchall()
        at = next(i for i, row in enumerate(chunks) if row["chunk_id"] == chunk_id)
        selected = chunks[slice(*_scope_range(chunks, at, scope, window))]

        pages: list[list[sqlite3.Row]] = []
        size = 0
        for row in selected:
            length = len(row["text"])
            if pages and size + length > max_chars:
                pages.append([row])
                size = length
            elif pages:
                pages[-1].append(row)
                size += length
            else:
                pages.append([row])
                size = length
        bound = request_binding("kb-read", kb_id, chunk_id, scope, window, max_chars)
        generation = kb.store.generation()
        index = (decode_cursor(cursor, bound, generation) if cursor
                 else (page or 1) - 1)
        if index >= len(pages):
            raise InvalidRequest(f"there are {len(pages)} page(s) for this read")

        truncated = False
        views = []
        for row in pages[index]:
            text = str(row["text"])
            if len(text) > max_chars:
                text, truncated = text[:max_chars], True
            views.append(ReadChunk(
                chunk_id=row["chunk_id"], ordinal=row["ordinal"], text=text,
                prefix=row["prefix"] or "", heading_path=json.loads(row["heading_path"]),
                page=row["page"], page_end=row["page_end"], location=row["location"]))
        return KbReadResponse(
            kb=kb_id, chunk_id=chunk_id, doc_id=anchor["doc_id"], scope=scope,
            title=anchor["title"] or anchor["rel_path"].rsplit("/", 1)[-1],
            citation=KbCitation(source_id=anchor["source_id"], path=anchor["rel_path"],
                                page=anchor["page"], page_end=anchor["page_end"],
                                location=anchor["location"],
                                heading_path=json.loads(anchor["heading_path"]),
                                origin_path=anchor["origin_path"]),
            indexed_at=anchor["indexed_at"], chunks=views, total_chunks=len(selected),
            truncated=truncated, page=index + 1,
            next_cursor=encode_cursor(index + 1, bound, generation)
            if index + 1 < len(pages) else None)
    finally:
        kb.close()


def _scope_range(chunks: list[sqlite3.Row], at: int, scope: ReadScope, window: int) -> tuple[int, int]:
    if scope == "chunk":
        return at, at + 1
    if scope == "neighbors":
        return max(0, at - window), min(len(chunks), at + window + 1)
    if scope == "document":
        return 0, len(chunks)
    path = chunks[at]["heading_path"]
    if path == "[]":
        return 0, len(chunks)
    start, end = at, at + 1
    while start > 0 and chunks[start - 1]["heading_path"] == path:
        start -= 1
    while end < len(chunks) and chunks[end]["heading_path"] == path:
        end += 1
    return start, end
