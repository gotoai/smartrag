"""`kb search`: hybrid keyword + vector search over one kb (design memo §3.7).

1. Keyword side: BM25 over the bigram FTS index (headings weigh 2, body 1, path 0.5).
2. Vector side: the query embedded as a query, cosine against the normalised chunk vectors.
3. Reciprocal rank fusion (k = 60) of the two candidate lists.
4. At most `max_per_doc` chunks per document, or one per document for `group_by=document`.
5. The requested page, with snippets around the best keyword match.

If the embedding endpoint cannot be reached, a hybrid search returns the keyword results
alone and says so in `degraded`; a vector-only search fails. Results are evidence only:
there is no answer field.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from collections.abc import Iterable, Sequence

import numpy as np

from smartrag.embedding import Embedder, HttpEmbedder
from smartrag.errors import ConfigError, EmbedderUnavailable, InvalidRequest
from smartrag.index import fts_query
from smartrag.kb.home import Home
from smartrag.paging import binding as request_binding
from smartrag.search.common import OpenKb, decode_cursor, encode_cursor, filter_sql, open_kb
from smartrag.search.models import (
    GroupBy,
    IndexDescription,
    KbCitation,
    KbSearchResponse,
    SearchFilter,
    SearchHit,
    SearchMode,
    SideRanks,
)
from smartrag.search.vectors import QueryVectorCache, load_vectors

DEPTH_CAP = 200
RRF_K = 60
BASE_POOL = 50
MAX_QUERY_CHARS = 2000
_BM25_WEIGHTS = "2.0, 1.0, 0.5"  # heading, body, path
_CJK = re.compile(r"[々぀-ヿ㐀-䶿一-鿿豈-﫿]+")


def search_kb(home: Home, kb_id: str, query: str, *, limit: int = 10, page: int | None = None,
              cursor: str | None = None, mode: SearchMode = "hybrid",
              group_by: GroupBy = "chunk", max_per_doc: int = 3,
              filters: SearchFilter | None = None, include: Iterable[str] = (),
              snippet_chars: int = 300, embedder: Embedder | None = None) -> KbSearchResponse:
    started = time.monotonic()
    filters = filters or SearchFilter()
    include = set(include)
    _validate(query, limit, page, cursor, max_per_doc, snippet_chars, include)
    kb = open_kb(home, kb_id, check_profile=True)
    try:
        generation = kb.store.generation()
        bound = request_binding("kb-search", kb_id, query, filters.model_dump(mode="json"),
                                mode, group_by, max_per_doc)
        offset = (decode_cursor(cursor, bound, generation) if cursor
                  else ((page or 1) - 1) * limit)
        if offset >= DEPTH_CAP:
            raise InvalidRequest(f"results are available to a depth of {DEPTH_CAP}; "
                                 "refine the query instead of paging further")
        pool = max(BASE_POOL, 2 * (offset + limit)) * (3 if group_by == "document" else 1)
        where, args = filter_sql(filters)

        lexical: list[int] = []
        vector: list[int] = []
        degraded: str | None = None
        ran: SearchMode = mode
        if mode in ("hybrid", "lexical"):
            lexical = _lexical(kb, query, where, args, pool)
        if mode in ("hybrid", "vector"):
            try:
                vector = _vector(kb, query, where, args, pool, filters, embedder)
            except EmbedderUnavailable as exc:
                if mode == "vector":
                    raise
                degraded = f"the embedding endpoint is unavailable ({exc.message}); keyword " \
                           "results only"
                ran = "lexical"

        fused = _fuse(lexical, vector)
        ordered = _limit_per_document(kb, fused, 1 if group_by == "document" else max_per_doc)
        window = ordered[offset:offset + limit]
        end = offset + len(window)
        more = end < len(ordered) and end < DEPTH_CAP
        hits = _hydrate(kb, window, offset, query, include, snippet_chars)
        assert kb.config.embedding is not None
        return KbSearchResponse(
            kb=kb_id, query_id="q_" + uuid.uuid4().hex[:12], query=query, mode=ran,
            degraded=degraded, group_by=group_by, results=hits, page=offset // limit + 1,
            next_cursor=encode_cursor(end, bound, generation) if more else None,
            index=IndexDescription(embedding_model=kb.store.meta("embedding_profile")
                                   or kb.config.embedding.profile),
            took_ms=int((time.monotonic() - started) * 1000))
    finally:
        kb.close()


def _validate(query: str, limit: int, page: int | None, cursor: str | None, max_per_doc: int,
              snippet_chars: int, include: set[str]) -> None:
    if not query.strip():
        raise InvalidRequest("the query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidRequest(f"the query is longer than {MAX_QUERY_CHARS} characters")
    if not 1 <= limit <= 50:
        raise InvalidRequest("limit must be between 1 and 50")
    if page is not None and cursor:
        raise InvalidRequest("give a page or a cursor, not both")
    if page is not None and page < 1:
        raise InvalidRequest("page must be 1 or more")
    if not 1 <= max_per_doc <= 50:
        raise InvalidRequest("max_per_doc must be between 1 and 50")
    if not 50 <= snippet_chars <= 2000:
        raise InvalidRequest("snippet_chars must be between 50 and 2000")
    unknown = include - {"highlights", "text"}
    if unknown:
        raise InvalidRequest(f"unknown include: {', '.join(sorted(unknown))}")


# --- the two sides -----------------------------------------------------------------


def _lexical(kb: OpenKb, query: str, where: str, args: list[object], pool: int) -> list[int]:
    match = fts_query(query)
    if not match:
        return []
    sql = (f"SELECT c.id FROM chunk_fts JOIN chunks c ON c.id = chunk_fts.rowid "
           f"JOIN documents d ON d.doc_id = c.doc_id "
           f"WHERE chunk_fts MATCH ? AND {where} "
           f"ORDER BY bm25(chunk_fts, {_BM25_WEIGHTS}) LIMIT ?")
    return [int(row[0]) for row in kb.store.db.execute(sql, (match, *args, pool))]


def _vector(kb: OpenKb, query: str, where: str, args: list[object], pool: int,
            filters: SearchFilter, embedder: Embedder | None) -> list[int]:
    assert kb.config.embedding is not None
    profile = kb.config.embedding.profile
    vectors = load_vectors(kb.store, kb.kb_dir / "cache", profile)
    if not vectors.rowids.size:
        return []
    cache = QueryVectorCache(kb.kb_dir / "cache")
    q = cache.get(profile, query)
    if q is None:
        if embedder is None:
            embedder = HttpEmbedder(kb.config.embedding)
            embedder.check()  # learns whether the model takes input_type
        q = np.asarray(embedder.embed([query], "query")[0], dtype=np.float32)
        cache.put(profile, query, q)
    if q.shape[0] != vectors.dimensions:
        raise ConfigError(f"the query vector has {q.shape[0]} dimensions but the index holds "
                          f"{vectors.dimensions}")
    norm = float(np.linalg.norm(q))
    scores = vectors.matrix @ (q / (norm or 1.0))
    if not filters.is_empty():
        allowed = np.fromiter(
            (row[0] for row in kb.store.db.execute(
                f"SELECT c.id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
                f"WHERE {where}", args)), dtype=np.int64)
        scores = np.where(np.isin(vectors.rowids, allowed), scores, -np.inf)
    count = min(pool, int(np.isfinite(scores).sum()))
    if count <= 0:
        return []
    top = np.argpartition(-scores, count - 1)[:count]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [int(vectors.rowids[i]) for i in top]


def _fuse(lexical: Sequence[int], vector: Sequence[int]) -> list[tuple[int, float, SideRanks]]:
    scores: dict[int, float] = {}
    ranks: dict[int, SideRanks] = {}
    for side, ids in (("lexical", lexical), ("vector", vector)):
        for rank, rowid in enumerate(ids, 1):
            scores[rowid] = scores.get(rowid, 0.0) + 1.0 / (RRF_K + rank)
            setattr(ranks.setdefault(rowid, SideRanks()), side, rank)
    ordered = sorted(scores, key=lambda r: (-scores[r], r))
    return [(r, scores[r], ranks[r]) for r in ordered]


def _limit_per_document(kb: OpenKb, fused: list[tuple[int, float, SideRanks]],
                        per_doc: int) -> list[tuple[int, float, SideRanks]]:
    doc_of: dict[int, str] = {}
    ids = [r for r, _s, _k in fused]
    for i in range(0, len(ids), 500):
        batch = ids[i:i + 500]
        marks = ",".join("?" * len(batch))
        doc_of.update((int(row[0]), str(row[1])) for row in kb.store.db.execute(
            f"SELECT id, doc_id FROM chunks WHERE id IN ({marks})", batch))
    taken: dict[str, int] = {}
    kept = []
    for item in fused:
        doc = doc_of.get(item[0])
        if doc is None or taken.get(doc, 0) >= per_doc:
            continue
        taken[doc] = taken.get(doc, 0) + 1
        kept.append(item)
    return kept


# --- results -----------------------------------------------------------------------


def _hydrate(kb: OpenKb, window: list[tuple[int, float, SideRanks]], offset: int, query: str,
             include: set[str], snippet_chars: int) -> list[SearchHit]:
    if not window:
        return []
    ids = [r for r, _s, _k in window]
    marks = ",".join("?" * len(ids))
    rows = {int(row["id"]): row for row in kb.store.db.execute(
        f"SELECT c.id, c.chunk_id, c.doc_id, c.prefix, c.heading_path, c.page, c.page_end, "
        f"c.location, d.source_id, d.rel_path, d.title, d.file_type, d.indexed_at, "
        f"{kb.origin_column} AS origin_path, "
        f"substr(ct.text, c.start + 1, c.end - c.start) AS text "
        f"FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        f"JOIN contents ct ON ct.content_hash = d.content_hash WHERE c.id IN ({marks})", ids)}
    terms = query_terms(query)
    hits = []
    for position, (rowid, score, ranks) in enumerate(window, offset + 1):
        row = rows[rowid]
        text = str(row["text"])
        snippet, highlights = make_snippet(text, terms, snippet_chars)
        prefix = str(row["prefix"] or "")
        hits.append(SearchHit(
            rank=position, chunk_id=row["chunk_id"], doc_id=row["doc_id"],
            score=round(score, 6), ranks=ranks,
            title=row["title"] or row["rel_path"].rsplit("/", 1)[-1], snippet=snippet,
            highlights=highlights if "highlights" in include else None,
            text=(f"{prefix}\n{text}" if prefix else text) if "text" in include else None,
            citation=KbCitation(source_id=row["source_id"], path=row["rel_path"],
                                page=row["page"], page_end=row["page_end"],
                                location=row["location"],
                                heading_path=json.loads(row["heading_path"]),
                                origin_path=row["origin_path"]),
            indexed_at=row["indexed_at"], metadata={"file_type": row["file_type"]}))
    return hits


def query_terms(query: str) -> list[str]:
    """What to look for in a chunk to place the snippet and the highlights: each word of
    the query, and each two-character piece of its Japanese runs, as typed and NFKC'd."""
    terms: set[str] = set()
    for form in {query, unicodedata.normalize("NFKC", query)}:
        for word in re.findall(r'"([^"]+)"|(\S+)', form):
            text = (word[0] or word[1]).strip('"').lower()
            if len(text) >= 2:
                terms.add(text)
            for run in _CJK.findall(text):
                terms.update(run[i:i + 2] for i in range(len(run) - 1))
    return sorted(terms, key=len, reverse=True)


def make_snippet(text: str, terms: Sequence[str], size: int) -> tuple[str, list[tuple[int, int]]]:
    flat = text.replace("\r", " ").replace("\n", " ")
    lower = flat.lower()
    found: list[tuple[int, int]] = []
    for term in terms:
        start = lower.find(term)
        while start != -1:
            found.append((start, start + len(term)))
            start = lower.find(term, start + 1)
    first = min((s for s, _e in found), default=0)
    start = max(0, first - size // 3)
    end = min(len(flat), start + size)
    start = max(0, end - size)
    lead = "…" if start > 0 else ""
    snippet = lead + flat[start:end] + ("…" if end < len(flat) else "")
    spans = sorted((s - start + len(lead), e - start + len(lead)) for s, e in found
                   if s >= start and e <= end)
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return snippet, merged
