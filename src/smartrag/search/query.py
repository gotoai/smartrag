"""`kb search`: hybrid keyword + vector search over one kb.

1. Keyword side: BM25 over the bigram FTS index (headings weigh 2, body 1, path 0.5).
2. Vector side: the query embedded as a query, cosine against the normalised chunk vectors.
3. Reciprocal rank fusion (k = 60) of the two candidate lists.
4. At most `max_per_doc` chunks per document, or one per document for `group_by=document`.
5. The requested page, with snippets around the best keyword match.

With `partition_by`, the documents the filters leave are split by a key (an attribute or
`source`), and each partition contributes its best `top` hits. The score stays global:
both sides rank every chunk the filters leave, not a pool of the best, so a partition's
hits sit where they rank overall, and pages run through all partitions in that order.

If the embedding endpoint cannot be reached, a hybrid search returns the keyword results
alone, with `search_mode` `keyword` and the reason in `fallback_reason`; a vector-only
search fails. Results are evidence only: there is no answer field.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from smartrag.embedding.api import Embedder, open_embedder, resolve
from smartrag.errors import ConfigError, EmbedderUnavailable, InvalidRequest
from smartrag.index import fts_query
from smartrag.kb.home import Home
from smartrag.kb.uris import uri_to_file_path
from smartrag.paging import binding as request_binding
from smartrag.search.attributes import (
    Partitioning,
    documents_clause,
    matching_documents,
    partitioning,
)
from smartrag.search.common import OpenKb, decode_cursor, encode_cursor, filter_sql, open_kb
from smartrag.search.models import (
    CompactHit,
    GroupBy,
    IndexDescription,
    KbCitation,
    KbSearchResponse,
    Partitions,
    SearchFilter,
    SearchHit,
    SearchMode,
    SideRanks,
)
from smartrag.search.vectors import QueryVectorCache, load_vectors

DEPTH_CAP = 200
PARTITION_DEPTH_CAP = 10_000
"""Hits a partitioned search pages through: 1,000 listed values (the filter's limit) at
`MAX_TOP` each."""
MAX_LIMIT = 50
MAX_COMPACT_LIMIT = 100
"""A page of compact hits (about 300 characters each) is smaller than 50 full ones."""
MAX_TOP = 10
"""Hits per partition: past ten, a search filtered to that one value serves better."""
COMPACT_SNIPPET = 120
RRF_K = 60
BASE_POOL = 50
MAX_QUERY_CHARS = 2000
_BM25_WEIGHTS = "2.0, 1.0, 0.5"  # heading, body, path
_CJK = re.compile(r"[々぀-ヿ㐀-䶿一-鿿豈-﫿]+")

Ranked = tuple[int, float, tuple[int | None, int | None]]
"""A chunk's id, fused score, and its (keyword, vector) ranks."""


def search_kb(home: Home, kb_id: str, query: str, *, limit: int = 10, page: int | None = None,
              cursor: str | None = None, search_mode: SearchMode = "hybrid",
              group_by: GroupBy = "chunk", max_per_doc: int = 3,
              filters: SearchFilter | None = None, include: Iterable[str] = (),
              snippet_chars: int | None = None, partition_by: str | None = None,
              top: int = 1, compact: bool | None = None,
              embedder: Embedder | None = None) -> KbSearchResponse:
    """`compact` is on by default with `partition_by`, off without; `snippet_chars`
    defaults to 300 characters, 120 for compact hits."""
    started = time.monotonic()
    filters = filters or SearchFilter()
    include = set(include)
    compact = partition_by is not None if compact is None else compact
    if snippet_chars is None:
        snippet_chars = COMPACT_SNIPPET if compact else 300
    _validate(query, limit, page, cursor, max_per_doc, snippet_chars, include, partition_by,
              top, compact)
    kb = open_kb(home, kb_id, check_space=True)
    try:
        generation = kb.store.generation()
        bound = request_binding("kb-search", kb_id, query, filters.model_dump(mode="json"),
                                search_mode, group_by, max_per_doc, partition_by, top)
        offset = (decode_cursor(cursor, bound, generation) if cursor
                  else ((page or 1) - 1) * limit)
        depth = PARTITION_DEPTH_CAP if partition_by else DEPTH_CAP
        if offset >= depth:
            raise InvalidRequest(f"results are available to a depth of {depth}; "
                                 "refine the query instead of paging further")
        where, args = filter_sql(filters)
        if filters.attributes:
            clause, extra = documents_clause(
                matching_documents(kb.store, filters.attributes))
            where, args = f"{where} AND {clause}", args + extra
        split: Partitioning | None = None
        pool: int | None = None
        if partition_by:
            docs = [row[0] for row in kb.store.db.execute(
                f"SELECT d.doc_id FROM documents d WHERE {where}", args)]
            split = partitioning(kb.store, partition_by, docs, filters.attributes)
            clause, extra = documents_clause(sorted(split.of))
            where, args = f"{where} AND {clause}", args + extra
        else:
            pool = max(BASE_POOL, 2 * (offset + limit)) * (3 if group_by == "document" else 1)
        left = (_Chunks.read(kb, where, args)
                if split is not None or not filters.is_empty() else None)

        keyword = vector = _NONE
        fallback_reason: str | None = None
        ran: SearchMode = search_mode
        if search_mode in ("hybrid", "keyword"):
            keyword = _keyword(kb, query, pool, left)
        if search_mode in ("hybrid", "vector"):
            try:
                vector = _vector(kb, query, pool, left, embedder)
            except EmbedderUnavailable as exc:
                if search_mode == "vector":
                    raise
                fallback_reason = f"the embedding endpoint is unavailable ({exc.message}); " \
                                  "keyword results only"
                ran = "keyword"

        per_doc = 1 if group_by == "document" else max_per_doc
        need = offset + limit + 1
        fused = _fuse(keyword, vector)
        if split is not None and left is not None:
            ordered = _select_partitioned(kb, fused, left, split, per_doc, top, need)
        else:
            ordered = _select(kb, fused, per_doc, need)
        window = ordered[offset:offset + limit]
        end = offset + len(window)
        more = end < len(ordered) and end < depth
        hits = _hydrate(kb, window, offset, query, include, snippet_chars, split, compact)
        assert kb.config.embedding is not None
        return KbSearchResponse(
            kb=kb_id, query_id="q_" + uuid.uuid4().hex[:12], query=query, search_mode=ran,
            fallback_reason=fallback_reason, group_by=group_by,
            partitions=None if split is None else Partitions(
                key=split.key, top=top, count=len(set(split.of.values())),
                documents_without_key=split.without_key,
                empty=[split.display(v) for v in split.empty]),
            compact=compact, results=hits,
            page=offset // limit + 1,
            next_cursor=encode_cursor(end, bound, generation) if more else None,
            index=IndexDescription(embedding_model=kb.store.meta("model_profile_id")
                                   or kb.config.embedding.model_profile_id),
            took_ms=int((time.monotonic() - started) * 1000))
    finally:
        kb.close()


def _validate(query: str, limit: int, page: int | None, cursor: str | None, max_per_doc: int,
              snippet_chars: int, include: set[str], partition_by: str | None, top: int,
              compact: bool) -> None:
    if not query.strip():
        raise InvalidRequest("the query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidRequest(f"the query is longer than {MAX_QUERY_CHARS} characters")
    most = MAX_COMPACT_LIMIT if compact else MAX_LIMIT
    if not 1 <= limit <= most:
        raise InvalidRequest(f"limit must be between 1 and {most}"
                             + ("" if compact else f" ({MAX_COMPACT_LIMIT} for compact results)"))
    if page is not None and cursor:
        raise InvalidRequest("give a page or a cursor, not both")
    if page is not None and page < 1:
        raise InvalidRequest("page must be 1 or more")
    if not 1 <= max_per_doc <= 50:
        raise InvalidRequest("max_per_doc must be between 1 and 50")
    if not 50 <= snippet_chars <= 2000:
        raise InvalidRequest("snippet_chars must be between 50 and 2000")
    if partition_by is not None and not partition_by.strip():
        raise InvalidRequest("partition_by is empty")
    if not 1 <= top <= MAX_TOP:
        raise InvalidRequest(f"top must be between 1 and {MAX_TOP}")
    if top != 1 and partition_by is None:
        raise InvalidRequest("top is the number of hits per partition: give partition_by")
    unknown = include - {"highlights", "text"}
    if unknown:
        raise InvalidRequest(f"unknown include: {', '.join(sorted(unknown))}")
    if compact and include:
        raise InvalidRequest("compact results carry no highlights or text; "
                             "give compact=false, or read the chunk")


# --- the two sides -----------------------------------------------------------------


_NONE = np.zeros(0, dtype=np.int64)


@dataclass
class _Chunks:
    """The chunks a search's filters leave, sorted by id, each with its document's rowid.
    Read from the chunks' `(doc_id)` index alone: a chunk's row holds its vector, and
    reading a row per chunk is what makes a join to millions of chunks slow."""

    ids: np.ndarray
    docs: np.ndarray

    @classmethod
    def read(cls, kb: OpenKb, where: str, args: list[object]) -> _Chunks:
        rows = kb.store.db.execute(
            f"SELECT c.id, d.rowid FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE {where}", args)
        pairs = np.fromiter(((row[0], row[1]) for row in rows),
                            dtype=[("id", np.int64), ("doc", np.int64)])
        pairs.sort(order="id")
        return cls(np.ascontiguousarray(pairs["id"]), np.ascontiguousarray(pairs["doc"]))

    def positions(self, ids: np.ndarray) -> np.ndarray:
        """Where `ids` sit in `self.ids`; -1 for those not left."""
        if not self.ids.size:
            return np.full(ids.size, -1, dtype=np.int64)
        at = np.minimum(np.searchsorted(self.ids, ids), self.ids.size - 1)
        return np.where(self.ids[at] == ids, at, -1)


def _keyword(kb: OpenKb, query: str, pool: int | None, left: _Chunks | None) -> np.ndarray:
    """The best `pool` matches among the chunks left, or all of them when `pool` is None.
    BM25 ranks every match in the FTS index alone; the filters apply to its ids after."""
    match = fts_query(query)
    if not match:
        return _NONE
    sql = f"SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY bm25(chunk_fts, " \
          f"{_BM25_WEIGHTS})"
    if left is None:
        cursor = kb.store.db.execute(f"{sql} LIMIT ?", (match, -1 if pool is None else pool))
        return np.fromiter((row[0] for row in cursor), dtype=np.int64)
    cursor = kb.store.db.execute(sql, (match,))
    found: list[np.ndarray] = []
    count = 0
    while batch := cursor.fetchmany(50_000):
        ids = np.fromiter((row[0] for row in batch), dtype=np.int64, count=len(batch))
        ids = ids[left.positions(ids) >= 0]
        found.append(ids)
        count += ids.size
        if pool is not None and count >= pool:
            break
    ranked = np.concatenate(found) if found else _NONE
    return ranked if pool is None else ranked[:pool]


def _vector(kb: OpenKb, query: str, pool: int | None, left: _Chunks | None,
            embedder: Embedder | None) -> np.ndarray:
    """The `pool` chunks nearest the query among those left, or all of them when `pool`
    is None."""
    assert kb.config.embedding is not None
    space = resolve(kb.config.embedding).space.key
    vectors = load_vectors(kb.store, kb.kb_dir / "cache", space)
    if not vectors.rows:
        return _NONE
    cache = QueryVectorCache(kb.kb_dir / "cache")
    q = cache.get(space, query)
    if q is None:
        if embedder is None:
            embedder = open_embedder(kb.config.embedding)
        q = np.asarray(embedder.embed([query], "query")[0], dtype=np.float32)
        cache.put(space, query, q)
    if q.shape[0] != vectors.dimensions:
        raise ConfigError(f"the query vector has {q.shape[0]} dimensions but the index holds "
                          f"{vectors.dimensions}")
    norm = float(np.linalg.norm(q))
    return np.asarray(vectors.top(q / (norm or 1.0), vectors.rows if pool is None else pool,
                                  None if left is None else left.ids), dtype=np.int64)


@dataclass
class _Fused:
    """Every chunk either side ranked, best fused score first (ties by id), with its rank
    on each side (0: not ranked there)."""

    ids: np.ndarray
    scores: np.ndarray
    keyword: np.ndarray
    vector: np.ndarray

    def item(self, i: int) -> Ranked:
        k, v = int(self.keyword[i]), int(self.vector[i])
        return int(self.ids[i]), float(self.scores[i]), (k or None, v or None)


def _fuse(keyword: np.ndarray, vector: np.ndarray) -> _Fused:
    ids, inverse = np.unique(np.concatenate([keyword, vector]), return_inverse=True)
    sides = np.concatenate([np.arange(1, keyword.size + 1), np.arange(1, vector.size + 1)])
    scores = np.bincount(inverse, weights=1.0 / (RRF_K + sides), minlength=ids.size)
    ranks = np.zeros((2, ids.size), dtype=np.int64)
    ranks[0, inverse[:keyword.size]] = np.arange(1, keyword.size + 1)
    ranks[1, inverse[keyword.size:]] = np.arange(1, vector.size + 1)
    order = np.lexsort((ids, -scores))
    return _Fused(ids[order], scores[order], ranks[0, order], ranks[1, order])


def _select(kb: OpenKb, fused: _Fused, per_doc: int, need: int) -> list[Ranked]:
    """The fused list, best first, with at most `per_doc` chunks per document; it stops
    once it holds `need`."""
    taken: dict[str, int] = {}
    kept: list[Ranked] = []
    for start in range(0, fused.ids.size, 500):
        ids = [int(i) for i in fused.ids[start:start + 500]]
        marks = ",".join("?" * len(ids))
        doc_of = {int(row[0]): str(row[1]) for row in kb.store.db.execute(
            f"SELECT c.id, c.doc_id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE c.id IN ({marks}) AND d.status = 'ok'", ids)}
        for offset, rowid in enumerate(ids):
            doc = doc_of.get(rowid)
            if doc is None or taken.get(doc, 0) >= per_doc:
                continue
            taken[doc] = taken.get(doc, 0) + 1
            kept.append(fused.item(start + offset))
            if len(kept) >= need:
                return kept
    return kept


def _select_partitioned(kb: OpenKb, fused: _Fused, left: _Chunks, split: Partitioning,
                        per_doc: int, top: int, need: int) -> list[Ranked]:
    """As `_select`, with at most `top` chunks per partition. Every ranked chunk is among
    those left, so its document comes from `left`, without reading chunk rows."""
    value_of = {int(row[0]): split.of[row[1]] for row in kb.store.db.execute(
        "SELECT rowid, doc_id FROM documents WHERE doc_id IN (SELECT value FROM json_each(?))",
        (json.dumps(sorted(split.of)),))}
    docs = left.docs[left.positions(fused.ids)]
    taken: dict[int, int] = {}
    filled: dict[object, int] = {}
    kept: list[Ranked] = []
    for i, doc in enumerate(docs.tolist()):
        if taken.get(doc, 0) >= per_doc:
            continue
        value = value_of[doc]
        if filled.get(value, 0) >= top:
            continue
        taken[doc] = taken.get(doc, 0) + 1
        filled[value] = filled.get(value, 0) + 1
        kept.append(fused.item(i))
        if len(kept) >= need:
            break
    return kept


# --- results -----------------------------------------------------------------------


def _hydrate(kb: OpenKb, window: list[Ranked], offset: int, query: str, include: set[str],
             snippet_chars: int, split: Partitioning | None,
             compact: bool) -> list[SearchHit | CompactHit]:
    if not window:
        return []
    ids = [r for r, _s, _k in window]
    marks = ",".join("?" * len(ids))
    rows = {int(row["id"]): row for row in kb.store.db.execute(
        f"SELECT c.id, c.chunk_id, c.doc_id, c.prefix, c.heading_path, c.page, c.page_end, "
        f"c.location, d.source_id, d.rel_path, d.uri, d.title, d.file_type, d.indexed_at, "
        f"{kb.origin_column} AS origin_path, "
        f"substr(ct.text, c.start + 1, c.end - c.start) AS text "
        f"FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        f"JOIN contents ct ON ct.content_hash = d.content_hash WHERE c.id IN ({marks})", ids)}
    attributes = kb.attributes((row["source_id"], row["rel_path"]) for row in rows.values())
    terms = query_terms(query)
    hits: list[SearchHit | CompactHit] = []
    for position, (rowid, score, (keyword_rank, vector_rank)) in enumerate(window, offset + 1):
        row = rows[rowid]
        ranks = SideRanks(keyword=keyword_rank, vector=vector_rank)
        doc = attributes.get((row["source_id"], row["rel_path"]))
        text = str(row["text"])
        snippet, highlights = make_snippet(text, terms, snippet_chars)
        title = (doc.title if doc else None) or row["title"] or row["rel_path"].rsplit("/", 1)[-1]
        partition = None if split is None else split.display(split.of[row["doc_id"]])
        headings = json.loads(row["heading_path"])
        if compact:
            hits.append(CompactHit(
                rank=position, partition=partition, score=round(score, 6), ranks=ranks,
                chunk_id=row["chunk_id"], doc_id=row["doc_id"], title=title,
                page=row["page"], heading=headings[-1] if headings else None,
                snippet=snippet, link=doc.link if doc else None))
            continue
        prefix = str(row["prefix"] or "")
        hits.append(SearchHit(
            rank=position, chunk_id=row["chunk_id"], doc_id=row["doc_id"],
            partition=partition, score=round(score, 6), ranks=ranks, title=title,
            snippet=snippet, highlights=highlights if "highlights" in include else None,
            text=(f"{prefix}\n{text}" if prefix else text) if "text" in include else None,
            citation=KbCitation(source_id=row["source_id"], path=row["rel_path"],
                                file_path=uri_to_file_path(row["uri"]), page=row["page"],
                                page_end=row["page_end"], location=row["location"],
                                heading_path=headings,
                                origin_path=row["origin_path"], link=doc.link if doc else None),
            indexed_at=row["indexed_at"], metadata={"file_type": row["file_type"]},
            attributes=doc.attributes if doc else {}))
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
