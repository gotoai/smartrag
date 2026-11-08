"""Query-plane endpoints 1–6: the only operations agents, the UI and MCP get.

    GET  /v1/sources               describe_sources
    POST /v1/search                search (canonical form)
    GET  /v1/search?q=             search for the search box (bookmarkable URLs)
    GET  /v1/chunks/{chunk_id}     read: chunk -> neighbours -> section -> document
    GET  /v1/documents/{doc_id}    metadata, outline, text in pages
    POST /v1/feedback              click / relevant / irrelevant

Admin operations (sources, sync, jobs, index, query logs, API keys) are not here by
design: they run through the CLI on the operator host.

Every search is logged with its `query_id` (the server-side query log, CommonRAG P3),
so a wrong answer can be traced to retrieval or to the agent.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Path, Query, Request

from smartrag.api.auth import request_context
from smartrag.engine.base import RequestContext, RetrievalEngine
from smartrag.models import (
    DocumentResponse,
    FeedbackRequest,
    FeedbackResponse,
    ReadResponse,
    ReadScope,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SourcesResponse,
)
from smartrag.paging import encode_cursor, search_binding

log = logging.getLogger("smartrag.query")

router = APIRouter(prefix="/v1", dependencies=[Depends(request_context)])


def _engine(request: Request) -> RetrievalEngine:
    return request.app.state.engine


def _document_link(request: Request, doc_id: str, page: int | None) -> str:
    base = request.app.state.settings.public_base_url or str(request.base_url)
    link = f"{base.rstrip('/')}/v1/documents/{doc_id}"
    return f"{link}?page={page}" if page is not None else link


@router.get("/sources", response_model=SourcesResponse, response_model_exclude_none=True)
async def describe_sources(
    request: Request, ctx: RequestContext = Depends(request_context)
) -> SourcesResponse:
    """what is indexed, which fields can be filtered, and how fresh it is"""
    return await _engine(request).describe_sources(ctx)


@router.post("/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search(
    body: SearchRequest, request: Request, ctx: RequestContext = Depends(request_context)
) -> SearchResponse:
    """search with a natural-language query; returns evidence with citations"""
    return await _search(body, request, ctx)


@router.get("/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search_get(
    request: Request,
    q: str = Query(min_length=1, max_length=2000),
    source: list[str] | None = Query(default=None),
    type: list[str] | None = Query(default=None),  # the URL's own name
    limit: int = Query(default=10, ge=1, le=50),
    page: int = Query(default=1, ge=1, le=100),
    ctx: RequestContext = Depends(request_context),
) -> SearchResponse:
    """search for the search box: one result per document, highlights on"""
    filters = SearchFilters(source_id=source, file_type=type) if (source or type) else None
    body = SearchRequest(query=q, filters=filters, limit=limit, group_by="document",
                         include=["highlights"])
    if page > 1:
        body.cursor = encode_cursor((page - 1) * limit, search_binding(body))
    return await _search(body, request, ctx)


async def _search(body: SearchRequest, request: Request, ctx: RequestContext) -> SearchResponse:
    query_id = f"q_{uuid.uuid4().hex}"
    started = time.perf_counter()
    response = await _engine(request).search(body, ctx, query_id)
    response.took_ms = round((time.perf_counter() - started) * 1000)
    for result in response.results:
        result.citation.link = _document_link(request, result.doc_id, result.citation.page)
    log.info("search %s", json.dumps({
        "query_id": query_id,
        "key_id": ctx.key_id,
        "on_behalf_of": ctx.on_behalf_of,
        "query": body.query,
        "filters": body.filters.model_dump(mode="json", exclude_none=True)
        if body.filters else None,
        "mode": body.mode,
        "group_by": body.group_by,
        "results": [r.chunk_id for r in response.results],
        "took_ms": response.took_ms,
    }, ensure_ascii=False))
    return response


@router.get("/chunks/{chunk_id}", response_model=ReadResponse, response_model_exclude_none=True)
async def read_chunk(
    request: Request,
    chunk_id: str = Path(max_length=200),
    scope: ReadScope = "chunk",
    window: int = Query(default=1, ge=0, le=5),
    max_chars: int = Query(default=8000, ge=500, le=50000),
    cursor: str | None = None,
    ctx: RequestContext = Depends(request_context),
) -> ReadResponse:
    """read a chunk, widened to its neighbours, section or whole document"""
    response = await _engine(request).read_chunk(chunk_id, scope, window, max_chars, cursor, ctx)
    response.citation.link = _document_link(request, response.doc_id, response.citation.page)
    return response


@router.get("/documents/{doc_id}", response_model=DocumentResponse,
            response_model_exclude_none=True)
async def get_document(
    request: Request,
    doc_id: str = Path(max_length=100),
    include: list[Literal["text"]] = Query(default=[]),
    section: str | None = Query(default=None, max_length=50),
    max_chars: int = Query(default=8000, ge=500, le=50000),
    cursor: str | None = None,
    ctx: RequestContext = Depends(request_context),
) -> DocumentResponse:
    """document metadata and outline; text in pages with include=text"""
    response = await _engine(request).get_document(
        doc_id, "text" in include, section, max_chars, cursor, ctx)
    response.link = _document_link(request, response.doc_id, None)
    return response


@router.post("/feedback", status_code=202, response_model=FeedbackResponse)
async def feedback(
    body: FeedbackRequest, request: Request, ctx: RequestContext = Depends(request_context)
) -> FeedbackResponse:
    """record whether a result was clicked, relevant or irrelevant"""
    await _engine(request).record_feedback(body, ctx)
    return FeedbackResponse()
