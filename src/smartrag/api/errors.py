"""One error shape for every failure: `{"error": {"code", "message", "details"}}`.

Validation errors are `400 invalid_request` (not FastAPI's default 422), unknown paths
`404 not_found`, and unexpected exceptions `500 internal_error` with a generic message
(the traceback goes to the log, never to the client).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from smartrag.errors import EmbedderUnavailable, SmartRAGError, Unauthorized

log = logging.getLogger("smartrag.api")

_HTTP_CODES = {404: "not_found", 405: "method_not_allowed"}


def error_response(status: int, code: str, message: str,
                   details: dict[str, Any] | None = None,
                   headers: dict[str, str] | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details or {}}}
    return JSONResponse(body, status_code=status, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(SmartRAGError)
    async def _smartrag_error(_: Request, exc: SmartRAGError) -> JSONResponse:
        headers = None
        if isinstance(exc, Unauthorized):
            headers = {"WWW-Authenticate": "Bearer"}
        elif isinstance(exc, EmbedderUnavailable) and exc.retry_after:
            headers = {"Retry-After": str(exc.retry_after)}
        return error_response(exc.status, exc.code, exc.message, exc.details, headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""),
                   "type": e.get("type", "")} for e in exc.errors()]
        return error_response(400, "invalid_request", "the request is not valid",
                              {"errors": errors})

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _HTTP_CODES.get(exc.status_code, "http_error")
        return error_response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(500, "internal_error", "an internal error occurred")
