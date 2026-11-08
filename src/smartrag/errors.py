"""Errors shared by the engine and the Web API, each with a stable `code`.

The API renders every error in one shape, `{"error": {"code", "message", "details"}}`
(see `smartrag.api.errors`). Codes are part of the contract: clients and agents branch
on them, so a code is never renamed once released.
"""

from __future__ import annotations

from typing import Any


class SmartRAGError(Exception):
    """Base error: an HTTP status, a stable code, a user-safe message and details."""

    status = 500
    code = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidRequest(SmartRAGError):
    status = 400
    code = "invalid_request"


class Unauthorized(SmartRAGError):
    status = 401
    code = "unauthorized"


class Forbidden(SmartRAGError):
    status = 403
    code = "forbidden"


class NotFound(SmartRAGError):
    status = 404
    code = "not_found"


class ChunkSuperseded(SmartRAGError):
    """The chunk belongs to an older version of its document (it was re-ingested).

    `details` carries `doc_id` and `current_version`, so an agent can re-read the
    document instead of failing silently.
    """

    status = 410
    code = "chunk_superseded"


class IndexNotReady(SmartRAGError):
    status = 503
    code = "index_not_ready"


class EmbedderUnavailable(SmartRAGError):
    """The embedding service (OumiGo `/v1/embeddings`) cannot serve right now."""

    status = 503
    code = "embedder_unavailable"

    def __init__(self, message: str, retry_after: int | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details)
        self.retry_after = retry_after
