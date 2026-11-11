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


class Conflict(SmartRAGError):
    """The target already exists (a kb id, a source id, a source root)."""

    status = 409
    code = "conflict"


class KbLocked(SmartRAGError):
    """Another process holds the kb's lock: a sync, or a change to its config.

    `details` carries the holder's `pid` and `command`, so the message can say what to
    wait for rather than only that something is busy.
    """

    status = 409
    code = "kb_locked"


class ConfigError(SmartRAGError):
    """A kb's `config.yaml` is missing, unreadable or invalid."""

    status = 500
    code = "config_error"


class Unsupported(SmartRAGError):
    """A valid request for something not built yet, such as an S3 source in v1."""

    status = 501
    code = "unsupported"


class CursorExpired(SmartRAGError):
    """The index changed (a sync wrote to it) since this cursor was issued. Pages from
    two different indexes could repeat or skip results, so start again from page 1."""

    status = 410
    code = "cursor_expired"
