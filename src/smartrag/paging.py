"""Opaque pagination cursors, bound to the request they page through.

A cursor is base64url JSON `{"o": offset, "b": binding}`. The binding is a short hash of
the request's identity (query, filters, mode, grouping), so a cursor from one search
can't be replayed against another — that is answered `400 invalid_request` instead of
returning a silently wrong page. Clients must treat cursors as opaque.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

from smartrag.errors import InvalidRequest
from smartrag.models import SearchRequest


def binding(*parts: Any) -> str:
    """Short, stable hash of the parts that define a result list."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def search_binding(request: SearchRequest) -> str:
    """The binding of a search: everything that changes its result list, not `limit`."""
    filters = request.filters.model_dump(mode="json") if request.filters else None
    return binding("search", request.query, filters, request.mode, request.group_by)


def encode_cursor(offset: int, bound_to: str) -> str:
    raw = json.dumps({"o": offset, "b": bound_to}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(cursor: str | None, bound_to: str) -> int:
    """The offset a cursor points at (0 without one). Invalid or foreign cursors raise."""
    if not cursor:
        return 0
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode()))
        offset, bound = data["o"], data["b"]
    except (ValueError, KeyError, TypeError) as exc:
        raise InvalidRequest("cursor is not valid") from exc
    if not isinstance(offset, int) or offset < 0:
        raise InvalidRequest("cursor is not valid")
    if bound != bound_to:
        raise InvalidRequest("cursor belongs to a different request; start without a cursor")
    return offset
