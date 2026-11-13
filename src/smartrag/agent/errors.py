"""Errors as an agent sees them: a stable code, a message of our own, and a hint.

The library's messages are written for the person running the CLI and may name the home,
a kb's directory or the embedding endpoint's URL. None of that may reach an agent
(design memo §3.9), so each code gets a fixed message here and only `invalid_request`,
which describes the caller's own arguments, passes the library's text through.
"""

from __future__ import annotations

from smartrag.agent.models import AgentError
from smartrag.errors import (
    ChunkSuperseded,
    ConfigError,
    CursorExpired,
    IndexNotReady,
    InvalidRequest,
    NotFound,
    SmartRAGError,
)


def agent_error(err: Exception, kb_id: str | None = None) -> AgentError:
    if isinstance(err, NotFound) and "chunk_id" in err.details:
        return AgentError(code="not_found", message="No passage with this chunk_id exists.",
                          hint="Use a chunk_id from a search result; search again if needed.")
    if isinstance(err, (NotFound, IndexNotReady, ConfigError)):
        # A kb that stopped qualifying is refused like one that never existed.
        return AgentError(code="not_found",
                          message=f"No knowledge base {kb_id!r} is available.",
                          hint="Use a kb_id from list_kbs.")
    if isinstance(err, ChunkSuperseded):
        return AgentError(code="chunk_superseded",
                          message="This passage has changed: its document was updated since "
                                  "the search.",
                          hint="Search again.",
                          details={"doc_id": str(err.details.get("doc_id", ""))})
    if isinstance(err, CursorExpired):
        return AgentError(code="cursor_expired",
                          message="The knowledge base was updated since this cursor was issued.",
                          hint="Repeat the request without a cursor.")
    if isinstance(err, InvalidRequest):
        return AgentError(code="invalid_request", message=err.message,
                          hint="Correct the arguments and try again.")
    code = err.code if isinstance(err, SmartRAGError) else "internal_error"
    return AgentError(code=code, message="The request failed inside Smart RAG.",
                      hint="Try again later; if it keeps failing, tell the user.")
