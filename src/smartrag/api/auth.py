"""API-key authentication and the end-user identity slot.

Clients send `Authorization: Bearer <api key>`. Keys are stored hashed (SHA-256) in a
JSON file that the CLI writes, so the query server never holds a plaintext key:

    [{"key_id": "ui-search", "sha256": "<hex>", "role": "reader", "delegate": true}]

The file is re-read when it changes, so a key issued or revoked by the CLI takes effect
without restarting the server.

`X-SmartRAG-On-Behalf-Of: <user id>` names the end user an agent or UI acts for
(CommonRAG P2). Only keys with `delegate: true` may send it; v1 logs it and passes it to
the engine, and document permissions will use it later.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from fastapi import Header, Request

from smartrag.engine.base import RequestContext
from smartrag.errors import Forbidden, Unauthorized

log = logging.getLogger("smartrag.api.auth")

ON_BEHALF_OF_HEADER = "X-SmartRAG-On-Behalf-Of"

Role = Literal["reader", "admin"]


@dataclass(frozen=True)
class Principal:
    key_id: str
    role: Role
    delegate: bool = False


ANONYMOUS = Principal(key_id="anonymous", role="reader", delegate=True)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class KeyStore(Protocol):
    def authenticate(self, key: str) -> Principal | None: ...


class FileKeyStore:
    """Hashed keys from a JSON file, reloaded when the file's mtime changes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._entries: list[tuple[str, Principal]] = []
        self._reload()

    def _reload(self) -> None:
        mtime = self.path.stat().st_mtime
        if mtime == self._mtime:
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            entries = [(str(e["sha256"]).lower(),
                        Principal(key_id=str(e["key_id"]), role=e.get("role", "reader"),
                                  delegate=bool(e.get("delegate", False))))
                       for e in raw]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid API key file {self.path}: {exc}") from exc
        self._entries, self._mtime = entries, mtime
        log.info("loaded %d API keys from %s", len(entries), self.path)

    def authenticate(self, key: str) -> Principal | None:
        with self._lock:
            try:
                self._reload()
            except (OSError, ValueError):
                log.exception("cannot reload %s; keeping the previous keys", self.path)
        digest = hash_key(key)
        match = None
        for stored, principal in self._entries:  # compare all: no early exit on timing
            if hmac.compare_digest(stored, digest):
                match = principal
        return match


async def request_context(
    request: Request,
    authorization: str | None = Header(default=None),
    on_behalf_of: str | None = Header(default=None, alias=ON_BEHALF_OF_HEADER),
) -> RequestContext:
    """FastAPI dependency: authenticate the caller and build the engine's context."""
    store: KeyStore | None = request.app.state.key_store
    if store is None:  # auth disabled (development)
        principal = ANONYMOUS
    else:
        scheme, _, key = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not key.strip():
            raise Unauthorized("an API key is required: Authorization: Bearer <key>")
        found = store.authenticate(key.strip())
        if found is None:
            raise Unauthorized("the API key is not valid")
        principal = found
    if on_behalf_of is not None and not principal.delegate:
        raise Forbidden(f"this API key may not act on behalf of users ({ON_BEHALF_OF_HEADER})")
    return RequestContext(key_id=principal.key_id, on_behalf_of=on_behalf_of or None)
