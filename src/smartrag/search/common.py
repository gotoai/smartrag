"""What searching and reading share: opening a kb's index, cursors, filters."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from smartrag.errors import ConfigError, CursorExpired, IndexNotReady, InvalidRequest
from smartrag.index import SqliteIndex
from smartrag.kb.config import KbConfig
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.search.models import SearchFilter


@dataclass
class OpenKb:
    config: KbConfig
    kb_dir: Path
    store: SqliteIndex

    @property
    def origin_column(self) -> str:
        """`d.origin_path`, or NULL for an index synced before that column existed (a
        read-only connection cannot add it; the next sync does)."""
        return "d.origin_path" if self.store.has_column("documents", "origin_path") else "NULL"

    def close(self) -> None:
        self.store.close()


def open_kb(home: Home, kb_id: str, *, check_profile: bool) -> OpenKb:
    """A local kb's index, opened read-only. Raises if the kb is remote, has no index,
    or (when searching) its index was built with a different embedding profile."""
    config = home.load(kb_id)
    if config.kind == "remote":
        raise InvalidRequest(f"kb {kb_id!r} is remote; v1 searches and reads local kbs only",
                             {"kb": kb_id})
    kb_dir = home.kb_dir(kb_id)
    path = kb_dir / INDEX_FILE
    if not path.is_file():
        raise IndexNotReady(f"kb {kb_id!r} has no index yet: run "
                            f"`smartrag run kb sync --kb {kb_id}`", {"kb": kb_id})
    store = SqliteIndex.open(path, readonly=True)
    if check_profile:
        assert config.embedding is not None
        held = store.meta("embedding_profile")
        if held is not None and held != config.embedding.profile:
            store.close()
            raise ConfigError(
                f"the index holds vectors from {held!r} but the config says "
                f"{config.embedding.profile!r}; restore the profile or rebuild the index",
                {"index_profile": held, "config_profile": config.embedding.profile})
    return OpenKb(config, kb_dir, store)


# --- cursors -----------------------------------------------------------------------


def encode_cursor(offset: int, binding: str, generation: int) -> str:
    raw = json.dumps({"o": offset, "b": binding, "g": generation}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(cursor: str, binding: str, generation: int) -> int:
    """The offset a cursor points at. A cursor from another request is invalid; one from
    before the index changed has expired."""
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded.encode()))
        offset, bound, issued = data["o"], data["b"], data["g"]
    except (ValueError, KeyError, TypeError) as exc:
        raise InvalidRequest("the cursor is not valid") from exc
    if not isinstance(offset, int) or offset < 0:
        raise InvalidRequest("the cursor is not valid")
    if bound != binding:
        raise InvalidRequest("the cursor belongs to a different request; start without one")
    if issued != generation:
        raise CursorExpired("the index has changed since this cursor was issued; start again "
                            "from the first page")
    return offset


# --- filters -----------------------------------------------------------------------


def _ns(moment: datetime) -> int:
    return int(moment.astimezone().timestamp() * 1_000_000_000)


def filter_sql(filters: SearchFilter) -> tuple[str, list[object]]:
    """A WHERE clause over `documents d` (and the chunks joined to it)."""
    clauses = ["d.status = 'ok'"]
    args: list[object] = []
    if filters.source:
        clauses.append(f"d.source_id IN ({','.join('?' * len(filters.source))})")
        args += filters.source
    if filters.file_type:
        types = [t.lower().lstrip(".") for t in filters.file_type]
        clauses.append(f"d.file_type IN ({','.join('?' * len(types))})")
        args += types
    if filters.path_prefix:
        prefix = filters.path_prefix.lstrip("/")
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        clauses.append("d.rel_path LIKE ? ESCAPE '\\'")
        args.append(escaped + "%")
    if filters.modified_after:
        clauses.append("d.mtime_ns >= ?")
        args.append(_ns(filters.modified_after))
    if filters.modified_before:
        clauses.append("d.mtime_ns < ?")
        args.append(_ns(filters.modified_before))
    return " AND ".join(clauses), args
