"""Which kbs an agent sees, and what `describe_sources` tells it (design memo §3.9).

An agent sees only usable kbs: local, indexed with at least one document, built with
the embedding profile its config names, and described. Managing kbs is not the agent's
job, so an unfinished or broken kb is simply absent. A kb being synced is still usable;
so is one whose embedding endpoint is down (search falls back to keywords).

The visible set is fixed when the server starts, since the server instructions and the
`kb_id` enum are built from it, and re-checked on every call: a kb that stops
qualifying is refused from then on, with the same `not_found` as a kb that never existed.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime

from smartrag.agent.models import AgentKb, FilterInfo, KbDescription, SourceSummary
from smartrag.errors import NotFound
from smartrag.index import SqliteIndex
from smartrag.kb.config import MAX_DESCRIPTION
from smartrag.kb.home import INDEX_FILE, Home


def agent_description(text: str) -> str:
    """The description as one paragraph, cut at `MAX_DESCRIPTION` characters."""
    flat = " ".join(text.split())
    return flat if len(flat) <= MAX_DESCRIPTION else flat[:MAX_DESCRIPTION - 1] + "…"


class AgentCatalog:
    def __init__(self, home: Home, allow: Collection[str] | None = None) -> None:
        self.home = home
        self.allow = frozenset(allow) if allow else None
        self.visible: tuple[str, ...] = tuple(kb.kb_id for kb in self.kbs(initial=True))

    def kbs(self, *, initial: bool = False) -> list[AgentKb]:
        """The usable kbs, in id order. After startup, only those visible since then."""
        ids = [s.kb for s in self.home.list().kbs]
        candidates = ids if initial else [i for i in ids if i in self.visible]
        if self.allow is not None:
            candidates = [i for i in candidates if i in self.allow]
        return [kb for kb in map(self._usable, candidates) if kb is not None]

    def require(self, kb_id: str) -> AgentKb:
        kb = self._usable(kb_id) if kb_id in self.visible else None
        if kb is None:
            raise NotFound(f"no knowledge base {kb_id!r}", {"kb": kb_id})
        return kb

    def describe(self, kb_id: str) -> KbDescription:
        kb = self.require(kb_id)
        store = SqliteIndex.open(self.home.kb_dir(kb_id) / INDEX_FILE, readonly=True)
        try:
            rows = store.db.execute(
                "SELECT source_id, file_type, COUNT(*) AS n, MIN(mtime_ns) AS first, "
                "MAX(mtime_ns) AS last FROM documents WHERE status = 'ok' "
                "GROUP BY source_id, file_type ORDER BY source_id, file_type").fetchall()
            as_of = store.meta("last_sync_at") or store.db.execute(
                "SELECT MAX(indexed_at) FROM documents").fetchone()[0]
        finally:
            store.close()
        sources: dict[str, SourceSummary] = {}
        for row in rows:
            summary = sources.setdefault(row["source_id"], SourceSummary(
                source_id=row["source_id"], documents=0, file_types={}))
            summary.documents += row["n"]
            summary.file_types[row["file_type"]] = row["n"]
            first, last = _day(row["first"]), _day(row["last"])
            summary.modified_from = min(filter(None, (summary.modified_from, first)), default=None)
            summary.modified_to = max(filter(None, (summary.modified_to, last)), default=None)
        types = sorted({t for s in sources.values() for t in s.file_types})
        dates = "YYYY-MM-DD, or an ISO datetime"
        return KbDescription(
            kb_id=kb_id, description=kb.description, as_of=as_of,
            documents=sum(s.documents for s in sources.values()), sources=list(sources.values()),
            filters=[FilterInfo(name="source", values=list(sources)),
                     FilterInfo(name="file_type", values=types),
                     FilterInfo(name="modified_after", format=dates),
                     FilterInfo(name="modified_before", format=dates)])

    def _usable(self, kb_id: str) -> AgentKb | None:
        try:
            config = self.home.load(kb_id)
            if config.kind != "local" or config.embedding is None:
                return None  # a remote kb: not until forwarding to remote nodes is built
            description = agent_description(config.description)
            path = self.home.kb_dir(kb_id) / INDEX_FILE
            if not description or not path.is_file():
                return None
            store = SqliteIndex.open(path, readonly=True)
            try:
                held = store.meta("embedding_profile")
                has_documents = store.db.execute(
                    "SELECT EXISTS (SELECT 1 FROM documents WHERE status = 'ok')").fetchone()[0]
            finally:
                store.close()
        except Exception:  # noqa: BLE001 - a kb that cannot be read is not usable: hide it
            return None
        if not has_documents or (held is not None and held != config.embedding.profile):
            return None
        return AgentKb(kb_id=kb_id, description=description)


def _day(mtime_ns: int | None) -> str | None:
    return None if mtime_ns is None else datetime.fromtimestamp(mtime_ns / 1e9).date().isoformat()
