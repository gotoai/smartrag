"""`table add|remove|list`: a kb's table locations.

A location is a file or a folder; it is identified by itself, with no id of its own, since
agents see only the tables, named after their files. `add` registers it, and the next
`kb sync` imports it; `remove` drops its tables at once.
"""

from __future__ import annotations

from smartrag.errors import Conflict, InvalidRequest, NotFound
from smartrag.kb.config import KbConfig, TableEntryConfig
from smartrag.kb.home import Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.uris import location_to_uri, uri_to_path
from smartrag.tables.models import TableAdded, TableEntryView, TableList, TableRemoved
from smartrag.tables.store import drop_entry_tables, read_meta


def _local(config: KbConfig, kb_id: str) -> KbConfig:
    if config.kind != "local":
        raise InvalidRequest(f"kb {kb_id!r} is remote; its tables are managed by its server",
                             {"kb": kb_id})
    return config


def add_table(home: Home, kb_id: str, location: str, *,
              allow_missing: bool = False) -> TableAdded:
    _local(home.load(kb_id), kb_id)
    uri, path = location_to_uri(location)
    kb_dir = home.kb_dir(kb_id).resolve()
    if path == kb_dir or kb_dir in path.parents:
        raise InvalidRequest("a table location cannot be inside the kb's own directory")
    if not allow_missing and not path.exists():
        raise InvalidRequest(f"{path} does not exist; check the path, or pass "
                             "--allow-missing=true for a share that is not mounted right now",
                             {"path": str(path)})
    with kb_lock(home.kb_dir(kb_id), "table add"):
        document = home.document(kb_id)
        if any(entry.uri == uri for entry in _local(document.model, kb_id).tables):
            raise Conflict(f"{path} is already a table location of kb {kb_id!r}",
                           {"kb": kb_id, "uri": uri})
        document.add_table(TableEntryConfig(uri=uri))
        document.save()
    return TableAdded(kb=kb_id, uri=uri)


def remove_table(home: Home, kb_id: str, location: str) -> TableRemoved:
    uri, _path = location_to_uri(location)
    with kb_lock(home.kb_dir(kb_id), "table remove"):
        document = home.document(kb_id)
        entries = _local(document.model, kb_id).tables
        if not any(entry.uri == uri for entry in entries):
            listed = ", ".join(entry.uri for entry in entries) or "none"
            raise NotFound(f"kb {kb_id!r} has no table location {uri}; its locations are: "
                           f"{listed}", {"kb": kb_id, "uri": uri})
        removed = drop_entry_tables(home.kb_dir(kb_id), uri)
        document.remove_table(uri)
        document.save()
    return TableRemoved(kb=kb_id, uri=uri, tables_removed=removed)


def list_tables(home: Home, kb_id: str) -> TableList:
    config = _local(home.load(kb_id), kb_id)
    tables = read_meta(home.kb_dir(kb_id)).tables
    entries = []
    for entry in config.tables:
        path = uri_to_path(entry.uri)
        entries.append(TableEntryView(
            uri=entry.uri, exists=bool(path and path.exists()),
            tables=[t.name for t in tables if t.entry == entry.uri]))
    return TableList(kb=kb_id, entries=entries, tables=tables)
