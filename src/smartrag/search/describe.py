"""`kb describe`: what a kb holds, as its owner sees it.

Tables, when it has any, follow its sources: each with its file, rows and columns.

The kb's basics (as `kb status` gives them), then per source: where it is, its documents
by file type and date, and its attributes as the last sync applied them: the folder they
came from, each key with its type, format, meaning and, for a string or boolean key with
few values, those values. Unlike an agent's `describe_sources`, paths are shown. Reads the
index only; makes no network calls.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from smartrag.index import SqliteIndex
from smartrag.kb.attributes import attributes_dir
from smartrag.kb.config import SINK_ID
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.kb.models import KbStatus
from smartrag.kb.uris import uri_to_path
from smartrag.search.attributes import FilterableKey, key_values
from smartrag.search.models import Scalar
from smartrag.tables.models import TableMeta
from smartrag.tables.store import read_meta


class AttributeKey(BaseModel):
    name: str = Field(description="As stored: a name Smart RAG uses itself is escaped to "
                                  "`_attr_<name>`.")
    type: str | None = None
    format: str | None = None
    description: str | None = None
    enum: list[Scalar] | None = None
    values: list[Scalar] | None = Field(default=None, description="A string or boolean key's "
                                        "distinct values, when there are at most 50.")


class SourceAttributes(BaseModel):
    folder: str
    rows: int
    applied_at: str
    keys: list[AttributeKey]


class SourceDescription(BaseModel):
    source: str
    uri: str | None = Field(default=None, description="None for the sink.")
    documents: int
    file_types: dict[str, int] = Field(default_factory=dict)
    modified_from: str | None = None
    modified_to: str | None = None
    attributes: SourceAttributes | None = Field(default=None, description="None when the "
                                                "last sync found no attributes folder.")


class KbDescribed(BaseModel):
    status: KbStatus = Field(description="The kb's basics, as `kb status` gives them.")
    sources: list[SourceDescription] = Field(default_factory=list)
    tables: list[TableMeta] = Field(default_factory=list, description="The tables the last "
                                    "sync imported (`smartrag.tables`).")


def describe_kb(home: Home, kb_id: str) -> KbDescribed:
    status = home.status(kb_id)
    described = KbDescribed(status=status)
    if status.kind != "local":
        return described
    config = home.load(kb_id)
    described.tables = read_meta(home.kb_dir(kb_id)).tables
    index = home.kb_dir(kb_id) / INDEX_FILE
    store = SqliteIndex.open(index, readonly=True) if index.is_file() else None
    try:
        counts = _documents(store) if store is not None else {}
        applied = store.attribute_sets() if store is not None else {}
        for source in [*config.sources, None]:
            source_id = source.id if source is not None else SINK_ID
            types = counts.get(source_id, {})
            entry = SourceDescription(
                source=source_id, uri=source.uri if source is not None else None,
                documents=sum(n for n, _first, _last in types.values()),
                file_types={t: n for t, (n, _first, _last) in types.items()},
                modified_from=_day(min((f for _n, f, _l in types.values() if f), default=None)),
                modified_to=_day(max((last for _n, _f, last in types.values() if last),
                                     default=None)))
            attribute_set = applied.get(source_id)
            if attribute_set is not None and store is not None and source is not None:
                root = uri_to_path(source.uri)
                properties = attribute_set.schema.get("properties", {})
                entry.attributes = SourceAttributes(
                    folder=str(attributes_dir(root)) if root is not None else "",
                    rows=attribute_set.rows, applied_at=attribute_set.applied_at,
                    keys=[_key(store, source_id, name, properties.get(name, {}))
                          for name in _ordered(attribute_set.keys)])
            described.sources.append(entry)
    finally:
        if store is not None:
            store.close()
    return described


def _documents(store: SqliteIndex) -> dict[str, dict[str, tuple[int, int | None, int | None]]]:
    """Per source and file type: indexed documents, oldest and newest file time."""
    found: dict[str, dict[str, tuple[int, int | None, int | None]]] = {}
    for row in store.db.execute(
            "SELECT source_id, file_type, COUNT(*), MIN(mtime_ns), MAX(mtime_ns) "
            "FROM documents WHERE status = 'ok' GROUP BY source_id, file_type "
            "ORDER BY source_id, file_type"):
        found.setdefault(row[0], {})[row[1]] = (row[2], row[3], row[4])
    return found


def _ordered(keys: list[str]) -> list[str]:
    return [k for k in ("title", "link") if k in keys] + \
        [k for k in keys if k not in ("title", "link")]


def _key(store: SqliteIndex, source_id: str, name: str, spec: dict[str, object]) -> AttributeKey:
    kind = spec.get("type")
    fmt = spec.get("format")
    values = None
    if name not in ("title", "link") and isinstance(kind, str):
        values = key_values(store, FilterableKey(
            name, kind, fmt if isinstance(fmt, str) else None, None, sources=[source_id]))
    enum = spec.get("enum")
    description = spec.get("description")
    return AttributeKey(name=name, type=kind if isinstance(kind, str) else None,
                        format=fmt if isinstance(fmt, str) else None,
                        description=description if isinstance(description, str) else None,
                        enum=enum if isinstance(enum, list) else None, values=values)


def _day(mtime_ns: int | None) -> str | None:
    return None if mtime_ns is None else datetime.fromtimestamp(mtime_ns / 1e9).date().isoformat()
