"""Reading a kb's tables: list, describe, and read-only SQL.

Every call opens `tables.duckdb` read-only, with the configuration locked so a query cannot
loosen it:

* no file or network access (`enable_external_access = false`): DuckDB can otherwise read
  any file the process can see (`read_csv('/etc/passwd')`) and reach URLs through its
  extensions, which would make `query` a way to read the machine;
* no extensions installed or loaded;
* 2 GB of memory and 4 threads at most;
* one statement per call, stopped after `QUERY_TIMEOUT` seconds. The stop is a timer that
  interrupts the connection: a watchdog, not parallel work.

A query answers with its columns once and its rows as arrays, at most `max_rows` of them
(100 unless asked, 1,000 at most), the number of rows it produced, and the tables it read,
to be cited. An error comes back as DuckDB words it: it names the bad column or table and
suggests close names.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from smartrag.errors import InvalidRequest, NotFound
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.tables.models import (
    ColumnLink,
    QueryColumn,
    QueryResult,
    TableCitation,
    TableDescription,
    TableMeta,
)
from smartrag.tables.store import TABLES_FILE, duckdb_module, json_value, quote, read_meta

QUERY_TIMEOUT = 30
"""Seconds: well under an MCP host's usual 300 s, and enough for a GROUP BY over tens of
millions of rows."""
MEMORY_LIMIT = "2GB"
"""Safe on a 16 GB machine running other things."""
THREADS = 4
DEFAULT_ROWS = 100
MAX_ROWS = 1000
SAMPLE_ROWS = 5

_LOCKED = {
    "enable_external_access": False,
    "autoinstall_known_extensions": False,
    "autoload_known_extensions": False,
    "memory_limit": MEMORY_LIMIT,
    "threads": THREADS,
    "lock_configuration": True,
}


def tables_of(home: Home, kb_id: str) -> dict[str, TableMeta]:
    """The kb's imported tables by name; empty when it has none."""
    return {t.name: t for t in read_meta(home.kb_dir(kb_id)).tables}


def _connect(home: Home, kb_id: str) -> Any:
    path = home.kb_dir(kb_id) / TABLES_FILE
    if not path.is_file():
        raise NotFound(f"kb {kb_id!r} has no tables", {"kb": kb_id})
    duckdb = duckdb_module()
    return duckdb.connect(str(path), read_only=True, config=dict(_LOCKED))


def _citation(meta: TableMeta, show_paths: bool) -> TableCitation:
    return TableCitation(name=meta.name, file=meta.path if show_paths else meta.file_name,
                         sheet=meta.sheet, imported_at=meta.imported_at)


# --- query -------------------------------------------------------------------------


def query_tables(home: Home, kb_id: str, sql: str, *, max_rows: int = DEFAULT_ROWS,
                 show_paths: bool = True) -> QueryResult:
    if not sql.strip():
        raise InvalidRequest("the SQL is empty")
    if not 1 <= max_rows <= MAX_ROWS:
        raise InvalidRequest(f"max_rows must be between 1 and {MAX_ROWS}")
    duckdb = duckdb_module()
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as exc:
        raise InvalidRequest(str(exc)) from None
    if len(statements) != 1:
        raise InvalidRequest("give one SQL statement")
    known = tables_of(home, kb_id)
    try:
        read = sorted(duckdb.get_table_names(sql, qualified=False))
    except (duckdb.Error, TypeError):
        read = []
    started = time.monotonic()
    db = _connect(home, kb_id)
    watchdog = threading.Timer(QUERY_TIMEOUT, db.interrupt)
    watchdog.daemon = True
    watchdog.start()
    try:
        relation = db.sql(sql)  # binds the query, for its column types; runs nothing
        if relation is None:
            raise InvalidRequest("the statement returns no rows; give a SELECT")
        columns = [QueryColumn(name=name, type=str(kind))
                   for name, kind in zip(relation.columns, relation.types)]
        cursor = db.execute(sql)
        rows = [[json_value(v) for v in row] for row in cursor.fetchmany(max_rows)]
        count = len(rows)
        if count == max_rows:
            while batch := cursor.fetchmany(10_000):
                count += len(batch)
    except duckdb.InterruptException:
        raise InvalidRequest(f"the query took longer than {QUERY_TIMEOUT} s and was stopped; "
                             "narrow it with WHERE, or aggregate") from None
    except duckdb.Error as exc:
        raise InvalidRequest(str(exc)) from None
    finally:
        watchdog.cancel()
        db.close()
    return QueryResult(kb=kb_id, sql=sql, columns=columns, rows=rows, row_count=count,
                       truncated=count > len(rows),
                       tables=[_citation(known[name], show_paths) for name in read
                               if name in known],
                       took_ms=int((time.monotonic() - started) * 1000))


# --- describe ----------------------------------------------------------------------


def attribute_keys(home: Home, kb_id: str) -> set[str]:
    """Every attribute key the kb's documents have, as stored."""
    path = home.kb_dir(kb_id) / INDEX_FILE
    if not path.is_file():
        return set()
    from smartrag.index import SqliteIndex

    store = SqliteIndex.open(path, readonly=True)
    try:
        return {key for applied in store.attribute_sets().values() for key in applied.keys}
    finally:
        store.close()


def links(columns: list[str], keys: set[str]) -> list[ColumnLink]:
    """Columns named like a document attribute, directly or in its escaped `_attr_` form."""
    found = []
    for column in columns:
        if column in keys:
            found.append(ColumnLink(column=column, attribute=column))
        elif f"_attr_{column}" in keys:
            found.append(ColumnLink(column=column, attribute=f"_attr_{column}"))
    return found


def describe_table(home: Home, kb_id: str, name: str, *,
                   show_paths: bool = True) -> TableDescription:
    known = tables_of(home, kb_id)
    meta = known.get(name)
    if meta is None:
        listed = ", ".join(sorted(known)) or "none"
        raise NotFound(f"kb {kb_id!r} has no table {name!r}; its tables are: {listed}",
                       {"kb": kb_id, "table": name})
    db = _connect(home, kb_id)
    try:
        sample = [[json_value(v) for v in row] for row in
                  db.execute(f"SELECT * FROM {quote(name)} LIMIT {SAMPLE_ROWS}").fetchall()]
    finally:
        db.close()
    return TableDescription(
        kb=kb_id, name=meta.name, original=meta.original,
        file=meta.path if show_paths else meta.file_name, sheet=meta.sheet,
        description=meta.description, rows=meta.rows, columns=meta.columns,
        links=links([c.name for c in meta.columns], attribute_keys(home, kb_id)),
        sample=sample, warnings=meta.warnings, imported_at=meta.imported_at)


def table_path(home: Home, kb_id: str) -> Path:
    return home.kb_dir(kb_id) / TABLES_FILE
