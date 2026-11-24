"""Importing a kb's tables into its DuckDB file at sync.

The kb's tables live in `<kb>/tables.duckdb`, and what Smart RAG knows about them (where
each came from, its columns' meanings and profile, the import's warnings) in
`<kb>/tables.json`, beside it rather than inside it, since agents may read every table in
the DuckDB file and paths must stay out of their reach when a kb hides them.

* **What is imported:** every table file under each `tables:` location (a file, or a
  folder taken whole and recursively). Files are re-read when their size or time changed
  and their bytes did too, or their schema file changed; others are carried over as they
  are. Tables whose files are gone are dropped.
* **Built beside, then swapped in:** a new DuckDB file is written next to the live one and
  renamed over it, since DuckDB lets a file be written only while no other process has it
  open, and the MCP server reads it. Readers open it per call.
* **Two names may not clash:** the second table to take a name is skipped, with a warning
  naming both files.
* **A location that is missing**, or a file that cannot be read, keeps its tables from the
  last import, like an incomplete listing of documents.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartrag.errors import ConfigError
from smartrag.kb.config import KbConfig
from smartrag.kb.uris import uri_to_path
from smartrag.tables.models import ColumnInfo, TableMeta, TablesFile, TableSync
from smartrag.tables.read import (
    DUCK_TYPES,
    SCHEMA_SUFFIX,
    TABLE_EXTENSIONS,
    ReadTable,
    TableReadError,
    read_tables,
)

TABLES_FILE = "tables.duckdb"
META_FILE = "tables.json"
IMPORTER_VERSION = "1"
STORAGE_VERSION = "v1.1.0"
"""The DuckDB file format written: the oldest DuckDB the package accepts (`duckdb>=1.1`),
so a file written by a newer DuckDB opens in an older one, as in an application's own
Python environment. A file in a newer DuckDB's default format cannot be read by 1.x."""
_LISTED_VALUES = 10


def duckdb_module() -> Any:
    try:
        import duckdb
    except ImportError:
        raise ConfigError("tables need DuckDB: install smartrag with the `tables` extra "
                          "(pip install 'smartrag[tables]')") from None
    return duckdb


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_meta(kb_dir: Path) -> TablesFile:
    path = kb_dir / META_FILE
    try:
        return TablesFile.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return TablesFile()


def _write_meta(kb_dir: Path, meta: TablesFile) -> None:
    path = kb_dir / META_FILE
    tmp = path.with_name(f".{META_FILE}.tmp-{os.getpid()}")
    tmp.write_text(meta.model_dump_json(indent=1), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _schema_stamp(path: Path) -> str | None:
    schema = path.with_name(path.name + SCHEMA_SUFFIX)
    try:
        stat = schema.stat()
    except OSError:
        return None
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def table_files(root: Path) -> list[Path]:
    """The table files of a location: the file itself, or every one under the folder."""
    if root.is_file():
        return [root] if root.suffix.lower() in TABLE_EXTENSIONS else []
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and p.suffix.lower() in TABLE_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


# --- sync ---------------------------------------------------------------------------


def sync_tables(config: KbConfig, kb_dir: Path, *, dry_run: bool = False,
                force: bool = False, only_stale: bool = False) -> list[TableSync]:
    """Bring `<kb>/tables.duckdb` in line with the kb's `tables:` locations. `force`
    re-reads every file (a rebuild); `only_stale` re-reads only tables imported by an older
    importer, adds nothing and drops nothing."""
    old = {t.name: t for t in read_meta(kb_dir).tables}
    if not config.tables and not old:
        return []
    reports: list[TableSync] = []
    keep: dict[str, TableMeta] = {}      # carried over from the live file as they are
    fresh: dict[str, tuple[ReadTable, TableMeta]] = {}
    owner: dict[str, str] = {}           # table name -> the file that took it
    by_path: dict[str, list[TableMeta]] = {}
    for meta in old.values():
        by_path.setdefault(meta.path, []).append(meta)

    def carry(metas: Iterable[TableMeta], status: str, warning: str | None = None) -> None:
        for meta in metas:
            keep[meta.name] = meta
            owner.setdefault(meta.name, meta.path)
            reports.append(TableSync(name=meta.name, original=meta.original,
                                     status=status, rows=meta.rows,  # type: ignore[arg-type]
                                     columns=len(meta.columns),
                                     warnings=[warning] if warning else []))

    seen_paths: set[str] = set()
    for entry in config.tables:
        root = uri_to_path(entry.uri)
        if root is None or not root.exists():
            stale = [m for m in old.values() if m.entry == entry.uri]
            carry(stale, "kept", f"{root or entry.uri} does not exist (not mounted?); the "
                                 "tables from the last import are kept")
            seen_paths.update(m.path for m in stale)
            continue
        for path in table_files(root):
            key = str(path)
            seen_paths.add(key)
            previous = by_path.get(key, [])
            stat = path.stat()
            stamp = _schema_stamp(path)
            if only_stale and not previous:
                continue  # new files are the next sync's
            if previous and not force and _unchanged(previous, path, stat, stamp, only_stale):
                for meta in previous:
                    meta.size, meta.mtime_ns = stat.st_size, stat.st_mtime_ns
                carry(previous, "unchanged")
                continue
            try:
                tables = read_tables(path)
            except TableReadError as exc:
                carry(previous, "kept", f"{exc}; the tables from the last import are kept")
                continue
            digest = _digest(path)
            for table in tables:
                if table.name in owner:
                    reports.append(TableSync(
                        name=table.name, original=table.original, status="skipped",
                        warnings=[(f"the name {table.name!r} is already taken by "
                                   f"{owner[table.name]}; rename one of the files")]))
                    continue
                owner[table.name] = key
                meta = TableMeta(
                    name=table.name, entry=entry.uri, path=key, file_name=path.name,
                    sheet=table.sheet, original=table.original, description=table.description,
                    rows=len(table.rows), columns=[
                        ColumnInfo(name=c.name, original=c.original, type=c.type,
                                   unit=c.unit, description=c.description)
                        for c in table.columns],
                    warnings=table.warnings, imported_at=_now(), importer=IMPORTER_VERSION,
                    size=stat.st_size, mtime_ns=stat.st_mtime_ns, digest=digest,
                    schema_stamp=stamp)
                fresh[table.name] = (table, meta)
                reports.append(TableSync(
                    name=table.name, original=table.original,
                    status="updated" if table.name in old else "added", rows=meta.rows,
                    columns=len(meta.columns), warnings=table.warnings))
    if only_stale:
        carry([m for m in old.values() if m.path not in seen_paths], "unchanged")
    else:  # files gone, and sheets or tables gone from files still there
        reports += [TableSync(name=m.name, original=m.original, status="removed")
                    for m in old.values() if m.name not in keep and m.name not in fresh]
    if dry_run:
        return reports
    changed = fresh or any(r.status == "removed" for r in reports)
    if changed:
        _build(kb_dir, keep, fresh)
    _write_meta(kb_dir, TablesFile(tables=sorted(
        [*keep.values(), *(meta for _t, meta in fresh.values())], key=lambda m: m.name)))
    return reports


def _unchanged(previous: list[TableMeta], path: Path, stat: os.stat_result,
               stamp: str | None, only_stale: bool) -> bool:
    """Whether a file's tables can be carried over: same importer and schema file, and the
    same size and time, or else the same bytes."""
    if only_stale:
        return all(m.importer == IMPORTER_VERSION for m in previous)
    if any(m.importer != IMPORTER_VERSION or m.schema_stamp != stamp for m in previous):
        return False
    first = previous[0]
    if (first.size, first.mtime_ns) == (stat.st_size, stat.st_mtime_ns):
        return True
    return first.digest == _digest(path)


def _build(kb_dir: Path, keep: dict[str, TableMeta],
           fresh: dict[str, tuple[ReadTable, TableMeta]]) -> None:
    """Write a new DuckDB file beside the live one, then rename it over it."""
    duckdb = duckdb_module()
    live, new = kb_dir / TABLES_FILE, kb_dir / f"{TABLES_FILE}.new"
    for leftover in (new, Path(f"{new}.wal")):
        leftover.unlink(missing_ok=True)
    db = duckdb.connect(str(new), config={"storage_compatibility_version": STORAGE_VERSION})
    try:
        if keep and live.is_file():
            db.execute(f"ATTACH {_literal(str(live))} AS live (READ_ONLY)")
            for name in keep:
                db.execute(f"CREATE TABLE {quote(name)} AS SELECT * FROM live.{quote(name)}")
            db.execute("DETACH live")
        for table, meta in fresh.values():
            _import(db, kb_dir, table)
            _profile(db, meta)
        db.execute("CHECKPOINT")
    finally:
        db.close()
    os.chmod(new, 0o600)
    os.replace(new, live)


def _literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _import(db: Any, kb_dir: Path, table: ReadTable) -> None:
    """Rows go through a temporary CSV with the column types spelled out, which DuckDB reads
    far faster than row-by-row inserts."""
    tmp = kb_dir / f".table-{os.getpid()}.csv"
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([c.name for c in table.columns])
            for row in table.rows:
                writer.writerow(["" if v is None else _csv_value(v) for v in row])
        types = ", ".join(f"{_literal(c.name)}: {_literal(DUCK_TYPES[c.type])}"
                          for c in table.columns)
        db.execute(f"CREATE TABLE {quote(table.name)} AS SELECT * FROM read_csv("
                   f"{_literal(str(tmp))}, header = true, delim = ',', quote = '\"', "
                   f"escape = '\"', nullstr = '', auto_detect = false, "
                   f"columns = {{{types}}})")
    finally:
        tmp.unlink(missing_ok=True)


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def _profile(db: Any, meta: TableMeta) -> None:
    """Nulls, approximate distinct values, min and max per column (DuckDB's SUMMARIZE), and
    the values themselves when there are few."""
    if not meta.columns:
        return
    rows = db.execute(f"SUMMARIZE {quote(meta.name)}").fetchall()
    names = [d[0] for d in db.description]
    summary = {row[0]: dict(zip(names, row)) for row in rows}
    for column in meta.columns:
        info = summary.get(column.name)
        if info is None:
            continue
        count = int(info.get("count") or 0)
        null_pct = float(info.get("null_percentage") or 0)
        column.nulls = round(count * null_pct / 100)
        column.distinct = int(info["approx_unique"]) if info.get("approx_unique") is not None \
            else None
        column.min = None if info.get("min") is None else str(info["min"])
        column.max = None if info.get("max") is None else str(info["max"])
        if column.type in ("string", "boolean") and (column.distinct or 0) <= _LISTED_VALUES + 2:
            values = [row[0] for row in db.execute(
                f"SELECT DISTINCT {quote(column.name)} FROM {quote(meta.name)} WHERE "
                f"{quote(column.name)} IS NOT NULL ORDER BY 1 LIMIT {_LISTED_VALUES + 1}"
            ).fetchall()]
            if len(values) <= _LISTED_VALUES:
                column.values = values


def drop_entry_tables(kb_dir: Path, uri: str) -> list[str]:
    """Remove a location's tables now (`table remove`), keeping every other table."""
    meta = read_meta(kb_dir)
    gone = [t.name for t in meta.tables if t.entry == uri]
    if not gone:
        return []
    keep = {t.name: t for t in meta.tables if t.entry != uri}
    _build(kb_dir, keep, {})
    _write_meta(kb_dir, TablesFile(tables=list(keep.values())))
    return gone


def json_value(value: Any) -> Any:
    """A DuckDB value as JSON gives it: dates and times as ISO text, decimals as numbers."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return json.dumps(value, default=str)
