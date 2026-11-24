"""Reading table files into typed columns: CSV, TSV, Excel sheets and SQLite tables.

Best effort, and Japanese files first:

* **One table per CSV or TSV file, per non-empty Excel sheet, per SQLite table** (views
  are skipped). The first non-empty row of a CSV, TSV or sheet is its header.
* **Encodings:** a BOM, then UTF-8, then what the bytes look like (CP932 for most
  Japanese exports); EDINET's own CSV is UTF-16 with a BOM.
* **Types are inferred from every value of a column**: a column is an integer, a number,
  a boolean, a date or a timestamp only when all its values read as one; otherwise text.
  Numbers may carry thousands separators (`1,234`), full-width digits, or a leading `△`
  or `▲`, or parentheses, for a negative. `-`, `－`, `—`, `N/A` and an empty cell are
  null. **Codes stay text:** a column with a value that has a leading zero (`0123`) is
  text, so codes keep their zeros, and so is a column mixing `7203` with `627A`.
* **A schema overrides inference**: `<file>.schema.json` beside the data file, a subset of
  JSON Schema with a table `description` and a column `unit` added. For a file with
  several tables (a workbook, a SQLite database), it maps each sheet or table name to its
  schema. A value that does not read as its declared type is dropped (null), with a
  warning.
* **Errors in two levels:** a bad row or value is skipped or dropped, with a warning;
  a file that cannot be read at all raises `TableReadError`, and its tables from the
  last import are kept.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smartrag.extract.encoding import detect_encoding
from smartrag.extract.tabular import _delimiter, cell_text

DELIMITED = (".csv", ".tsv")
EXCEL = (".xlsx", ".xlsm", ".xls", ".xlsb")
SQLITE = (".sqlite", ".sqlite3", ".db")
TABLE_EXTENSIONS = DELIMITED + EXCEL + SQLITE
SCHEMA_SUFFIX = ".schema.json"

_SQLITE_MAGIC = b"SQLite format 3\x00"
_SAMPLE_BYTES = 1 << 20
_NULLS = {"", "-", "－", "—", "―", "‐", "n/a", "na", "#n/a", "null", "none"}
_INT = re.compile(r"^-?\d+$")
_FLOAT = re.compile(r"^-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?$")
_DATE = re.compile(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")
_INT64 = 2 ** 63 - 1
_EXAMPLES = 3

#: The DuckDB type each inferred or declared type becomes.
DUCK_TYPES = {"integer": "BIGINT", "number": "DOUBLE", "boolean": "BOOLEAN",
              "date": "DATE", "timestamp": "TIMESTAMP", "string": "VARCHAR"}
_SCHEMA_TYPES = {"integer": "integer", "number": "number", "boolean": "boolean",
                 "string": "string"}


class TableReadError(Exception):
    """A file that cannot be read as tables at all."""


@dataclass
class Column:
    name: str
    """Safe for SQL: see `safe_name`."""
    original: str
    """As the file's header (or database) names it."""
    type: str = "string"
    """integer, number, boolean, date, timestamp or string."""
    description: str | None = None
    unit: str | None = None


@dataclass
class ReadTable:
    name: str
    original: str
    """The file name, plus the sheet or table for a file with several."""
    sheet: str | None
    columns: list[Column]
    rows: list[list[Any]]
    """Typed values: int, float, bool, date, datetime, str or None."""
    description: str | None = None
    warnings: list[str] = field(default_factory=list)


# --- names -------------------------------------------------------------------------


def safe_name(text: str, fallback: str) -> str:
    """A name SQL takes without quotes: lower case, anything but letters (Japanese
    included), digits and `_` becomes `_`; one that would start with a digit gets the
    fallback's first letter in front."""
    lowered = unicodedata.normalize("NFKC", text).strip().lower()
    name = re.sub(r"_+", "_", "".join(ch if ch.isalnum() or ch == "_" else "_"
                                      for ch in lowered)).strip("_")
    if not name:
        return fallback
    return f"{fallback[0]}_{name}" if name[0].isdigit() else name


def _column_names(header: list[str]) -> tuple[list[str], list[str]]:
    """Safe, unique names for a header row, and the originals (generated when blank)."""
    originals: list[str] = []
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, cell in enumerate(header, 1):
        original = cell.strip() or f"column_{index}"
        name = safe_name(original, f"c{index}")
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        seen.setdefault(name, 1)
        originals.append(original)
        names.append(name)
    return names, originals


# --- values ------------------------------------------------------------------------


def _number_text(text: str) -> str:
    """A cell as a plain number would be written: NFKC (full-width digits), no thousands
    separators, `△`/`▲` and parentheses as a minus sign."""
    value = unicodedata.normalize("NFKC", text).strip().replace(",", "")
    if value[:1] in ("△", "▲"):
        value = "-" + value[1:].strip()
    elif value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1].strip()
    return value


def _is_null(text: str) -> bool:
    return unicodedata.normalize("NFKC", text).strip().lower() in _NULLS


def _as(kind: str, text: str) -> Any:
    """`text` read as `kind`; raises ValueError when it does not read as one."""
    if kind == "string":
        return text
    if kind in ("integer", "number"):
        number = _number_text(text)
        if kind == "integer":
            if not _INT.match(number) or abs(int(number)) > _INT64:
                raise ValueError(text)
            return int(number)
        if not _FLOAT.match(number) or not math.isfinite(float(number)):
            raise ValueError(text)
        return float(number)
    if kind == "boolean":
        lowered = text.strip().lower()
        if lowered not in ("true", "false"):
            raise ValueError(text)
        return lowered == "true"
    if kind == "date":
        match = _DATE.match(text.strip())
        if not match:
            raise ValueError(text)
        return dt.date(int(match[1]), int(match[2]), int(match[3]))
    if kind == "timestamp":
        value = text.strip()
        if not _DATETIME.match(value):
            raise ValueError(text)
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z")
                                           else value)
        return parsed.astimezone(dt.timezone.utc).replace(tzinfo=None) \
            if parsed.tzinfo else parsed
    raise ValueError(kind)


def _infer(values: list[str]) -> tuple[str, str | None]:
    """The column's type, and why it was left as text when most values were numbers."""
    present = [v for v in values if not _is_null(v)]
    if not present:
        return "string", None
    for kind in ("boolean", "date", "timestamp"):
        if all(_reads(kind, v) for v in present):
            return kind, None
    numbers = [v for v in present if _reads("number", v)]
    if len(numbers) == len(present):
        if any(re.match(r"^-?0\d", _number_text(v)) for v in present):
            return "string", None  # codes keep their leading zeros
        if all(_reads("integer", v) for v in present):
            return "integer", None
        return "number", None
    if len(numbers) >= 0.9 * len(present):
        odd = [v for v in present if not _reads("number", v)]
        return "string", (f"{len(odd):,} of {len(present):,} values are not numbers "
                          f"(e.g. {', '.join(repr(v) for v in odd[:_EXAMPLES])}); kept as text")
    return "string", None


def _reads(kind: str, text: str) -> bool:
    try:
        _as(kind, text)
    except (ValueError, OverflowError):
        return False
    return True


# --- schema ------------------------------------------------------------------------


def read_schema(path: Path, tables: list[str], warnings: list[str]) -> dict[str, dict[str, Any]]:
    """Each table's schema (keyed by sheet or table name; `""` for a single table), from
    `<file>.schema.json`. A schema that cannot be read is ignored, with a warning."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        warnings.append(f"{path.name} cannot be read ({exc}); ignored")
        return {}
    if not isinstance(data, dict):
        warnings.append(f"{path.name} is not a JSON object; ignored")
        return {}
    if "properties" in data or "description" in data:
        return {"": data}
    schemas: dict[str, dict[str, Any]] = {}
    for name, spec in data.items():
        if isinstance(spec, dict) and name in tables:
            schemas[name] = spec
        else:
            warnings.append(f"{path.name}: no sheet or table {name!r}; its entry is ignored")
    return schemas


def _declared(spec: dict[str, Any] | None) -> str | None:
    if not isinstance(spec, dict):
        return None
    if spec.get("format") == "date":
        return "date"
    if spec.get("format") == "date-time":
        return "timestamp"
    kind = spec.get("type")
    return _SCHEMA_TYPES.get(kind) if isinstance(kind, str) else None


# --- building a table from a grid ----------------------------------------------------


def _table(name: str, original: str, sheet: str | None, grid: list[list[str]],
           schema: dict[str, Any] | None, warnings: list[str]) -> ReadTable | None:
    """A table from rows of cell text: the first non-empty row is the header."""
    rows = [row for row in grid if any(cell.strip() for cell in row)]
    if not rows:
        return None
    header, body = rows[0], rows[1:]
    width = len(header)
    long_rows = sum(1 for row in body if any(cell.strip() for cell in row[width:]))
    if long_rows:
        warnings.append(f"{long_rows:,} rows have more cells than the header; the extra "
                        "cells were dropped")
    body = [(row + [""] * width)[:width] for row in body]
    names, originals = _column_names(header)
    properties = (schema or {}).get("properties") or {}
    columns: list[Column] = []
    typed_columns: list[list[Any]] = []
    for index, (col_name, col_original) in enumerate(zip(names, originals)):
        spec = properties.get(col_original) or properties.get(col_name)
        values = [row[index] for row in body]
        kind = _declared(spec)
        note = None
        if kind is None:
            kind, note = _infer(values)
            if note:
                warnings.append(f"column {col_original!r}: {note}")
        converted: list[Any] = []
        dropped: list[str] = []
        for value in values:
            if _is_null(value):
                converted.append(None)
                continue
            try:
                converted.append(_as(kind, value))
            except (ValueError, OverflowError):
                converted.append(None)
                dropped.append(value)
        if dropped:
            warnings.append(f"column {col_original!r}: {len(dropped):,} values are not "
                            f"{kind} (schema), dropped (e.g. "
                            f"{', '.join(repr(v) for v in dropped[:_EXAMPLES])})")
        column = Column(name=col_name, original=col_original, type=kind)
        if isinstance(spec, dict):
            if isinstance(spec.get("description"), str):
                column.description = " ".join(spec["description"].split()) or None
            if isinstance(spec.get("unit"), str):
                column.unit = spec["unit"].strip() or None
        columns.append(column)
        typed_columns.append(converted)
    description = (schema or {}).get("description")
    return ReadTable(name=name, original=original, sheet=sheet, columns=columns,
                     rows=[list(row) for row in zip(*typed_columns)] if typed_columns else [],
                     description=" ".join(description.split()) if isinstance(description, str)
                     else None, warnings=warnings)


# --- formats -----------------------------------------------------------------------


def read_tables(path: Path) -> list[ReadTable]:
    """Every table in a file. Raises `TableReadError` when the file cannot be read."""
    ext = path.suffix.lower()
    stem = safe_name(path.stem, "table")
    schema_path = path.with_name(path.name + SCHEMA_SUFFIX)
    if ext in DELIMITED:
        warnings: list[str] = []
        schemas = read_schema(schema_path, [], warnings)
        table = _table(stem, path.name, None, _delimited_grid(path, ext, warnings),
                       schemas.get(""), warnings)
        return [table] if table else []
    if ext in EXCEL:
        return _workbook_tables(path, ext, stem, schema_path)
    if ext in SQLITE:
        return _sqlite_tables(path, stem, schema_path)
    raise TableReadError(f"{path.name}: not a table file")


def _delimited_grid(path: Path, ext: str, warnings: list[str]) -> list[list[str]]:
    import csv

    try:
        with open(path, "rb") as handle:
            sample = handle.read(_SAMPLE_BYTES)
            complete = len(sample) < _SAMPLE_BYTES
        encoding = detect_encoding(sample, complete)
        text = sample.decode(encoding, errors="replace")
        delimiter = _delimiter(text if complete else text.rsplit("\n", 1)[0], ext)
        csv.field_size_limit(max(csv.field_size_limit(), 1 << 24))
        with open(path, encoding=encoding, errors="replace", newline="") as handle:
            grid = [[cell_text(cell) for cell in record]
                    for record in csv.reader(handle, delimiter=delimiter)]
    except (OSError, csv.Error, LookupError) as exc:
        raise TableReadError(f"{path.name} cannot be read: {exc}") from exc
    if any("�" in cell for row in grid for cell in row):
        warnings.append(f"some bytes could not be decoded as {encoding} and were replaced")
    return grid


def _workbook_tables(path: Path, ext: str, stem: str, schema_path: Path) -> list[ReadTable]:
    import python_calamine as pc

    try:
        workbook = pc.CalamineWorkbook.from_path(str(path))
        sheets = [s.name for s in workbook.sheets_metadata
                  if s.typ == pc.SheetTypeEnum.WorkSheet
                  and s.visible == pc.SheetVisibleEnum.Visible]  # users hide sheets on purpose
    except (pc.CalamineError, OSError) as exc:
        raise TableReadError(f"{path.name} cannot be read as an Excel file: {exc}") from exc
    shared: list[str] = []
    schemas = read_schema(schema_path, sheets, shared)
    tables: list[ReadTable] = []
    for sheet in sheets:
        try:
            grid = [[cell_text(v) for v in row]
                    for row in workbook.get_sheet_by_name(sheet).to_python()]
        except pc.CalamineError as exc:
            shared.append(f"sheet {sheet!r} cannot be read ({exc}); skipped")
            continue
        warnings = list(shared)
        table = _table(f"{stem}_{safe_name(sheet, 'sheet')}", f"{path.name} › {sheet}", sheet,
                       grid, schemas.get(sheet), warnings)
        if table:
            tables.append(table)
    return tables


def _sqlite_tables(path: Path, stem: str, schema_path: Path) -> list[ReadTable]:
    try:
        with open(path, "rb") as handle:
            if handle.read(len(_SQLITE_MAGIC)) != _SQLITE_MAGIC:
                raise TableReadError(f"{path.name} is not a SQLite database")
        db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except (OSError, sqlite3.Error) as exc:
        raise TableReadError(f"{path.name} cannot be opened: {exc}") from exc
    try:
        names = [row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        shared: list[str] = []
        schemas = read_schema(schema_path, names, shared)
        tables: list[ReadTable] = []
        for name in names:
            warnings = list(shared)
            cursor = db.execute(f'SELECT * FROM "{name.replace(chr(34), chr(34) * 2)}"')
            header = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            blobs = {i for row in rows for i, v in enumerate(row) if isinstance(v, bytes)}
            if blobs:
                warnings.append("binary columns skipped: "
                                + ", ".join(header[i] for i in sorted(blobs)))
            keep = [i for i in range(len(header)) if i not in blobs]
            grid = [[header[i] for i in keep]] + [[cell_text(row[i]) for i in keep]
                                                    for row in rows]
            table = _table(f"{stem}_{safe_name(name, 'table')}", f"{path.name} › {name}", name,
                           grid, schemas.get(name), warnings)
            if table:
                tables.append(table)
        return tables
    except sqlite3.Error as exc:
        raise TableReadError(f"{path.name} cannot be read: {exc}") from exc
    finally:
        db.close()
