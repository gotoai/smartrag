"""`source apply`: per-document attributes from a publisher's manifest (design memo §3.10).

A manifest is one JSONL file, or a set applied together; each line describes one document
of one source:

* `path` (required, unique across the set): the document's path relative to the source
  root, `/`-separated, compared after Unicode NFC normalisation;
* every other key is an attribute with a flat value: string, number, boolean or null;
* reserved: `title` (a non-empty string, shown instead of the file's own title) and `link`
  (an `http(s)` URL, which fills `citation.link`).

An optional schema, a small subset of JSON Schema, says what each attribute means. With
one, every key a row uses must be declared and every value must match its type and format.

Applying replaces the source's previous manifest, all or nothing: everything is validated
before anything changes. Nothing is re-extracted or re-embedded; rows are joined to
documents by path when results are built, so a file synced later gets its row.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from smartrag.errors import InvalidRequest, NotFound
from smartrag.kb.config import SINK_ID
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.models import ManifestApplied

RESERVED = ("title", "link")
"""Attribute keys with a meaning of their own; `path` is the row's key, not an attribute."""

TYPES = ("string", "number", "integer", "boolean")
FORMATS = ("date", "date-time", "uri")
MAX_DESCRIPTION = 200
"""Characters of a schema description: publisher text that reaches agents (§3.10)."""

MAX_PROBLEMS = 20
"""Problems listed in one refusal; the count says how many more there were."""

_SCHEMA_KEYS = {"$schema", "$id", "title", "description", "type", "properties"}
_PROPERTY_KEYS = {"type", "format", "description", "title"}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

Attribute = str | int | float | bool | None


def normalize_path(path: str) -> str:
    """The form manifest paths are stored and matched in."""
    return unicodedata.normalize("NFC", path)


# --- schema ------------------------------------------------------------------------


def read_schema(path: Path) -> dict[str, Any]:
    """A validated schema: `{"properties": {key: {"type", "format"?, "description"?}}}`."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise InvalidRequest(f"cannot read the schema {path}: {exc.strerror or exc}",
                             {"file": str(path)}) from exc
    except ValueError as exc:
        raise InvalidRequest(f"{path}: not valid JSON ({exc})", {"file": str(path)}) from exc
    problems: list[str] = []
    if not isinstance(data, dict):
        raise InvalidRequest(f"{path}: a schema is a JSON object", {"file": str(path)})
    for key in sorted(set(data) - _SCHEMA_KEYS):
        problems.append(f"`{key}` is not in the supported subset of JSON Schema "
                        f"({', '.join(sorted(_SCHEMA_KEYS))})")
    if data.get("type", "object") != "object":
        problems.append("`type` must be \"object\"")
    properties = data.get("properties")
    if not isinstance(properties, dict) or not properties:
        problems.append("`properties` must be an object with at least one attribute")
        properties = {}
    clean: dict[str, dict[str, str]] = {}
    for key, spec in properties.items():
        where = f"property `{key}`"
        if key == "path":
            continue  # the row's key, not an attribute: declaring it is harmless
        if not isinstance(spec, dict):
            problems.append(f"{where} must be an object")
            continue
        for extra in sorted(set(spec) - _PROPERTY_KEYS):
            problems.append(f"{where}: `{extra}` is not supported "
                            f"({', '.join(sorted(_PROPERTY_KEYS))})")
        kind = spec.get("type")
        if kind not in TYPES:
            problems.append(f"{where}: `type` must be one of {', '.join(TYPES)}")
            continue
        entry = {"type": kind}
        fmt = spec.get("format")
        if fmt is not None:
            if fmt not in FORMATS or kind != "string":
                problems.append(f"{where}: `format` must be one of {', '.join(FORMATS)}, "
                                "on a string")
                continue
            entry["format"] = fmt
        description = spec.get("description")
        if description is not None:
            if not isinstance(description, str):
                problems.append(f"{where}: `description` must be a string")
                continue
            description = " ".join(description.split())
            if len(description) > MAX_DESCRIPTION:
                problems.append(f"{where}: the description is {len(description)} characters; "
                                f"the limit is {MAX_DESCRIPTION}")
                continue
            if description:
                entry["description"] = description
        if key in RESERVED and kind != "string":
            problems.append(f"{where}: `{key}` is reserved and must be a string")
            continue
        clean[key] = entry
    if problems:
        _refuse([f"{path}: {p}" for p in problems], {"file": str(path)})
    return {"properties": clean}


# --- rows --------------------------------------------------------------------------


@dataclass
class _Rows:
    rows: dict[str, dict[str, Attribute]] = field(default_factory=dict)
    where: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not a JSON number")


def _check_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`path` must be a non-empty string")
    if "\\" in value:
        raise ValueError(f"`path` uses `/` separators, not `\\`: {value!r}")
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"`path` is relative to the source root, not absolute: {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"`path` has an empty, `.` or `..` segment: {value!r}")
    return normalize_path(value)


def _check_value(key: str, value: object, spec: dict[str, str] | None) -> None:
    if isinstance(value, (dict, list)):
        raise TypeError(f"`{key}` is nested; values must be a string, number, boolean or "
                         "null")
    if key == "title" and (not isinstance(value, str) or not value.strip()):
        raise ValueError("`title` must be a non-empty string")
    if key == "link":
        parsed = urlparse(value) if isinstance(value, str) else None
        if parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"`link` must be an http(s) URL, not {value!r}")
    if spec is None or value is None:
        return
    kind = spec["type"]
    ok = {"string": isinstance(value, str),
          "boolean": isinstance(value, bool),
          "number": isinstance(value, (int, float)) and not isinstance(value, bool),
          "integer": (isinstance(value, int) and not isinstance(value, bool))
          or (isinstance(value, float) and value.is_integer())}[kind]
    if not ok:
        raise ValueError(f"`{key}` must be {kind} (schema), not {value!r}")
    fmt = spec.get("format")
    assert isinstance(value, str) or fmt is None
    if fmt == "date" and not _is_date(str(value)):
        raise ValueError(f"`{key}` must be a date, YYYY-MM-DD (schema), not {value!r}")
    if fmt == "date-time" and not _is_datetime(str(value)):
        raise ValueError(f"`{key}` must be an ISO 8601 date-time (schema), not {value!r}")
    if fmt == "uri" and not urlparse(str(value)).scheme:
        raise ValueError(f"`{key}` must be a URI (schema), not {value!r}")


def _is_date(text: str) -> bool:
    try:
        return bool(_DATE.match(text)) and date.fromisoformat(text) is not None
    except ValueError:
        return False


def _is_datetime(text: str) -> bool:
    if "T" not in text and " " not in text:
        return False
    try:
        datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text)
    except ValueError:
        return False
    return True


def read_manifest(paths: Sequence[Path], schema: dict[str, Any] | None) -> dict[str, dict[str, Attribute]]:
    """Every row of a manifest set, by NFC path, without the `path` key. Refuses the whole
    set on the first problems found (up to `MAX_PROBLEMS`, all counted)."""
    if not paths:
        raise InvalidRequest("give at least one --manifest file")
    properties: dict[str, dict[str, str]] | None = schema["properties"] if schema else None
    result = _Rows()
    total = 0
    for path in paths:
        try:
            handle = path.open(encoding="utf-8-sig")
        except OSError as exc:
            raise InvalidRequest(f"cannot read the manifest {path}: {exc.strerror or exc}",
                                 {"file": str(path)}) from exc
        with handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                where = f"{path}:{number}"
                try:
                    row = json.loads(line, parse_constant=_reject_constant)
                    if not isinstance(row, dict):
                        raise TypeError("each line must be a JSON object")
                    key = _check_path(row.get("path"))
                    for name, value in row.items():
                        if name == "path":
                            continue
                        if not name.strip():
                            raise ValueError("an attribute name is empty")
                        if isinstance(value, float) and not math.isfinite(value):
                            raise ValueError(f"`{name}` is not a finite number")
                        if properties is not None and name not in properties:
                            raise ValueError(f"`{name}` is not declared in the schema")
                        _check_value(name, value, properties.get(name) if properties else None)
                    if key in result.rows:
                        raise ValueError(f"`path` {key!r} is also at {result.where[key]}")
                except (ValueError, TypeError) as exc:
                    total += 1
                    if len(result.problems) < MAX_PROBLEMS:
                        result.problems.append(f"{where}: {exc}")
                    continue
                result.rows[key] = {k: v for k, v in row.items() if k != "path"}
                result.where[key] = where
    if result.problems:
        more = total - len(result.problems)
        _refuse(result.problems + ([f"... and {more} more"] if more else []),
                {"problems": total})
    if not result.rows:
        raise InvalidRequest("the manifest has no rows; to remove a source's manifest, use "
                             "--clear")
    return result.rows


def _refuse(problems: list[str], details: dict[str, Any]) -> None:
    head = problems[0] if len(problems) == 1 else f"{len(problems)} problems; nothing was applied"
    message = head if len(problems) == 1 else head + ":\n" + "\n".join(
        f"  {p}" for p in problems)
    if len(problems) == 1:
        message += "\nNothing was applied."
    raise InvalidRequest(message, {**details, "messages": problems})


# --- apply -------------------------------------------------------------------------


def apply_manifest(home: Home, kb_id: str, source_id: str, manifests: Sequence[Path], *,
                   schema: Path | None = None, dry_run: bool = False) -> ManifestApplied:
    _check_source(home, kb_id, source_id)
    parsed_schema = read_schema(schema) if schema is not None else None
    rows = read_manifest(manifests, parsed_schema)
    keys = sorted({k for attributes in rows.values() for k in attributes})
    report = ManifestApplied(kb=kb_id, source=source_id, dry_run=dry_run, rows=len(rows),
                             attributes=keys, has_schema=parsed_schema is not None)
    payload = [(path, json.dumps(attributes, ensure_ascii=False))
               for path, attributes in sorted(rows.items())]
    _write(home, kb_id, source_id, report, set(rows), payload,
           json.dumps(parsed_schema, ensure_ascii=False) if parsed_schema else None, keys)
    return report


def clear_manifest(home: Home, kb_id: str, source_id: str, *,
                   dry_run: bool = False) -> ManifestApplied:
    _check_source(home, kb_id, source_id)
    report = ManifestApplied(kb=kb_id, source=source_id, dry_run=dry_run, cleared=True)
    _write(home, kb_id, source_id, report, set(), [], None, [])
    return report


def _check_source(home: Home, kb_id: str, source_id: str) -> None:
    if source_id == SINK_ID:
        raise InvalidRequest("the sink takes no manifest: `ingest` has no attributes for now")
    config = home.load(kb_id)
    if config.kind != "local":
        raise InvalidRequest(f"kb {kb_id!r} is remote; its sources are managed by its server",
                             {"kb": kb_id})
    if not any(s.id == source_id for s in config.sources):
        raise NotFound(f"kb {kb_id!r} has no source {source_id!r}",
                       {"kb": kb_id, "source": source_id})


def _write(home: Home, kb_id: str, source_id: str, report: ManifestApplied, paths: set[str],
           payload: list[tuple[str, str]], schema: str | None, keys: list[str]) -> None:
    """Count matches against the source's indexed documents, then (unless a dry run)
    replace the manifest under the kb lock."""
    from smartrag.index import SqliteIndex

    kb_dir = home.kb_dir(kb_id)
    index_path = kb_dir / INDEX_FILE
    with kb_lock(kb_dir, "source apply"):
        if not index_path.is_file() and report.dry_run:
            report.note = "the kb has no index yet; every row waits for its file"
            report.rows_without_document = len(paths)
            return
        store = SqliteIndex.open(index_path, readonly=report.dry_run)
        try:
            documents = {normalize_path(row.rel_path)
                         for row in store.documents(source_id).values() if row.status == "ok"}
            previous = store.manifests().get(source_id)
            report.previous_rows = previous.rows if previous else 0
            report.documents = len(documents)
            if not report.cleared:
                report.matched = len(paths & documents)
                report.rows_without_document = len(paths - documents)
                report.documents_without_row = len(documents - paths)
                report.unmatched_examples = sorted(paths - documents)[:5]
            if report.cleared and previous is None:
                report.note = "the source had no manifest"
                return
            if not report.dry_run:
                store.replace_manifest(source_id, payload, schema, keys,
                                       datetime.now(timezone.utc).isoformat(timespec="seconds"))
        finally:
            store.close()
