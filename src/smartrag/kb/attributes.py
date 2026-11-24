"""Attributes: what a publisher says about each document, in a folder beside the source.

A source folder `<name>` has its attributes in the sibling folder `<name>.attributes/`,
named after the folder, not the kb's source id. The location is fixed; nothing configures
it. The folder holds two files:

* `attributes.jsonl`: one JSON object per line, one document each.
  * `path` (required): the document's path relative to the source folder, `/`-separated,
    matched after Unicode NFC normalisation (macOS stores Japanese names decomposed).
  * `title` replaces the document's own title; `link`, an http(s) URL, is the citation's
    web link.
  * Every other key is an attribute. Values are flat (string, number, boolean or null),
    at most `MAX_VALUE_CHARS` characters.
* `schema.json` (optional): a small subset of JSON Schema saying what each attribute
  means: `type`, `format` (`date`, `date-time`, `uri`), `enum` and `description`. Anything
  else in it is ignored, so a schema written for other tools still works. A key the schema
  does not declare, or every key when there is no schema, is read as a string.

**Every attribute reaches agents.** The publisher decides what is disclosed by what it
writes in `attributes.jsonl`.

`sync` and `reindex` read the folder on every run, before any document, and replace the
source's attributes when they changed. Names that Smart RAG uses for its own result fields
are escaped to `_attr_<name>`, so an agent never sees two `doc_id`s with different
meanings; the `_attr_` prefix is reserved.

Errors are handled in two levels:

1. **Only the bad part is skipped.** A line that is not a JSON object, has a bad `path`,
   repeats a path, or names a file the source does not have is skipped. A value that is
   nested, too long, or does not match the schema is dropped, and the rest of its row is
   kept.
2. **The whole folder is ignored** only when `attributes.jsonl` cannot be read at all;
   the attributes applied last time are then kept, since the usual cause is a file being
   rewritten. A `schema.json` that cannot be read is ignored on its own.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from smartrag.kb.models import AttributesSync

if TYPE_CHECKING:
    from smartrag.index import SqliteIndex

FOLDER_SUFFIX = ".attributes"
ROWS_FILE = "attributes.jsonl"
SCHEMA_FILE = "schema.json"

ROLES = ("title", "link")
"""Keys with a role of their own; `path` is the row's key, not an attribute."""

ESCAPE = "_attr_"
CLASHING = frozenset({
    "kb_id", "doc_id", "chunk_id", "source_id", "file_path", "file_type", "page", "page_end",
    "location", "heading_path", "origin_path", "indexed_at", "rank", "score", "snippet",
    "text", "citation", "attributes", "source", "modified_after", "modified_before",
})
"""Smart RAG's own result and filter fields: an attribute of the same name is escaped."""

TYPES = ("string", "number", "integer", "boolean")
FORMATS = ("date", "date-time", "uri")
MAX_DESCRIPTION = 200
"""Characters of a schema description, publisher text that reaches agents; longer ones
are cut, as a kb's description is."""
MAX_VALUE_CHARS = 500
"""Characters of one value. Every attribute travels with every search hit, so a summary
paragraph in an attribute would multiply the cost of every result; 500 is about a chunk."""
EXAMPLES = 3
"""Line numbers given per kind of problem."""

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

Attribute = str | int | float | bool | None


def normalize_path(path: str) -> str:
    """The form attribute paths are stored and matched in."""
    return unicodedata.normalize("NFC", path)


def attributes_dir(root: Path) -> Path:
    """Where a source folder's attributes live."""
    return root.parent / f"{root.name}{FOLDER_SUFFIX}"


def is_date(text: str) -> bool:
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


# --- problems ----------------------------------------------------------------------


@dataclass
class _Problems:
    """Problems by kind, counted, with the first few places each."""

    counts: dict[str, int] = field(default_factory=dict)
    places: dict[str, list[str]] = field(default_factory=dict)

    def add(self, kind: str, place: str) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        listed = self.places.setdefault(kind, [])
        if len(listed) < EXAMPLES:
            listed.append(place)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def lines(self, what: str) -> list[str]:
        return [f"{count:,} {what}: {kind} ({', '.join(self.places[kind])}"
                f"{', …' if count > len(self.places[kind]) else ''})"
                for kind, count in self.counts.items()]


class Unreadable(Exception):
    """`attributes.jsonl` cannot be read at all."""


# --- schema ------------------------------------------------------------------------


def _scalar_of(kind: str, value: object) -> bool:
    if value is None:
        return True
    return {"string": isinstance(value, str),
            "boolean": isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": (isinstance(value, int) and not isinstance(value, bool))
            or (isinstance(value, float) and value.is_integer())}[kind]


def read_schema(path: Path, warnings: list[str]) -> dict[str, dict[str, Any]] | None:
    """The declared properties, keyed by the publisher's names; None if there is no usable
    schema. Each problem drops only what it concerns, with a warning."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
        warnings.append(f"{SCHEMA_FILE} cannot be read ({exc}); the attributes are read "
                        "without it")
        return None
    properties = data.get("properties") if isinstance(data, dict) else None
    if not isinstance(properties, dict):
        warnings.append(f"{SCHEMA_FILE} has no `properties` object; the attributes are read "
                        "without it")
        return None
    declared: dict[str, dict[str, Any]] = {}
    for key, spec in properties.items():
        where = f"{SCHEMA_FILE}: `{key}`"
        if key == "path":
            continue  # the row's key, not an attribute: declaring it is harmless
        if key.startswith(ESCAPE):
            warnings.append(f"{where} uses the reserved prefix `{ESCAPE}`; ignored")
            continue
        if not isinstance(spec, dict) or spec.get("type") not in TYPES:
            warnings.append(f"{where} has no `type` of {', '.join(TYPES)}; read as a string")
            continue
        kind = spec["type"]
        if key in ROLES and kind != "string":
            warnings.append(f"{where} is reserved and always a string; its entry is ignored")
            continue
        entry: dict[str, Any] = {"type": kind}
        fmt = spec.get("format")
        if fmt is not None:
            if fmt in FORMATS and kind == "string":
                entry["format"] = fmt
            else:
                warnings.append(f"{where}: `format` {fmt!r} ignored (one of "
                                f"{', '.join(FORMATS)}, on a string)")
        enum = spec.get("enum")
        if enum is not None:
            if isinstance(enum, list) and enum and all(
                    not isinstance(v, (dict, list)) and _scalar_of(kind, v) for v in enum):
                entry["enum"] = enum
            else:
                warnings.append(f"{where}: `enum` ignored (a non-empty list of {kind} values)")
        description = spec.get("description")
        if isinstance(description, str):
            flat = " ".join(description.split())
            if len(flat) > MAX_DESCRIPTION:
                warnings.append(f"{where}: the description is {len(flat)} characters; cut to "
                                f"{MAX_DESCRIPTION}")
                flat = flat[:MAX_DESCRIPTION - 1] + "…"
            if flat:
                entry["description"] = flat
        elif description is not None:
            warnings.append(f"{where}: `description` is not a string; ignored")
        declared[key] = entry
    return declared


# --- rows --------------------------------------------------------------------------


@dataclass
class ReadAttributes:
    rows: dict[str, dict[str, Attribute]]
    """By NFC path, with escaped names and typed values; `path` itself left out."""
    schema: dict[str, Any]
    """`{"properties": ...}` as applied: declared keys (escaped) plus undeclared ones as
    strings."""
    has_schema: bool
    keys: list[str]
    renamed: dict[str, str]
    untyped: list[str]
    skipped_lines: int
    dropped_values: int
    warnings: list[str]

    @property
    def digest(self) -> str:
        payload = json.dumps([self.schema, sorted(self.rows.items())], ensure_ascii=False,
                             sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reject_constant(name: str) -> float:
    raise ValueError(f"{name} is not a JSON number")


def _check_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("no `path`")
    if "\\" in value or value.startswith("/") or re.match(r"^[A-Za-z]:", value) or any(
            part in ("", ".", "..") for part in value.split("/")):
        raise ValueError("a bad `path`")
    return normalize_path(value)


def _value_problem(name: str, value: object, spec: dict[str, Any] | None) -> str | None:
    """Why a value is dropped, or None to keep it."""
    if isinstance(value, (dict, list)):
        return "a nested value"
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        return f"a value over {MAX_VALUE_CHARS} characters"
    if name == "title" and (not isinstance(value, str) or not value.strip()):
        return "a `title` that is not a non-empty string"
    if name == "link":
        parsed = urlparse(value) if isinstance(value, str) else None
        if parsed is None or parsed.scheme not in ("http", "https") or not parsed.netloc:
            return "a `link` that is not an http(s) URL"
    if spec is None or value is None:
        return None
    if not _scalar_of(spec["type"], value):
        return f"a value that is not {spec['type']} (schema)"
    fmt = spec.get("format")
    if fmt == "date" and not is_date(str(value)):
        return "a value that is not a date, YYYY-MM-DD (schema)"
    if fmt == "date-time" and not _is_datetime(str(value)):
        return "a value that is not an ISO 8601 date-time (schema)"
    if fmt == "uri" and not urlparse(str(value)).scheme:
        return "a value that is not a URI (schema)"
    if "enum" in spec and value not in spec["enum"]:
        return "a value not in the schema's `enum`"
    return None


def _as_string(value: Attribute) -> Attribute:
    """An undeclared key's value, read as a string; null stays null."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)  # true, 7203, 1.5: the form the publisher wrote


def read_folder(folder: Path, files: set[str] | None) -> ReadAttributes:
    """Read and check a source's attributes folder. `files` are the source's files as NFC
    paths; a row for any other path is skipped (None: no such check). Raises `Unreadable`
    when `attributes.jsonl` cannot be read at all."""
    warnings: list[str] = []
    schema_path = folder / SCHEMA_FILE
    declared = read_schema(schema_path, warnings) if schema_path.is_file() else None
    try:
        text = (folder / ROWS_FILE).read_text(encoding="utf-8-sig")
    except (OSError, ValueError) as exc:
        raise Unreadable(f"{ROWS_FILE} cannot be read ({exc})") from exc

    lines, values = _Problems(), _Problems()
    rows: dict[str, dict[str, Attribute]] = {}
    renamed: dict[str, str] = {}
    untyped: set[str] = set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, parse_constant=_reject_constant)
        except ValueError:
            lines.add("not valid JSON", str(number))
            continue
        if not isinstance(row, dict):
            lines.add("not a JSON object", str(number))
            continue
        try:
            path = _check_path(row.get("path"))
        except ValueError as exc:
            lines.add(str(exc), str(number))
            continue
        if path in rows:
            lines.add("a path already given", str(number))
            continue
        if files is not None and path not in files:
            lines.add("no such file in the source", str(number))
            continue
        attributes: dict[str, Attribute] = {}
        for name, value in row.items():
            if name == "path":
                continue
            if not name.strip():
                values.add("an empty name", f"line {number}")
                continue
            if name.startswith(ESCAPE):
                values.add(f"a name with the reserved prefix `{ESCAPE}`",
                           f"line {number} `{name}`")
                continue
            spec = declared.get(name) if declared else None
            problem = _value_problem(name, value, spec)
            if problem is not None:
                values.add(problem, f"line {number} `{name}`")
                continue
            if spec is None and name not in ROLES:
                untyped.add(name)
                value = _as_string(value)
            stored = ESCAPE + name if name in CLASHING else name
            if stored != name:
                renamed[name] = stored
            attributes[stored] = value
        rows[path] = attributes

    properties: dict[str, dict[str, Any]] = {}
    for name, spec in (declared or {}).items():
        properties[ESCAPE + name if name in CLASHING else name] = spec
    for name in untyped:
        properties.setdefault(ESCAPE + name if name in CLASHING else name, {"type": "string"})
    if declared is not None and untyped:
        warnings.append("not in the schema, read as strings: " + ", ".join(sorted(untyped)))
    warnings += lines.lines("lines skipped") + values.lines("values dropped")
    keys = sorted({key for attributes in rows.values() for key in attributes})
    return ReadAttributes(rows=rows, schema={"properties": dict(sorted(properties.items()))},
                          has_schema=declared is not None, keys=keys, renamed=renamed,
                          untyped=sorted(untyped), skipped_lines=lines.total,
                          dropped_values=values.total, warnings=warnings)


# --- applying during sync -----------------------------------------------------------


def sync_attributes(store: SqliteIndex | None, source_id: str, root: Path,
                    files: set[str] | None, *, dry_run: bool) -> AttributesSync:
    """Bring a source's attributes in the index in line with its folder. `files` are the
    source's listed files as NFC paths; None when the listing was incomplete, in which case
    nothing changes."""
    folder = attributes_dir(root)
    report = AttributesSync(source=source_id, folder=str(folder), status="none")
    previous = store.attribute_sets().get(source_id) if store is not None else None
    if files is None:
        report.status = "kept"
        report.warnings.append("the source's listing was incomplete; its attributes were "
                               "not checked")
        return report
    if not folder.exists():
        if previous is not None:
            report.status = "removed"
            if store is not None and not dry_run:
                store.delete_attributes(source_id)
        return report
    try:
        read = read_folder(folder, files)
    except Unreadable as exc:
        report.status = "kept"
        report.warnings.append(f"{exc}; " + ("the attributes applied before are kept"
                                             if previous is not None else "none applied"))
        return report
    report.rows = len(read.rows)
    report.keys = read.keys
    report.has_schema = read.has_schema
    report.renamed = read.renamed
    report.untyped = read.untyped
    report.skipped_lines = read.skipped_lines
    report.dropped_values = read.dropped_values
    report.warnings = read.warnings
    report.documents_without_row = len(files - set(read.rows))
    digest = read.digest
    if previous is not None and previous.digest == digest:
        report.status = "unchanged"
        return report
    report.status = "applied"
    if store is not None and not dry_run:
        store.replace_attributes(
            source_id,
            [(path, json.dumps(attributes, ensure_ascii=False))
             for path, attributes in sorted(read.rows.items())],
            json.dumps(read.schema, ensure_ascii=False), read.keys, digest,
            datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return report
