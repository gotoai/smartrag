"""Filters on attributes (`smartrag.kb.attributes`).

Every attribute key is filterable: the schema's type decides what a filter can say, and a
key the schema does not declare is a string. Strings and booleans take one value or a list; numbers and dates also take
a range, both ends included. A date-time attribute is matched on the date it is written
with. Lists mean OR, keys combine with AND, and a document matches only when its
attributes have a matching value: no row, no key or `null` never matches.

Matching runs before ranking. The documents a filter leaves are found first, from the
attribute rows (JSON, read with SQLite's JSON functions) joined to documents by NFC path,
and the search's WHERE clause then reads their ids, so both the keyword and the vector
side rank only their chunks.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from smartrag.errors import InvalidRequest
from smartrag.index import SqliteIndex
from smartrag.kb.attributes import ROLES, is_date, normalize_path
from smartrag.kb.home import Home
from smartrag.search.common import open_kb
from smartrag.search.models import AttributeCondition, AttributeRange, Scalar

MAX_VALUES = 1000
"""Values in one key's list: a set of entities chosen by SQL over a kb's tables (a
screen of listed companies passes about 600), while a filter stays a filter rather than
a document list. A list reaches SQLite as one JSON array, not one parameter per value."""

MAX_LISTED_VALUES = 50
"""`describe_sources` lists a key's values when it has at most this many: the 33 TSE
industries fit, the 4,169 securities codes do not."""


@dataclass
class FilterableKey:
    """A key some sources' attributes have, with the type they agree on."""

    name: str
    type: str
    format: str | None
    description: str | None
    sources: list[str] = field(default_factory=list)
    conflict: bool = False
    """Two sources give the key different types or formats: not filterable."""

    @property
    def ranges(self) -> bool:
        return self.type in ("integer", "number") or self.format in ("date", "date-time")

    @property
    def shapes(self) -> list[str]:
        if self.type == "boolean":
            return ["value"]
        return ["value", "list", "range"] if self.ranges else ["value", "list"]


def filterable_keys(store: SqliteIndex) -> dict[str, FilterableKey]:
    """Every key a source's attributes have, except `title` and `link`, which are display
    text."""
    keys: dict[str, FilterableKey] = {}
    for source_id, applied in store.attribute_sets().items():
        for name, spec in (applied.schema.get("properties") or {}).items():
            if name in ROLES or '"' in name:  # a `"` cannot be quoted in a JSON path
                continue
            key = keys.get(name)
            if key is None:
                key = keys[name] = FilterableKey(name, spec["type"], spec.get("format"),
                                                 spec.get("description"))
            elif (key.type, key.format) != (spec["type"], spec.get("format")):
                key.conflict = True
            key.sources.append(source_id)
    return keys


def key_values(store: SqliteIndex, key: FilterableKey) -> list[Scalar] | None:
    """A string or boolean key's distinct values, when there are few enough to list."""
    if key.conflict or key.ranges:
        return None
    marks = ",".join("?" * len(key.sources))
    rows = store.db.execute(
        f"SELECT DISTINCT {_value(key)} AS v FROM attribute_rows "
        f"WHERE source_id IN ({marks}) AND v IS NOT NULL ORDER BY v LIMIT ?",
        (*key.sources, MAX_LISTED_VALUES + 1)).fetchall()
    if len(rows) > MAX_LISTED_VALUES:
        return None
    return [bool(row[0]) if key.type == "boolean" else row[0] for row in rows]


def matching_documents(store: SqliteIndex,
                       conditions: Mapping[str, AttributeCondition]) -> list[str]:
    """The ids of the documents every condition matches."""
    keys = filterable_keys(store)
    usable = {name: key for name, key in keys.items() if not key.conflict}
    clauses: list[str] = []
    args: list[object] = []
    sources: set[str] | None = None
    for name, condition in conditions.items():
        key = usable.get(name)
        if key is None:
            raise _not_filterable(name, keys, usable)
        clause, values = _clause(key, condition)
        clauses.append(clause)
        args += values
        sources = set(key.sources) if sources is None else sources & set(key.sources)
    if not sources:
        return []
    where = " AND ".join(clauses)
    docs: list[str] = []
    for source_id in sorted(sources):
        paths = {row[0] for row in store.db.execute(
            f"SELECT path FROM attribute_rows WHERE source_id = ? AND {where}",
            (source_id, *args))}
        if not paths:
            continue
        docs += [row[0] for row in store.db.execute(
            "SELECT doc_id, rel_path FROM documents WHERE source_id = ? AND status = 'ok'",
            (source_id,)) if normalize_path(row[1]) in paths]
    return docs


SOURCE_KEY = "source"
"""Partitioning by `source` splits by the documents' source; no attribute has this name
(it is escaped to `_attr_source`)."""


@dataclass
class Partitioning:
    """Each document's value of a partition key, among the documents a search's filters
    leave."""

    key: str
    of: dict[str, Any]
    """doc_id → value, as SQL holds it; documents without a value are not here."""
    without_key: int
    empty: list[Any]
    """Listed values of the key's filter that no document has, as SQL holds them."""
    boolean: bool = False

    def display(self, value: Any) -> Any:
        return bool(value) if self.boolean and value is not None else value


def partitioning(store: SqliteIndex, name: str, docs: list[str],
                 conditions: Mapping[str, AttributeCondition]) -> Partitioning:
    """`docs` (the ids the filters leave) split by `name`: an attribute key or `source`."""
    left = set(docs)
    of: dict[str, Any] = {}
    boolean = False
    if name == SOURCE_KEY:
        of = {doc_id: source_id for doc_id, source_id in store.db.execute(
            "SELECT doc_id, source_id FROM documents WHERE status = 'ok'") if doc_id in left}
    else:
        keys = filterable_keys(store)
        key = keys.get(name)
        if key is None or key.conflict:
            usable = {n: k for n, k in keys.items() if not k.conflict}
            known = ", ".join([SOURCE_KEY, *sorted(usable)])
            reason = ("its sources give it different types" if key is not None
                      else "no source has this attribute")
            raise InvalidRequest(f"cannot partition by {name!r}: {reason}; partition by one "
                                 f"of: {known}", {"partition_by": name, "keys": known})
        boolean = key.type == "boolean"
        for source_id in key.sources:
            values = {row[0]: row[1] for row in store.db.execute(
                f"SELECT path, {_value(key)} AS v FROM attribute_rows "
                f"WHERE source_id = ? AND v IS NOT NULL", (source_id,))}
            for doc_id, rel_path in store.db.execute(
                    "SELECT doc_id, rel_path FROM documents WHERE source_id = ? "
                    "AND status = 'ok'", (source_id,)):
                if doc_id in left:
                    value = values.get(normalize_path(rel_path))
                    if value is not None:
                        of[doc_id] = value
    empty: list[Any] = []
    listed = conditions.get(name)
    if name != SOURCE_KEY and listed is not None and not isinstance(listed, AttributeRange):
        key = filterable_keys(store)[name]
        present = set(of.values())
        for item in listed if isinstance(listed, list) else [listed]:
            value = _scalar(key, item)
            if value not in present and value not in empty:
                empty.append(value)
    return Partitioning(key=name, of=of, without_key=len(left) - len(of), empty=empty,
                        boolean=boolean)


def documents_clause(doc_ids: list[str]) -> tuple[str, list[object]]:
    """A WHERE clause over `documents d` for a set of ids of any size: one JSON array,
    not one parameter per id, which SQLite caps."""
    return ("d.doc_id IN (SELECT value FROM json_each(?))",
            [json.dumps(doc_ids, ensure_ascii=False)])


# --- conditions --------------------------------------------------------------------


def _value(key: FilterableKey) -> str:
    """The SQL expression for a row's value of the key."""
    extract = f"json_extract(attributes, '$.\"{key.name.replace(chr(39), chr(39) * 2)}\"')"
    return f"substr({extract}, 1, 10)" if key.format == "date-time" else extract


def _clause(key: FilterableKey, condition: AttributeCondition) -> tuple[str, list[object]]:
    value = _value(key)
    if isinstance(condition, AttributeRange):
        if not key.ranges:
            raise InvalidRequest(f"attribute {key.name!r} is a {_kind(key)}: give a value or "
                                 "a list, not a range", {"attribute": key.name})
        start = None if condition.start is None else _scalar(key, condition.start)
        end = None if condition.end is None else _scalar(key, condition.end)
        if start is None and end is None:
            raise InvalidRequest(f"the range for attribute {key.name!r} has neither `from` "
                                 "nor `to`", {"attribute": key.name})
        parts: list[str] = []
        args: list[object] = []
        if start is not None:
            parts.append(f"{value} >= ?")
            args.append(start)
        if end is not None:
            parts.append(f"{value} <= ?")
            args.append(end)
        return "(" + " AND ".join(parts) + ")", args
    if isinstance(condition, list):
        if key.type == "boolean":
            raise InvalidRequest(f"attribute {key.name!r} is a boolean: give true or false, "
                                 "not a list", {"attribute": key.name})
        if not condition:
            raise InvalidRequest(f"the list for attribute {key.name!r} is empty",
                                 {"attribute": key.name})
        if len(condition) > MAX_VALUES:
            raise InvalidRequest(f"the list for attribute {key.name!r} has {len(condition)} "
                                 f"values; the limit is {MAX_VALUES}", {"attribute": key.name})
        values = [_scalar(key, item) for item in condition]
        return f"{value} IN (SELECT value FROM json_each(?))", [
            json.dumps(values, ensure_ascii=False)]
    return f"{value} = ?", [_scalar(key, condition)]


def _scalar(key: FilterableKey, given: Scalar) -> object:
    """A filter value as the SQL value the key's rows hold. A number given for a string
    key is read as its digits, and a numeric string for a number key as the number, since
    callers (agents above all) blur the two; anything else of the wrong type is refused."""
    wrong = InvalidRequest(f"attribute {key.name!r} is a {_kind(key)}; {given!r} is not one",
                           {"attribute": key.name})
    if key.type == "boolean":
        if isinstance(given, bool):
            return int(given)
        if isinstance(given, str) and given.lower() in ("true", "false"):
            return int(given.lower() == "true")
        raise wrong
    if key.type in ("integer", "number"):
        number: float | int
        if isinstance(given, bool):
            raise wrong
        if isinstance(given, (int, float)):
            number = given
        else:
            try:
                number = float(given) if key.type == "number" else int(given)
            except ValueError:
                raise wrong from None
        if isinstance(number, float) and not math.isfinite(number):
            raise wrong
        return number
    if isinstance(given, (bool, float)):
        raise wrong
    text = str(given)
    if key.format in ("date", "date-time") and not is_date(text):
        raise InvalidRequest(f"attribute {key.name!r} is a {_kind(key)}: give a date as "
                             f"YYYY-MM-DD, not {given!r}", {"attribute": key.name})
    return text


def _kind(key: FilterableKey) -> str:
    return {"date": "date", "date-time": "date-time"}.get(key.format or "", key.type)


def _not_filterable(name: str, keys: Mapping[str, FilterableKey],
                    usable: Mapping[str, FilterableKey]) -> InvalidRequest:
    known = ", ".join(sorted(usable)) or "none"
    reason = ("its sources give it different types" if name in keys and keys[name].conflict
              else "no source has this attribute")
    return InvalidRequest(f"attribute {name!r} cannot be filtered on: {reason}; the "
                          f"filterable attributes are: {known}",
                          {"attribute": name, "filterable": known})


# --- the CLI's --attr --------------------------------------------------------------


def attr_options(home: Home, kb_id: str, options: list[str]) -> dict[str, AttributeCondition]:
    """`--attr` options for a kb, read with its keys' types."""
    if not options:
        return {}
    kb = open_kb(home, kb_id, check_space=False)
    try:
        return parse_options(kb.store, options)
    finally:
        kb.close()


def parse_options(store: SqliteIndex, options: list[str]) -> dict[str, AttributeCondition]:
    """`--attr KEY=VALUE` options as conditions. A repeated key is a list; for a number
    or date key, `FROM..TO`, `FROM..` or `..TO` is a range. Values stay strings: the key's
    type reads them when the filter is matched."""
    keys = filterable_keys(store)
    values: dict[str, list[str]] = {}
    ranges: dict[str, AttributeRange] = {}
    for option in options:
        name, sep, value = option.partition("=")
        name = name.strip()
        if not sep or not name:
            raise InvalidRequest(f"--attr takes KEY=VALUE, not {option!r}")
        key = keys.get(name)
        if key is not None and key.ranges and ".." in value:
            start, _, end = value.partition("..")
            if name in ranges or name in values:
                raise InvalidRequest(f"--attr {name}: a range cannot be combined with other "
                                     "values for the same key", {"attribute": name})
            ranges[name] = AttributeRange.model_validate(
                {"from": start.strip() or None, "to": end.strip() or None})
            continue
        if name in ranges:
            raise InvalidRequest(f"--attr {name}: a range cannot be combined with other "
                                 "values for the same key", {"attribute": name})
        values.setdefault(name, []).append(value)
    conditions: dict[str, AttributeCondition] = dict(ranges)
    for name, given in values.items():
        conditions[name] = given[0] if len(given) == 1 else list(given)
    return conditions
