"""JSON, JSON Lines, YAML and XML: data flattened to `path: value` lines.

Read as plain text, a JSON file is one long paragraph whose keys sit far from most of
their values. Flattened, every line carries its own path (`orders[3].customer.name:
山田`), so any chunk of it still says what its values are.

**Records stay together.** Each element of an array, each line of a JSON Lines file, and
the plain values of each object become one unit, so the chunker breaks between records
rather than inside one. In a file that is one array, or a JSON Lines file, paths start
inside each record.
Nulls and empty containers carry nothing searchable and are left out. A YAML anchor used
twice is flattened once, so a file built from aliases cannot multiply itself.

**XML** keeps text and attribute values. An element whose children are all leaves is one
record (`name: …`, `price@currency: …`); an element with text between its children is
prose (DocBook, XHTML-like content) and becomes one unit of its joined text, with no
spaces inserted at inline tags. Comments and processing instructions are dropped.

A file that does not parse is indexed as plain text, with a warning.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path
from typing import Any

from smartrag.extract.encoding import decode_bytes
from smartrag.extract.text import paragraphs
from smartrag.extract.units import Extracted, Unit


def _plain(text: str, why: str) -> Extracted:
    return Extracted(units=paragraphs(text), warnings=[f"{why}; indexed as plain text"])


def _decoded(path: Path) -> tuple[str, list[str]]:
    text, encoding = decode_bytes(path.read_bytes())
    warnings = [f"some bytes could not be decoded as {encoding}"] if "�" in text else []
    return text, warnings


# --- JSON, JSON Lines, YAML --------------------------------------------------------


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _lines(value: Any, path: str, seen: set[int]) -> Iterator[str]:
    if isinstance(value, (dict, list)):
        if id(value) in seen:
            return
        seen.add(id(value))
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, child in items:
            if isinstance(value, dict):
                child_path = f"{path}.{key}" if path else str(key)
            else:
                child_path = f"{path}[{key}]"
            yield from _lines(child, child_path, seen)
    elif value is not None:
        text = _scalar(value)
        yield f"{path}: {text}" if path else text


def _records(value: Any, path: str, seen: set[int]) -> Iterator[Unit]:
    if isinstance(value, list):
        if id(value) in seen:
            return
        seen.add(id(value))
        for index, element in enumerate(value):
            prefix = f"{path}[{index}]" if path else ""
            body = "\n".join(_lines(element, prefix, seen))
            if body:
                yield Unit(body)
    elif isinstance(value, dict):
        if id(value) in seen:
            return
        seen.add(id(value))
        plain: list[str] = []
        nested: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(child, (dict, list)):
                nested.append((child_path, child))
            elif child is not None:
                plain.append(f"{child_path}: {_scalar(child)}")
        if plain:
            yield Unit("\n".join(plain))
        for child_path, child in nested:
            yield from _records(child, child_path, seen)
    elif value is not None:
        yield Unit(_scalar(value))


def data_units(value: Any) -> list[Unit]:
    """A parsed JSON or YAML value as units, one per record."""
    return list(_records(value, "", set()))


def extract_json(path: Path) -> Extracted:
    text, warnings = _decoded(path)
    try:
        value = json.loads(text)
    except (ValueError, RecursionError) as exc:
        return _plain(text, f"not valid JSON ({str(exc) or type(exc).__name__})")
    return Extracted(units=data_units(value), warnings=warnings)


def extract_jsonl(path: Path) -> Extracted:
    text, warnings = _decoded(path)
    units: list[Unit] = []
    invalid = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except (ValueError, RecursionError):
            invalid += 1
            units.append(Unit(line.strip()))
            continue
        body = "\n".join(_lines(value, "", set()))
        if body:
            units.append(Unit(body))
    if invalid:
        warnings.append(f"{invalid} line{'' if invalid == 1 else 's'} not valid JSON; "
                        "indexed as plain text")
    return Extracted(units=units, warnings=warnings)


def extract_yaml(path: Path) -> Extracted:
    from ruamel.yaml import YAML
    from ruamel.yaml.error import YAMLError

    text, warnings = _decoded(path)
    try:
        documents = list(YAML(typ="safe", pure=True).load_all(text))
    except (YAMLError, ValueError, RecursionError) as exc:
        reason = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return _plain(text, f"not valid YAML ({reason})")
    units: list[Unit] = []
    for document in documents:
        units += data_units(document)
    return Extracted(units=units, warnings=warnings)


# --- XML ---------------------------------------------------------------------------


def _local(name: str) -> str:
    """A tag or attribute name without its `{namespace}`."""
    return name.rsplit("}", 1)[-1]


def _space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _attributes(element: Any) -> list[str]:
    tag = _local(element.tag)
    return [f"{tag}@{_local(name)}: {_space(value)}"
            for name, value in element.attrib.items() if _space(value)]


def _leaf_lines(element: Any) -> list[str]:
    text = _space(element.text)
    return _attributes(element) + ([f"{_local(element.tag)}: {text}"] if text else [])


def _xml_units(element: Any, units: list[Unit]) -> None:
    children = [child for child in element if isinstance(child.tag, str)]
    if not children:
        lines = _leaf_lines(element)
        if lines:
            units.append(Unit("\n".join(lines)))
        return
    if _space(element.text) or any(_space(child.tail) for child in children):
        prose = re.sub(r"\s+", " ", "".join(element.itertext())).strip()
        if prose:
            units.append(Unit(prose))
        return
    own = _attributes(element)
    if all(len(child) == 0 for child in children):
        lines = own + [line for child in children for line in _leaf_lines(child)]
        if lines:
            units.append(Unit("\n".join(lines)))
        return
    if own:
        units.append(Unit("\n".join(own)))
    for child in children:
        _xml_units(child, units)


_XML_DECLARATION = re.compile(r"^\ufeff?\s*<\?xml[^>]*\?>")
_DECLARED_ENCODING = re.compile(rb"^\s*<\?xml[^>]*?encoding\s*=\s*[\"']([A-Za-z0-9._-]+)[\"']")


def _parse_xml(data: bytes) -> Any:
    import xml.etree.ElementTree as ET

    # Expat (2.4 and later) refuses entity expansion attacks, and ElementTree never fetches
    # external entities.
    try:
        return ET.fromstring(data)  # bytes: expat follows the file's encoding declaration
    except ValueError:
        # Expat reads no multi-byte encoding but UTF-8 and UTF-16, so Shift_JIS and EUC-JP
        # files are decoded here, by their declaration, and parsed without it.
        declared = _DECLARED_ENCODING.match(data)
        try:
            text = data.decode(declared.group(1).decode("ascii")) if declared \
                else decode_bytes(data)[0]
        except (LookupError, UnicodeDecodeError):
            text = decode_bytes(data)[0]
        return ET.fromstring(_XML_DECLARATION.sub("", text, count=1))


def extract_xml(path: Path) -> Extracted:
    import xml.etree.ElementTree as ET

    data = path.read_bytes()
    try:
        root = _parse_xml(data)
    except ET.ParseError as exc:
        text, _encoding = decode_bytes(data)
        return _plain(text, f"not well-formed XML ({exc})")
    units: list[Unit] = []
    _xml_units(root, units)
    return Extracted(units=units)
