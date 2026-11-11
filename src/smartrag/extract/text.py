"""Plain text, Markdown and HTML.

Plain text and Markdown split into paragraphs at blank lines. Markdown `#` headings and
Japanese-style heading lines (`第3章`, `第12条`, `【…】`, `■…`) become headings; fenced
code stays one unit, never parsed for headings.

HTML is reduced to block-level text with the standard library's parser: `h1`-`h6` become
headings, `tr` rows become TSV, and `script`, `style`, `nav`, `header`, `footer` and
`aside` are dropped. Text inside one block is joined **without** inserting spaces at inline
tags, so `東京<b>都</b>庁` stays `東京都庁`.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from smartrag.extract.encoding import decode_bytes
from smartrag.extract.units import Extracted, Unit, UnitKind, heading_level

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def extract_text(path: Path, *, markdown: bool) -> Extracted:
    text, encoding = decode_bytes(path.read_bytes())
    extracted = Extracted(units=_paragraphs(text, markdown))
    if "�" in text:
        extracted.warnings.append(f"some bytes could not be decoded as {encoding}")
    headings = [u for u in extracted.units if u.kind == "heading" and u.level == 1]
    if headings:
        extracted.title = headings[0].text
    return extracted


def _paragraphs(text: str, markdown: bool) -> list[Unit]:
    units: list[Unit] = []
    buffer: list[str] = []
    fence: list[str] | None = None

    def flush() -> None:
        if buffer:
            units.append(Unit("\n".join(buffer).strip()))
            buffer.clear()

    for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if markdown and _FENCE.match(line):
            if fence is None:
                flush()
                fence = [line]
            else:
                fence.append(line)
                units.append(Unit("\n".join(fence)))
                fence = None
            continue
        if fence is not None:
            fence.append(line)
            continue
        if not line.strip():
            flush()
            continue
        md = _MD_HEADING.match(line) if markdown else None
        level = len(md.group(1)) if md else heading_level(line)
        if level:
            flush()
            units.append(Unit(md.group(2) if md else line.strip(), "heading", level=level))
            continue
        buffer.append(line.rstrip())
    if fence is not None:
        units.append(Unit("\n".join(fence)))
    flush()
    return [u for u in units if u.text]


# --- HTML --------------------------------------------------------------------------

_SKIP = frozenset({"script", "style", "nav", "header", "footer", "aside", "noscript",
                   "template", "svg"})
_BLOCK = frozenset({"p", "div", "li", "dd", "dt", "blockquote", "pre", "section",
                    "article", "main", "figcaption", "caption", "td", "th", "tr", "table",
                    "ul", "ol", "dl", "form", "address", "body", "html"})
_HEADINGS = {f"h{n}": n for n in range(1, 7)}


class _Blocks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.units: list[Unit] = []
        self.title: str | None = None
        self._text: list[str] = []
        self._skip = 0
        self._heading = 0
        self._in_title = False
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._group = 0
        self._pre = 0

    def _flush(self) -> None:
        text = "".join(self._text)
        self._text = []
        text = text if self._pre else re.sub(r"[ \t\r\n]+", " ", text)
        text = text.strip()
        if text:
            kind: UnitKind = "heading" if self._heading else "text"
            self.units.append(Unit(text, kind, level=self._heading))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = True
        elif tag == "br":
            (self._cell if self._cell is not None else self._text).append("\n")
        elif tag == "tr":
            self._flush()
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag in _HEADINGS:
            self._flush()
            self._heading = _HEADINGS[tag]
        elif tag in _BLOCK:
            if tag == "pre":
                self._pre += 1
            if self._cell is None:
                self._flush()
        if tag == "table":
            self._group += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "title":
            self._in_title = False
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                from smartrag.extract.tabular import tsv_line

                self.units.append(Unit(tsv_line(self._row), "row", group=self._group))
            self._row = None
        elif tag in _HEADINGS:
            self._flush()
            self._heading = 0
        elif tag in _BLOCK and self._cell is None:
            self._flush()
            if tag == "pre":
                self._pre = max(0, self._pre - 1)

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self.title = ((self.title or "") + data).strip() or None
        elif self._cell is not None:
            self._cell.append(data)
        else:
            self._text.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def extract_html(path: Path) -> Extracted:
    text, encoding = decode_bytes(path.read_bytes())
    parser = _Blocks()
    parser.feed(text)
    parser.close()
    units = _mark_table_headers(parser.units)
    extracted = Extracted(units=units, title=parser.title)
    if "�" in text:
        extracted.warnings.append(f"some bytes could not be decoded as {encoding}")
    return extracted


def _mark_table_headers(units: list[Unit]) -> list[Unit]:
    """The first row of each HTML table becomes its header, repeated on every chunk."""
    seen: set[int] = set()
    for unit in units:
        if unit.kind == "row" and unit.group is not None and unit.group not in seen:
            seen.add(unit.group)
            unit.kind = "header"
    return units
