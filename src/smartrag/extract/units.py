"""What every extractor produces: an ordered list of `Unit`s.

A unit is the smallest piece the chunker never splits unless it has to: a heading, a
paragraph (a PDF text box, a DOCX paragraph, a slide's text frame), a table's header
row, or one table row in TSV. The chunker packs units into chunks; the document's
canonical text is the units joined by newlines.

`extract(path)` dispatches on the file's extension. `EXTRACTOR_VERSION` goes into the
pipeline version: bump it whenever an extractor's output changes, and `kb reindex
--refresh-stale-only` will re-extract what an older extractor produced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from smartrag.errors import Unsupported

EXTRACTOR_VERSION = "3"  # 2: PDF lines joined; 3: deterministic PDF reading order

UnitKind = Literal["heading", "text", "header", "row"]


@dataclass
class Unit:
    text: str
    kind: UnitKind = "text"
    level: int = 0
    """Heading level, 1 = top."""
    page: int | None = None
    """PDF page or PPTX slide, from 1."""
    location: str | None = None
    """Where the unit's table or frame sits: `見積!B4:E40`, `slide 3 notes`."""
    group: int | None = None
    """Table id within the document: a table's rows share chunks only with each other."""
    row: int | None = None
    """Source row number of a table row."""


@dataclass
class Extracted:
    units: list[Unit]
    title: str | None = None
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def has_text(self) -> bool:
        return any(u.kind != "heading" and u.text.strip() for u in self.units)


# --- headings in plain text and PDFs -----------------------------------------------

_NUM = r"[0-9０-９一二三四五六七八九十百千]+"
_HEADING_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(rf"^第{_NUM}[編章](?:\s|　|$)"), 1),
    (re.compile(rf"^第{_NUM}節(?:\s|　|$)"), 2),
    (re.compile(rf"^第{_NUM}(?:\s|　)"), 2),
    (re.compile(rf"^第{_NUM}条(?:の{_NUM})?(?:\s|　|$)"), 3),
    (re.compile(r"^【[^】]{1,40}】$"), 2),
    (re.compile(r"^[■◆□◇]\s*\S"), 3),
]
_MAX_HEADING_CHARS = 60


def heading_level(line: str) -> int:
    """The level of a Japanese-style heading line, or 0. A heading is short and does not
    end a sentence: `第12条 年次有給休暇` is one, `第12条 社員は…取得できる。` is not."""
    text = line.strip()
    if not text or len(text) > _MAX_HEADING_CHARS or text.endswith(("。", ".")):
        return 0
    for pattern, level in _HEADING_PATTERNS:
        if pattern.match(text):
            return level
    return 0


# --- dispatch ----------------------------------------------------------------------

TEXT_EXTENSIONS = frozenset({".txt", ".text", ".md", ".markdown"})
HTML_EXTENSIONS = frozenset({".html", ".htm", ".xhtml"})


def extract(path: Path) -> Extracted:
    """Extract a supported file. Raises `Unsupported` for other types and
    `ExtractError` for a file that cannot be read as what its name says."""
    ext = path.suffix.lower()
    if ext in TEXT_EXTENSIONS:
        from smartrag.extract.text import extract_text

        return extract_text(path, markdown=ext in (".md", ".markdown"))
    if ext in HTML_EXTENSIONS:
        from smartrag.extract.text import extract_html

        return extract_html(path)
    if ext == ".pdf":
        from smartrag.extract.pdf import extract_pdf

        return extract_pdf(path)
    if ext == ".docx":
        from smartrag.extract.office import extract_docx

        return extract_docx(path)
    if ext == ".pptx":
        from smartrag.extract.office import extract_pptx

        return extract_pptx(path)
    from smartrag.extract.tabular import TABULAR_EXTENSIONS, convert_tabular

    if ext in TABULAR_EXTENSIONS:
        return tabular_units(convert_tabular(path))
    raise Unsupported(f"no extractor for {ext or 'files without an extension'}")


def tabular_units(doc: object) -> Extracted:
    """Table blocks as units: a heading per sheet and per table title, then the header
    row and the data rows of each table, grouped so they chunk together."""
    from smartrag.extract.tabular import TabularDocument

    assert isinstance(doc, TabularDocument)
    units: list[Unit] = []
    sheet: str | None = None
    for block in doc.blocks:
        if block.sheet is not None and block.sheet != sheet:
            sheet = block.sheet
            units.append(Unit(sheet, "heading", level=1))
        if block.title:
            units.append(Unit(block.title, "heading", level=2))
        header = block.header_tsv()
        if header is not None:
            units.append(Unit(header, "header", location=block.location, group=block.index,
                              row=block.header_row))
        for i, number in enumerate(block.row_numbers):
            units.append(Unit(block.row_tsv(i), "row", location=block.location,
                              group=block.index, row=number))
    return Extracted(units=units, warnings=list(doc.warnings))
