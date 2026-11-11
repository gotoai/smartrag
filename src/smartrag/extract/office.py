"""Word (.docx) and PowerPoint (.pptx).

DOCX: body paragraphs and tables in document order. Paragraph styles named `Heading N`,
`見出し N` or `Title` become headings. A table's first row is its header, repeated at
the top of every chunk of its rows.

PPTX: one page per slide. The slide title is a heading; other text frames follow in
position order (top to bottom, then left to right), not in the order they were added;
tables become rows; speaker notes follow the slide's own text.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any

from smartrag.extract.tabular import ExtractError, tsv_line
from smartrag.extract.units import Extracted, Unit, UnitKind

_HEADING_STYLE = re.compile(r"^(?:heading|見出し)\s*(\d)$", re.IGNORECASE)


def _open_error(path: Path, exc: Exception) -> ExtractError:
    return ExtractError(f"{path.name} cannot be read: {exc}", {"path": str(path)})


# --- DOCX --------------------------------------------------------------------------


def _style_level(style_name: str) -> int:
    if style_name.lower() == "title":
        return 1
    match = _HEADING_STYLE.match(style_name.strip())
    return int(match.group(1)) if match else 0


def extract_docx(path: Path) -> Extracted:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(str(path))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise _open_error(path, exc) from exc

    extracted = Extracted(units=[], title=document.core_properties.title or None)
    group = 0
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = paragraph.style.name if paragraph.style is not None else ""
            level = _style_level(style or "")
            extracted.units.append(Unit(text, "heading" if level else "text", level=level))
        elif tag == "tbl":
            group += 1
            rows = [[cell.text.strip() for cell in row.cells]
                    for row in Table(child, document).rows]
            extracted.units += _table_units(rows, group)
    return extracted


def _table_units(rows: list[list[str]], group: int, page: int | None = None) -> list[Unit]:
    rows = [row for row in rows if any(cell for cell in row)]
    units: list[Unit] = []
    for i, row in enumerate(rows):
        kind: UnitKind = "header" if i == 0 and len(rows) > 1 else "row"
        units.append(Unit(tsv_line(row), kind, group=group, page=page, row=i + 1))
    return units


# --- PPTX --------------------------------------------------------------------------


def _shapes(shapes: Any) -> list[Any]:
    """Shapes in reading position, with grouped shapes flattened."""
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    flat: list[Any] = []
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            flat += _shapes(shape.shapes)
        else:
            flat.append(shape)
    return sorted(flat, key=lambda s: (s.top or 0, s.left or 0))


def extract_pptx(path: Path) -> Extracted:
    import pptx

    try:
        presentation = pptx.Presentation(str(path))
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise _open_error(path, exc) from exc

    extracted = Extracted(units=[], title=presentation.core_properties.title or None,
                          pages=len(presentation.slides))
    group = 0
    for number, slide in enumerate(presentation.slides, 1):
        title_shape = slide.shapes.title
        title = title_shape.text_frame.text.strip() if title_shape is not None else ""
        extracted.units.append(Unit(title or f"Slide {number}", "heading", level=1,
                                    page=number, location=f"slide {number}"))
        for shape in _shapes(slide.shapes):
            if title_shape is not None and shape.shape_id == title_shape.shape_id:
                continue
            if getattr(shape, "has_table", False) and shape.has_table:
                group += 1
                rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                extracted.units += _table_units(rows, group, page=number)
            elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                text = shape.text_frame.text.strip()
                if text:
                    extracted.units.append(Unit(text, page=number, location=f"slide {number}"))
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            text = notes.text.strip() if notes is not None else ""
            if text:
                extracted.units.append(Unit(text, page=number,
                                            location=f"slide {number} notes"))
    return extracted
