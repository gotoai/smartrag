"""PDF text layers, with pdfminer.six.

Each text box pdfminer finds becomes a unit, in its reading order, with its page number.
Lines inside a box are joined: Japanese without a space, Latin words with one.

**Repeated page furniture is dropped.** A short box whose text (with digits ignored) appears
on most pages of the same PDF — a running header, a page number, a document reference — is
kept once, where it first appears, and removed elsewhere. This looks only at the one
document, so it adds no corpus-wide state.

A PDF with no text layer (a scan) yields no text; the caller records it as skipped. Text
dominated by `(cid:N)` placeholders means the PDF's fonts carry no character mapping;
that is reported, since it reads as noise.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from smartrag.extract.tabular import ExtractError
from smartrag.extract.units import Extracted, Unit, heading_level

_LATIN_END = re.compile(r"[A-Za-z0-9,.;:)\]]$")
_LATIN_START = re.compile(r"^[A-Za-z0-9(\[]")
_CID = re.compile(r"\(cid:\d+\)")
_DIGITS = re.compile(r"[0-9０-９]+")
#: Running headers and page numbers are short; longer boxes are body text.
_MAX_FURNITURE_CHARS = 60


def _join_lines(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    joined = lines[0]
    for line in lines[1:]:
        space = " " if _LATIN_END.search(joined) and _LATIN_START.match(line) else ""
        joined += space + line
    return joined


#: Text that ends a paragraph: a sentence end or a closing bracket after one.
_PARAGRAPH_END = re.compile(r"[。．.！？!?」』）)]$")
#: Lines that start a new item even when the previous line runs on: （１）, (1), ①, ア,
#: (ｱ), 第１, or a bare number followed by a space, as Japanese legal documents number them.
_ITEM_START = re.compile(
    r"^(?:[（(][0-9０-９一二三四五六七八九十]+[）)]|[①-⑳]|[ア-ンｱ-ﾝ][ 　]"
    r"|[（(][ア-ンｱ-ﾝ][）)]|第[0-9０-９一二三四五六七八九十]|[0-9０-９]+[ 　])")
_MAX_PARAGRAPH_CHARS = 4000


def _join_wrapped_lines(units: list[Unit], keep_apart: set[int] | None = None) -> list[Unit]:
    """Join lines that one paragraph was wrapped into. pdfminer often returns every
    printed line as its own box; a sentence split across them loses the character pairs
    at each break (keyword search) and reads as fragments (embedding). Lines join on the
    same page only, and never across a sentence end, a heading, a new numbered item, or a
    running header kept from the page furniture (`keep_apart`, by position in `units`)."""
    keep_apart = keep_apart or set()
    joined: list[Unit] = []
    previous_apart = False
    for position, unit in enumerate(units):
        previous = joined[-1] if joined else None
        apart = position in keep_apart
        if (previous is not None and not apart and not previous_apart
                and previous.kind == "text" and unit.kind == "text"
                and previous.page == unit.page
                and not _PARAGRAPH_END.search(previous.text)
                and not _ITEM_START.match(unit.text)
                and len(previous.text) + len(unit.text) <= _MAX_PARAGRAPH_CHARS):
            previous.text = _join_lines(previous.text + "\n" + unit.text)
        else:
            joined.append(unit)
            previous_apart = apart
    return joined


def _furniture_key(text: str) -> str:
    return _DIGITS.sub("#", re.sub(r"\s+", "", text))


_tie_break_patched = False


def _deterministic_tie_breaks() -> None:
    """Make pdfminer's reading order the same in every process.

    `LTLayoutContainer.group_textboxes` ranks pairs of boxes by distance and breaks exact
    ties with `id(obj)`, a memory address that differs between processes. Wherever two
    distances are equal (grid-like table cells, single-character boxes), the reading
    order then changes from run to run: about 1 in 7 of a real 158-PDF corpus did. That
    gives the same file different chunks, different chunk ids and needless re-embedding.

    For the duration of that one method, `id` in pdfminer's layout module is replaced by
    a counter in first-seen order. The pairs are built in a fixed order, so the counter is
    fixed too. The method is otherwise untouched, and the replacement is removed after
    every call; the objects stay alive throughout, so their addresses cannot be reused."""
    global _tie_break_patched
    if _tie_break_patched:
        return
    import builtins
    import functools

    from pdfminer import layout

    original = layout.LTLayoutContainer.group_textboxes

    @functools.wraps(original)
    def group_textboxes(self, laparams, boxes):  # type: ignore[no-untyped-def]
        order: dict[int, int] = {}
        layout.id = lambda obj: order.setdefault(builtins.id(obj), len(order))  # type: ignore[attr-defined]
        try:
            return original(self, laparams, boxes)
        finally:
            del layout.id  # type: ignore[attr-defined]

    layout.LTLayoutContainer.group_textboxes = group_textboxes  # type: ignore[method-assign]
    _tie_break_patched = True


def extract_pdf(path: Path) -> Extracted:
    _deterministic_tie_breaks()
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LAParams, LTTextContainer
    from pdfminer.pdfdocument import PDFEncryptionError, PDFPasswordIncorrect
    from pdfminer.pdfparser import PDFSyntaxError
    from pdfminer.psparser import PSException

    boxes: list[tuple[int, str]] = []
    pages = 0
    try:
        for page_no, layout in enumerate(extract_pages(str(path), laparams=LAParams()), 1):
            pages = page_no
            for element in layout:
                if isinstance(element, LTTextContainer):
                    text = _join_lines(element.get_text())
                    if text:
                        boxes.append((page_no, text))
    except (PDFPasswordIncorrect, PDFEncryptionError) as exc:
        raise ExtractError(f"{path.name} is encrypted", {"path": str(path)}) from exc
    except (PDFSyntaxError, PSException) as exc:
        raise ExtractError(f"{path.name} is not a readable PDF: {exc}",
                           {"path": str(path)}) from exc

    extracted = Extracted(units=[], pages=pages)
    pages_with: Counter[str] = Counter()
    for _page, furniture in {(page, _furniture_key(text)) for page, text in boxes}:
        pages_with[furniture] += 1
    threshold = max(3, (pages + 1) // 2)
    seen: set[str] = set()
    kept_headers: set[int] = set()
    dropped = 0
    for page, text in boxes:
        key = _furniture_key(text)
        if pages >= 3 and len(text) <= _MAX_FURNITURE_CHARS and pages_with[key] >= threshold:
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept_headers.add(len(extracted.units))
        level = heading_level(text)
        extracted.units.append(Unit(text, "heading" if level else "text", level=level,
                                    page=page))
    extracted.units = _join_wrapped_lines(extracted.units, kept_headers)
    if dropped:
        extracted.warnings.append(f"removed {dropped} repeated header/footer boxes")
    total = sum(len(u.text) for u in extracted.units)
    cid = sum(len(m) for u in extracted.units for m in _CID.findall(u.text))
    if total and cid / total > 0.3:
        extracted.warnings.append("most text is unmapped glyphs (cid:N); the PDF's fonts "
                                  "carry no character mapping")
    return extracted
