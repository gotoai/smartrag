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


def _furniture_key(text: str) -> str:
    return _DIGITS.sub("#", re.sub(r"\s+", "", text))


def extract_pdf(path: Path) -> Extracted:
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
    dropped = 0
    for page, text in boxes:
        key = _furniture_key(text)
        if pages >= 3 and len(text) <= _MAX_FURNITURE_CHARS and pages_with[key] >= threshold:
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
        level = heading_level(text)
        extracted.units.append(Unit(text, "heading" if level else "text", level=level,
                                    page=page))
    if dropped:
        extracted.warnings.append(f"removed {dropped} repeated header/footer boxes")
    total = sum(len(u.text) for u in extracted.units)
    cid = sum(len(m) for u in extracted.units for m in _CID.findall(u.text))
    if total and cid / total > 0.3:
        extracted.warnings.append("most text is unmapped glyphs (cid:N); the PDF's fonts "
                                  "carry no character mapping")
    return extracted
