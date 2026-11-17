"""Rich Text Format (.rtf), read by a small parser of its own rather than a dependency.

RTF is 7-bit text: groups in braces, control words such as `\\par`, and the document's
characters written either as `\\uN` or as `\\'hh` bytes in a code page. Japanese
documents from Word and WordPad carry Shift_JIS (cp932) byte pairs, so bytes are decoded
per font: the font table's `\\fcharset` (or `\\cpg`) gives each font's code page, falling
back to the document's `\\ansicpg`. Word marks the font for double-byte text with
`\\dbch\\afN`, so a run of bytes is decoded with a double-byte code page whenever either
the current font or that associated font has one.

Kept: body text, paragraphs (`\\par`), line breaks, tabs, table rows (cells joined by
tabs, one row per paragraph), list numbers, field results (link texts), and headings. A
paragraph is a heading by its own `\\outlinelevelN`, or by its style (`\\sN`) when the
style sheet names that style `Heading N`, `見出し N` or `Title`, or gives it an outline
level; Word and LibreOffice mark headings the second way. Where a writer gives text twice,
in ANSI and in Unicode (`\\upr` with `\\*\\ud`, as LibreOffice does for the title), the
Unicode copy is kept. Skipped: the font, colour and list tables, pictures, embedded
objects, field instructions, headers, footers, footnotes, comments, and every other `\\*`
destination; from the document info only the title is kept.
"""

from __future__ import annotations

import codecs
import re
from dataclasses import dataclass, replace
from pathlib import Path

from smartrag.extract.tabular import ExtractError
from smartrag.extract.units import Extracted, Unit

_TOKEN = re.compile(
    r"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?"   # control word, optional parameter
    r"|\\'([0-9a-fA-F]{2})"               # one byte in the current code page
    r"|\\([^a-zA-Z'])"                    # control symbol
    r"|([{}])"
    r"|[\r\n]+"                           # line breaks in the file are not text
    r"|([^\\{}\r\n]+)",
    re.DOTALL)

#: Destinations whose text is never body text.
_SKIPPED = frozenset({
    "colortbl", "listtable", "listoverridetable", "revtbl", "rsidtbl",
    "filetbl", "pgdsctbl", "latentstyles", "themedata", "colorschememapping", "datastore",
    "xmlnstbl", "xmlnsdecl", "generator", "mmathPr", "pict", "object", "objdata",
    "shppict", "nonshppict", "fldinst", "header", "headerl", "headerr", "headerf",
    "footer", "footerl", "footerr", "footerf", "footnote", "annotation", "atnid",
    "atnauthor", "bkmkstart", "bkmkend", "template", "userprops", "docvar",
})

#: `\fcharset` values and their code pages; 0 (ANSI) and 1 (default) use `\ansicpg`.
_CHARSETS = {
    2: "cp1252", 77: "mac_roman", 128: "cp932", 129: "cp949", 130: "johab", 134: "gbk",
    136: "big5", 161: "cp1253", 162: "cp1254", 163: "cp1258", 177: "cp1255",
    178: "cp1256", 186: "cp1257", 204: "cp1251", 222: "cp874", 238: "cp1250",
}
_DOUBLE_BYTE = frozenset({"cp932", "cp949", "johab", "gbk", "big5"})

_SYMBOLS = {
    "emdash": "\u2014", "endash": "\u2013", "bullet": "\u2022", "lquote": "\u2018",
    "rquote": "\u2019", "ldblquote": "\u201c", "rdblquote": "\u201d", "emspace": " ",
    "enspace": " ", "qmspace": " ", "tab": "\t", "line": "\n", "cell": "\t",
    "nestcell": "\t",
}
_ESCAPES = {"\\": "\\", "{": "{", "}": "}", "~": " ", "_": "-", "-": "", "\n": None,
            "\r": None, "\t": "\t"}
_PARAGRAPH_ENDS = frozenset({"par", "row", "nestrow", "sect", "page"})
_HEADING_STYLE = re.compile(r"^(?:heading|見出し)\s*(\d)$", re.IGNORECASE)


def _codec(name: str) -> str | None:
    try:
        return codecs.lookup(name).name
    except LookupError:
        return None


@dataclass
class _Group:
    sink: str = "body"
    """Where text goes: body, title, info (dropped), fonttbl, stylesheet, or skip."""
    before_star: str | None = None
    """The sink before a `\\*` made this group skipped, in case it turns out to be `\\ud`."""
    uc: int = 1
    font: int = -1
    dbch_font: int = -1
    selector: str = "loch"


class _Reader:
    def __init__(self) -> None:
        self.codepage = "cp1252"
        self.fonts: dict[int, str] = {}
        self.defining = -1
        self.default_font = -1
        self.style = 0
        """The paragraph style (`\\sN`), or in the style sheet the style being defined."""
        self.style_names: dict[int, str] = {}
        self.style_outlines: dict[int, int] = {}
        self.upr_mark = (0, 0)
        self.group = _Group()
        self.stack: list[_Group] = []
        self.pending = bytearray()
        self.skip = 0
        self.outline: int | None = None
        self.parts: list[str] = []
        self.title: list[str] = []
        self.units: list[Unit] = []

    # --- text --------------------------------------------------------------------

    def font_codec(self, font: int) -> str | None:
        return self.fonts.get(font)

    def flush_bytes(self) -> None:
        if not self.pending:
            return
        candidates = [self.font_codec(self.group.font), self.font_codec(self.group.dbch_font)]
        codec = next((c for c in candidates if c in _DOUBLE_BYTE), None) \
            or candidates[0] or self.codepage
        self.emit(bytes(self.pending).decode(codec, errors="replace"))
        self.pending.clear()

    def emit(self, text: str) -> None:
        if self.group.sink == "body":
            self.parts.append(text)
        elif self.group.sink == "title":
            self.title.append(text)
        elif self.group.sink == "stylesheet":
            self.style_names[self.style] = self.style_names.get(self.style, "") + text

    def heading_level(self) -> int:
        if self.outline is not None:
            return min(self.outline + 1, 6) if self.outline < 9 else 0
        outline = self.style_outlines.get(self.style)
        if outline is not None:
            return min(outline + 1, 6) if outline < 9 else 0
        name = self.style_names.get(self.style, "").strip().rstrip(";").strip()
        match = _HEADING_STYLE.match(name)
        if match:
            return min(int(match.group(1)), 6)
        return 1 if name.lower() == "title" else 0

    def end_paragraph(self) -> None:
        text = _joined(self.parts).strip("\t \n")
        self.parts = []
        if not text:
            return
        level = self.heading_level()
        self.units.append(Unit(text, "heading", level=level) if level else Unit(text))

    # --- tokens ------------------------------------------------------------------

    def destination(self, word: str) -> None:
        group = self.group
        if group.sink in ("skip", "fonttbl"):
            return
        if word in _SKIPPED:
            group.sink = "skip"
        elif word == "fonttbl":
            group.sink = "fonttbl"
        elif word == "stylesheet":
            group.sink = "stylesheet"
        elif word == "info":
            group.sink = "info"
        elif word == "title" and group.sink == "info":
            group.sink = "title"

    def control(self, word: str, param: int | None) -> None:
        group = self.group
        if word == "ud" and group.before_star is not None:
            # The Unicode copy of text an `\upr` group has just given in ANSI: keep this
            # one, and take back the other.
            group.sink, group.before_star = group.before_star, None
            del self.parts[self.upr_mark[0]:]
            del self.title[self.upr_mark[1]:]
        elif word == "upr":
            self.upr_mark = (len(self.parts), len(self.title))
        elif word in _SKIPPED or word in ("fonttbl", "stylesheet", "info", "title"):
            self.destination(word)
        elif word == "ansicpg" and param is not None:
            self.codepage = _codec(f"cp{param}") or self.codepage
        elif group.sink == "fonttbl":
            if word == "f" and param is not None:
                self.defining = param
            elif word == "fcharset" and param is not None and param in _CHARSETS:
                self.fonts.setdefault(self.defining, _CHARSETS[param])
            elif word == "cpg" and param is not None:
                codec = _codec(f"cp{param}")
                if codec:
                    self.fonts[self.defining] = codec
        elif group.sink == "stylesheet" and word in ("s", "outlinelevel"):
            if word == "s" and param is not None:
                self.style = param
            elif param is not None:
                self.style_outlines[self.style] = param
        elif word == "s" and param is not None:
            self.style = param
        elif word == "deff" and param is not None:
            self.default_font = group.font = param
        elif word == "plain":
            group.font = self.default_font
        elif word == "uc" and param is not None:
            group.uc = max(param, 0)
        elif word == "u" and param is not None:
            self.emit(chr(param + 65536 if param < 0 else param))
            self.skip = group.uc
        elif word in ("loch", "hich", "dbch"):
            group.selector = word
        elif word == "f" and param is not None:
            group.font = param
            if group.selector == "dbch":
                group.dbch_font = param
        elif word == "af" and param is not None and group.selector == "dbch":
            group.dbch_font = param
        elif word == "pard":
            self.outline = None
            self.style = 0
        elif word == "outlinelevel" and param is not None:
            self.outline = param
        elif word in _PARAGRAPH_ENDS:
            if group.sink == "body":
                self.end_paragraph()
        elif word in _SYMBOLS:
            self.emit(_SYMBOLS[word])

    def read(self, text: str) -> None:
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if match is None:  # a lone backslash at the very end
                break
            position = match.end()
            word, param, byte, symbol, brace, run = match.groups()
            if byte is not None:
                if self.skip:
                    self.skip -= 1
                else:
                    self.pending.append(int(byte, 16))
                continue
            self.flush_bytes()
            if brace is not None:
                self.skip = 0
                if brace == "{":
                    self.stack.append(self.group)
                    self.group = replace(self.group, before_star=None)
                    if self.group.sink == "stylesheet":
                        self.style = 0  # an entry without `\sN` defines the Normal style
                elif self.stack:
                    self.group = self.stack.pop()
            elif word is not None:
                number = int(param) if param is not None else None
                if word == "bin" and number:
                    position += number  # raw binary data, inside a picture
                elif self.skip:
                    self.skip -= 1
                else:
                    self.control(word, number)
            elif symbol is not None:
                if self.skip:
                    self.skip -= 1
                elif symbol == "*":
                    if self.group.sink not in ("fonttbl", "skip"):
                        self.group.before_star = self.group.sink
                        self.group.sink = "skip"
                else:
                    escaped = _ESCAPES.get(symbol, "")
                    if escaped is not None:
                        self.emit(escaped)
                    elif self.group.sink == "body":  # a backslash ending a line: \par
                        self.end_paragraph()
            elif run is not None:
                if self.skip:
                    dropped = min(self.skip, len(run))
                    self.skip -= dropped
                    run = run[dropped:]
                if any(ord(char) > 0x7F for char in run):
                    self.pending.extend(run.encode("latin-1"))
                    self.flush_bytes()
                elif run:
                    self.emit(run)
        self.flush_bytes()
        self.end_paragraph()


def _joined(parts: list[str]) -> str:
    """Parts joined, with `\\uN` surrogate pairs combined into one character."""
    return "".join(parts).encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def extract_rtf(path: Path) -> Extracted:
    data = path.read_bytes()
    if not data.lstrip().startswith(b"{\\rtf"):
        raise ExtractError(f"{path.name} is not an RTF file", {"path": str(path)})
    reader = _Reader()
    reader.read(data.decode("latin-1"))
    title = _joined(reader.title).strip() or None
    return Extracted(units=reader.units, title=title)
