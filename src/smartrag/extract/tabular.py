"""CSV, TSV and Excel files as TSV table blocks — best effort (design memo §3.4).

A file becomes a list of `TableBlock`s: one per table found. Each block keeps its header
row apart from its data rows, and the source position of every row, so the chunker can

* break only between rows, never inside one;
* repeat the header row at the top of every chunk, so a chunk from row 500 still says
  what its columns are;
* cite the sheet and the cell range a passage came from.

**Finding tables (Excel).** Excel's own Table objects are authoritative: their range and
column names are used as they are. Everything else on a sheet is split into blocks at
blank rows. Within a block, a first row whose cells are all text becomes the header; a
lone cell on its own above a block becomes that block's title. Vertically merged cells
are filled down, so every row stands on its own ("区分: A" on each row it covers);
horizontally merged cells keep their value in the first cell only.

**Known limits.** Tables placed side by side with no blank row between them become one
block. Multi-row headers become a header plus data rows. Formulas are read as the values
Excel cached when it saved the file; a workbook written by a program that stores no
cached values yields empty cells for its formulas. Hidden sheets are skipped unless asked
for, because users hide sheets for a reason.

**Cell text.** Values are written as they would read: `1250000` not `1250000.0`, dates as
`2026-11-01`, times as `09:30:00`. In the TSV, a tab, newline or backslash inside a cell
is escaped as `\\t`, `\\n` or `\\\\` (the convention of PostgreSQL's text format), so one
row is always one line. Text is not otherwise normalised here; that happens downstream,
the same way for every format.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from smartrag.errors import SmartRAGError, Unsupported
from smartrag.extract.encoding import SAMPLE_BYTES, detect_encoding

TabularFormat = Literal["csv", "tsv", "xlsx", "xlsm", "xls", "xlsb"]
TABULAR_EXTENSIONS: frozenset[str] = frozenset(
    {".csv", ".tsv", ".xlsx", ".xlsm", ".xls", ".xlsb"})
_EXCEL_WITH_TABLE_OBJECTS = frozenset({".xlsx", ".xlsm"})

#: Rows read per sheet (Excel) or per file (CSV) unless the caller says otherwise.
DEFAULT_MAX_ROWS = 100_000

#: How much of a CSV is read to decide its encoding and delimiter.
_SAMPLE_BYTES = SAMPLE_BYTES
_CSV_DELIMITERS = ",\t;|"
#: Cells longer than the csv module's default limit (128 KiB) are legitimate in exports.
_CSV_FIELD_LIMIT = 16 * 1024 * 1024

_NUMERIC = re.compile(
    r"^[+\-]?[¥$€£]?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%?$"
    r"|^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$"
    r"|^\d{1,2}:\d{2}(?::\d{2})?$"
)


class ExtractError(SmartRAGError):
    """The file exists but cannot be read as what its name says (corrupt, encrypted)."""

    status = 422
    code = "extract_failed"


# --- the result --------------------------------------------------------------------


def column_letters(index: int) -> str:
    """1-based column index → Excel letters (1 → A, 27 → AA)."""
    letters = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def escape_cell(value: str) -> str:
    return (value.replace("\\", "\\\\").replace("\t", "\\t")
            .replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n"))


def tsv_line(cells: Sequence[str]) -> str:
    return "\t".join(escape_cell(cell) for cell in cells)


class TableBlock(BaseModel):
    """One table: an optional header row, data rows, and where they came from.

    Rows and columns are 1-based, as Excel shows them. A CSV has no sheet and no cell
    letters; its "rows" are record numbers (a quoted cell may span several lines)."""

    index: int = Field(description="Position of this block in the document, from 0.")
    sheet: str | None = None
    table: str | None = Field(default=None, description="Excel Table object's name.")
    title: str | None = Field(default=None, description="A caption found above the table.")
    header: list[str] | None = None
    header_row: int | None = None
    rows: list[list[str]]
    row_numbers: list[int]
    first_col: int
    last_col: int

    @property
    def first_row(self) -> int:
        return self.header_row if self.header_row is not None else self.row_numbers[0]

    @property
    def last_row(self) -> int:
        return self.row_numbers[-1] if self.row_numbers else self.first_row

    @property
    def cell_range(self) -> str:
        return (f"{column_letters(self.first_col)}{self.first_row}:"
                f"{column_letters(self.last_col)}{self.last_row}")

    @property
    def location(self) -> str:
        """For citations: `見積!B4:E6`, `部品表 (Tbl!A1:C4)`, or `rows 1-120` for a CSV."""
        if self.sheet is None:
            return f"rows {self.first_row}-{self.last_row}"
        place = f"{self.sheet}!{self.cell_range}"
        return f"{self.table} ({place})" if self.table else place

    def header_tsv(self) -> str | None:
        return tsv_line(self.header) if self.header is not None else None

    def row_tsv(self, i: int) -> str:
        return tsv_line(self.rows[i])

    def to_tsv(self, include_header: bool = True) -> str:
        lines = [self.row_tsv(i) for i in range(len(self.rows))]
        header = self.header_tsv() if include_header else None
        return "\n".join([header, *lines] if header is not None else lines)


class TabularDocument(BaseModel):
    path: str
    format: TabularFormat
    encoding: str | None = Field(default=None, description="CSV/TSV only.")
    delimiter: str | None = Field(default=None, description="CSV/TSV only.")
    sheets: list[str] = Field(default_factory=list, description="Sheets read.")
    skipped_sheets: list[str] = Field(default_factory=list)
    blocks: list[TableBlock] = Field(default_factory=list)
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)

    def to_tsv(self) -> str:
        """Every block, each under a `# location — title` line: for reading, not chunking."""
        parts = []
        for block in self.blocks:
            label = block.location + (f" — {block.title}" if block.title else "")
            parts.append(f"# {label}\n{block.to_tsv()}")
        return "\n\n".join(parts)


# --- entry point -------------------------------------------------------------------


def is_tabular(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TABULAR_EXTENSIONS


def convert_tabular(path: str | Path, *, max_rows: int = DEFAULT_MAX_ROWS,
                    include_hidden: bool = False) -> TabularDocument:
    """Read a CSV, TSV or Excel file into TSV table blocks."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in TABULAR_EXTENSIONS:
        raise Unsupported(f"{path.name} is not a CSV, TSV or Excel file", {"path": str(path)})
    if not path.is_file():
        raise ExtractError(f"{path} does not exist or is not a file", {"path": str(path)})
    if ext in (".csv", ".tsv"):
        return _convert_delimited(path, ext, max_rows)
    return _convert_workbook(path, ext, max_rows, include_hidden)


# --- cell values -------------------------------------------------------------------


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    if isinstance(value, dt.datetime):
        if value.time() == dt.time(0, 0):
            return value.date().isoformat()
        return value.isoformat(sep=" ", timespec="seconds" if not value.microsecond
                               else "auto")
    if isinstance(value, (dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n")
    return str(value)


def _numeric(text: str) -> bool:
    return bool(_NUMERIC.match(text.strip()))


# --- splitting a grid into blocks --------------------------------------------------

Grid = list[list[str]]


def _occupied(row: Sequence[str]) -> list[int]:
    return [j for j, cell in enumerate(row) if cell.strip()]


def _bands(grid: Grid) -> list[tuple[int, int]]:
    """Runs of non-blank rows, as (first, last) indexes into the grid."""
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for i, row in enumerate(grid):
        if _occupied(row):
            if start is None:
                start = i
        elif start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(grid) - 1))
    return bands


def _looks_like_header(row: Sequence[str], width: int, more_rows: bool) -> bool:
    cells = [cell.strip() for cell in row if cell.strip()]
    return (more_rows and len(cells) >= max(1, (width + 1) // 2)
            and not any(_numeric(cell) for cell in cells))


def _blocks_from_grid(grid: Grid, row_of: Sequence[int], col_offset: int,
                      sheet: str | None) -> list[TableBlock]:
    """Blocks from a grid of cell text. `row_of[i]` is grid row i's source row number;
    grid column j is source column `col_offset + j` (both 1-based)."""
    blocks: list[TableBlock] = []
    # A lone cell on its own row: the title of the table below it, if one follows.
    pending: tuple[str, int, int] | None = None
    for first, last in _bands(grid):
        band = grid[first:last + 1]
        cols = sorted({j for row in band for j in _occupied(row)})
        lo, hi = cols[0], cols[-1]
        if first == last and len(cols) == 1:
            if pending is not None:
                blocks.append(_lone_cell_block(pending, sheet))
            pending = (band[0][lo].strip(), row_of[first], col_offset + lo)
            continue
        rows = [row[lo:hi + 1] + [""] * max(0, hi + 1 - len(row)) for row in band]
        numbers = list(row_of[first:last + 1])
        title = pending[0] if pending is not None else None
        pending = None
        if (title is None and len(rows) > 2 and len(_occupied(rows[0])) == 1
                and len(_occupied(rows[1])) > 1):
            title = rows[0][_occupied(rows[0])[0]].strip()
            rows, numbers = rows[1:], numbers[1:]
        header: list[str] | None = None
        header_row: int | None = None
        if _looks_like_header(rows[0], hi - lo + 1, len(rows) > 1):
            header, header_row = [cell.strip() for cell in rows[0]], numbers[0]
            rows, numbers = rows[1:], numbers[1:]
        blocks.append(TableBlock(index=0, sheet=sheet, title=title, header=header,
                                 header_row=header_row, rows=rows, row_numbers=numbers,
                                 first_col=col_offset + lo, last_col=col_offset + hi))
    if pending is not None:
        blocks.append(_lone_cell_block(pending, sheet))
    return blocks


def _lone_cell_block(cell: tuple[str, int, int], sheet: str | None) -> TableBlock:
    text, row, col = cell
    return TableBlock(index=0, sheet=sheet, rows=[[text]], row_numbers=[row],
                      first_col=col, last_col=col)


# --- CSV / TSV ---------------------------------------------------------------------


def _delimiter(sample_text: str, ext: str) -> str:
    """The csv module's sniffer first. It gives up when a file holds tables with different
    column counts, so then: the candidate found on the most non-blank lines."""
    if ext == ".tsv":
        return "\t"
    try:
        return csv.Sniffer().sniff(sample_text, delimiters=_CSV_DELIMITERS).delimiter
    except csv.Error:
        pass
    lines = [line for line in sample_text.splitlines()[:200] if line.strip()]
    scores = {d: sum(1 for line in lines if d in line) for d in _CSV_DELIMITERS}
    best = max(scores, key=lambda d: (scores[d], d == ","))
    return best if scores[best] else ","


def _convert_delimited(path: Path, ext: str, max_rows: int) -> TabularDocument:
    with open(path, "rb") as handle:
        sample = handle.read(_SAMPLE_BYTES)
        complete = len(sample) < _SAMPLE_BYTES
    encoding = detect_encoding(sample, complete)
    sample_text = sample.decode(encoding, errors="replace")
    if not complete:
        sample_text = sample_text.rsplit("\n", 1)[0]
    delimiter = _delimiter(sample_text, ext)
    doc = TabularDocument(path=str(path), format="csv" if ext == ".csv" else "tsv",
                          encoding=encoding, delimiter=delimiter)

    csv.field_size_limit(max(csv.field_size_limit(), _CSV_FIELD_LIMIT))
    grid: Grid = []
    with open(path, encoding=encoding, errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            for record in reader:
                if len(grid) >= max_rows:
                    doc.truncated = True
                    break
                grid.append([cell_text(cell) for cell in record])
        except csv.Error as exc:
            raise ExtractError(f"{path.name}: {exc} (record {reader.line_num})",
                               {"path": str(path)}) from exc
    if any("\ufffd" in cell for row in grid for cell in row):
        doc.warnings.append(f"some bytes could not be decoded as {encoding} and were "
                            "replaced; the detected encoding may be wrong")
    if doc.truncated:
        doc.warnings.append(f"only the first {max_rows:,} records were read")
    width = max((len(row) for row in grid), default=0)
    grid = [row + [""] * (width - len(row)) for row in grid]
    doc.blocks = _number(_blocks_from_grid(grid, range(1, len(grid) + 1), 1, None))
    return doc


# --- Excel -------------------------------------------------------------------------


def _open_workbook(path: Path, ext: str) -> tuple[Any, bool]:
    import python_calamine as pc

    want_tables = ext in _EXCEL_WITH_TABLE_OBJECTS
    try:
        try:
            return pc.CalamineWorkbook.from_path(str(path), load_tables=want_tables), want_tables
        except pc.TablesNotSupported:
            return pc.CalamineWorkbook.from_path(str(path)), False
    except pc.PasswordError as exc:
        raise ExtractError(f"{path.name} is password-protected", {"path": str(path)}) from exc
    except pc.CalamineError as exc:
        raise ExtractError(f"{path.name} cannot be read as an Excel file: {exc}",
                           {"path": str(path)}) from exc
    except OSError as exc:
        raise ExtractError(f"{path.name} cannot be read: {exc}", {"path": str(path)}) from exc


def _sheets_to_read(workbook: Any, include_hidden: bool) -> tuple[list[str], list[str]]:
    import python_calamine as pc

    read, skipped = [], []
    for meta in workbook.sheets_metadata:
        worksheet = meta.typ == pc.SheetTypeEnum.WorkSheet
        visible = meta.visible == pc.SheetVisibleEnum.Visible
        if not worksheet or not (visible or include_hidden):
            skipped.append(meta.name)
        else:
            read.append(meta.name)
    return read, skipped


def _fill_down_merges(grid: Grid, merges: Iterable[Any], top: int, left: int) -> None:
    """Copy a vertically merged cell's value into every row it covers."""
    for (r0, c0), (r1, _c1) in merges:
        if r1 <= r0:
            continue
        i0, j = r0 - top, c0 - left
        if not (0 <= i0 < len(grid) and 0 <= j < len(grid[i0])):
            continue
        value = grid[i0][j]
        for i in range(i0 + 1, min(r1 - top, len(grid) - 1) + 1):
            if j < len(grid[i]) and not grid[i][j]:
                grid[i][j] = value


def _convert_workbook(path: Path, ext: str, max_rows: int,
                      include_hidden: bool) -> TabularDocument:
    workbook, tables_loaded = _open_workbook(path, ext)
    doc = TabularDocument(path=str(path), format=ext[1:])  # type: ignore[arg-type]
    try:
        doc.sheets, doc.skipped_sheets = _sheets_to_read(workbook, include_hidden)
        tables = _tables_by_sheet(workbook) if tables_loaded else {}
        blocks: list[TableBlock] = []
        for name in doc.sheets:
            blocks += _sheet_blocks(workbook, name, tables.get(name, []), max_rows, doc)
        doc.blocks = _number(blocks)
    finally:
        workbook.close()
    if doc.skipped_sheets:
        doc.warnings.append(f"skipped {len(doc.skipped_sheets)} hidden or non-worksheet "
                            "sheet(s): " + ", ".join(doc.skipped_sheets))
    return doc


def _tables_by_sheet(workbook: Any) -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    for name in workbook.table_names:
        table = workbook.get_table_by_name(name)
        found.setdefault(table.sheet, []).append(table)
    return found


def _sheet_blocks(workbook: Any, name: str, tables: list[Any], max_rows: int,
                  doc: TabularDocument) -> list[TableBlock]:
    sheet = workbook.get_sheet_by_name(name)
    if sheet.start is None:
        return []
    top, left = sheet.start
    raw = sheet.to_python(nrows=max_rows + 1)
    if len(raw) > max_rows:
        raw = raw[:max_rows]
        doc.truncated = True
        doc.warnings.append(f"sheet {name!r}: only the first {max_rows:,} rows were read")
    width = max((len(row) for row in raw), default=0)
    grid: Grid = [[cell_text(v) for v in row] + [""] * (width - len(row)) for row in raw]
    _fill_down_merges(grid, sheet.merged_cell_ranges or [], top, left)

    blocks: list[TableBlock] = []
    for table in tables:
        block = _table_block(table, grid, top, left, name)
        if block is not None:
            blocks.append(block)
    # Whatever the Table objects do not cover is split at blank rows.
    rows_of = [top + i + 1 for i in range(len(grid))]
    blocks += _blocks_from_grid(grid, rows_of, left + 1, name)
    return sorted(blocks, key=lambda b: (b.first_row, b.first_col))


def _table_block(table: Any, grid: Grid, top: int, left: int, sheet: str) -> TableBlock | None:
    """An Excel Table object as a block; its cells are then blanked in `grid` so the
    blank-row splitting does not read them a second time."""
    if table.start is None or table.end is None:
        return None
    (r0, c0), (r1, c1) = table.start, table.end
    header_r = r0 - 1  # the Table reports its data area; the header row is just above
    rows: list[list[str]] = []
    numbers: list[int] = []
    for r in range(max(header_r, top), r1 + 1):
        i = r - top
        if i >= len(grid):
            break
        cells = [grid[i][c - left] if 0 <= c - left < len(grid[i]) else ""
                 for c in range(c0, c1 + 1)]
        if r >= r0:
            rows.append(cells)
            numbers.append(r + 1)
        for c in range(c0, c1 + 1):
            if 0 <= c - left < len(grid[i]):
                grid[i][c - left] = ""
    if not numbers:
        return None
    return TableBlock(index=0, sheet=sheet, table=table.name,
                      header=[str(col) for col in table.columns],
                      header_row=header_r + 1 if header_r >= top else None,
                      rows=rows, row_numbers=numbers, first_col=c0 + 1, last_col=c1 + 1)


def _number(blocks: list[TableBlock]) -> list[TableBlock]:
    for i, block in enumerate(blocks):
        block.index = i
    return blocks


# --- a quick look from the command line --------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m smartrag.extract FILE [--json] [--include-hidden]`."""
    import argparse

    parser = argparse.ArgumentParser(prog="python -m smartrag.extract",
                                     description="Show a CSV/TSV/Excel file as TSV blocks.")
    parser.add_argument("file")
    parser.add_argument("--json", action="store_true", help="print the blocks as JSON")
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    args = parser.parse_args(argv)
    try:
        doc = convert_tabular(args.file, max_rows=args.max_rows,
                              include_hidden=args.include_hidden)
    except SmartRAGError as err:
        print(f"error: {err.message}")
        return 1
    if args.json:
        print(doc.model_dump_json(indent=2))
    else:
        out = io.StringIO()
        out.write(doc.to_tsv())
        for warning in doc.warnings:
            out.write(f"\n# warning: {warning}")
        print(out.getvalue())
    return 0
