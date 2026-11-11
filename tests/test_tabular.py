"""CSV, TSV and Excel files as TSV table blocks."""

from __future__ import annotations

import datetime as dt

import pytest
from openpyxl import Workbook
from openpyxl.worksheet.table import Table

from smartrag.errors import Unsupported
from smartrag.extract import convert_tabular, is_tabular
from smartrag.extract.tabular import ExtractError, cell_text, column_letters, escape_cell


def _workbook(path, build):
    wb = Workbook()
    build(wb)
    wb.save(path)
    return path


# --- cells -------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "text"), [
    (None, ""), (True, "TRUE"), (1250000.0, "1250000"), (99.5, "99.5"), (1e20, "1e+20"),
    (dt.date(2026, 11, 1), "2026-11-01"),
    (dt.datetime(2026, 11, 1, 0, 0), "2026-11-01"),
    (dt.datetime(2026, 11, 2, 9, 30), "2026-11-02 09:30:00"),
    (dt.time(9, 30), "09:30:00"), ("a\r\nb", "a\nb"), (7, "7"),
])
def test_cell_text(value, text):
    assert cell_text(value) == text


def test_escaping_keeps_one_row_on_one_line():
    assert escape_cell("a\tb\nc\\d") == "a\\tb\\nc\\\\d"


def test_column_letters():
    assert [column_letters(n) for n in (1, 2, 26, 27, 52, 703)] == ["A", "B", "Z", "AA",
                                                                    "AZ", "AAA"]


def test_is_tabular():
    assert is_tabular("x/見積.CSV") and is_tabular("a.xlsb") and not is_tabular("a.pdf")


# --- Excel -------------------------------------------------------------------------


@pytest.fixture
def quotes_xlsx(tmp_path):
    def build(wb):
        ws = wb.active
        ws.title = "見積"
        ws["B2"] = "2026年 見積一覧"
        ws.append([])
        for row in ([None, "見積番号", "顧客名", "金額", "日付"],
                    [None, "Q-0412", "A社", 1250000, dt.date(2026, 11, 1)],
                    [None, "Q-0413", "B社", 99.5, dt.datetime(2026, 11, 2, 9, 30)]):
            ws.append(row)
        ws.merge_cells("B8:C8")
        ws["B8"] = "合計"
        ws["D8"] = 1250099.5

        tbl = wb.create_sheet("Tbl")
        for row in (["区分", "品名", "数量"], ["A", "ボルト", 3], ["", "ナット", 5],
                    ["B", "座金", 7]):
            tbl.append(row)
        tbl.add_table(Table(displayName="部品表", ref="A1:C4"))
        tbl["E1"] = "メモ"
        tbl["E2"] = "loose cell"

        hidden = wb.create_sheet("隠し")
        hidden["A1"] = "secret"
        hidden.sheet_state = "hidden"

    return _workbook(tmp_path / "見積.xlsx", build)


def test_workbook_blocks_titles_tables_and_locations(quotes_xlsx):
    doc = convert_tabular(quotes_xlsx)
    assert doc.format == "xlsx" and doc.sheets == ["見積", "Tbl"]
    assert doc.skipped_sheets == ["隠し"] and "隠し" in doc.warnings[0]
    locations = [b.location for b in doc.blocks]
    assert locations == ["見積!B4:E6", "見積!B8:D8", "部品表 (Tbl!A1:C4)", "Tbl!E1:E2"]
    assert [b.index for b in doc.blocks] == [0, 1, 2, 3]

    quotes = doc.blocks[0]
    assert quotes.title == "2026年 見積一覧"
    assert quotes.header == ["見積番号", "顧客名", "金額", "日付"] and quotes.header_row == 4
    assert quotes.row_numbers == [5, 6]
    assert quotes.to_tsv().splitlines() == [
        "見積番号\t顧客名\t金額\t日付",
        "Q-0412\tA社\t1250000\t2026-11-01",
        "Q-0413\tB社\t99.5\t2026-11-02 09:30:00",
    ]
    total = doc.blocks[1]
    assert total.header is None and total.rows == [["合計", "", "1250099.5"]]

    parts = doc.blocks[2]
    assert parts.table == "部品表" and parts.header == ["区分", "品名", "数量"]
    assert parts.row_numbers == [2, 3, 4] and parts.rows[1] == ["", "ナット", "5"]
    assert doc.blocks[3].header == ["メモ"] and doc.blocks[3].rows == [["loose cell"]]


def test_hidden_sheets_on_request(quotes_xlsx):
    doc = convert_tabular(quotes_xlsx, include_hidden=True)
    assert "隠し" in doc.sheets and doc.blocks[-1].rows == [["secret"]]


def test_vertical_merges_fill_down_horizontal_ones_do_not(tmp_path):
    def build(wb):
        ws = wb.active
        ws.title = "S"
        for row in (["区分", "品名", "数量"], ["機械", "ボルト", 3], [None, "ナット", 5],
                    [None, "座金", 7], ["小計", None, 15]):
            ws.append(row)
        ws.merge_cells("A2:A4")
        ws.merge_cells("A5:B5")

    doc = convert_tabular(_workbook(tmp_path / "m.xlsx", build))
    rows = doc.blocks[0].rows
    assert [r[0] for r in rows[:3]] == ["機械", "機械", "機械"]
    assert rows[3] == ["小計", "", "15"]


def test_title_in_the_same_band_and_numeric_first_rows(tmp_path):
    def build(wb):
        ws = wb.active
        ws.title = "S"
        for row in (["月次売上"], ["月", "売上"], ["4月", 100], ["5月", 120]):
            ws.append(row)
        other = wb.create_sheet("N")
        for row in ([1, 2], [3, 4]):
            other.append(row)

    doc = convert_tabular(_workbook(tmp_path / "t.xlsx", build))
    first, numbers = doc.blocks
    assert first.title == "月次売上" and first.header == ["月", "売上"]
    assert first.row_numbers == [3, 4]
    assert numbers.header is None and numbers.rows == [["1", "2"], ["3", "4"]]


def test_cells_with_tabs_and_newlines(tmp_path):
    def build(wb):
        ws = wb.active
        ws.append(["項目", "備考"])
        ws.append(["A", "一行目\n二行目\tタブ"])

    block = convert_tabular(_workbook(tmp_path / "n.xlsx", build)).blocks[0]
    assert block.rows == [["A", "一行目\n二行目\tタブ"]]
    assert block.row_tsv(0) == "A\t一行目\\n二行目\\tタブ"
    assert len(block.to_tsv().splitlines()) == 2


def test_row_limit(tmp_path):
    def build(wb):
        ws = wb.active
        ws.append(["n"])
        for n in range(10):
            ws.append([f"r{n}"])

    doc = convert_tabular(_workbook(tmp_path / "big.xlsx", build), max_rows=5)
    assert doc.truncated and doc.blocks[0].row_numbers == [2, 3, 4, 5]
    assert "first 5 rows" in doc.warnings[0]


def test_xlsm_is_read_like_xlsx(tmp_path):
    def build(wb):
        wb.active.append(["a", "b"])
        wb.active.append(["1", "2"])

    doc = convert_tabular(_workbook(tmp_path / "macro.xlsm", build))
    assert doc.format == "xlsm" and doc.blocks[0].header == ["a", "b"]


def test_unreadable_files(tmp_path):
    with pytest.raises(Unsupported):
        convert_tabular(tmp_path / "x.pdf")
    with pytest.raises(ExtractError, match="does not exist"):
        convert_tabular(tmp_path / "missing.xlsx")
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not a zip file")
    with pytest.raises(ExtractError, match="cannot be read"):
        convert_tabular(broken)


# --- CSV / TSV ---------------------------------------------------------------------


def test_cp932_csv_with_quoted_newlines(tmp_path):
    path = tmp_path / "見積.csv"
    text = '見積番号,顧客名,備考\r\nQ-0412,A社,"一行目\r\n二行目"\r\nQ-0413,B社,通常\r\n'
    path.write_bytes(text.encode("cp932"))
    doc = convert_tabular(path)
    block = doc.blocks[0]
    assert doc.delimiter == "," and doc.encoding
    assert block.header == ["見積番号", "顧客名", "備考"]
    assert block.rows == [["Q-0412", "A社", "一行目\n二行目"], ["Q-0413", "B社", "通常"]]
    assert block.row_numbers == [2, 3] and block.location == "rows 1-3"
    assert not doc.warnings


def test_utf8_bom_semicolons_and_two_tables(tmp_path):
    path = tmp_path / "two.csv"
    path.write_text("品名;単価;数量\nボルト;10;3\nナット;5;4\n\n区分;件数\nA;2\nB;1\n",
                    encoding="utf-8-sig")
    doc = convert_tabular(path)
    assert doc.encoding == "utf-8-sig" and doc.delimiter == ";"
    first, second = doc.blocks
    assert first.header == ["品名", "単価", "数量"] and first.rows[0] == ["ボルト", "10", "3"]
    assert second.header == ["区分", "件数"] and second.row_numbers == [6, 7]
    assert first.header and first.header[0] == "品名"  # no BOM left on the first cell


def test_tsv_and_headerless_csv(tmp_path):
    tsv = tmp_path / "export.tsv"
    tsv.write_text("id\tname\n1\t太郎\n", encoding="utf-8")
    doc = convert_tabular(tsv)
    assert doc.format == "tsv" and doc.delimiter == "\t"
    assert doc.blocks[0].rows == [["1", "太郎"]]

    numbers = tmp_path / "numbers.csv"
    numbers.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
    block = convert_tabular(numbers).blocks[0]
    assert block.header is None and len(block.rows) == 2


def test_csv_row_limit(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text("n\n" + "".join(f"r{i}\n" for i in range(20)), encoding="utf-8")
    doc = convert_tabular(path, max_rows=6)
    assert doc.truncated and len(doc.blocks[0].rows) == 5
    assert "first 6 records" in doc.warnings[0]


def test_command_line_preview(tmp_path, capsys):
    from smartrag.extract.tabular import main

    path = tmp_path / "a.csv"
    path.write_text("品名,数量\nボルト,3\n", encoding="utf-8")
    assert main([str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == ["# rows 1-2", "品名\t数量", "ボルト\t3"]
