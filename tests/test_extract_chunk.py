"""Extractors (text, Markdown, HTML, PDF, DOCX, PPTX) and the chunker."""

from __future__ import annotations

import pytest

from smartrag.chunking import chunk_units, estimate_tokens
from smartrag.extract.units import Unit, extract, heading_level
from smartrag.kb.config import ChunkConfig

# --- headings ----------------------------------------------------------------------


@pytest.mark.parametrize(("line", "level"), [
    ("第3章 休暇", 1), ("第１２条 年次有給休暇", 3), ("第２　事案の概要", 2),
    ("【対象】", 2), ("■ 申請方法", 3),
    ("第12条 社員は年休を取得できる。", 0), ("普通の文", 0), ("x" * 61, 0),
])
def test_heading_level(line, level):
    assert heading_level(line) == level


# --- text, Markdown, HTML ----------------------------------------------------------


def test_plain_text_paragraphs_and_japanese_headings(tmp_path):
    path = tmp_path / "規則.txt"
    path.write_bytes("第3章 休暇\n\n第12条 年次有給休暇\n社員は、10日の\n年休を取得できる。\n\n対象は全拠点。\n"
                     .encode("cp932"))
    units = extract(path).units
    assert [(u.kind, u.level) for u in units] == [("heading", 1), ("heading", 3), ("text", 0),
                                                  ("text", 0)]
    assert units[2].text == "社員は、10日の\n年休を取得できる。"


def test_markdown_headings_and_fences(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# Title\n\npara\n\n```\n# not a heading\n```\n## Sub\ntext\n",
                    encoding="utf-8")
    extracted = extract(path)
    assert extracted.title == "Title"
    kinds = [(u.kind, u.text) for u in extracted.units]
    assert ("text", "```\n# not a heading\n```") in kinds
    assert ("heading", "Sub") in kinds


def test_html_blocks_tables_and_no_spaces_at_inline_tags(tmp_path):
    path = tmp_path / "p.html"
    path.write_text("<html><head><title>規程</title><style>x{}</style></head><body>"
                    "<nav>menu</nav><h1>就業規則</h1><p>東京<b>都</b>庁で<br>勉強会</p>"
                    "<table><tr><th>項目</th><th>日数</th></tr><tr><td>年休</td><td>10</td></tr>"
                    "</table><footer>©</footer></body></html>", encoding="utf-8")
    extracted = extract(path)
    assert extracted.title == "規程"
    texts = [(u.kind, u.text) for u in extracted.units]
    assert texts == [("heading", "就業規則"), ("text", "東京都庁で 勉強会"),
                     ("header", "項目\t日数"), ("row", "年休\t10")]


# --- PDF ---------------------------------------------------------------------------


def _pdf(path, pages):
    """A minimal PDF, one line of Helvetica text per entry in each page's list."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    font = 3 + 2 * len(pages)
    for i, lines in enumerate(pages):
        page_obj, content_obj = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_obj} 0 R")
        ops = "".join(f"BT /F1 12 Tf 72 {720 - 60 * n} Td ({line}) Tj ET\n"
                      for n, line in enumerate(lines))
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                       f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content_obj} 0 R >>")
        objects.append(f"<< /Length {len(ops)} >>\nstream\n{ops}endstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = b"%PDF-1.4\n"
    offsets = []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(out)
    return path


def test_pdf_pages_and_repeated_headers(tmp_path):
    bodies = ["Alpha opening facts.", "Bravo arguments.", "Charlie reasoning.", "Delta ruling."]
    pages = [["Court Record 274", bodies[n - 1], f"- {n} -"] for n in range(1, 5)]
    extracted = extract(_pdf(tmp_path / "a.pdf", pages))
    texts = [(u.page, u.text) for u in extracted.units]
    assert extracted.pages == 4
    assert texts.count((1, "Court Record 274")) == 1
    assert sum(1 for _p, t in texts if t == "Court Record 274") == 1
    assert sum(1 for _p, t in texts if t.startswith("- ")) == 1
    assert (3, "Charlie reasoning.") in texts
    assert any("repeated" in w for w in extracted.warnings)


def test_pdf_without_text_has_none(tmp_path):
    assert not extract(_pdf(tmp_path / "scan.pdf", [[], []])).has_text


def test_broken_pdf_is_an_extract_error(tmp_path):
    from smartrag.extract.tabular import ExtractError

    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(ExtractError):
        extract(path)


# --- DOCX, PPTX --------------------------------------------------------------------


def test_docx_headings_paragraphs_and_tables(tmp_path):
    import docx

    document = docx.Document()
    document.core_properties.title = "就業規則"
    document.add_heading("第3章 休暇", level=1)
    document.add_paragraph("社員は年休を取得できる。")
    table = document.add_table(rows=2, cols=2)
    for r, row in enumerate([["項目", "日数"], ["年休", "10"]]):
        for c, value in enumerate(row):
            table.cell(r, c).text = value
    path = tmp_path / "a.docx"
    document.save(str(path))
    extracted = extract(path)
    assert extracted.title == "就業規則"
    assert [(u.kind, u.text) for u in extracted.units] == [
        ("heading", "第3章 休暇"), ("text", "社員は年休を取得できる。"),
        ("header", "項目\t日数"), ("row", "年休\t10")]


def test_pptx_slides_in_position_order_with_notes(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches

    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "売上報告"
    lower = slide.shapes.add_textbox(Inches(1), Inches(5), Inches(4), Inches(1))
    lower.text_frame.text = "下の箱"
    upper = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(4), Inches(1))
    upper.text_frame.text = "上の箱"
    slide.notes_slide.notes_text_frame.text = "話すこと"
    path = tmp_path / "a.pptx"
    deck.save(str(path))
    units = extract(path).units
    assert [(u.kind, u.text, u.page) for u in units] == [
        ("heading", "売上報告", 1), ("text", "上の箱", 1), ("text", "下の箱", 1),
        ("text", "話すこと", 1)]
    assert units[-1].location == "slide 1 notes"


# --- chunking ----------------------------------------------------------------------

CONFIG = ChunkConfig(target_tokens=60, max_tokens=100, min_tokens=20)


def test_token_estimate():
    assert estimate_tokens("あ" * 30) == 20 and estimate_tokens("a" * 40) == 10


def test_headings_set_paths_and_cut_full_chunks():
    units = [Unit("第3章 休暇", "heading", level=1), Unit("あ" * 60),
             Unit("第12条 年休", "heading", level=3), Unit("い" * 60),
             Unit("第4章 賃金", "heading", level=1), Unit("う" * 60)]
    text, chunks = chunk_units(units, CONFIG)
    assert [c.heading_path for c in chunks] == [["第3章 休暇"], ["第3章 休暇", "第12条 年休"],
                                                ["第4章 賃金"]]
    assert all(text[c.start:c.end] == c.text for c in chunks)
    assert chunks[1].embed_text() == "第3章 休暇 > 第12条 年休\n" + "い" * 60


def test_table_rows_stay_together_and_repeat_the_header():
    units = [Unit("intro text")]
    units.append(Unit("品名\t数量", "header", group=1, location="S!A1:B9", row=1))
    units += [Unit(f"品{n}\t{n}" + "x" * 80, "row", group=1, location="S!A1:B9", row=n + 1)
              for n in range(1, 7)]
    _text, chunks = chunk_units(units, CONFIG)
    assert chunks[0].text == "intro text" and chunks[0].prefix == ""
    rows = chunks[1:]
    assert len(rows) > 1 and all(c.prefix == "品名\t数量" for c in rows)
    assert rows[0].location == "S!A1:B9, rows 2-4"  # header 3 + rows 22 each: target 60
    assert all("intro" not in c.text for c in rows)


def test_oversized_paragraphs_split_at_sentences_then_windows():
    long = "。".join("文" * 40 for _ in range(10)) + "。"
    _text, chunks = chunk_units([Unit(long)], CONFIG)
    assert len(chunks) > 1 and all(c.tokens <= CONFIG.max_tokens for c in chunks)
    _text, windows = chunk_units([Unit("字" * 1000)], CONFIG)
    assert len(windows) > 5 and all(c.tokens <= CONFIG.max_tokens for c in windows)


def test_small_tail_merges_into_the_previous_chunk():
    units = [Unit("あ" * 80), Unit("い" * 10)]
    _text, chunks = chunk_units(units, ChunkConfig(target_tokens=50, max_tokens=100,
                                                   min_tokens=20))
    assert len(chunks) == 1 and chunks[0].text.endswith("い" * 10)
