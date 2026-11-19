"""Extractors added for more file types: lightweight markups, data files, RTF, e-mail,
macro-enabled Office files, and text files found by their bytes."""

from __future__ import annotations

import base64
import json
import zipfile

import pytest

from smartrag.errors import Unsupported
from smartrag.extract.sniff import looks_binary
from smartrag.extract.units import extract
from smartrag.kb.config import ChunkConfig

#: The table tests' sizes are set against these budgets, not the defaults.
TABLE_CHUNKS = ChunkConfig(target_tokens=350, max_tokens=512)


def _pairs(units):
    return [(u.kind, u.level, u.text) for u in units]


# --- sniffing ----------------------------------------------------------------------


@pytest.mark.parametrize(("sample", "binary"), [
    (b"", False),
    (b"def main():\n\treturn 0\n", False),
    ("設定ファイル\r\n".encode("cp932"), False),
    ("[section]\r\nkey=値\r\n".encode("utf-16-le"), True),  # no byte-order mark
    (b"\xff\xfe" + "[section]\r\nkey=値\r\n".encode("utf-16-le"), False),
    (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", True),
    (bytes(range(1, 32)) * 4, True),
    (b"log line\x1b[31m red \x1b[0m\n", False),
])
def test_looks_binary(sample, binary):
    assert looks_binary(sample) is binary


def test_text_files_without_an_extractor_read_as_plain_text(tmp_path):
    code = tmp_path / "main.py"
    code.write_text("import os\n\n\ndef main():\n    return os.getcwd()\n", encoding="utf-8")
    assert _pairs(extract(code).units) == [
        ("text", 0, "import os"), ("text", 0, "def main():\n    return os.getcwd()")]
    readme = tmp_path / "README"
    readme.write_text("■ 使い方\n\n実行する。\n", encoding="utf-8")
    assert _pairs(extract(readme).units) == [("heading", 3, "■ 使い方"), ("text", 0, "実行する。")]


def test_binary_and_never_text_files_are_unsupported(tmp_path):
    blob = tmp_path / "data.dat"
    blob.write_bytes(b"\x00\x01\x02" * 100)
    with pytest.raises(Unsupported):
        extract(blob)
    svg = tmp_path / "logo.svg"
    svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    with pytest.raises(Unsupported):
        extract(svg)


def test_log_files_are_plain_text(tmp_path):
    path = tmp_path / "app.log"
    path.write_text("2026-11-17 INFO start\n2026-11-17 ERROR 接続失敗\n", encoding="utf-8")
    assert extract(path).units[0].text.endswith("ERROR 接続失敗")


# --- reStructuredText and AsciiDoc -------------------------------------------------


def test_rst_titles_are_levelled_by_first_appearance(tmp_path):
    path = tmp_path / "guide.rst"
    path.write_text("=========\n  Title\n=========\n\nIntro.\n\nSection A\n---------\n\nText a.\n\n"
                    "Sub\n~~~\n\nText sub.\n\nSection B\n---------\n\nText b.\n",
                    encoding="utf-8")
    extracted = extract(path)
    assert extracted.title == "Title"
    assert [(u.level, u.text) for u in extracted.units if u.kind == "heading"] == [
        (1, "Title"), (2, "Section A"), (3, "Sub"), (2, "Section B")]
    assert ("text", 0, "Text sub.") in _pairs(extracted.units)
    assert not any(set(u.text) <= set("=-~ ") for u in extracted.units)


def test_asciidoc_headings_and_listing_blocks(tmp_path):
    path = tmp_path / "manual.adoc"
    path.write_text("= 手引き\n:toc:\n\n== 申請\n\n申請する。\n\n----\n== not a heading\n----\n",
                    encoding="utf-8")
    extracted = extract(path)
    assert extracted.title == "手引き"
    assert _pairs(extracted.units) == [
        ("heading", 1, "手引き"), ("text", 0, ":toc:"), ("heading", 2, "申請"),
        ("text", 0, "申請する。"), ("text", 0, "----\n== not a heading\n----")]


# --- Markdown tables ---------------------------------------------------------------


def test_markdown_headings_are_its_hash_lines_only(tmp_path):
    path = tmp_path / "hanki.md"
    path.write_text(
        "# 半期報告書\n\n## 第一部【企業情報】\n\n##### 【注記事項】\n\n"
        "【セグメント情報】\n\n単一セグメントである。\n\n■ 補足\n\n"
        "##### （２）【中間損益計算書】\n", encoding="utf-8")
    assert _pairs(extract(path).units) == [
        ("heading", 1, "半期報告書"), ("heading", 2, "第一部【企業情報】"),
        ("heading", 5, "【注記事項】"), ("text", 0, "【セグメント情報】"),
        ("text", 0, "単一セグメントである。"), ("text", 0, "■ 補足"),
        ("heading", 5, "（２）【中間損益計算書】")]


def test_markdown_pipe_tables_are_tables(tmp_path):
    path = tmp_path / "yuho.md"
    path.write_text(
        "# 有価証券報告書\n\n連結貸借対照表は次のとおり。\n"
        "| 科目 | 前期 | 当期 |\n"
        "|:---|---:|---:|\n"
        "| 現金\\|預金 | 1,000 | 1,200 |\n"
        "| 売掛金<br>（純額） | 500 | 450 |\n"
        "| \u3000\u3000流動資産合計 | 1,500 | 1,650 |\n"
        "|  |  |  |\n"
        "\n"
        "注記は省略。\n\n"
        "部門 | 人数\n"
        "--- | ---\n"
        "営業 | 12\n"
        "以上\n", encoding="utf-8")
    units = extract(path).units
    assert _pairs(units) == [
        ("heading", 1, "有価証券報告書"), ("text", 0, "連結貸借対照表は次のとおり。"),
        ("header", 0, "科目\t前期\t当期"), ("row", 0, "現金|預金\t1,000\t1,200"),
        ("row", 0, "売掛金 （純額）\t500\t450"),
        ("row", 0, "\u3000\u3000流動資産合計\t1,500\t1,650"), ("text", 0, "注記は省略。"),
        ("header", 0, "部門\t人数"), ("row", 0, "営業\t12"), ("text", 0, "以上")]
    assert [u.group for u in units if u.kind in ("header", "row")] == [1, 1, 1, 1, 2, 2]


@pytest.mark.parametrize("text", [
    "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```\n",  # inside fenced code
    "見出し\n---\n",  # a rule under a line without pipes
    "| a | b |\nno delimiter here\n",
])
def test_what_is_not_a_markdown_table(tmp_path, text):
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    assert {u.kind for u in extract(path).units} <= {"text", "heading"}


def test_a_long_markdown_table_keeps_its_header_on_every_chunk(tmp_path):
    from smartrag.chunking import chunk_units

    rows = "".join(f"| 項目{i} | {i * 1000:,} | {i * 1100:,} |\n" for i in range(1, 121))
    path = tmp_path / "bs.md"
    path.write_text("## 連結貸借対照表\n\n| 科目 | 前連結会計年度 | 当連結会計年度 |\n"
                    "|---|---:|---:|\n" + rows, encoding="utf-8")
    _text, chunks = chunk_units(extract(path).units, TABLE_CHUNKS)
    assert len(chunks) > 1
    assert {c.prefix for c in chunks} == {"科目\t前連結会計年度\t当連結会計年度"}
    assert all(c.heading_path == ["連結貸借対照表"] for c in chunks)
    assert "|" not in "".join(c.text for c in chunks)


def test_a_small_table_is_packed_whole_with_the_text_around_it(tmp_path):
    from smartrag.chunking import chunk_units

    path = tmp_path / "note.md"
    path.write_text("## 注記\n\n（単位：百万円）\n\n| 区分 | 金額 |\n|---|---:|\n| 土地 | 100 |\n"
                    "| 建物 | 200 |\n\n上記は担保に供している資産である。\n", encoding="utf-8")
    _text, chunks = chunk_units(extract(path).units, TABLE_CHUNKS)
    assert len(chunks) == 1 and chunks[0].prefix == ""
    assert chunks[0].text == ("（単位：百万円）\n区分\t金額\n土地\t100\n建物\t200\n"
                              "上記は担保に供している資産である。")


def test_a_small_table_is_never_split_between_chunks(tmp_path):
    from smartrag.chunking import chunk_units

    # About 300 tokens of prose, below the target, then a table of about 350: together
    # they pass max_tokens, so the table starts the next chunk, whole.
    rows = "".join(f"| 項目{i} | {i} |\n" for i in range(120))
    path = tmp_path / "doc.md"
    path.write_text("前置きの文章です。" * 50 + "\n\n| 項目 | 値 |\n|---|---|\n" + rows,
                    encoding="utf-8")
    _text, chunks = chunk_units(extract(path).units, TABLE_CHUNKS)
    assert len(chunks) == 2 and "項目" not in chunks[0].text
    assert chunks[1].text.startswith("項目\t値\n項目0\t0") and "項目119" in chunks[1].text


def test_a_caption_line_travels_with_the_table_header(tmp_path):
    from smartrag.chunking import chunk_units

    rows = "".join(f"| 項目{i} | {i:,} |\n" for i in range(200))
    path = tmp_path / "bs.md"
    path.write_text("## 連結貸借対照表\n\n（単位：百万円）\n\n| 科目 | 当期 |\n|---|---:|\n" + rows,
                    encoding="utf-8")
    units = extract(path).units
    assert _pairs(units)[1] == ("header", 0, "（単位：百万円）\n科目\t当期")
    _text, chunks = chunk_units(units, TABLE_CHUNKS)
    assert len(chunks) > 1 and {c.prefix for c in chunks} == {"（単位：百万円）\n科目\t当期"}


@pytest.mark.parametrize(("above", "header"), [
    ("表1 地域別売上高", "表1 地域別売上高\n地域\t売上"),
    ("（単位：百万円）\n\n報告セグメント", "（単位：百万円）\n報告セグメント\n地域\t売上"),
    ("前置き\n\n（単位：百万円）\n\n報告セグメント", "（単位：百万円）\n報告セグメント\n地域\t売上"),
    ("地域別の売上は次のとおりです。", "地域\t売上"),  # a sentence introducing the table
    ("# 売上", "地域\t売上"),  # a heading is not a caption
])
def test_what_counts_as_a_markdown_caption(tmp_path, above, header):
    path = tmp_path / "t.md"
    path.write_text(f"{above}\n\n| 地域 | 売上 |\n|---|---|\n| 東京 | 10 |\n", encoding="utf-8")
    assert [u.text for u in extract(path).units if u.kind == "header"] == [header]


def test_a_table_without_a_header_row_takes_its_caption_as_header(tmp_path):
    path = tmp_path / "t.md"
    path.write_text("（単位：千円）\n|  |  |\n|---|---|\n| 現金 | 10 |\n", encoding="utf-8")
    assert _pairs(extract(path).units) == [("header", 0, "（単位：千円）"),
                                           ("row", 0, "現金\t10")]


def test_an_html_caption_joins_the_header(tmp_path):
    path = tmp_path / "t.html"
    path.write_text("<p>単位の注記</p><table><caption>（単位：百万円）</caption>"
                    "<tr><th>科目</th><th>当期</th></tr><tr><td>現金</td><td>5</td></tr>"
                    "</table>", encoding="utf-8")
    assert _pairs(extract(path).units) == [
        ("text", 0, "単位の注記"), ("header", 0, "（単位：百万円）\n科目\t当期"),
        ("row", 0, "現金\t5")]


# --- JSON, JSON Lines, YAML, XML ---------------------------------------------------


def test_json_is_flattened_one_record_per_unit(tmp_path):
    path = tmp_path / "orders.json"
    path.write_text(json.dumps({
        "company": "A社", "active": True,
        "orders": [{"id": 1, "customer": {"name": "山田"}, "note": None}, {"id": 2, "items": []}],
        "meta": {"version": 2},
    }, ensure_ascii=False), encoding="utf-8")
    assert [u.text for u in extract(path).units] == [
        "company: A社\nactive: true", "orders[0].id: 1\norders[0].customer.name: 山田",
        "orders[1].id: 2", "meta.version: 2"]


def test_a_json_array_has_paths_inside_each_record(tmp_path):
    path = tmp_path / "list.json"
    path.write_text('[{"name": "a", "tags": ["x", "y"]}, {"name": "b"}]', encoding="utf-8")
    assert [u.text for u in extract(path).units] == [
        "name: a\ntags[0]: x\ntags[1]: y", "name: b"]


def test_invalid_json_is_indexed_as_plain_text(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{\n  // a comment\n  "a": 1\n}\n', encoding="utf-8")
    extracted = extract(path)
    assert "a comment" in extracted.units[0].text
    assert "not valid JSON" in extracted.warnings[0]


def test_jsonl_is_one_unit_per_line(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"user": "山田", "event": {"type": "login"}}\n\nnot json\n{"user": "佐藤"}\n',
                    encoding="utf-8")
    extracted = extract(path)
    assert [u.text for u in extracted.units] == [
        "user: 山田\nevent.type: login", "not json", "user: 佐藤"]
    assert "1 line not valid JSON" in extracted.warnings[0]


def test_yaml_documents_dates_and_aliases(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("name: 申請\nsince: 2026-04-01\nsteps:\n  - 記入\n  - 提出\n---\nother: 2\n",
                    encoding="utf-8")
    assert [u.text for u in extract(path).units] == [
        "name: 申請\nsince: 2026-04-01", "steps[0]: 記入", "steps[1]: 提出", "other: 2"]


def test_yaml_aliases_cannot_multiply_the_text(tmp_path):
    path = tmp_path / "bomb.yml"
    path.write_text("a: &a [x, x, x, x, x, x, x, x, x]\nb: &b [*a, *a, *a, *a, *a, *a]\n"
                    "c: &c [*b, *b, *b, *b, *b, *b]\nd: [*c, *c, *c, *c, *c, *c]\n",
                    encoding="utf-8")
    text = "\n".join(u.text for u in extract(path).units)
    assert text.count("x") == 9


def test_invalid_yaml_is_indexed_as_plain_text(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("key: [unclosed\n", encoding="utf-8")
    extracted = extract(path)
    assert extracted.units[0].text == "key: [unclosed"
    assert "not valid YAML" in extracted.warnings[0]


def test_xml_records_attributes_and_prose(tmp_path):
    path = tmp_path / "catalog.xml"
    path.write_bytes('<?xml version="1.0" encoding="Shift_JIS"?>\n'
                     '<catalog xmlns="urn:x"><!-- a comment -->'
                     '<item id="1"><name>りんご</name><price currency="JPY">100</price></item>'
                     '<doc><para>東京<b>都</b>庁の\n  案内</para></doc></catalog>'
                     .encode("shift_jis"))
    assert [u.text for u in extract(path).units] == [
        "item@id: 1\nname: りんご\nprice@currency: JPY\nprice: 100", "東京都庁の 案内"]


def test_malformed_xml_is_indexed_as_plain_text(tmp_path):
    path = tmp_path / "broken.xml"
    path.write_text("<a><b>text</a>", encoding="utf-8")
    extracted = extract(path)
    assert extracted.units[0].text == "<a><b>text</a>"
    assert "not well-formed XML" in extracted.warnings[0]


# --- RTF ---------------------------------------------------------------------------


def _bytes(text: str, codec: str = "cp932") -> str:
    return "".join(f"\\'{byte:02x}" for byte in text.encode(codec))


def test_rtf_text_headings_tables_and_code_pages(tmp_path):
    rtf = (
        r"{\rtf1\ansi\ansicpg1252\deff0"
        r"{\fonttbl{\f0\fswiss\fcharset0 Arial;}{\f1\fnil\fcharset128 MS Mincho;}}"
        r"{\colortbl;\red0\green0\blue0;}{\stylesheet{\s1 heading 1;}}"
        r"{\info{\title Rules}{\author Someone}}"
        r"{\*\generator Riched20;}{\header hidden header}"
        "\n\\pard\\outlinelevel0\\f1 " + _bytes("第3章 休暇") + r"\par"
        "\n\\pard\\f0 Plain {\\b bold} text\\par"
        "\n\\pard\\hich\\af0\\dbch\\af1\\loch\\f0 " + _bytes("年休を取得") + r"\par"
        "\n\\pard\\uc1\\u26085?\\u26412?\\par"
        f"\n\\pard\\uc2\\u{ord('規') - 65536}" + _bytes("規") + r"\par"
        r"{\field{\*\fldinst HYPERLINK http://x}{\fldrslt link}}\par"
        "\n\\trowd\\cellx1000\\cellx2000 \\pard\\intbl A\\cell B\\cell\\row"
        "\n}")
    path = tmp_path / "doc.rtf"
    path.write_bytes(rtf.encode("ascii"))
    extracted = extract(path)
    assert extracted.title == "Rules"
    assert _pairs(extracted.units) == [
        ("heading", 1, "第3章 休暇"), ("text", 0, "Plain bold text"), ("text", 0, "年休を取得"),
        ("text", 0, "日本"), ("text", 0, "規"), ("text", 0, "link"), ("text", 0, "A\tB")]


def test_rtf_headings_by_style_and_unicode_copies(tmp_path):
    rtf = (r"{\rtf1\ansi\ansicpg1252{\fonttbl{\f0 Arial;}{\f1\fcharset128 MS Mincho;}}"
           r"{\stylesheet{\ql Normal;}{\s1\sbasedon0 Heading 1;}{\s2\f1 " + _bytes("見出し 2")
           + r";}{\s3\outlinelevel2 Custom;}{\*\cs10 Heading 1 Char;}}"
           r"{\info{\upr{\title \'3f\'3f}{\*\ud{\title \u35215?\u21063?}}}}"
           r"\pard\plain\s1 One\par\pard\plain\s2 Two\par\pard\s3 Three\par"
           r"\pard Body\par}")
    path = tmp_path / "styled.rtf"
    path.write_bytes(rtf.encode("ascii"))
    extracted = extract(path)
    assert extracted.title == "規則"
    assert _pairs(extracted.units) == [("heading", 1, "One"), ("heading", 2, "Two"),
                                       ("heading", 3, "Three"), ("text", 0, "Body")]


def test_rtf_document_code_page(tmp_path):
    path = tmp_path / "wordpad.rtf"
    path.write_bytes((r"{\rtf1\ansi\ansicpg932{\fonttbl{\f0\fnil MS UI Gothic;}}\f0 "
                      + _bytes("社内規程") + r"\par}").encode("ascii"))
    assert extract(path).units[0].text == "社内規程"


def test_a_file_named_rtf_that_is_not_rtf_fails(tmp_path):
    from smartrag.extract.tabular import ExtractError

    path = tmp_path / "fake.rtf"
    path.write_text("plain words", encoding="utf-8")
    with pytest.raises(ExtractError):
        extract(path)


# --- e-mail ------------------------------------------------------------------------


def test_eml_iso_2022_jp_headers_and_body(tmp_path):
    subject = "=?ISO-2022-JP?B?" + base64.b64encode("会議の件".encode("iso-2022-jp")).decode() + "?="
    head = (f"Subject: {subject}\r\nFrom: Yamada <yamada@example.com>\r\n"
            "To: sato@example.com\r\nDate: Tue, 17 Nov 2026 10:00:00 +0900\r\n"
            "MIME-Version: 1.0\r\nContent-Type: text/plain; charset=ISO-2022-JP\r\n"
            "Content-Transfer-Encoding: 7bit\r\n\r\n")
    path = tmp_path / "mail.eml"
    path.write_bytes(head.encode("ascii") + "来週に延期します。\r\n".encode("iso-2022-jp"))
    extracted = extract(path)
    assert extracted.title == "会議の件"
    assert _pairs(extracted.units) == [
        ("heading", 1, "会議の件"),
        ("text", 0, ("From: Yamada <yamada@example.com>\nTo: sato@example.com\n"
                    "Date: Tue, 17 Nov 2026 10:00:00 +0900")),
        ("text", 0, "来週に延期します。")]


def test_eml_html_body_and_attachment_names(tmp_path):
    from email.message import EmailMessage

    message = EmailMessage()
    message["Subject"] = "見積"
    message["From"] = "a@example.com"
    message.set_content("<html><body><h2>見積書</h2><p>添付をご確認ください。</p></body></html>",
                        subtype="html")
    message.add_attachment(b"%PDF-1.4", maintype="application", subtype="pdf",
                           filename="見積.pdf")
    path = tmp_path / "quote.eml"
    path.write_bytes(bytes(message))
    assert _pairs(extract(path).units) == [
        ("heading", 1, "見積"), ("text", 0, "From: a@example.com"), ("heading", 2, "見積書"),
        ("text", 0, "添付をご確認ください。"), ("text", 0, "Attachments: 見積.pdf")]


# --- macro-enabled Office files ----------------------------------------------------


def _macro_enabled(source, target, plain_type, macro_type):
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as copy:
        for item in original.infolist():
            data = original.read(item.filename)
            if item.filename == "[Content_Types].xml":
                data = data.replace(plain_type.encode(), macro_type.encode())
            copy.writestr(item, data)


def test_docm_reads_like_docx(tmp_path):
    import docx

    document = docx.Document()
    document.add_heading("第1章 総則", level=1)
    document.add_paragraph("マクロ付きの文書。")
    document.save(str(tmp_path / "a.docx"))
    _macro_enabled(tmp_path / "a.docx", tmp_path / "a.docm",
                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                   ".main+xml", "application/vnd.ms-word.document.macroEnabled.main+xml")
    assert _pairs(extract(tmp_path / "a.docm").units) == [
        ("heading", 1, "第1章 総則"), ("text", 0, "マクロ付きの文書。")]


def test_pptm_reads_like_pptx(tmp_path):
    from pptx import Presentation

    deck = Presentation()
    deck.slides.add_slide(deck.slide_layouts[5]).shapes.title.text = "売上"
    deck.save(str(tmp_path / "a.pptx"))
    _macro_enabled(tmp_path / "a.pptx", tmp_path / "a.pptm",
                   "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                   ".main+xml", "application/vnd.ms-powerpoint.presentation.macroEnabled.main+xml")
    assert _pairs(extract(tmp_path / "a.pptm").units) == [("heading", 1, "売上")]
