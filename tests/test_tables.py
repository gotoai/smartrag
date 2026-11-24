"""Tables: files of rows imported into DuckDB at sync, and read-only SQL over them."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest
from typer.testing import CliRunner

pytest.importorskip("duckdb")

from smartrag.errors import Conflict, InvalidRequest, NotFound
from smartrag.kb.reindex import reindex_rebuild
from smartrag.kb.sync import sync_kb
from smartrag.tables import query as query_module
from smartrag.tables.manage import add_table, list_tables, remove_table
from smartrag.tables.query import describe_table, query_tables
from smartrag.tables.read import read_tables, safe_name
from tests.test_attributes import publish
from tests.test_search_read import CharEmbedder

CSV = (
    "証券コード,会社名,決算期,売上高,営業利益,上場\n"
    "7203,トヨタ自動車,2026-03-31,\"48,036,704\",\"5,352,934\",true\n"
    "627A,テスト,2026/03/31,１２３,△45,false\n"
    "0123,ゼロ始まり,2025-03-31,100,-,true\n"
)
SCHEMA = {"description": "主要な財務数値（会社×決算期）", "properties": {
    "売上高": {"type": "integer", "unit": "百万円", "description": "連結売上高"},
    "営業利益": {"type": "integer", "unit": "百万円"}}}


def _sync(home, **options):
    report = sync_kb(home, "legal", embedder=CharEmbedder(), **options)
    assert not report.error, report.error
    return {t.name: t for t in report.tables}


@pytest.fixture
def data(tmp_path):
    folder = tmp_path / "tables"
    folder.mkdir()
    (folder / "財務.csv").write_bytes(CSV.encode("cp932"))
    (folder / "財務.csv.schema.json").write_text(json.dumps(SCHEMA, ensure_ascii=False),
                                                 encoding="utf-8")
    db = sqlite3.connect(folder / "master.sqlite")
    db.execute("CREATE TABLE companies (sec_code TEXT, name TEXT, capital INTEGER)")
    db.executemany("INSERT INTO companies VALUES (?, ?, ?)",
                   [("7203", "トヨタ自動車", 635401), ("9064", "ヤマト", 127234)])
    db.execute("CREATE VIEW v AS SELECT * FROM companies")
    db.commit()
    db.close()
    return folder


@pytest.fixture
def home(agent_home, data):
    add_table(agent_home, "legal", str(data))
    return agent_home


# --- reading ------------------------------------------------------------------------


def test_names_are_safe_for_sql():
    assert safe_name("Net Sales (円)", "c1") == "net_sales_円"
    assert safe_name("2024年度", "t") == "t_2024年度"
    assert safe_name("---", "c3") == "c3"


def test_values_and_types_are_read_best_effort(data):
    (table,) = read_tables(data / "財務.csv")
    assert table.name == "財務" and table.description == "主要な財務数値（会社×決算期）"
    types = {c.original: c.type for c in table.columns}
    assert types == {"証券コード": "string", "会社名": "string", "決算期": "date",
                     "売上高": "integer", "営業利益": "integer", "上場": "boolean"}
    by_code = {row[0]: row for row in table.rows}
    assert by_code["7203"][3] == 48036704 and by_code["627A"][3] == 123  # full-width
    assert by_code["627A"][4] == -45 and by_code["0123"][4] is None  # △ and -
    assert str(by_code["627A"][2]) == "2026-03-31"  # 2026/03/31
    sales = next(c for c in table.columns if c.original == "売上高")
    assert (sales.name, sales.unit, sales.description) == ("売上高", "百万円", "連結売上高")


def test_codes_keep_their_leading_zeros(tmp_path):
    path = tmp_path / "codes.csv"
    path.write_text("code,n\n0123,1\n4567,2\n", encoding="utf-8")
    (table,) = read_tables(path)
    assert [c.type for c in table.columns] == ["string", "integer"]
    assert table.rows[0][0] == "0123"


def test_a_mostly_numeric_column_stays_text_with_a_warning(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("v\n" + "".join(f"{n}\n" for n in range(20)) + "n/a?\n", encoding="utf-8")
    (table,) = read_tables(path)
    assert table.columns[0].type == "string"
    assert "1 of 21 values are not numbers" in table.warnings[0]


def test_values_breaking_the_schema_are_dropped(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("v\n1\nx\n3\n", encoding="utf-8")
    (tmp_path / "s.csv.schema.json").write_text(
        json.dumps({"properties": {"v": {"type": "integer"}}}), encoding="utf-8")
    (table,) = read_tables(path)
    assert [row[0] for row in table.rows] == [1, None, 3]
    assert "1 values are not integer (schema), dropped" in table.warnings[0]


def test_sqlite_tables_and_excel_sheets(data, tmp_path):
    (companies,) = read_tables(data / "master.sqlite")  # the view is skipped
    assert companies.name == "master_companies" and companies.sheet == "companies"
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "売上"
    workbook.active.append(["年", "金額"])
    workbook.active.append([2025, 100])
    hidden = workbook.create_sheet("隠し")
    hidden.append(["x"])
    hidden.sheet_state = "hidden"
    workbook.save(tmp_path / "book.xlsx")
    (sheet,) = read_tables(tmp_path / "book.xlsx")
    assert (sheet.name, sheet.original) == ("book_売上", "book.xlsx › 売上")
    assert sheet.rows == [[2025, 100]]


# --- sync ---------------------------------------------------------------------------


def test_sync_imports_tables_and_query_reads_them(home):
    tables = _sync(home)
    assert {n: t.status for n, t in tables.items()} == {
        "財務": "added", "master_companies": "added"}
    r = query_tables(home, "legal", "SELECT 会社名, 売上高 FROM 財務 WHERE 売上高 > 1000 "
                                    "ORDER BY 売上高 DESC")
    assert r.rows == [["トヨタ自動車", 48036704]] and r.row_count == 1 and not r.truncated
    assert [c.name for c in r.columns] == ["会社名", "売上高"]
    assert [t.name for t in r.tables] == ["財務"] and r.tables[0].file.endswith("財務.csv")
    listed = list_tables(home, "legal")
    assert sorted(listed.entries[0].tables) == ["master_companies", "財務"]


def test_unchanged_files_are_carried_over_and_changes_reimported(home, data):
    _sync(home)
    assert {t.status for t in _sync(home).values()} == {"unchanged"}
    os.utime(data / "財務.csv", ns=(1, 1))  # touched, same bytes
    assert _sync(home)["財務"].status == "unchanged"
    (data / "財務.csv").write_bytes((CSV + "9064,ヤマト,2026-03-31,1,1,true\n").encode("cp932"))
    tables = _sync(home)
    assert tables["財務"].status == "updated" and tables["財務"].rows == 4
    assert tables["master_companies"].status == "unchanged"
    assert query_tables(home, "legal", "SELECT count(*) FROM master_companies").rows == [[2]]
    (data / "財務.csv.schema.json").write_text("{}", encoding="utf-8")
    assert _sync(home)["財務"].status == "updated"  # the schema changed
    (data / "master.sqlite").unlink()
    assert _sync(home)["master_companies"].status == "removed"
    with pytest.raises(InvalidRequest, match="master_companies"):
        query_tables(home, "legal", "SELECT * FROM master_companies")


def test_a_clashing_name_is_skipped(home, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "財務.csv").write_text("a\n1\n", encoding="utf-8")
    add_table(home, "legal", str(other))
    tables = _sync(home)
    skipped = [t for t in tables.values() if t.status == "skipped"]
    assert skipped and "already taken" in skipped[0].warnings[0]


def test_a_missing_location_keeps_its_tables(home, data):
    _sync(home)
    data.rename(data.with_name("gone"))
    tables = _sync(home)
    assert {t.status for t in tables.values()} == {"kept"}
    assert query_tables(home, "legal", "SELECT count(*) FROM 財務").rows == [[3]]


def test_an_unreadable_file_keeps_its_last_table(home, data):
    _sync(home)
    (data / "master.sqlite").write_bytes(b"not a database")
    table = _sync(home)["master_companies"]
    assert table.status == "kept" and "not a SQLite database" in table.warnings[0]


def test_dry_run_writes_nothing(home):
    report = sync_kb(home, "legal", dry_run=True)
    assert {t.status for t in report.tables} == {"added"}
    assert not (home.kb_dir("legal") / "tables.duckdb").exists()


def test_rebuild_reimports_and_remove_drops(home, data):
    _sync(home)
    report = reindex_rebuild(home, "legal", embedder=CharEmbedder())
    assert {t.status for t in report.tables} == {"updated"}
    removed = remove_table(home, "legal", str(data))
    assert sorted(removed.tables_removed) == ["master_companies", "財務"]
    assert list_tables(home, "legal").tables == []
    with pytest.raises(NotFound):
        remove_table(home, "legal", str(data))


def test_add_refuses_duplicates_and_missing_paths(home, data, tmp_path):
    with pytest.raises(Conflict):
        add_table(home, "legal", str(data))
    with pytest.raises(InvalidRequest, match="does not exist"):
        add_table(home, "legal", str(tmp_path / "nowhere"))
    assert add_table(home, "legal", str(tmp_path / "later"), allow_missing=True)


# --- query safety -------------------------------------------------------------------


@pytest.mark.parametrize(("sql", "message"), [
    ("SELECT * FROM read_csv('/etc/passwd')", "disabled"),
    ("COPY 財務 TO '/tmp/x.csv'", "disabled"),
    ("SET enable_external_access = true", "Cannot change"),
    ("CREATE TABLE t (a INT)", "read-only"),
    ("SELECT 1; SELECT 2", "one SQL statement"),
    ("SELECT nope FROM 財務", "nope"),
    ("", "empty"),
])
def test_queries_cannot_reach_outside_or_write(home, sql, message):
    _sync(home)
    with pytest.raises(InvalidRequest, match=message):
        query_tables(home, "legal", sql)


def test_a_long_query_is_stopped(home, monkeypatch):
    _sync(home)
    monkeypatch.setattr(query_module, "QUERY_TIMEOUT", 0.3)
    with pytest.raises(InvalidRequest, match="was stopped"):
        query_tables(home, "legal", "SELECT count(*) FROM range(100000000000)")


def test_rows_are_capped_and_counted(home):
    _sync(home)
    r = query_tables(home, "legal", "SELECT * FROM range(2500)", max_rows=10)
    assert len(r.rows) == 10 and r.row_count == 2500 and r.truncated
    with pytest.raises(InvalidRequest, match="max_rows"):
        query_tables(home, "legal", "SELECT 1", max_rows=1001)


# --- describe -----------------------------------------------------------------------


def test_describe_gives_meanings_profile_sample_and_links(home, tmp_path):
    publish(tmp_path / "docs", [{"path": "tax/相続.md", "証券コード": "7203"}])
    _sync(home)
    d = describe_table(home, "legal", "財務")
    columns = {c.name: c for c in d.columns}
    assert d.rows == 3 and len(d.sample) == 3 and d.description.startswith("主要な")
    assert columns["売上高"].unit == "百万円" and columns["営業利益"].nulls == 1
    assert columns["上場"].values == [False, True]
    assert [(link.column, link.attribute) for link in d.links] == [("証券コード", "証券コード")]
    with pytest.raises(NotFound, match="its tables are"):
        describe_table(home, "legal", "nope")


def test_links_also_match_escaped_attribute_names():
    from smartrag.tables.query import links

    found = links(["doc_id", "sec_code", "other"], {"_attr_doc_id", "sec_code"})
    assert [(f.column, f.attribute) for f in found] == [
        ("doc_id", "_attr_doc_id"), ("sec_code", "sec_code")]


# --- CLI ----------------------------------------------------------------------------


def test_cli_table_commands_and_query(agent_home, data):
    from smartrag.cli.main import app

    runner = CliRunner()
    base = ["run", "--home", str(agent_home.root)]
    added = runner.invoke(app, [*base, "table", "add", "--kb", "legal", str(data)])
    assert added.exit_code == 0, added.output
    dry = runner.invoke(app, [*base, "kb", "sync", "--kb", "legal", "--dry-run"])
    assert "tables     (would be) 2 added" in dry.stdout, dry.stdout
    _sync(agent_home)  # the real sync needs an embedding endpoint; the library takes a fake
    listed = runner.invoke(app, [*base, "table", "list", "--kb", "legal"])
    assert "財務  3 rows, 6 columns  from 財務.csv" in listed.stdout
    found = runner.invoke(app, [*base, "kb", "query", "--kb", "legal", "--sql",
                                "SELECT 会社名 FROM 財務 ORDER BY 1"])
    assert found.exit_code == 0, found.output
    assert "トヨタ自動車" in found.stdout and "3 rows" in found.stdout
    described = runner.invoke(app, [*base, "table", "describe", "--kb", "legal", "財務"])
    assert described.exit_code == 0, described.output
    assert "integer [百万円]" in described.stdout and "連結売上高" in described.stdout
    bad = runner.invoke(app, [*base, "kb", "query", "--kb", "legal", "--sql", "SELECT x"])
    assert bad.exit_code != 0 and "x" in bad.output
    removed = runner.invoke(app, [*base, "table", "remove", "--kb", "legal", str(data),
                                  "--yes"])
    assert removed.exit_code == 0 and "dropped 2 tables" in removed.stdout


# --- agents -------------------------------------------------------------------------


async def test_agents_list_describe_and_query_tables_over_mcp(home, tmp_path):
    from mcp import Client

    from smartrag.agent import QueryEmbedders
    from smartrag.mcp import build_server

    publish(tmp_path / "docs", [{"path": "tax/相続.md", "証券コード": "7203"}])
    _sync(home)
    server = build_server(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))
    async with Client(server) as client:
        assert "list_tables, describe_table" in (client.instructions or "")
        described = await client.call_tool("describe_sources", {"kb_id": "legal"})
        assert sorted(described.structured_content["tables"]) == ["master_companies", "財務"]
        listed = await client.call_tool("list_tables", {"kb_id": "legal"})
        names = {t["name"]: t for t in listed.structured_content["tables"]}
        assert names["財務"]["origin"] == "財務.csv" and names["財務"]["rows"] == 3
        table = await client.call_tool("describe_table", {"kb_id": "legal", "table": "財務"})
        assert table.structured_content["links"] == [
            {"column": "証券コード", "attribute": "証券コード"}]
        result = await client.call_tool("query", {
            "kb_id": "legal", "sql": "SELECT 証券コード FROM 財務 WHERE 営業利益 > 0"})
        assert not result.is_error and result.structured_content["rows"] == [["7203"]]
        assert result.structured_content["tables"][0]["file"].endswith("/財務.csv")
        refused = await client.call_tool("query", {
            "kb_id": "legal", "sql": "SELECT * FROM read_csv('/etc/passwd')"})
        assert refused.is_error and "disabled" in refused.content[0].text


def test_hidden_paths_give_file_names_only(home):
    from smartrag.agent import AgentService, QueryEmbedders

    _sync(home)
    config = home.kb_dir("legal") / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "show_paths: true", "show_paths: false"), encoding="utf-8")
    service = AgentService(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))
    result = service.query("legal", "SELECT count(*) FROM 財務")
    assert result.tables[0].file == "財務.csv"
    assert service.describe_table("legal", "財務").file == "財務.csv"


def test_a_kb_with_tables_but_no_documents_is_usable(agent_home, data):
    from smartrag.agent import AgentService

    add_table(agent_home, "empty", str(data))
    sync_kb(agent_home, "empty", embedder=CharEmbedder())
    service = AgentService(agent_home)
    assert "empty" in {kb.kb_id for kb in service.list_kbs().kbs}
    assert service.describe_sources("empty").documents == 0
    assert {t.name for t in service.list_tables("empty").tables} == {"財務", "master_companies"}
