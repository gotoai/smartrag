"""Filters on attributes."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from smartrag.agent import AgentService, QueryEmbedders
from smartrag.errors import InvalidRequest
from smartrag.kb import add_source
from smartrag.kb.sync import sync_kb
from smartrag.search import SearchFilter, search_kb
from smartrag.search.attributes import MAX_VALUES
from tests.test_attributes import publish
from tests.test_search_read import CharEmbedder

ROWS = [
    {"path": "tax/相続.md", "sec_code": "7203", "industry": "輸送用機器",
     "fiscal_year_end": "2026-03-31", "fiscal_year": 2026, "amended": False,
     "submitted_at": "2026-06-20T15:00:00+09:00"},
    {"path": "tax/電子.md", "sec_code": "9064", "industry": "陸運業",
     "fiscal_year_end": "2025-03-31", "fiscal_year": 2025, "amended": True,
     "submitted_at": "2025-06-19T23:30:00-05:00"},
    {"path": "hr/規程.txt", "sec_code": "7267", "industry": "輸送用機器",
     "fiscal_year_end": "2024-03-31", "fiscal_year": 2024, "amended": False,
     "submitted_at": None},
]
SCHEMA = {"type": "object", "properties": {
    "sec_code": {"type": "string", "description": "証券コード"},
    "industry": {"type": "string", "description": "東証33業種"},
    "fiscal_year_end": {"type": "string", "format": "date", "description": "決算期末日"},
    "fiscal_year": {"type": "integer"},
    "amended": {"type": "boolean"},
    "submitted_at": {"type": "string", "format": "date-time"},
}}


@pytest.fixture
def home(agent_home, tmp_path):
    publish(tmp_path / "docs", ROWS, SCHEMA)
    sync_kb(agent_home, "legal", embedder=CharEmbedder())
    return agent_home


def _paths(home, query="資料", **attributes):
    r = search_kb(home, "legal", query, filters=SearchFilter(attributes=attributes),
                  embedder=CharEmbedder())
    return {h.citation.path for h in r.results}


# --- matching ----------------------------------------------------------------------


def test_values_lists_and_ranges(home):
    assert _paths(home, sec_code="7203") == {"tax/相続.md"}
    assert _paths(home, sec_code=["7203", "7267"]) == {"tax/相続.md", "hr/規程.txt"}
    assert _paths(home, industry="輸送用機器", fiscal_year_end={"from": "2025-01-01"}) == {
        "tax/相続.md"}
    both_ends = {"from": "2024-03-31", "to": "2025-03-31"}
    assert _paths(home, fiscal_year_end=both_ends) == {"tax/電子.md", "hr/規程.txt"}
    assert _paths(home, fiscal_year={"to": 2024}) == {"hr/規程.txt"}
    assert _paths(home, sec_code="0000") == set()


def test_values_are_read_by_the_keys_type(home):
    assert _paths(home, fiscal_year="2025") == {"tax/電子.md"}  # a string for an integer
    assert _paths(home, sec_code=7203) == {"tax/相続.md"}  # a number for a string
    assert _paths(home, amended=True) == {"tax/電子.md"}
    assert _paths(home, amended="false") == {"tax/相続.md", "hr/規程.txt"}


def test_a_date_time_matches_by_the_date_it_is_written_with(home):
    # 2025-06-19T23:30-05:00 is 06-20 in UTC, but the filing says 06-19.
    assert _paths(home, submitted_at="2025-06-19") == {"tax/電子.md"}
    assert _paths(home, submitted_at={"from": "2025-06-20"}) == {"tax/相続.md"}


def test_null_and_missing_values_never_match(home):
    assert "hr/規程.txt" not in _paths(home, submitted_at={"from": "1900-01-01"})


def test_filters_narrow_before_ranking_on_both_sides(home):
    # The best matches for the query are in 相続.md; the filter leaves only 規程.txt,
    # and still fills the page from it.
    r = search_kb(home, "legal", "相続税の評価", embedder=CharEmbedder(),
                  filters=SearchFilter(attributes={"sec_code": "7267"}))
    assert r.results and {h.citation.path for h in r.results} == {"hr/規程.txt"}
    for mode in ("keyword", "vector"):
        r = search_kb(home, "legal", "賃金", search_mode=mode, embedder=CharEmbedder(),
                      filters=SearchFilter(attributes={"industry": "陸運業"}))
        assert {h.citation.path for h in r.results} <= {"tax/電子.md"}


def test_attributes_combine_with_the_other_filters(home):
    r = search_kb(home, "legal", "資料", embedder=CharEmbedder(), filters=SearchFilter(
        file_type=["md"], attributes={"industry": "輸送用機器"}))
    assert {h.citation.path for h in r.results} == {"tax/相続.md"}


def test_cursors_bind_the_attribute_filters(home):
    filters = SearchFilter(attributes={"industry": "輸送用機器"})
    first = search_kb(home, "legal", "資料", limit=1, filters=filters, embedder=CharEmbedder())
    assert first.next_cursor
    with pytest.raises(InvalidRequest, match="different request"):
        search_kb(home, "legal", "資料", limit=1, cursor=first.next_cursor,
                  embedder=CharEmbedder(),
                  filters=SearchFilter(attributes={"industry": "陸運業"}))


# --- refusals ----------------------------------------------------------------------


@pytest.mark.parametrize(("attributes", "message"), [
    ({"court": "東京地方裁判所"}, "'court' cannot be filtered on: no source has this"),
    ({"title": "x"}, "'title' cannot be filtered on"),
    ({"sec_code": {"from": "7000"}}, "is a string: give a value or a list, not a range"),
    ({"fiscal_year_end": "2026/03/31"}, "give a date as YYYY-MM-DD"),
    ({"fiscal_year": "twenty"}, "is a integer; 'twenty' is not one"),
    ({"amended": [True, False]}, "is a boolean: give true or false"),
    ({"sec_code": []}, "the list for attribute 'sec_code' is empty"),
    ({"fiscal_year": {}}, "has neither `from` nor `to`"),
    ({"sec_code": [str(n) for n in range(MAX_VALUES + 1)]}, f"the limit is {MAX_VALUES}"),
])
def test_mistakes_fail_loudly(home, attributes, message):
    with pytest.raises(InvalidRequest, match=message) as caught:
        _paths(home, **attributes)
    if "cannot be filtered" in message:
        assert "fiscal_year_end" in caught.value.message  # names what can be filtered


def test_without_a_schema_keys_are_filtered_as_strings(agent_home, tmp_path):
    publish(tmp_path / "docs", ROWS)
    sync_kb(agent_home, "legal", embedder=CharEmbedder())
    assert _paths(agent_home, sec_code="7203") == {"tax/相続.md"}
    assert _paths(agent_home, fiscal_year=2025) == {"tax/電子.md"}  # read as "2025"
    assert _paths(agent_home, amended="true") == {"tax/電子.md"}
    with pytest.raises(InvalidRequest, match="is a string: give a value or a list"):
        _paths(agent_home, fiscal_year={"to": 2024})


def test_two_schemas_must_agree_on_a_key(home, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "report.md").write_text("# 報告\n\n資料の報告。\n", encoding="utf-8")
    add_source(home, "legal", "other", str(other))
    publish(other, [{"path": "report.md", "fiscal_year": "2026", "sec_code": "7203"}],
            {"type": "object", "properties": {
                "fiscal_year": {"type": "string"}, "sec_code": {"type": "string"}}})
    sync_kb(home, "legal", embedder=CharEmbedder())
    assert _paths(home, sec_code="7203") == {"tax/相続.md", "report.md"}
    with pytest.raises(InvalidRequest, match="its sources give it different types"):
        _paths(home, fiscal_year=2026)


# --- agents ------------------------------------------------------------------------


def _service(home):
    return AgentService(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))


def test_agents_filter_on_every_attribute(home):
    service = _service(home)
    r = service.search("legal", "資料", attributes={
        "sec_code": ["7203", "9064"], "fiscal_year_end": {"from": "2026-01-01"}})
    assert [h.attributes["sec_code"] for h in r.results] == ["7203"]
    r = service.search("legal", "資料", attributes={"industry": "陸運業"})
    assert [h.attributes["sec_code"] for h in r.results] == ["9064"]
    with pytest.raises(InvalidRequest, match="no source has this attribute") as caught:
        service.search("legal", "資料", attributes={"court": "x"})
    assert "industry" in caught.value.message.split("are:")[1]
    with pytest.raises(InvalidRequest, match="a value, a list of values"):
        service.search("legal", "資料", attributes={"sec_code": {"is": "7203"}})


def test_describe_sources_lists_the_attribute_filters(home):
    filters = {f.name: f for f in _service(home).describe_sources("legal").filters}
    industry = filters["attributes.industry"]
    assert (industry.type, industry.shapes, industry.description) == (
        "string", ["value", "list"], "東証33業種")
    assert industry.values == ["輸送用機器", "陸運業"]
    date = filters["attributes.fiscal_year_end"]
    assert (date.format, date.shapes, date.values) == ("date", ["value", "list", "range"],
                                                       None)
    assert filters["attributes.amended"].values == [False, True]
    assert filters["attributes.sec_code"].values is not None  # three codes: listed


async def test_mcp_search_takes_an_attributes_object(home):
    from mcp import Client

    from smartrag.mcp import build_server

    server = build_server(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))
    async with Client(server) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert "attributes" in tools["search"].input_schema["properties"]
        found = await client.call_tool("search", {"kb_id": "legal", "query": "資料",
                                                  "attributes": {"sec_code": ["9064"]}})
        assert not found.is_error
        assert {r["attributes"]["sec_code"] for r in found.structured_content["results"]} == {
            "9064"}
        refused = await client.call_tool("search", {"kb_id": "legal", "query": "資料",
                                                    "attributes": {"court": "x"}})
        assert refused.is_error and "no source has this attribute" in refused.content[0].text


# --- CLI ---------------------------------------------------------------------------


def test_cli_attr_options(home):
    from smartrag.cli.main import app

    runner = CliRunner()
    base = ["run", "--home", str(home.root), "--format", "json", "kb", "search", "--kb",
            "legal", "相続税 電子申告 賃金", "--search-mode", "keyword"]

    def paths(*options):
        result = runner.invoke(app, [*base, *options])
        assert result.exit_code == 0, result.output
        return {r["citation"]["path"] for r in json.loads(result.stdout)["results"]}

    assert paths() == {"tax/相続.md", "tax/電子.md", "hr/規程.txt"}
    assert paths("--attr", "sec_code=7203", "--attr", "sec_code=7267") == {
        "tax/相続.md", "hr/規程.txt"}
    assert paths("--attr", "fiscal_year_end=2025-03-31..") == {"tax/相続.md", "tax/電子.md"}
    assert paths("--attr", "fiscal_year=..2024") == {"hr/規程.txt"}
    assert paths("--attr", "amended=true") == {"tax/電子.md"}
    bad = runner.invoke(app, [*base, "--attr", "fiscal_year=2024..", "--attr",
                              "fiscal_year=2026"])
    assert bad.exit_code == 1 and "cannot be combined" in bad.output
    assert runner.invoke(app, [*base, "--attr", "sec_code"]).exit_code == 1
