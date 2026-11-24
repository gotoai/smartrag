"""Partitioned search (`partition_by`, `top`) and compact results."""

from __future__ import annotations

import json
from collections import Counter

import pytest
from typer.testing import CliRunner

from smartrag.agent import AgentService, QueryEmbedders
from smartrag.errors import InvalidRequest
from smartrag.kb import Home, add_source
from smartrag.kb.sync import sync_kb
from smartrag.search import SearchFilter, search_kb
from smartrag.search.models import CompactHit, SearchHit
from tests.test_attributes import publish
from tests.test_search_read import CharEmbedder

FILES = {
    "a/1.md": "# 相続税\n\n相続税の評価について。土地の評価は路線価による。\n\n"
              "## 更正\n\n相続税の更正処分の取消しを求める訴え。\n",
    "a/2.md": "# 相続\n\n相続税の申告と評価の期限。\n",
    "b/1.md": "# 電子申告\n\n電子申告の義務化は特定法人に適用される。\n",
    "b/2.md": "# 賃金\n\n賃金の規程と相続税の扱い。\n",
    "c/1.md": "# 年休\n\n年休は10日。年休の規程。\n",
    "x/1.md": "# 相続税の評価\n\n相続税の評価と更正処分。\n",
}
ROWS = [
    {"path": "a/1.md", "sec_code": "7203", "fiscal_year": 2026, "link": "https://e.jp/a1"},
    {"path": "a/2.md", "sec_code": "7203", "fiscal_year": 2025, "link": "https://e.jp/a2"},
    {"path": "b/1.md", "sec_code": "9064", "fiscal_year": 2026},
    {"path": "b/2.md", "sec_code": "9064", "fiscal_year": 2025},
    {"path": "c/1.md", "sec_code": "6758", "fiscal_year": 2026},
    {"path": "x/1.md", "fiscal_year": 2026},  # no sec_code
]
SCHEMA = {"type": "object", "properties": {
    "sec_code": {"type": "string"}, "fiscal_year": {"type": "integer"},
    "link": {"type": "string", "format": "uri"}}}
CODES = ["7203", "9064", "6758"]
QUERY = "相続税の評価"


@pytest.fixture
def home(tmp_path):
    home = Home(tmp_path / "home")
    home.new("co", description="Companies' reports.")
    root = tmp_path / "docs"
    for path, text in FILES.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    add_source(home, "co", "docs", str(root))
    publish(root, ROWS, SCHEMA)
    report = sync_kb(home, "co", embedder=CharEmbedder())
    assert not report.error, report.error
    return home


def _search(home, query=QUERY, **options):
    return search_kb(home, "co", query, embedder=CharEmbedder(), **options)


# --- partitions --------------------------------------------------------------------


def test_each_partition_contributes_its_best_hit(home):
    plain = _search(home, limit=10)
    assert {h.citation.path.split("/")[0] for h in plain.results[:3]} <= {"a", "x"}
    r = _search(home, partition_by="sec_code")
    assert [h.partition for h in r.results] == ["7203", "9064", "6758"]
    assert r.partitions is not None
    assert (r.partitions.key, r.partitions.top, r.partitions.count,
            r.partitions.documents_without_key, r.partitions.empty) == (
        "sec_code", 1, 3, 1, [])


def test_scores_are_global_across_partitions(home):
    # The same chunks, ranked over the same documents without partitioning.
    every = _search(home, limit=50, max_per_doc=50, compact=False,
                    filters=SearchFilter(attributes={"sec_code": CODES}))
    score = {h.chunk_id: h.score for h in every.results}
    r = _search(home, partition_by="sec_code", top=2)
    assert r.results and all(h.score == score[h.chunk_id] for h in r.results)
    assert [h.score for h in r.results] == sorted((h.score for h in r.results), reverse=True)


def test_top_and_per_document_limits(home):
    r = _search(home, partition_by="sec_code", top=2)
    per = Counter(h.partition for h in r.results)
    assert per["7203"] == 2 and per["9064"] == 2 and max(per.values()) == 2
    docs = _search(home, partition_by="sec_code", top=2, group_by="document", compact=False)
    assert all(isinstance(h, SearchHit) for h in docs.results)
    by_doc = Counter(h.doc_id for h in docs.results)
    assert max(by_doc.values()) == 1 and Counter(h.partition for h in docs.results)["7203"] == 2


def test_listed_values_are_the_partitions_and_empty_ones_are_named(home):
    r = _search(home, partition_by="sec_code",
                filters=SearchFilter(attributes={"sec_code": ["7203", "6758", "0000"]}))
    assert {h.partition for h in r.results} == {"7203", "6758"}
    assert r.partitions.count == 2 and r.partitions.empty == ["0000"]
    # A value with documents, but none the other filters leave, is empty too.
    r = _search(home, partition_by="sec_code", filters=SearchFilter(
        attributes={"sec_code": ["7203", "6758"], "fiscal_year": 2025}))
    assert {h.partition for h in r.results} == {"7203"} and r.partitions.empty == ["6758"]


def test_pages_run_through_every_partition(home):
    first = _search(home, partition_by="sec_code", limit=2)
    assert len(first.results) == 2 and first.next_cursor
    second = _search(home, partition_by="sec_code", limit=2, cursor=first.next_cursor)
    assert [h.partition for h in first.results + second.results] == ["7203", "9064", "6758"]
    assert second.next_cursor is None
    with pytest.raises(InvalidRequest, match="different request"):
        _search(home, partition_by="sec_code", top=2, limit=2, cursor=first.next_cursor)


def test_partition_by_source_and_by_a_number(home):
    r = _search(home, partition_by="source", top=2)
    assert [h.partition for h in r.results] == ["docs", "docs"]
    assert r.partitions.documents_without_key == 0
    years = _search(home, partition_by="fiscal_year")
    assert sorted(h.partition for h in years.results) == [2025, 2026]


def test_attribute_lists_hold_a_thousand_values(home):
    codes = CODES + [f"{n:04d}" for n in range(997)]
    r = _search(home, filters=SearchFilter(attributes={"sec_code": codes}), limit=50,
                max_per_doc=50)
    assert {h.attributes["sec_code"] for h in r.results} == set(CODES)
    with pytest.raises(InvalidRequest, match="the limit is 1000"):
        _search(home, filters=SearchFilter(attributes={"sec_code": [*codes, "9999"]}))
    numbers = _search(home, filters=SearchFilter(attributes={"fiscal_year": [2025, 2024]}),
                      limit=50, max_per_doc=50)
    assert {h.attributes["fiscal_year"] for h in numbers.results} == {2025}


# --- compact -----------------------------------------------------------------------


def test_compact_hits_by_default_with_partition_by(home):
    r = _search(home, partition_by="sec_code")
    assert r.compact and all(isinstance(h, CompactHit) for h in r.results)
    hit = r.results[0]
    assert (hit.title, hit.heading, hit.link) == ("相続税", "相続税", "https://e.jp/a1")
    assert len(hit.snippet) <= 122
    dumped = hit.model_dump()
    assert "attributes" not in dumped and "citation" not in dumped
    full = _search(home, partition_by="sec_code", compact=False)
    assert not full.compact and isinstance(full.results[0], SearchHit)
    assert full.results[0].partition == "7203" and full.results[0].attributes["sec_code"]


def test_compact_without_partitions(home):
    r = _search(home, compact=True, limit=100)
    assert r.partitions is None and all(isinstance(h, CompactHit) for h in r.results)
    assert all(h.partition is None for h in r.results)
    assert isinstance(_search(home).results[0], SearchHit)


@pytest.mark.parametrize(("options", "message"), [
    ({"partition_by": "court"}, "cannot partition by 'court': no source has this attribute"),
    ({"top": 2}, "give partition_by"),
    ({"partition_by": "sec_code", "top": 11}, "top must be between 1 and 10"),
    ({"limit": 100}, "limit must be between 1 and 50 \\(100 for compact"),
    ({"compact": True, "include": ["text"]}, "compact results carry no highlights or text"),
])
def test_mistakes_fail_loudly(home, options, message):
    with pytest.raises(InvalidRequest, match=message) as caught:
        _search(home, **options)
    if "cannot partition" in message:
        assert "source, fiscal_year, sec_code" in caught.value.message


# --- agents and CLI ----------------------------------------------------------------


def _service(home):
    return AgentService(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))


def test_agents_get_compact_partitioned_hits_without_paths(home):
    r = _service(home).search("co", QUERY, partition_by="sec_code", k=100,
                              attributes={"sec_code": [*CODES, "0000"]})
    assert [h.partition for h in r.results] == ["7203", "9064", "6758"]
    assert r.partitions is not None and r.partitions.empty == ["0000"]
    text = r.model_dump_json()
    assert "file_path" not in text and str(home.root.parent) not in text
    full = _service(home).search("co", QUERY, partition_by="sec_code", compact=False)
    assert full.results[0].citation.file_path  # show_paths is on for this kb


async def test_mcp_search_takes_partition_by(home):
    from mcp import Client

    from smartrag.mcp import build_server

    server = build_server(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))
    async with Client(server) as client:
        found = await client.call_tool("search", {"kb_id": "co", "query": QUERY,
                                                  "partition_by": "sec_code", "top": 2})
        assert not found.is_error, found.content[0].text
        results = found.structured_content["results"]
        assert Counter(r["partition"] for r in results)["7203"] == 2
        assert "attributes" not in results[0] and "file_path" not in found.content[0].text
        refused = await client.call_tool("search", {"kb_id": "co", "query": QUERY,
                                                    "partition_by": "court"})
        assert refused.is_error and "cannot partition by" in refused.content[0].text


def test_cli_partition_options(home):
    from smartrag.cli.main import app

    runner = CliRunner()
    base = ["run", "--home", str(home.root), "kb", "search", "--kb", "co", QUERY,
            "--search-mode", "keyword", "--partition-by", "sec_code"]
    text = runner.invoke(app, base)
    assert text.exit_code == 0, text.output
    assert "partitions by sec_code: 3, top 1 each · 1 documents without sec_code" in text.output
    assert "1. [7203] 相続税" in text.output
    as_json = runner.invoke(app, [*base[:2], str(home.root), "--format", "json", *base[3:],
                                  "--top", "2", "--compact=false"])
    assert as_json.exit_code == 0, as_json.output
    data = json.loads(as_json.stdout)
    assert not data["compact"] and data["results"][0]["citation"]["path"] == "a/1.md"
    assert runner.invoke(app, [*base, "--compact"]).exit_code != 0  # always takes a value
