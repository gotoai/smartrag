"""`source apply`: manifest attributes (design memo §3.10)."""

from __future__ import annotations

import json
import unicodedata

import pytest
from typer.testing import CliRunner

from smartrag.agent import AgentService, QueryEmbedders
from smartrag.errors import InvalidRequest, NotFound
from smartrag.index import SqliteIndex
from smartrag.kb import remove_source
from smartrag.kb.manifest import apply_manifest, clear_manifest
from smartrag.kb.reindex import reindex_rebuild
from smartrag.search import read_chunk, search_kb
from tests.test_search_read import CharEmbedder

LINK = "https://example.jp/cases/1"
ROWS = [
    {"path": "tax/相続.md", "title": "相続税評価事件", "link": LINK, "court": "東京地方裁判所",
     "judge_date": "2024-03-01", "kind": "全文"},
    {"path": "hr/規程.txt", "court": "大阪地方裁判所", "judge_date": "2023-12-25"},
    {"path": "tax/未同期.md", "court": "名古屋地方裁判所"},
]
SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "事件名"},
        "link": {"type": "string", "format": "uri", "description": "判例詳細ページ"},
        "court": {"type": "string", "description": "裁判所名"},
        "judge_date": {"type": "string", "format": "date", "description": "裁判年月日"},
        "kind": {"type": "string"},
    },
}


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


@pytest.fixture
def files(tmp_path):
    manifest = _write(tmp_path / "manifest.jsonl", ROWS)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA, ensure_ascii=False), encoding="utf-8")
    return manifest, schema


def _hit(home, query, path):
    r = search_kb(home, "legal", query, search_mode="keyword")
    return next(h for h in r.results if h.citation.path == path)


def _allow(home, attributes, show_paths=True):
    config = home.kb_dir("legal") / "config.yaml"
    text = config.read_text(encoding="utf-8")
    text = text.replace("attributes: []", f"attributes: {json.dumps(attributes)}")
    if not show_paths:
        text = text.replace("show_paths: true", "show_paths: false")
    config.write_text(text, encoding="utf-8")


# --- apply -------------------------------------------------------------------------


def test_apply_reports_matches_and_results_carry_the_attributes(agent_home, files):
    manifest, schema = files
    before = SqliteIndex.open(agent_home.kb_dir("legal") / "index.db", readonly=True)
    generation = before.generation()
    before.close()
    report = apply_manifest(agent_home, "legal", "docs", [manifest], schema=schema)
    assert (report.rows, report.has_schema, report.documents) == (3, True, 3)
    assert (report.matched, report.rows_without_document, report.documents_without_row) == (
        2, 1, 1)
    assert report.unmatched_examples == ["tax/未同期.md"]
    assert report.attributes == ["court", "judge_date", "kind", "link", "title"]

    hit = _hit(agent_home, "相続税の評価", "tax/相続.md")
    assert hit.title == "相続税評価事件" and hit.citation.link == LINK
    assert hit.attributes == {"court": "東京地方裁判所", "judge_date": "2024-03-01",
                              "kind": "全文"}
    plain = _hit(agent_home, "電子申告", "tax/電子.md")
    assert plain.attributes == {} and plain.citation.link is None and plain.title == "電子申告"
    read = read_chunk(agent_home, "legal", hit.chunk_id)
    assert read.title == "相続税評価事件" and read.citation.link == LINK
    assert read.attributes["court"] == "東京地方裁判所"
    store = SqliteIndex.open(agent_home.kb_dir("legal") / "index.db", readonly=True)
    assert store.generation() == generation + 1  # old cursors expire
    store.close()


def test_a_new_manifest_replaces_the_old_one_and_clear_removes_it(agent_home, files, tmp_path):
    manifest, _schema = files
    apply_manifest(agent_home, "legal", "docs", [manifest])
    second = _write(tmp_path / "second.jsonl", [{"path": "hr/規程.txt", "court": "最高裁判所"}])
    report = apply_manifest(agent_home, "legal", "docs", [second])
    assert report.previous_rows == 3 and report.rows == 1
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").attributes == {}
    assert _hit(agent_home, "賃金", "hr/規程.txt").attributes == {"court": "最高裁判所"}
    cleared = clear_manifest(agent_home, "legal", "docs")
    assert cleared.cleared and cleared.previous_rows == 1
    assert _hit(agent_home, "賃金", "hr/規程.txt").attributes == {}
    assert clear_manifest(agent_home, "legal", "docs").note == "the source had no manifest"


def test_a_manifest_set_is_one_manifest(agent_home, tmp_path):
    a = _write(tmp_path / "a.jsonl", ROWS[:1])
    b = _write(tmp_path / "b.jsonl", ROWS[1:2])
    assert apply_manifest(agent_home, "legal", "docs", [a, b]).matched == 2
    dup = _write(tmp_path / "dup.jsonl", ROWS[:1])
    with pytest.raises(InvalidRequest, match="also at"):
        apply_manifest(agent_home, "legal", "docs", [a, dup])


def test_dry_run_changes_nothing(agent_home, files):
    manifest, schema = files
    report = apply_manifest(agent_home, "legal", "docs", [manifest], schema=schema,
                            dry_run=True)
    assert report.dry_run and report.matched == 2
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").attributes == {}


def test_paths_match_after_nfc_normalisation(agent_home, tmp_path):
    # Kanji have no decomposed form; a voiced kana (ガ) does.
    (tmp_path / "docs" / "tax" / "ガイド.md").write_text("# ガイド\n\n相続の手引き。\n",
                                                        encoding="utf-8")
    from smartrag.kb.sync import sync_kb

    sync_kb(agent_home, "legal", embedder=CharEmbedder())
    nfd = unicodedata.normalize("NFD", "tax/ガイド.md")
    assert nfd != "tax/ガイド.md"
    manifest = _write(tmp_path / "nfd.jsonl", [{"path": nfd, "court": "札幌地方裁判所"}])
    assert apply_manifest(agent_home, "legal", "docs", [manifest]).matched == 1
    assert _hit(agent_home, "手引き", "tax/ガイド.md").attributes == {"court": "札幌地方裁判所"}


# --- refusals: all or nothing ------------------------------------------------------


@pytest.mark.parametrize(("row", "message"), [
    ({"court": "x"}, "`path` must be a non-empty string"),
    ({"path": "/tax/相続.md"}, "not absolute"),
    ({"path": "tax\\相続.md"}, "separators"),
    ({"path": "tax/../相続.md"}, "segment"),
    ({"path": "tax/相続.md", "court": {"name": "x"}}, "nested"),
    ({"path": "tax/相続.md", "tags": ["a"]}, "nested"),
    ({"path": "tax/相続.md", "title": ""}, "`title` must be a non-empty string"),
    ({"path": "tax/相続.md", "link": "file:///etc/passwd"}, "http"),
])
def test_bad_rows_refuse_the_whole_manifest(agent_home, files, tmp_path, row, message):
    manifest, _schema = files
    apply_manifest(agent_home, "legal", "docs", [manifest])
    bad = _write(tmp_path / "bad.jsonl", [ROWS[1], row])
    with pytest.raises(InvalidRequest, match=message) as caught:
        apply_manifest(agent_home, "legal", "docs", [bad])
    assert "bad.jsonl:2" in caught.value.message and "Nothing was applied" in caught.value.message
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").citation.link == LINK  # unchanged


@pytest.mark.parametrize(("row", "message"), [
    ({"path": "tax/相続.md", "cuort": "x"}, "not declared in the schema"),
    ({"path": "tax/相続.md", "judge_date": "2024/03/01"}, "YYYY-MM-DD"),
    ({"path": "tax/相続.md", "court": 3}, "must be string"),
])
def test_the_schema_checks_every_row(agent_home, files, tmp_path, row, message):
    _manifest, schema = files
    bad = _write(tmp_path / "bad.jsonl", [row])
    with pytest.raises(InvalidRequest, match=message):
        apply_manifest(agent_home, "legal", "docs", [bad], schema=schema)


def test_problems_are_listed_together(agent_home, tmp_path):
    bad = _write(tmp_path / "bad.jsonl", [{"court": "x"}, {"path": "/a"}, ROWS[0]])
    with pytest.raises(InvalidRequest) as caught:
        apply_manifest(agent_home, "legal", "docs", [bad])
    assert caught.value.message.startswith("2 problems; nothing was applied")
    assert caught.value.details["problems"] == 2


def test_invalid_json_and_bad_schemas_are_refused(agent_home, files, tmp_path):
    manifest, _schema = files
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"path": "tax/相続.md"\n', encoding="utf-8")
    with pytest.raises(InvalidRequest, match="broken.jsonl:1"):
        apply_manifest(agent_home, "legal", "docs", [broken])
    for schema, message in (
            ({"properties": {"court": {"type": "string", "enum": ["a"]}}}, "`enum`"),
            ({"properties": {"court": {"type": "date"}}}, "`type` must be one of"),
            ({"properties": {"court": {"type": "string", "description": "あ" * 201}}},
             "the limit is 200"),
            ({"properties": {"title": {"type": "integer"}}}, "reserved"),
            ({"required": ["court"], "properties": {"court": {"type": "string"}}},
             "supported subset")):
        path = tmp_path / "schema.json"
        path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(InvalidRequest, match=message):
            apply_manifest(agent_home, "legal", "docs", [manifest], schema=path)


def test_empty_manifests_the_sink_and_unknown_sources_are_refused(agent_home, files,
                                                                  tmp_path):
    manifest, _schema = files
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(InvalidRequest, match="--clear"):
        apply_manifest(agent_home, "legal", "docs", [empty])
    with pytest.raises(InvalidRequest, match="sink"):
        apply_manifest(agent_home, "legal", "sink", [manifest])
    with pytest.raises(NotFound):
        apply_manifest(agent_home, "legal", "nope", [manifest])


# --- lifecycle ---------------------------------------------------------------------


def test_rebuild_keeps_and_source_remove_deletes_the_manifest(agent_home, files):
    manifest, schema = files
    apply_manifest(agent_home, "legal", "docs", [manifest], schema=schema)
    report = reindex_rebuild(agent_home, "legal", embedder=CharEmbedder())
    assert not report.error
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").citation.link == LINK
    remove_source(agent_home, "legal", "docs")
    store = SqliteIndex.open(agent_home.kb_dir("legal") / "index.db", readonly=True)
    assert store.manifests() == {} and store.manifest_paths("docs") == set()
    store.close()


def test_apply_before_the_first_sync(agent_home, files):
    manifest, _schema = files
    report = apply_manifest(agent_home, "later", "docs", [manifest])
    assert report.documents == 0 and report.rows_without_document == 3
    from smartrag.kb.sync import sync_kb

    sync_kb(agent_home, "later", embedder=CharEmbedder())
    r = search_kb(agent_home, "later", "相続税の評価", search_mode="keyword")
    top = next(h for h in r.results if h.citation.path == "tax/相続.md")
    assert top.title == "相続税評価事件"


# --- what agents see ---------------------------------------------------------------


def _service(home):
    return AgentService(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))


def test_agents_see_title_link_and_only_allow_listed_attributes(agent_home, files):
    manifest, schema = files
    apply_manifest(agent_home, "legal", "docs", [manifest], schema=schema)
    service = _service(agent_home)
    top = next(h for h in service.search("legal", "相続税の評価").results
               if h.citation.file_path.endswith("相続.md"))
    assert top.title == "相続税評価事件" and top.citation.link == LINK
    assert top.attributes == {}  # nothing allow-listed yet

    _allow(agent_home, ["court", "judge_date"])
    top = next(h for h in service.search("legal", "相続税の評価").results
               if h.citation.file_path.endswith("相続.md"))
    assert top.attributes == {"court": "東京地方裁判所", "judge_date": "2024-03-01"}
    read = service.read("legal", top.chunk_id)
    assert read.attributes == top.attributes and read.citation.link == LINK

    described = service.describe_sources("legal").sources[0].attributes
    assert [(a.name, a.description) for a in described] == [
        ("title", "事件名"), ("link", "判例詳細ページ"), ("court", "裁判所名"),
        ("judge_date", "裁判年月日")]
    assert described[3].format == "date"


def test_hidden_paths_hide_links_too(agent_home, files):
    manifest, _schema = files
    apply_manifest(agent_home, "legal", "docs", [manifest])
    _allow(agent_home, ["court"], show_paths=False)
    service = _service(agent_home)
    r = service.search("legal", "相続税の評価")
    assert LINK not in r.model_dump_json()
    top = next(h for h in r.results if h.title == "相続税評価事件")
    assert top.attributes == {"court": "東京地方裁判所"}
    assert [a.name for a in service.describe_sources("legal").sources[0].attributes] == [
        "title", "court"]


def test_the_allow_list_cannot_name_reserved_keys(agent_home):
    from smartrag.errors import ConfigError

    _allow(agent_home, ["link"])
    with pytest.raises(ConfigError, match="link"):
        agent_home.load("legal")


# --- CLI ---------------------------------------------------------------------------


def test_cli_source_apply(agent_home, files):
    from smartrag.cli.main import app

    manifest, schema = files
    runner = CliRunner()
    base = ["run", "--home", str(agent_home.root)]
    dry = runner.invoke(app, [*base, "source", "apply", "--kb", "legal", "--source", "docs",
                              "--manifest", str(manifest), "--schema", str(schema),
                              "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "Would apply 3 rows" in dry.stdout and "matched     2 of 3" in dry.stdout
    done = runner.invoke(app, [*base, "--format", "json", "source", "apply", "--kb", "legal",
                               "--source", "docs", "--manifest", str(manifest)])
    assert done.exit_code == 0, done.output
    assert json.loads(done.stdout)["matched"] == 2
    found = runner.invoke(app, [*base, "kb", "search", "--kb", "legal", "相続税の評価",
                                "--search-mode", "keyword"])
    assert "相続税評価事件" in found.stdout and LINK in found.stdout
    assert "court: 東京地方裁判所" in found.stdout
    both = runner.invoke(app, [*base, "source", "apply", "--kb", "legal", "--source", "docs",
                               "--manifest", str(manifest), "--clear"])
    assert both.exit_code == 2
    cleared = runner.invoke(app, [*base, "source", "apply", "--kb", "legal", "--source",
                                  "docs", "--clear"])
    assert cleared.exit_code == 0 and "removed its manifest (3 rows)" in cleared.stdout
