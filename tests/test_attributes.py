"""Attributes: `<folder>.attributes/` beside a source folder, applied by sync."""

from __future__ import annotations

import json
import shutil
import sqlite3
import unicodedata

import pytest
from typer.testing import CliRunner

from smartrag.agent import AgentService, QueryEmbedders
from smartrag.errors import ConfigError
from smartrag.index import SqliteIndex
from smartrag.kb import add_source, remove_source
from smartrag.kb.attributes import MAX_DESCRIPTION, MAX_VALUE_CHARS
from smartrag.kb.reindex import reindex_rebuild
from smartrag.kb.sources import builtin_excluded
from smartrag.kb.sync import sync_kb
from smartrag.search import SearchFilter, read_chunk, search_kb
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
        "kind": {"type": "string", "enum": ["全文", "要旨"]},
    },
}


def publish(root, rows, schema=None, *, raw=None):
    """Write `<root>.attributes/`, as a publisher would."""
    folder = root.parent / f"{root.name}.attributes"
    folder.mkdir(exist_ok=True)
    text = raw if raw is not None else "".join(
        (r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n" for r in rows)
    (folder / "attributes.jsonl").write_text(text, encoding="utf-8")
    if schema is not None:
        (folder / "schema.json").write_text(json.dumps(schema, ensure_ascii=False),
                                            encoding="utf-8")
    return folder


def _sync(home, kb="legal", **options):
    report = sync_kb(home, kb, embedder=CharEmbedder(), **options)
    assert not report.error, report.error
    return report


def _applied(report, source="docs"):
    return next(a for a in report.attributes if a.source == source)


def _hit(home, query, path):
    r = search_kb(home, "legal", query, search_mode="keyword")
    return next(h for h in r.results if h.citation.path == path)


def _generation(home):
    store = SqliteIndex.open(home.kb_dir("legal") / "index.db", readonly=True)
    try:
        return store.generation()
    finally:
        store.close()


@pytest.fixture
def docs(tmp_path):
    return tmp_path / "docs"


# --- applying ----------------------------------------------------------------------


def test_sync_applies_the_folder_and_results_carry_the_attributes(agent_home, docs):
    publish(docs, ROWS, SCHEMA)
    generation = _generation(agent_home)
    a = _applied(_sync(agent_home))
    assert (a.status, a.rows, a.has_schema, a.documents_without_row) == ("applied", 2, True, 1)
    assert a.keys == ["court", "judge_date", "kind", "link", "title"]
    assert a.skipped_lines == 1 and "no such file in the source (3)" in a.warnings[0]
    assert _generation(agent_home) > generation  # old cursors expire

    hit = _hit(agent_home, "相続税の評価", "tax/相続.md")
    assert hit.title == "相続税評価事件" and hit.citation.link == LINK
    assert hit.attributes == {"court": "東京地方裁判所", "judge_date": "2024-03-01",
                              "kind": "全文"}
    plain = _hit(agent_home, "電子申告", "tax/電子.md")
    assert plain.attributes == {} and plain.citation.link is None and plain.title == "電子申告"
    read = read_chunk(agent_home, "legal", hit.chunk_id)
    assert read.title == "相続税評価事件" and read.citation.link == LINK


def test_an_unchanged_folder_is_not_written_again(agent_home, docs):
    publish(docs, ROWS, SCHEMA)
    _sync(agent_home)
    generation = _generation(agent_home)
    assert _applied(_sync(agent_home)).status == "unchanged"
    assert _generation(agent_home) == generation


def test_a_changed_folder_replaces_and_a_removed_one_clears(agent_home, docs):
    folder = publish(docs, ROWS)
    _sync(agent_home)
    publish(docs, [{"path": "hr/規程.txt", "court": "最高裁判所"}])
    assert _applied(_sync(agent_home)).status == "applied"
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").attributes == {}
    assert _hit(agent_home, "賃金", "hr/規程.txt").attributes == {"court": "最高裁判所"}
    shutil.rmtree(folder)
    assert _applied(_sync(agent_home)).status == "removed"
    assert _hit(agent_home, "賃金", "hr/規程.txt").attributes == {}
    assert _applied(_sync(agent_home)).status == "none"


def test_an_unreadable_rows_file_keeps_the_last_good_attributes(agent_home, docs):
    folder = publish(docs, ROWS)
    _sync(agent_home)
    (folder / "attributes.jsonl").write_bytes(b'{"path": "tax/\xff"}\n')
    a = _applied(_sync(agent_home))
    assert a.status == "kept" and "cannot be read" in a.warnings[0]
    assert "kept" in a.warnings[0]
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").citation.link == LINK
    (folder / "attributes.jsonl").unlink()  # mid-rewrite: the folder is still there
    assert _applied(_sync(agent_home)).status == "kept"
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").citation.link == LINK


def test_dry_run_reports_and_writes_nothing(agent_home, docs):
    publish(docs, ROWS, SCHEMA)
    a = _applied(sync_kb(agent_home, "legal", dry_run=True))
    assert (a.status, a.rows) == ("applied", 2)
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").attributes == {}


def test_paths_match_after_nfc_normalisation(agent_home, docs):
    # Kanji have no decomposed form; a voiced kana (ガ) does.
    (docs / "tax" / "ガイド.md").write_text("# ガイド\n\n相続の手引き。\n", encoding="utf-8")
    nfd = unicodedata.normalize("NFD", "tax/ガイド.md")
    assert nfd != "tax/ガイド.md"
    publish(docs, [{"path": nfd, "court": "札幌地方裁判所"}])
    assert _applied(_sync(agent_home)).rows == 1
    assert _hit(agent_home, "手引き", "tax/ガイド.md").attributes == {"court": "札幌地方裁判所"}


def test_the_folder_is_never_indexed_as_documents(agent_home, tmp_path):
    assert builtin_excluded("docs.attributes/attributes.jsonl")
    assert builtin_excluded("a/b.attributes/schema.json")
    parent = tmp_path
    add_source(agent_home, "legal", "all", str(parent))
    publish(parent / "docs", ROWS)
    report = _sync(agent_home)
    assert not any(".attributes" in i.path for i in report.issues)
    store = SqliteIndex.open(agent_home.kb_dir("legal") / "index.db", readonly=True)
    paths = {rel for (_s, rel) in store.documents("all")}
    store.close()
    assert paths and not any(".attributes" in p for p in paths)


def test_the_sink_has_no_attributes(agent_home, docs):
    publish(docs, ROWS)
    assert [a.source for a in _sync(agent_home).attributes] == ["docs"]


# --- errors: skip the bad part ------------------------------------------------------


@pytest.mark.parametrize(("line", "kind"), [
    ('{"court": "x"}', "no `path`"),
    ('{"path": "/tax/相続.md"}', "a bad `path`"),
    ('{"path": "tax\\\\相続.md"}', "a bad `path`"),
    ('{"path": "tax/../相続.md"}', "a bad `path`"),
    ('{"path": "tax/相続.md"', "not valid JSON"),
    ('["tax/相続.md"]', "not a JSON object"),
    ('{"path": "hr/規程.txt", "court": "x"}', "a path already given"),
    ('{"path": "tax/相続.md", "n": NaN}', "not valid JSON"),
])
def test_a_bad_line_is_skipped_and_the_rest_applied(agent_home, docs, line, kind):
    publish(docs, [ROWS[1], line, ROWS[0]])
    a = _applied(_sync(agent_home))
    assert a.status == "applied" and a.skipped_lines == 1
    assert any(w.startswith(f"1 lines skipped: {kind} (2)") for w in a.warnings), a.warnings
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").citation.link == LINK
    assert _hit(agent_home, "賃金", "hr/規程.txt").attributes["court"] == "大阪地方裁判所"


@pytest.mark.parametrize(("extra", "kind"), [
    ({"court": {"name": "x"}}, "a nested value"),
    ({"tags": ["a"]}, "a nested value"),
    ({"title": ""}, "a `title` that is not a non-empty string"),
    ({"link": "file:///etc/passwd"}, "a `link` that is not an http(s) URL"),
    ({"court": "あ" * (MAX_VALUE_CHARS + 1)}, f"a value over {MAX_VALUE_CHARS} characters"),
    ({"court": 3}, "a value that is not string (schema)"),
    ({"judge_date": "2024/03/01"}, "a value that is not a date, YYYY-MM-DD (schema)"),
    ({"kind": "抄録"}, "a value not in the schema's `enum`"),
    ({"_attr_court": "x"}, "a name with the reserved prefix `_attr_`"),
])
def test_a_bad_value_is_dropped_and_the_rest_of_its_row_kept(agent_home, docs, extra, kind):
    row = {"path": "tax/相続.md", "court": "東京地方裁判所", "judge_date": "2024-03-01",
           "kind": "全文", **extra}
    publish(docs, [row], SCHEMA)
    a = _applied(_sync(agent_home))
    assert a.dropped_values == 1
    assert any(w.startswith(f"1 values dropped: {kind} (line 1 `") for w in a.warnings), \
        a.warnings
    kept = _hit(agent_home, "相続税の評価", "tax/相続.md").attributes
    dropped = next(iter(extra))
    assert dropped not in kept
    assert {"court", "judge_date", "kind"} - {dropped} <= set(kept)


def test_problems_are_counted_with_their_first_lines(agent_home, docs):
    publish(docs, ["{", "{", "{", "{", ROWS[0]])
    a = _applied(_sync(agent_home))
    assert "4 lines skipped: not valid JSON (1, 2, 3, …)" in a.warnings


# --- the schema --------------------------------------------------------------------


def test_undeclared_keys_fall_back_to_strings(agent_home, docs):
    publish(docs, [{"path": "tax/相続.md", "court": "東京地方裁判所", "cuort": "typo",
                    "sec_code": 7203, "listed": True, "ratio": 1.5, "none": None}],
            {"properties": {"court": {"type": "string"}}})
    a = _applied(_sync(agent_home))
    assert a.untyped == ["cuort", "listed", "none", "ratio", "sec_code"]
    assert "not in the schema, read as strings: cuort, listed, none, ratio, sec_code" \
        in a.warnings
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").attributes == {
        "court": "東京地方裁判所", "cuort": "typo", "sec_code": "7203", "listed": "true",
        "ratio": "1.5", "none": None}
    found = search_kb(agent_home, "legal", "資料", embedder=CharEmbedder(),
                      filters=SearchFilter(attributes={"sec_code": 7203}))
    assert {h.citation.path for h in found.results} == {"tax/相続.md"}


def test_without_a_schema_every_key_is_a_string(agent_home, docs):
    publish(docs, ROWS)
    a = _applied(_sync(agent_home))
    assert not a.has_schema and a.untyped == ["court", "judge_date", "kind"]
    assert not any("not in the schema" in w for w in a.warnings)


def test_a_broken_schema_is_ignored_on_its_own(agent_home, docs):
    folder = publish(docs, ROWS)
    (folder / "schema.json").write_text("{not json", encoding="utf-8")
    a = _applied(_sync(agent_home))
    assert a.status == "applied" and not a.has_schema
    assert "schema.json cannot be read" in a.warnings[0]
    assert _hit(agent_home, "賃金", "hr/規程.txt").attributes["court"] == "大阪地方裁判所"


def test_the_schema_is_read_leniently(agent_home, docs):
    publish(docs, [{"path": "tax/相続.md", "court": "x", "kind": "全文", "n": 3}], {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "required": ["court"], "additionalProperties": False,
        "properties": {
            "court": {"type": "string", "description": "あ" * (MAX_DESCRIPTION + 1),
                      "minLength": 1},
            "kind": {"type": "date"},
            "title": {"type": "integer"},
            "n": {"type": "integer", "format": "date"},
        }})
    a = _applied(_sync(agent_home))
    assert a.has_schema and a.untyped == ["kind"]
    joined = "\n".join(a.warnings)
    assert f"cut to {MAX_DESCRIPTION}" in joined
    assert "schema.json: `kind` has no `type`" in joined
    assert "schema.json: `title` is reserved" in joined
    assert "schema.json: `n`: `format` 'date' ignored" in joined
    assert "required" not in joined and "minLength" not in joined
    described = AgentService(agent_home, embedders=QueryEmbedders(
        lambda _config: CharEmbedder())).describe_sources("legal").sources[0].attributes
    court = next(x for x in described if x.name == "court")
    assert len(court.description) == MAX_DESCRIPTION and court.description.endswith("…")


def test_names_smart_rag_uses_are_escaped(agent_home, docs):
    publish(docs, [{"path": "tax/相続.md", "doc_id": "S100U9OX", "court": "x"}],
            {"properties": {"doc_id": {"type": "string", "description": "書類管理番号"}}})
    a = _applied(_sync(agent_home))
    assert a.renamed == {"doc_id": "_attr_doc_id"}
    hit = _hit(agent_home, "相続税の評価", "tax/相続.md")
    assert hit.attributes["_attr_doc_id"] == "S100U9OX" and hit.doc_id != "S100U9OX"
    found = search_kb(agent_home, "legal", "資料", embedder=CharEmbedder(),
                      filters=SearchFilter(attributes={"_attr_doc_id": "S100U9OX"}))
    assert {h.citation.path for h in found.results} == {"tax/相続.md"}


# --- lifecycle ---------------------------------------------------------------------


def test_rebuild_applies_the_folder_and_source_remove_deletes(agent_home, docs):
    publish(docs, ROWS, SCHEMA)
    _sync(agent_home)
    report = reindex_rebuild(agent_home, "legal", embedder=CharEmbedder())
    assert not report.error and _applied(report).status == "applied"
    assert _hit(agent_home, "相続税の評価", "tax/相続.md").citation.link == LINK
    remove_source(agent_home, "legal", "docs")
    store = SqliteIndex.open(agent_home.kb_dir("legal") / "index.db", readonly=True)
    assert store.attribute_sets() == {}
    store.close()


def test_an_index_with_manifest_tables_drops_them_on_the_next_write(agent_home):
    path = agent_home.kb_dir("legal") / "index.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE manifests (source_id TEXT)")
        db.execute("CREATE TABLE manifest_rows (source_id TEXT)")
    _sync(agent_home)
    with sqlite3.connect(path) as db:
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    assert not {"manifests", "manifest_rows"} & names


def test_a_config_with_the_old_allow_list_is_refused_clearly(agent_home):
    config = agent_home.kb_dir("legal") / "config.yaml"
    text = config.read_text(encoding="utf-8")
    config.write_text(text.replace("show_paths: true", "show_paths: true\n  attributes: []"),
                      encoding="utf-8")
    with pytest.raises(ConfigError, match="agent.attributes is no longer used"):
        agent_home.load("legal")


# --- what agents see ---------------------------------------------------------------


def _service(home):
    return AgentService(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))


def test_agents_see_every_attribute(agent_home, docs):
    publish(docs, ROWS, SCHEMA)
    _sync(agent_home)
    service = _service(agent_home)
    top = next(h for h in service.search("legal", "相続税の評価").results
               if h.citation.file_path.endswith("相続.md"))
    assert top.title == "相続税評価事件" and top.citation.link == LINK
    assert top.attributes == {"court": "東京地方裁判所", "judge_date": "2024-03-01",
                              "kind": "全文"}
    read = service.read("legal", top.chunk_id)
    assert read.attributes == top.attributes and read.citation.link == LINK
    described = service.describe_sources("legal").sources[0].attributes
    assert [(a.name, a.description) for a in described] == [
        ("title", "事件名"), ("link", "判例詳細ページ"), ("court", "裁判所名"),
        ("judge_date", "裁判年月日"), ("kind", None)]
    assert described[3].format == "date"


def test_hidden_paths_hide_file_paths_but_not_links(agent_home, docs):
    publish(docs, ROWS)
    _sync(agent_home)
    config = agent_home.kb_dir("legal") / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace(
        "show_paths: true", "show_paths: false"), encoding="utf-8")
    r = _service(agent_home).search("legal", "相続税の評価")
    top = next(h for h in r.results if h.title == "相続税評価事件")
    assert top.citation.file_path is None and top.citation.link == LINK
    assert str(docs) not in r.model_dump_json()


# --- CLI ---------------------------------------------------------------------------


def test_cli_sync_reports_the_attributes(agent_home, docs):
    from smartrag.cli.main import app

    publish(docs, [*ROWS, {"path": "tax/電子.md", "doc_id": "X", "court": {"a": 1}}], SCHEMA)
    runner = CliRunner()
    base = ["run", "--home", str(agent_home.root)]
    dry = runner.invoke(app, [*base, "kb", "sync", "--kb", "legal", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "attributes docs: 3 rows would be applied" in dry.stdout
    assert "renamed doc_id to _attr_doc_id" in dry.stdout
    assert "1 lines skipped: no such file in the source (3)" in dry.stdout
    assert "1 values dropped: a nested value (line 4 `court`)" in dry.stdout
    gone = runner.invoke(app, [*base, "source", "apply", "--kb", "legal", "--source", "docs"])
    assert gone.exit_code == 2  # the command no longer exists


def test_kb_describe_shows_each_source_and_its_attributes(agent_home, docs):
    from smartrag.cli.main import app
    from smartrag.search.describe import describe_kb

    publish(docs, ROWS, SCHEMA)
    _sync(agent_home)
    described = describe_kb(agent_home, "legal")
    source = next(s for s in described.sources if s.source == "docs")
    assert source.documents == 3 and source.file_types == {"md": 2, "txt": 1}
    assert source.attributes is not None and source.attributes.rows == 2
    assert source.attributes.folder == str(docs.parent / "docs.attributes")
    keys = {k.name: k for k in source.attributes.keys}
    assert list(keys) == ["title", "link", "court", "judge_date", "kind"]
    assert (keys["court"].description, keys["court"].values) == (
        "裁判所名", ["大阪地方裁判所", "東京地方裁判所"])
    assert keys["judge_date"].format == "date" and keys["judge_date"].values is None
    assert keys["kind"].enum == ["全文", "要旨"]

    runner = CliRunner()
    base = ["run", "--home", str(agent_home.root)]
    text = runner.invoke(app, [*base, "kb", "describe", "--kb", "legal"])
    assert text.exit_code == 0, text.output
    assert "source docs" in text.stdout and "attributes  2 rows from" in text.stdout
    assert f"path       {agent_home.kb_dir('legal')}" in text.stdout
    assert "values: 大阪地方裁判所, 東京地方裁判所" in text.stdout
    assert "judge_date  date" in text.stdout
    data = runner.invoke(app, [*base, "--format", "json", "kb", "describe", "--kb", "legal"])
    assert json.loads(data.stdout)["status"]["kb"] == "legal"

    shutil.rmtree(docs.parent / "docs.attributes")
    _sync(agent_home)
    bare = runner.invoke(app, [*base, "kb", "describe", "--kb", "legal"])
    assert "attributes  none (no attributes folder at the last sync)" in bare.stdout
