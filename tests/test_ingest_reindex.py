"""`kb ingest` and `kb reindex`."""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from smartrag.errors import ConfigError, CursorExpired, EmbedderUnavailable, InvalidRequest
from smartrag.index import SqliteIndex
from smartrag.kb import Home, add_source
from smartrag.kb.ingest import ingest_files
from smartrag.kb.reindex import reindex_rebuild, reindex_stale
from smartrag.kb.sync import sync_kb
from smartrag.search import search_kb


class Embedder:
    def __init__(self, profile: str = "ruri-v3-310m@8k", fail: bool = False) -> None:
        self.profile = profile
        self.fail = fail
        self.embedded = 0

    def check(self) -> int:
        return 4

    def embed(self, texts, input_type):
        if self.fail:
            raise EmbedderUnavailable("endpoint down")
        self.embedded += len(texts)
        return [[float(len(t) % 7 + 1), float(t.count("。") + 1), 1.0, 0.5] for t in texts]


@pytest.fixture
def home(tmp_path):
    home = Home(tmp_path / "home")
    home.new("legal")
    return home


@pytest.fixture
def originals(tmp_path):
    root = tmp_path / "営業見積"
    (root / "2026").mkdir(parents=True)
    (root / "2026" / "見積書.md").write_text("# 見積\n\nA社向けの見積書です。単価は千円。\n",
                                           encoding="utf-8")
    (root / "メモ.txt").write_text("会議のメモ。来週に再提出する。\n", encoding="utf-8")
    (root / "画像.png").write_bytes(b"\x89PNG")
    (root / "~$見積書.md").write_text("lock", encoding="utf-8")
    return root


def _documents(home):
    db = sqlite3.connect(home.root / "legal" / "index.db")
    return db.execute("SELECT source_id, rel_path, origin_path, status FROM documents "
                      "ORDER BY rel_path").fetchall()


# --- ingest ------------------------------------------------------------------------


def test_ingest_copies_records_the_origin_and_indexes(home, originals):
    report = ingest_files(home, "legal", [str(originals / "2026" / "見積書.md")],
                          embedder=Embedder())
    assert report.copied == 1 and report.sync is not None
    assert report.sync.counts.added == 1 and report.sync.operation == "ingest"
    item = report.files[0]
    folder, name = item.sink_path.split("/")
    sink = home.root / "legal" / "sink"
    assert (sink / folder / name).read_text(encoding="utf-8").startswith("# 見積")
    sidecar = json.loads((sink / folder / ".origin.json").read_text(encoding="utf-8"))
    assert sidecar["origin_path"] == str(originals / "2026" / "見積書.md")
    (row,) = _documents(home)
    assert row[0] == "sink" and row[1] == f"{folder}/{name}" and row[2] == sidecar["origin_path"]

    hit = search_kb(home, "legal", "単価", search_mode="keyword").results[0]
    assert hit.citation.origin_path == sidecar["origin_path"]
    by_folder = search_kb(home, "legal", "営業見積", search_mode="keyword")
    assert by_folder.results, "the original folder name is searchable"


def test_same_content_twice_is_a_no_op(home, originals, tmp_path):
    first = originals / "メモ.txt"
    copy = tmp_path / "copy-of-memo.txt"
    copy.write_bytes(first.read_bytes())
    report = ingest_files(home, "legal", [str(first), str(copy)], embedder=Embedder())
    assert report.copied == 1 and report.already == 1
    again = ingest_files(home, "legal", [str(first)], embedder=Embedder())
    assert again.copied == 0 and again.already == 1
    assert len(_documents(home)) == 1


def test_folders_need_recursive_and_skip_what_cannot_be_indexed(home, originals, tmp_path):
    with pytest.raises(InvalidRequest, match="--recursive"):
        ingest_files(home, "legal", [str(originals)], embedder=Embedder())
    assert not any((home.root / "legal" / "sink").iterdir())
    report = ingest_files(home, "legal", [str(originals), str(tmp_path / "missing.md")],
                          recursive=True, embedder=Embedder())
    statuses = {item.path.rsplit("/", 1)[-1]: item.status for item in report.files}
    assert statuses == {"見積書.md": "copied", "メモ.txt": "copied", "画像.png": "skipped",
                        "missing.md": "skipped"}
    assert report.sync is not None and report.sync.counts.skipped == 0
    assert {r[3] for r in _documents(home)} == {"ok"}


def test_dry_run_copies_nothing(home, originals):
    report = ingest_files(home, "legal", [str(originals)], recursive=True, dry_run=True)
    assert {i.status for i in report.files} == {"would_copy", "skipped"}
    assert report.tokens_estimated > 0 and report.sync is None
    assert not any((home.root / "legal" / "sink").iterdir())


def test_ingested_copies_are_snapshots(home, originals):
    source = originals / "メモ.txt"
    ingest_files(home, "legal", [str(source)], embedder=Embedder())
    source.write_text("書き換えた内容。\n", encoding="utf-8")
    report = sync_kb(home, "legal", embedder=Embedder())
    assert report.counts.unchanged == 1 and report.counts.updated == 0


def test_ingest_refuses_the_kb_itself_and_remote_kbs(home, originals):
    with pytest.raises(InvalidRequest, match="inside the kb"):
        ingest_files(home, "legal", [str(home.root / "legal" / "sink")], recursive=True)
    home.new("factory", remote="https://rag.example.jp", token_env="T")
    with pytest.raises(InvalidRequest, match="remote"):
        ingest_files(home, "factory", [str(originals / "メモ.txt")])


# --- reindex --refresh-stale-only ---------------------------------------------------------------


@pytest.fixture
def synced(home, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    for n in range(3):
        (root / f"doc{n}.txt").write_text(f"文書{n}の内容です。" * 20, encoding="utf-8")
    add_source(home, "legal", "docs", str(root))
    sync_kb(home, "legal", embedder=Embedder())
    return root


def test_stale_reprocesses_only_old_documents(home, synced, monkeypatch):
    import smartrag.kb.sync as sync_module

    nothing = reindex_stale(home, "legal", embedder=Embedder())
    assert nothing.counts.unchanged == 3 and nothing.counts.updated == 0

    (synced / "new.txt").write_text("新しい文書。", encoding="utf-8")
    (synced / "doc0.txt").unlink()
    monkeypatch.setattr(sync_module, "pipeline_version", lambda config: "next")
    report = reindex_stale(home, "legal", embedder=Embedder())
    assert report.operation == "reindex-stale"
    assert report.counts.updated == 2 and report.counts.deferred == 1
    assert report.counts.removed == 0 and report.counts.added == 0
    assert len(_documents(home)) == 3  # doc0 is kept until the next sync


# --- reindex --rebuild -------------------------------------------------------------


def test_rebuild_reuses_vectors_and_continues_the_generation(home, synced):
    before = search_kb(home, "legal", "文書 内容", search_mode="keyword", limit=1)
    store = SqliteIndex.open(home.root / "legal" / "index.db", readonly=True)
    old_generation = store.generation()
    store.close()

    embedder = Embedder()
    report = reindex_rebuild(home, "legal", embedder=embedder)
    assert report.operation == "reindex-rebuild" and report.error is None
    assert report.counts.added == 3 and embedder.embedded == 0
    assert report.chunks_reused == report.chunks_written

    store = SqliteIndex.open(home.root / "legal" / "index.db", readonly=True)
    assert store.generation() > old_generation
    store.close()
    assert not list((home.root / "legal").glob("index.db.rebuild*"))
    with pytest.raises(CursorExpired):
        search_kb(home, "legal", "文書 内容", search_mode="keyword", limit=1, cursor=before.next_cursor)
    assert search_kb(home, "legal", "文書", embedder=Embedder()).results


def test_rebuild_is_how_a_kb_changes_its_model(home, synced):
    path = home.root / "legal" / "config.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("ruri-v3-310m@8k", "other@2"),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="reindex --kb <id> --rebuild"):
        sync_kb(home, "legal", embedder=Embedder("other@2"))
    embedder = Embedder("other@2")
    report = reindex_rebuild(home, "legal", embedder=embedder)
    assert embedder.embedded == report.chunks_written > 0
    store = SqliteIndex.open(home.root / "legal" / "index.db", readonly=True)
    assert store.meta("embedding_profile") == "other@2"
    store.close()


def test_a_failed_rebuild_keeps_the_old_index(home, synced, tmp_path):
    (synced / "more.txt").write_text("追加の文書。" * 10, encoding="utf-8")
    report = reindex_rebuild(home, "legal", embedder=Embedder(fail=True))
    assert report.error and "old index was kept" in report.error
    assert len(_documents(home)) == 3
    assert not list((home.root / "legal").glob("index.db.rebuild*"))


def test_rebuild_dry_run_changes_nothing(home, synced):
    stamp = (home.root / "legal" / "index.db").stat().st_mtime_ns
    report = reindex_rebuild(home, "legal", dry_run=True)
    assert report.dry_run and report.counts.added == 3
    assert (home.root / "legal" / "index.db").stat().st_mtime_ns == stamp


# --- CLI ---------------------------------------------------------------------------


def test_cli_rules(home, synced, originals):
    from smartrag.cli.main import app

    runner = CliRunner()
    base = ["run", "--home", str(home.root)]
    neither = runner.invoke(app, [*base, "kb", "reindex", "--kb", "legal"])
    both = runner.invoke(app, [*base, "kb", "reindex", "--kb", "legal", "--refresh-stale-only", "--rebuild"])
    assert neither.exit_code == 2 and both.exit_code == 2
    unconfirmed = runner.invoke(app, [*base, "kb", "reindex", "--kb", "legal", "--rebuild"])
    assert unconfirmed.exit_code == 1 and "--yes" in unconfirmed.stderr
    dry = runner.invoke(app, [*base, "--format", "json", "kb", "ingest", "--kb", "legal",
                              str(originals), "-r", "--dry-run"])
    assert dry.exit_code == 3  # 画像.png would be skipped
    assert json.loads(dry.stdout.splitlines()[-1])["dry_run"] is True
