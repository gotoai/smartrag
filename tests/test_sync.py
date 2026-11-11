"""`kb sync`: change detection, the deletion sweep, vector reuse, and the CLI."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3

import pytest
from typer.testing import CliRunner

from smartrag.errors import ConfigError
from smartrag.index import SqliteIndex
from smartrag.kb import Home, add_source, remove_source
from smartrag.kb.sync import sync_kb


class FakeEmbedder:
    def __init__(self, profile: str = "ruri-v3-310m@8k", dims: int = 8) -> None:
        self.profile = profile
        self.dims = dims
        self.calls: list[int] = []

    def check(self) -> int:
        return self.dims

    def embed(self, texts, input_type):
        assert input_type == "document"
        self.calls.append(len(texts))
        return [[b / 255 for b in hashlib.sha256(t.encode()).digest()[:self.dims]]
                for t in texts]


@pytest.fixture
def home(tmp_path):
    home = Home(tmp_path / "home")
    home.new("legal")
    return home


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "規程.txt").write_text("第3章 休暇\n\n" + "社員は年休を取得できる。" * 20 + "\n",
                                   encoding="utf-8")
    (root / "notes.md").write_text("# Notes\n\nShort note about holidays.\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    return root


def _store(home):
    return SqliteIndex.open(home.root / "legal" / "index.db", readonly=True)


def test_first_sync_indexes_and_records_skips(home, docs):
    add_source(home, "legal", "docs", str(docs))
    embedder = FakeEmbedder()
    report = sync_kb(home, "legal", embedder=embedder)
    assert report.counts.added == 2 and report.counts.skipped == 1
    assert report.issues[0].path == "image.png" and "unsupported" in report.issues[0].reason
    assert report.chunks_embedded == report.chunks_written > 0
    stats = _store(home).stats()
    assert (stats.ok, stats.skipped, stats.profile, stats.dimensions) == \
        (2, 1, "ruri-v3-310m@8k", 8)
    assert report.report_path
    with open(report.report_path, encoding="utf-8") as handle:
        assert json.load(handle)["kb"] == "legal"


def test_unchanged_files_are_not_reprocessed_or_retried(home, docs):
    add_source(home, "legal", "docs", str(docs))
    sync_kb(home, "legal", embedder=FakeEmbedder())
    embedder = FakeEmbedder()
    again = sync_kb(home, "legal", embedder=embedder)
    assert again.counts.unchanged == 3 and again.counts.skipped == 0
    assert embedder.calls == []


def test_a_touched_file_with_the_same_bytes_is_only_restamped(home, docs):
    add_source(home, "legal", "docs", str(docs))
    sync_kb(home, "legal", embedder=FakeEmbedder())
    stat = (docs / "notes.md").stat()
    os.utime(docs / "notes.md", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    embedder = FakeEmbedder()
    report = sync_kb(home, "legal", embedder=embedder)
    assert report.counts.unchanged == 3 and embedder.calls == []


def test_an_edit_reembeds_only_changed_chunks(home, docs):
    add_source(home, "legal", "docs", str(docs))
    first = sync_kb(home, "legal", embedder=FakeEmbedder())
    with open(docs / "notes.md", "a", encoding="utf-8") as handle:
        handle.write("\n## Added\n\nA new paragraph.\n")
    embedder = FakeEmbedder()
    report = sync_kb(home, "legal", embedder=embedder)
    assert report.counts.updated == 1 and report.counts.unchanged == 2
    assert 0 < report.chunks_embedded < first.chunks_written + 1
    assert sum(embedder.calls) == report.chunks_embedded


def test_deleted_files_are_swept(home, docs):
    add_source(home, "legal", "docs", str(docs))
    sync_kb(home, "legal", embedder=FakeEmbedder())
    (docs / "notes.md").unlink()
    report = sync_kb(home, "legal", embedder=FakeEmbedder())
    assert report.counts.removed == 1 and _store(home).stats().ok == 1


def test_an_unavailable_source_removes_nothing(home, docs, tmp_path):
    add_source(home, "legal", "docs", str(docs))
    sync_kb(home, "legal", embedder=FakeEmbedder())
    docs.rename(tmp_path / "unmounted")
    report = sync_kb(home, "legal", embedder=FakeEmbedder())
    assert report.counts.removed == 0
    source = next(s for s in report.sources if s.source == "docs")
    assert not source.complete and "kept" in (source.note or "")
    assert _store(home).stats().ok == 2


def test_dry_run_writes_nothing(home, docs):
    add_source(home, "legal", "docs", str(docs))
    report = sync_kb(home, "legal", dry_run=True)
    assert report.dry_run and report.counts.added == 2 and report.tokens_estimated > 0
    assert not (home.root / "legal" / "index.db").exists()
    assert list((home.root / "legal" / "runs").iterdir()) == []


def test_profile_mismatch_stops_the_sync(home, docs):
    add_source(home, "legal", "docs", str(docs))
    sync_kb(home, "legal", embedder=FakeEmbedder())
    path = home.root / "legal" / "config.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("ruri-v3-310m@8k", "other@1"),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot be mixed"):
        sync_kb(home, "legal", embedder=FakeEmbedder(profile="other@1"))


def test_removing_a_source_removes_its_documents(home, docs):
    add_source(home, "legal", "docs", str(docs))
    sync_kb(home, "legal", embedder=FakeEmbedder())
    removed = remove_source(home, "legal", "docs")
    assert removed.documents_removed == 3
    db = sqlite3.connect(home.root / "legal" / "index.db")
    assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM contents").fetchone()[0] == 0


def test_identical_files_share_vectors(home, tmp_path):
    root = tmp_path / "dup"
    root.mkdir()
    for name in ("a.txt", "b.txt"):
        (root / name).write_text("同じ内容の文書です。" * 30, encoding="utf-8")
    add_source(home, "legal", "dup", str(root))
    embedder = FakeEmbedder()
    report = sync_kb(home, "legal", embedder=embedder)
    assert report.counts.added == 2 and report.chunks_reused == report.chunks_written // 2


def test_keyword_search_over_the_index(home, docs):
    from smartrag.index import fts_query

    add_source(home, "legal", "docs", str(docs))
    sync_kb(home, "legal", embedder=FakeEmbedder())
    db = sqlite3.connect(home.root / "legal" / "index.db")
    hits = db.execute("SELECT rowid FROM chunk_fts WHERE chunk_fts MATCH ?",
                      (fts_query("年休"),)).fetchall()
    assert hits


def test_cli_dry_run_json_lines_and_exit_codes(home, docs, monkeypatch):
    from smartrag.cli.main import app

    add_source(home, "legal", "docs", str(docs))
    result = CliRunner().invoke(app, ["run", "--home", str(home.root), "--format", "json",
                                      "kb", "sync", "--kb", "legal", "--dry-run"])
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert lines[0]["event"] == "source"
    assert any(e.get("event") == "document" for e in lines)
    assert lines[-1]["counts"]["added"] == 2
    assert result.exit_code == 3  # image.png was skipped


def test_a_new_pipeline_version_never_reads_stale_text(home, tmp_path, monkeypatch):
    """Regression: extracted text was keyed by the file's bytes alone, so re-extracting
    an unchanged file with a newer extractor kept the old text under the new chunks."""
    import smartrag.kb.sync as sync_module
    from smartrag.extract.units import Extracted, Unit
    from smartrag.search import read_chunk

    root = tmp_path / "one"
    root.mkdir()
    (root / "a.txt").write_text("placeholder", encoding="utf-8")
    add_source(home, "legal", "one", str(root))
    monkeypatch.setattr(sync_module, "extract",
                        lambda path: Extracted(units=[Unit("old wording of the text")]))
    sync_kb(home, "legal", embedder=FakeEmbedder())

    monkeypatch.setattr(sync_module, "extract",
                        lambda path: Extracted(units=[Unit("brand new wording, longer than before")]))
    monkeypatch.setattr(sync_module, "pipeline_version", lambda config: "v-next")
    report = sync_kb(home, "legal", embedder=FakeEmbedder())
    assert report.counts.updated == 1

    db = sqlite3.connect(home.root / "legal" / "index.db")
    chunk_id = db.execute("SELECT chunk_id FROM chunks").fetchone()[0]
    read = read_chunk(home, "legal", chunk_id)
    assert read.chunks[0].text == "brand new wording, longer than before"
