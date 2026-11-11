"""`kb search` and `kb read` over a small synced kb."""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from smartrag.errors import (
    ChunkSuperseded,
    ConfigError,
    CursorExpired,
    EmbedderUnavailable,
    IndexNotReady,
    InvalidRequest,
    NotFound,
)
from smartrag.kb import Home, add_source
from smartrag.kb.sync import sync_kb
from smartrag.search import SearchFilter, read_chunk, search_kb

ALPHABET = "相続税評価更正処分電子申告義務消費年休規程賃金"


class CharEmbedder:
    """Vectors from character counts: texts sharing words with the query score higher."""

    profile = "ruri-v3-310m@8k"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def check(self) -> int:
        return len(ALPHABET) + 1

    def embed(self, texts, input_type):
        self.calls.append(input_type)
        if self.fail:
            raise EmbedderUnavailable("endpoint down")
        return [[float(t.count(ch)) for ch in ALPHABET] + [1.0] for t in texts]


@pytest.fixture
def home(tmp_path):
    home = Home(tmp_path / "home")
    home.new("legal")
    root = tmp_path / "docs"
    (root / "tax").mkdir(parents=True)
    (root / "hr").mkdir()
    (root / "tax" / "相続.md").write_text(
        "# 相続税\n\n相続税の評価について。土地の評価は路線価による。\n\n"
        "## 更正\n\n更正処分の取消しを求める訴え。\n", encoding="utf-8")
    (root / "tax" / "電子.md").write_text(
        "# 電子申告\n\n電子申告の義務化は特定法人に適用される。\n\n消費税の申告も電子で行う。\n",
        encoding="utf-8")
    (root / "hr" / "規程.txt").write_text(
        "第3章 休暇\n\n年休は10日。\n\n第4章 賃金\n\n賃金は毎月25日に支払う。\n", encoding="utf-8")
    add_source(home, "legal", "docs", str(root))
    sync_kb(home, "legal", embedder=CharEmbedder())
    return home


def _titles(response):
    return [h.citation.path for h in response.results]


# --- search ------------------------------------------------------------------------


def test_hybrid_search_ranks_and_cites(home):
    embedder = CharEmbedder()
    r = search_kb(home, "legal", "相続税の評価", embedder=embedder)
    assert r.mode == "hybrid" and r.degraded is None and embedder.calls == ["query"]
    top = r.results[0]
    assert top.citation.path == "tax/相続.md" and top.rank == 1
    assert top.ranks.lexical is not None and top.ranks.vector is not None
    assert top.citation.heading_path == ["相続税"]
    assert top.indexed_at and top.metadata == {"file_type": "md"}
    assert "answer" not in r.model_dump()


def test_lexical_mode_makes_no_embedding_call(home):
    embedder = CharEmbedder()
    r = search_kb(home, "legal", '"更正処分"', mode="lexical", embedder=embedder)
    assert embedder.calls == [] and r.mode == "lexical"
    assert r.results and all(h.ranks.vector is None for h in r.results)
    assert "更正処分" in r.results[0].snippet


def test_endpoint_down_degrades_hybrid_and_fails_vector(home):
    r = search_kb(home, "legal", "賃金", embedder=CharEmbedder(fail=True))
    assert r.mode == "lexical" and "endpoint down" in (r.degraded or "")
    assert r.results[0].citation.path == "hr/規程.txt"
    with pytest.raises(EmbedderUnavailable):
        search_kb(home, "legal", "賃金", mode="vector", embedder=CharEmbedder(fail=True))


def test_query_vectors_are_cached(home):
    first = CharEmbedder()
    search_kb(home, "legal", "電子申告", embedder=first)
    second = CharEmbedder()
    search_kb(home, "legal", "電子申告", embedder=second)
    assert first.calls == ["query"] and second.calls == []


def test_filters(home):
    by_type = search_kb(home, "legal", "申告", filters=SearchFilter(file_type=["txt"]),
                        embedder=CharEmbedder())
    assert set(_titles(by_type)) <= {"hr/規程.txt"}
    by_path = search_kb(home, "legal", "相続税", filters=SearchFilter(path_prefix="hr/"),
                        mode="lexical")
    assert by_path.results == []
    by_source = search_kb(home, "legal", "相続税", filters=SearchFilter(source=["sink"]),
                          embedder=CharEmbedder())
    assert by_source.results == []


def test_grouping_and_per_document_cap(home):
    chunks = search_kb(home, "legal", "相続税 評価 更正処分 電子申告", max_per_doc=1, limit=10,
                       embedder=CharEmbedder())
    paths = _titles(chunks)
    assert len(paths) == len(set(paths))
    docs = search_kb(home, "legal", "相続税 評価 更正処分 電子申告", group_by="document",
                     embedder=CharEmbedder())
    assert len(_titles(docs)) == len(set(_titles(docs)))


def test_paging_with_pages_and_cursors(home):
    full = search_kb(home, "legal", "申告 相続税 賃金 年休", limit=50, embedder=CharEmbedder())
    first = search_kb(home, "legal", "申告 相続税 賃金 年休", limit=2, embedder=CharEmbedder())
    assert first.next_cursor and first.page == 1
    by_cursor = search_kb(home, "legal", "申告 相続税 賃金 年休", limit=2,
                          cursor=first.next_cursor, embedder=CharEmbedder())
    by_page = search_kb(home, "legal", "申告 相続税 賃金 年休", limit=2, page=2,
                        embedder=CharEmbedder())
    ids = [h.chunk_id for h in full.results]
    assert [h.chunk_id for h in by_cursor.results] == ids[2:4]
    assert [h.chunk_id for h in by_page.results] == ids[2:4] and by_page.results[0].rank == 3
    with pytest.raises(InvalidRequest, match="different request"):
        search_kb(home, "legal", "別の質問", cursor=first.next_cursor, embedder=CharEmbedder())
    with pytest.raises(InvalidRequest, match="depth"):
        search_kb(home, "legal", "申告", page=21, limit=10, embedder=CharEmbedder())


def test_a_sync_expires_open_cursors(home, tmp_path):
    first = search_kb(home, "legal", "申告 相続税 賃金", limit=1, embedder=CharEmbedder())
    (tmp_path / "docs" / "hr" / "追加.txt").write_text("新しい文書。", encoding="utf-8")
    sync_kb(home, "legal", embedder=CharEmbedder())
    with pytest.raises(CursorExpired):
        search_kb(home, "legal", "申告 相続税 賃金", limit=1, cursor=first.next_cursor,
                  embedder=CharEmbedder())


def test_highlights_and_text(home):
    r = search_kb(home, "legal", "更正処分", include=["highlights", "text"], mode="lexical")
    hit = r.results[0]
    assert hit.highlights and all(hit.snippet[a:b] in ("更正処分", "更正", "正処", "処分")
                                  for a, b in hit.highlights)
    assert hit.text and "更正処分" in hit.text


def test_search_errors(home, tmp_path):
    home.new("empty")
    with pytest.raises(IndexNotReady):
        search_kb(home, "empty", "x")
    home.new("factory", remote="https://rag.example.jp", token_env="T")
    with pytest.raises(InvalidRequest, match="remote"):
        search_kb(home, "factory", "x")
    with pytest.raises(InvalidRequest):
        search_kb(home, "legal", "   ")
    path = home.root / "legal" / "config.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("ruri-v3-310m@8k", "other@1"),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="other@1"):
        search_kb(home, "legal", "相続税", embedder=CharEmbedder())


# --- read --------------------------------------------------------------------------


def _chunk_id(home, path_like: str, ordinal: int = 0) -> str:
    db = sqlite3.connect(home.root / "legal" / "index.db")
    return db.execute("SELECT c.chunk_id FROM chunks c JOIN documents d USING (doc_id) "
                      "WHERE d.rel_path = ? AND c.ordinal = ?", (path_like, ordinal)).fetchone()[0]


def test_read_scopes(home):
    anchor = _chunk_id(home, "hr/規程.txt", 0)
    one = read_chunk(home, "legal", anchor)
    assert one.scope == "chunk" and [c.chunk_id for c in one.chunks] == [anchor]
    assert one.citation.path == "hr/規程.txt"
    document = read_chunk(home, "legal", anchor, scope="document")
    assert document.total_chunks >= len(one.chunks)
    section = read_chunk(home, "legal", anchor, scope="section")
    assert all(c.heading_path == one.chunks[0].heading_path for c in section.chunks)
    neighbours = read_chunk(home, "legal", anchor, scope="neighbors", window=1)
    assert neighbours.chunks[0].chunk_id == anchor


def test_read_pages_by_characters(home, tmp_path):
    root = tmp_path / "docs" / "long"
    root.mkdir()
    (root / "long.md").write_text("\n\n".join(f"## 節{i}\n\n" + "本文です。" * 60
                                              for i in range(6)), encoding="utf-8")
    sync_kb(home, "legal", embedder=CharEmbedder())
    anchor = _chunk_id(home, "long/long.md", 0)
    first = read_chunk(home, "legal", anchor, scope="document", max_chars=400)
    assert first.next_cursor and first.page == 1
    second = read_chunk(home, "legal", anchor, scope="document", max_chars=400,
                        cursor=first.next_cursor)
    assert second.page == 2
    assert {c.ordinal for c in first.chunks}.isdisjoint({c.ordinal for c in second.chunks})


def test_read_errors(home, tmp_path):
    with pytest.raises(NotFound):
        read_chunk(home, "legal", "nope:123")
    anchor = _chunk_id(home, "hr/規程.txt", 0)
    (tmp_path / "docs" / "hr" / "規程.txt").write_text("全く別の内容に変わった。", encoding="utf-8")
    sync_kb(home, "legal", embedder=CharEmbedder())
    with pytest.raises(ChunkSuperseded) as raised:
        read_chunk(home, "legal", anchor)
    assert raised.value.details["doc_id"] == anchor.split(":")[0]


# --- CLI ---------------------------------------------------------------------------


def test_cli_search_and_read(home):
    from smartrag.cli.main import app

    runner = CliRunner()
    base = ["run", "--home", str(home.root)]
    lexical = runner.invoke(app, [*base, "--format", "json", "kb", "search", "--kb", "legal",
                                  "賃金", "--mode", "lexical", "--type", "txt"])
    assert lexical.exit_code == 0, lexical.output
    data = json.loads(lexical.stdout)
    assert data["results"][0]["citation"]["path"] == "hr/規程.txt"
    text = runner.invoke(app, [*base, "kb", "search", "--kb", "legal", "賃金", "--mode",
                               "lexical"])
    assert "1. " in text.stdout and "lexical" in text.stdout
    read = runner.invoke(app, [*base, "kb", "read", "--kb", "legal", "--chunk",
                               data["results"][0]["chunk_id"], "--scope", "document"])
    assert read.exit_code == 0 and "規程.txt" in read.stdout
    both = runner.invoke(app, [*base, "kb", "search", "--kb", "legal", "x", "--page", "2",
                               "--cursor", "abc"])
    assert both.exit_code == 2
    missing = runner.invoke(app, [*base, "--format", "json", "kb", "search", "--kb", "nope", "x"])
    assert missing.exit_code == 1 and json.loads(missing.stdout)["error"]["code"] == "not_found"


def test_search_after_the_wal_files_are_gone(home):
    """Regression: a read-only open failed ("unable to open database file") once SQLite
    had removed index.db-wal and index.db-shm, as it does when the last writer closes."""
    kb_dir = home.root / "legal"
    for name in ("index.db-wal", "index.db-shm"):
        (kb_dir / name).unlink(missing_ok=True)
    result = search_kb(home, "legal", "賃金", mode="lexical")
    assert result.results
    assert read_chunk(home, "legal", result.results[0].chunk_id).chunks
    assert home.status("legal").index.documents == 3


def test_reading_never_creates_or_changes_an_index(home, tmp_path):
    from smartrag.index import SqliteIndex

    missing = tmp_path / "nothing.db"
    with pytest.raises(sqlite3.OperationalError):
        SqliteIndex.open(missing, readonly=True)
    assert not missing.exists()
    store = SqliteIndex.open(home.root / "legal" / "index.db", readonly=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        store.db.execute("DELETE FROM documents")
    store.close()
