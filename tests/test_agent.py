"""The agent's view of a home (design memo §3.9): usable kbs only, no paths, fail fast."""

from __future__ import annotations

import pytest

from smartrag.agent import (
    DEGRADED,
    MAX_DESCRIPTION,
    AgentService,
    QueryEmbedders,
    agent_description,
    agent_error,
    safe_title,
)
from smartrag.errors import (
    ChunkSuperseded,
    ConfigError,
    CursorExpired,
    InvalidRequest,
    NotFound,
)
from smartrag.kb.sync import sync_kb
from tests.test_search_read import CharEmbedder


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _service(home, embedder=None, clock=None) -> AgentService:
    embedder = embedder or CharEmbedder()
    embedders = QueryEmbedders(lambda _config: embedder, clock=clock or Clock())
    return AgentService(home, embedders=embedders)


# --- which kbs ---------------------------------------------------------------------


def test_only_usable_kbs_are_listed(agent_home):
    service = _service(agent_home)
    assert service.visible == ("legal",)
    assert [kb.model_dump() for kb in service.list_kbs().kbs] == [
        {"kb_id": "legal", "description": "税務と人事の資料: 相続税、電子申告、休暇と賃金の規程。"}]


def test_description_is_one_paragraph_cut_to_the_limit():
    text = agent_description("規程" * 200)
    assert len(text) == MAX_DESCRIPTION and text.endswith("…")
    assert agent_description("  a\n\n b  ") == "a b"


@pytest.mark.parametrize("kb_id", ["nodesc", "empty", "later", "factory", "nope", "../legal"])
def test_hidden_and_unknown_kbs_are_refused_alike(agent_home, kb_id):
    service = _service(agent_home)
    for call in (lambda: service.search(kb_id, "相続税"), lambda: service.describe_sources(kb_id),
                 lambda: service.read(kb_id, "d_x:y")):
        with pytest.raises(NotFound) as caught:
            call()
        assert agent_error(caught.value, kb_id).code == "not_found"


def test_allow_kb_limits_the_set(agent_home):
    assert AgentService(agent_home, allow=["nodesc", "legal"]).visible == ("legal",)
    assert AgentService(agent_home, allow=["nodesc"]).visible == ()


def test_a_kb_that_stops_qualifying_is_refused(agent_home):
    service = _service(agent_home)
    for leftover in agent_home.kb_dir("legal").glob("index.db*"):
        leftover.unlink()
    assert service.list_kbs().kbs == []
    with pytest.raises(NotFound):
        service.search("legal", "相続税")


def test_a_kb_that_qualifies_later_waits_for_a_restart(agent_home):
    service = _service(agent_home)
    sync_kb(agent_home, "later", embedder=CharEmbedder())
    assert [kb.kb_id for kb in service.list_kbs().kbs] == ["legal"]
    with pytest.raises(NotFound):
        service.search("later", "相続税")
    assert _service(agent_home).visible == ("later", "legal")


# --- describe, search, read ----------------------------------------------------------


def test_describe_sources(agent_home):
    d = _service(agent_home).describe_sources("legal")
    assert d.documents == 3 and d.as_of
    assert [(s.source_id, s.documents, s.file_types) for s in d.sources] == [
        ("docs", 3, {"md": 2, "txt": 1})]
    assert d.sources[0].modified_from and d.sources[0].modified_to
    filters = {f.name: f for f in d.filters}
    assert filters["source"].values == ["docs"] and filters["file_type"].values == ["md", "txt"]
    assert set(filters) == {"source", "file_type", "modified_after", "modified_before"}


def test_search_and_read_carry_no_paths(agent_home, tmp_path):
    service = _service(agent_home)
    r = service.search("legal", "相続税の評価")
    top = r.results[0]
    assert top.title == "相続.docx" and top.citation.heading_path == ["相続税"]
    assert top.file_type == "md" and top.indexed_at and r.degraded is None
    read = service.read("legal", top.chunk_id, scope="document")
    assert read.title == "相続.docx" and read.scope == "document"
    assert read.chunks and len(read.chunks) == read.total_chunks
    for model in (r, read):
        assert not _keys(model.model_dump()) & {"path", "origin_path", "rel_path", "uri", "link"}
        dumped = model.model_dump_json()
        assert "tax/" not in dumped and "C:\\" not in dumped
        assert str(tmp_path) not in dumped and "Users" not in dumped


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_search_filters_and_dates(agent_home):
    service = _service(agent_home)
    only_txt = service.search("legal", "賃金", file_type=["txt"])
    assert {h.file_type for h in only_txt.results} == {"txt"}
    assert service.search("legal", "賃金", modified_after="2999-01-01").results == []
    assert service.search("legal", "賃金", modified_before="2999-01-01T00:00:00Z").results
    with pytest.raises(InvalidRequest, match="modified_after must be a date"):
        service.search("legal", "賃金", modified_after="last week")


def test_unknown_chunk_is_not_found(agent_home):
    with pytest.raises(NotFound) as caught:
        _service(agent_home).read("legal", "d_nothing:here")
    error = agent_error(caught.value, "legal")
    assert error.code == "not_found" and "chunk_id" in error.message


# --- the embedding endpoint ------------------------------------------------------------


def test_endpoint_down_degrades_with_a_fixed_sentence_and_fails_fast(agent_home):
    clock = Clock()
    down = CharEmbedder(fail=True)
    service = _service(agent_home, down, clock=clock)
    first = service.search("legal", "賃金")
    assert first.degraded == DEGRADED and first.results and len(down.calls) == 1
    assert service.search("legal", "相続税").degraded == DEGRADED
    assert len(down.calls) == 1  # skipped during the cooldown, no wait
    clock.now += 61
    service.search("legal", "電子申告")
    assert len(down.calls) == 2


def test_a_missing_credential_degrades_instead_of_failing(agent_home, monkeypatch):
    monkeypatch.delenv("SMARTRAG_EMBED_BASE_URL", raising=False)
    monkeypatch.delenv("SMARTRAG_EMBED_API_KEY", raising=False)
    r = AgentService(agent_home).search("legal", "賃金")
    assert r.degraded == DEGRADED and r.results


def test_the_embedder_is_checked_once_per_kb(agent_home):
    class Counting(CharEmbedder):
        checks = 0

        def check(self) -> int:
            Counting.checks += 1
            return super().check()

    service = _service(agent_home, Counting())
    service.search("legal", "賃金")
    service.search("legal", "相続税")
    assert Counting.checks == 1


# --- errors and titles -----------------------------------------------------------------


def test_errors_carry_no_library_text(agent_home):
    root = str(agent_home.root)
    cases = {
        NotFound(f"no kb 'x' in {root}", {"kb": "x", "home": root}): "not_found",
        ConfigError(f"{root}/x/config.yaml: broken"): "not_found",
        ChunkSuperseded("changed", {"doc_id": "d_1"}): "chunk_superseded",
        CursorExpired("expired"): "cursor_expired",
        RuntimeError(f"disk I/O error at {root}"): "internal_error",
    }
    for err, code in cases.items():
        error = agent_error(err, "x")
        assert error.code == code and error.hint
        assert root not in error.model_dump_json()
    assert agent_error(ChunkSuperseded("changed", {"doc_id": "d_1"})).details == {"doc_id": "d_1"}


@pytest.mark.parametrize(("title", "expected"), [
    ("C:\\Users\\tanaka\\見積.docx", "見積.docx"),
    ("\\\\fileserver\\share\\規程.pdf", "規程.pdf"),
    ("/home/tanaka/報告書.pdf", "報告書.pdf"),
    ("~/Documents/a.md", "a.md"),
    ("2024/03 決算報告", "2024/03 決算報告"),
    ("就業規則", "就業規則"),
])
def test_safe_title(title, expected):
    assert safe_title(title) == expected
