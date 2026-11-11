"""The kb library: homes, configs, locks."""

from __future__ import annotations

import json
import os
import stat

import pytest

from smartrag.errors import ConfigError, Conflict, InvalidRequest, KbLocked, NotFound
from smartrag.kb import Home, resolve_home
from smartrag.kb.config import CONFIG_FILE, KbConfig, load_config
from smartrag.kb.lock import LOCK_FILE, kb_lock, read_lock
from smartrag.kb.uris import location_to_uri, path_to_uri, uri_to_path


@pytest.fixture
def home(tmp_path):
    return Home(tmp_path / "home")


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- home resolution ---------------------------------------------------------------


def test_home_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("SMARTRAG_HOME", str(tmp_path / "from-env"))
    assert resolve_home(tmp_path / "explicit") == tmp_path / "explicit"
    assert resolve_home(None) == tmp_path / "from-env"
    monkeypatch.delenv("SMARTRAG_HOME")
    assert resolve_home(None).name == ".smartrag"


# --- kb new ------------------------------------------------------------------------


def test_new_local_kb_layout_and_permissions(home):
    created = home.new("legal", title="法務", description="契約書")
    kb_dir = home.root / "legal"
    assert created.kind == "local"
    assert {p.name for p in kb_dir.iterdir()} == {"config.yaml", "sink", "cache", "runs"}
    assert stat.S_IMODE(kb_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((kb_dir / CONFIG_FILE).stat().st_mode) == 0o600
    config = load_config(kb_dir / CONFIG_FILE)
    assert config.embedding is not None
    assert config.embedding.api_key == "${SMARTRAG_EMBED_API_KEY}"
    assert config.sink == "sink" and config.sources == []


def test_new_rejects_duplicates_and_bad_ids(home):
    home.new("legal")
    with pytest.raises(Conflict):
        home.new("legal")
    for bad in ("Legal", "-x", "has space", "a" * 65, "", "../up"):
        with pytest.raises(InvalidRequest):
            home.new(bad)


def test_new_remote_kb_needs_a_token_reference_and_tls(home):
    with pytest.raises(InvalidRequest, match="--token-env"):
        home.new("factory", remote="https://rag.example.jp")
    with pytest.raises(InvalidRequest, match="https"):
        home.new("factory", remote="http://rag.example.jp", token_env="TOKEN")
    home.new("local-dev", remote="http://127.0.0.1:8710", token_env="TOKEN")
    created = home.new("factory", remote="https://rag.example.jp", token_env="FACTORY_TOKEN")
    assert created.kind == "remote"
    assert [p.name for p in (home.root / "factory").iterdir()] == [CONFIG_FILE]
    text = (home.root / "factory" / CONFIG_FILE).read_text(encoding="utf-8")
    assert "${FACTORY_TOKEN}" in text
    assert "sources" not in text and "sink" not in text and "embedding" not in text


def test_new_rejects_remote_options_without_remote(home):
    with pytest.raises(InvalidRequest):
        home.new("legal", token_env="TOKEN")


# --- config rules ------------------------------------------------------------------


def _base(**extra):
    data = {"version": 1, "id": "legal", "title": "t", "kind": "local",
            "embedding": {"base_url": "${U}", "api_key": "${K}", "profile": "p"}}
    data.update(extra)
    return data


def test_literal_secrets_are_rejected():
    with pytest.raises(ValueError, match="environment variable reference"):
        KbConfig.model_validate(_base(embedding={"base_url": "${U}", "api_key": "sk-123"}))
    with pytest.raises(ValueError, match="environment variable reference"):
        KbConfig.model_validate({"id": "f", "title": "f", "kind": "remote", "remote": {
            "endpoint": "https://x.example", "kb": "f", "token": "abc"}})


def test_remote_kb_cannot_carry_sources_or_a_further_hop():
    remote = {"endpoint": "https://x.example", "kb": "f", "token": "${T}"}
    with pytest.raises(ValueError, match="managed by its server"):
        KbConfig.model_validate({"id": "f", "title": "f", "kind": "remote", "remote": remote,
                                 "sources": [{"id": "a", "uri": "file:///tmp"}]})
    with pytest.raises(ValueError):
        KbConfig.model_validate({"id": "f", "title": "f", "kind": "remote",
                                 "remote": {**remote, "cascade": "https://y.example"}})


def test_unknown_keys_and_versions_are_refused():
    with pytest.raises(ValueError):
        KbConfig.model_validate(_base(cascade=True))
    with pytest.raises(ValueError, match="version 2"):
        KbConfig.model_validate(_base(version=2))


def test_env_refs_and_missing():
    config = KbConfig.model_validate(_base())
    assert config.env_refs() == ["U", "K"]
    assert config.missing_env({"U": "http://x"}) == ["K"]


def test_config_id_must_match_directory(home):
    home.new("legal")
    path = home.root / "legal" / CONFIG_FILE
    path.write_text(path.read_text(encoding="utf-8").replace("id: legal", "id: other"),
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="does not match"):
        home.load("legal")


# --- kb list / status --------------------------------------------------------------


def test_list_reports_each_kb_without_failing_on_a_broken_one(home):
    assert home.list().kbs == [] and not home.list().home_exists
    home.new("legal", title="法務")
    home.new("broken")
    _write(home.root / "broken" / CONFIG_FILE, "id: [unclosed")
    _write(home.root / "stray" / "notes.txt", "not a kb")
    kbs = {k.kb: k for k in home.list().kbs}
    assert set(kbs) == {"legal", "broken"}
    assert kbs["legal"].status == "ok" and kbs["legal"].title == "法務"
    assert kbs["broken"].status == "config_error" and "YAML" in (kbs["broken"].error or "")


def test_status_warns_about_what_needs_doing(home, monkeypatch, tmp_path):
    monkeypatch.delenv("SMARTRAG_EMBED_API_KEY", raising=False)
    monkeypatch.setenv("SMARTRAG_EMBED_BASE_URL", "http://oumigo.local/v1")
    home.new("legal")
    status = home.status("legal")
    assert status.env.missing == ["SMARTRAG_EMBED_API_KEY"]
    assert status.index is not None and not status.index.exists
    joined = " ".join(status.warnings)
    for expected in ("no description", "SMARTRAG_EMBED_API_KEY", "no index yet", "no sources"):
        assert expected in joined
    with pytest.raises(NotFound):
        home.status("nope")


# --- kb remove ---------------------------------------------------------------------


def test_remove_deletes_the_kb_but_never_source_files(home, tmp_path):
    from smartrag.kb import add_source

    originals = tmp_path / "docs"
    _write(originals / "a.txt", "keep me")
    home.new("legal")
    add_source(home, "legal", "docs", str(originals))
    _write(home.root / "legal" / "sink" / "abc" / "copy.txt", "copied")
    removed = home.remove("legal")
    assert removed.sink_files_deleted == 1
    assert not (home.root / "legal").exists()
    assert (originals / "a.txt").read_text(encoding="utf-8") == "keep me"
    with pytest.raises(NotFound):
        home.remove("legal")


def test_a_broken_kb_can_still_be_removed(home):
    home.new("broken")
    _write(home.root / "broken" / CONFIG_FILE, "not: [valid")
    _write(home.root / "broken" / "sink" / "h" / "f.txt", "x")
    assert home.remove("broken").sink_files_deleted == 1


def test_remove_refuses_while_locked(home):
    home.new("legal")
    with kb_lock(home.root / "legal", "kb sync"), pytest.raises(KbLocked):
        home.remove("legal")
    assert (home.root / "legal").exists()


# --- locks -------------------------------------------------------------------------


def test_lock_is_released_and_stale_locks_are_taken_over(tmp_path):
    with kb_lock(tmp_path, "test"):
        holder = read_lock(tmp_path)
        assert holder is not None and holder.pid == os.getpid() and not holder.stale
    assert not (tmp_path / LOCK_FILE).exists()

    (tmp_path / LOCK_FILE).write_text(json.dumps({"pid": 2**22 + 12345, "command": "x"}))
    holder = read_lock(tmp_path)
    assert holder is not None and holder.stale
    with kb_lock(tmp_path, "test"):
        pass


def test_live_lock_blocks(tmp_path):
    (tmp_path / LOCK_FILE).write_text(json.dumps({"pid": os.getpid(), "command": "kb sync"}))
    with pytest.raises(KbLocked, match="kb sync"), kb_lock(tmp_path, "test"):
        pass


# --- uris --------------------------------------------------------------------------


def test_uris_stay_readable_and_round_trip(tmp_path):
    folder = tmp_path / "規程 2026"
    uri, path = location_to_uri(str(folder))
    assert uri == "file://" + folder.resolve().as_posix()
    assert "%" not in uri and uri_to_path(uri) == path
    typed = "file://" + folder.resolve().as_posix().replace(" ", "%20")
    assert location_to_uri(typed)[1] == path
    assert path_to_uri(path) == uri
    assert uri_to_path("s3://bucket/x") is None


# --- editing config.yaml in place --------------------------------------------------


def test_commands_keep_hand_written_comments_and_formatting(home, tmp_path):
    from smartrag.kb import add_source, remove_source

    docs = tmp_path / "docs"
    (docs / "sub").mkdir(parents=True)
    home.new("legal", title="法務")
    path = home.root / "legal" / CONFIG_FILE
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Smart RAG knowledge base.")
    assert "# What this kb contains" in text
    text = text.replace("description: ''", "description: |\n  契約書と規程。\n  判例は含まない。")
    text = text.replace("  profile: ruri-v3-310m@8k", "  profile: ruri-v3-310m@8k   # chosen by eval")
    path.write_text(text + "# TODO: add the 人事 share\n", encoding="utf-8")

    add_source(home, "legal", "docs", str(docs), include=["**/*.pdf"])
    add_source(home, "legal", "sub", str(docs / "sub"))
    remove_source(home, "legal", "sub")

    after = path.read_text(encoding="utf-8")
    for kept in ("# chosen by eval", "# TODO: add the 人事 share", "description: |",
                 "  契約書と規程。", "# What this kb contains"):
        assert kept in after, kept
    assert "  - id: docs" in after and "id: sub" not in after
    assert f"uri: file://{docs.resolve().as_posix()}" in after  # one line, unencoded
    assert home.load("legal").sources[0].include == ["**/*.pdf"]


def test_hand_edits_that_break_the_rules_are_caught_on_load(home):
    home.new("legal")
    path = home.root / "legal" / CONFIG_FILE
    path.write_text(path.read_text(encoding="utf-8").replace(
        "${SMARTRAG_EMBED_API_KEY}", "sk-live-123"), encoding="utf-8")
    with pytest.raises(ConfigError, match="environment variable reference"):
        home.load("legal")
