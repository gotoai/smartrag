from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smartrag.api import create_app
from smartrag.api.auth import hash_key
from smartrag.engine import demo_engine
from smartrag.settings import Settings

READER_KEY = "test-reader-key"
DELEGATE_KEY = "test-delegate-key"


@pytest.fixture
def keys_file(tmp_path):
    path = tmp_path / "api_keys.json"
    path.write_text(json.dumps([
        {"key_id": "agent-1", "sha256": hash_key(READER_KEY), "role": "reader"},
        {"key_id": "ui", "sha256": hash_key(DELEGATE_KEY), "role": "reader", "delegate": True},
    ]))
    return path


@pytest.fixture
def engine():
    return demo_engine()


@pytest.fixture
def client(engine, keys_file):
    app = create_app(engine, Settings(api_keys_file=keys_file))
    with TestClient(app, headers={"Authorization": f"Bearer {READER_KEY}"}) as c:
        yield c


@pytest.fixture
def anon(engine, keys_file):
    """A client without credentials."""
    app = create_app(engine, Settings(api_keys_file=keys_file))
    with TestClient(app) as c:
        yield c


@pytest.fixture
def agent_home(tmp_path):
    """A home as an agent would find it: one usable kb (`legal`) and four it must not see
    (`nodesc` has no description, `empty` no index, `factory` is remote, `later` is
    described but not synced yet). One title is a Windows path, as Office files often
    carry, to check that no path reaches an agent."""
    import sqlite3

    from smartrag.kb import Home, add_source
    from smartrag.kb.sync import sync_kb
    from tests.test_search_read import CharEmbedder

    home = Home(tmp_path / "home")
    root = tmp_path / "docs"
    (root / "tax").mkdir(parents=True)
    (root / "hr").mkdir()
    (root / "tax" / "相続.md").write_text(
        "# 相続税\n\n相続税の評価について。土地の評価は路線価による。\n\n"
        "## 更正\n\n更正処分の取消しを求める訴え。\n", encoding="utf-8")
    (root / "tax" / "電子.md").write_text(
        "# 電子申告\n\n電子申告の義務化は特定法人に適用される。\n", encoding="utf-8")
    (root / "hr" / "規程.txt").write_text(
        "第3章 休暇\n\n年休は10日。\n\n第4章 賃金\n\n賃金は毎月25日に支払う。\n", encoding="utf-8")
    home.new("legal", description="税務と人事の資料:\n相続税、電子申告、休暇と賃金の規程。")
    home.new("nodesc")
    home.new("empty", description="Nothing synced yet.")
    home.new("later", description="Synced only after the server starts.")
    home.new("factory", description="A remote kb.", remote="https://rag.example.jp",
             token_env="FACTORY_TOKEN")
    for kb in ("legal", "nodesc", "later"):
        add_source(home, kb, "docs", str(root))
    for kb in ("legal", "nodesc"):
        sync_kb(home, kb, embedder=CharEmbedder())
    with sqlite3.connect(home.kb_dir("legal") / "index.db") as db:
        db.execute("UPDATE documents SET title = ? WHERE rel_path = 'tax/相続.md'",
                   ("C:\\Users\\tanaka\\Documents\\相続.docx",))
    return home
