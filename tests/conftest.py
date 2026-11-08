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
