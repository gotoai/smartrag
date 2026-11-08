from __future__ import annotations

import json
import os

import pytest

from smartrag.api import create_app
from smartrag.api.auth import ON_BEHALF_OF_HEADER, hash_key
from smartrag.engine import demo_engine
from smartrag.settings import Settings
from tests.conftest import DELEGATE_KEY


def test_query_endpoints_require_a_key(anon):
    for method, path in [("get", "/v1/sources"), ("get", "/v1/search?q=休暇"),
                         ("get", "/v1/documents/d_0"), ("get", "/v1/chunks/x")]:
        resp = getattr(anon, method)(path)
        assert resp.status_code == 401, path
        assert resp.json()["error"]["code"] == "unauthorized"
        assert resp.headers["www-authenticate"] == "Bearer"


def test_wrong_key_and_wrong_scheme(anon):
    assert anon.get("/v1/sources", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert anon.get("/v1/sources", headers={"Authorization": "Basic abc"}).status_code == 401


def test_ops_endpoints_are_open(anon):
    assert anon.get("/healthz").json() == {"status": "ok"}
    assert anon.get("/version").json()["api"] == "v1"
    assert anon.get("/readyz").status_code == 200


def test_on_behalf_of_needs_a_delegate_key(client):
    resp = client.get("/v1/sources", headers={ON_BEHALF_OF_HEADER: "tanaka@example.jp"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"

    resp = client.get("/v1/sources", headers={"Authorization": f"Bearer {DELEGATE_KEY}",
                                              ON_BEHALF_OF_HEADER: "tanaka@example.jp"})
    assert resp.status_code == 200


def test_key_file_is_reloaded_when_it_changes(client, keys_file):
    new_key = "issued-later"
    entries = json.loads(keys_file.read_text())
    entries.append({"key_id": "late", "sha256": hash_key(new_key), "role": "reader"})
    keys_file.write_text(json.dumps(entries))
    stat = keys_file.stat()
    os.utime(keys_file, (stat.st_atime, stat.st_mtime + 5))  # make the change visible

    resp = client.get("/v1/sources", headers={"Authorization": f"Bearer {new_key}"})
    assert resp.status_code == 200


def test_app_refuses_to_start_without_keys():
    with pytest.raises(ValueError, match="no API keys configured"):
        create_app(demo_engine(), Settings(api_keys_file=None, auth_disabled=False))
