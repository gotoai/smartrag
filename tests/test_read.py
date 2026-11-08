from __future__ import annotations

import pytest

from smartrag.engine import RequestContext


@pytest.fixture
def carryover(client):
    """The chunk about carrying over paid leave (2nd chunk of 就業規則, page 14)."""
    resp = client.post("/v1/search", json={"query": "有給休暇の繰り越し"})
    return resp.json()["results"][0]


def test_read_scopes(client, carryover):
    cid = carryover["chunk_id"]
    chunk = client.get(f"/v1/chunks/{cid}").json()
    assert [c["chunk_id"] for c in chunk["chunks"]] == [cid]
    assert chunk["citation"]["link"].endswith("?page=14")

    neighbors = client.get(f"/v1/chunks/{cid}", params={"scope": "neighbors"}).json()
    assert len(neighbors["chunks"]) == 3  # one before, the anchor, one after

    section = client.get(f"/v1/chunks/{cid}", params={"scope": "section"}).json()
    assert len(section["chunks"]) == 2  # both chunks of 第22条
    assert all(c["heading_path"][-1] == "第22条 年次有給休暇" for c in section["chunks"])

    document = client.get(f"/v1/chunks/{cid}", params={"scope": "document"}).json()
    assert len(document["chunks"]) == 4


async def test_read_pages_by_max_chars(engine, carryover):
    # Through the engine: the API's minimum max_chars (500) exceeds the demo document.
    ctx = RequestContext(key_id="test")
    cid = carryover["chunk_id"]
    pages, cursor = [], None
    while True:
        page = await engine.read_chunk(cid, "document", 1, 80, cursor, ctx)
        pages.append([c.chunk_id for c in page.chunks])
        cursor = page.next_cursor
        if not cursor:
            break
    seen = [cid for page in pages for cid in page]
    assert len(pages) > 1
    assert len(seen) == len(set(seen)) == 4

    cut = await engine.read_chunk(cid, "chunk", 1, 10, None, ctx)
    assert cut.truncated and len(cut.chunks[0].text) == 10


def test_superseded_chunk(client, carryover):
    doc_id = carryover["doc_id"]
    old = carryover["chunk_id"].replace(":v3:", ":v2:")
    resp = client.get(f"/v1/chunks/{old}")
    assert resp.status_code == 410
    error = resp.json()["error"]
    assert error["code"] == "chunk_superseded"
    assert error["details"] == {"doc_id": doc_id, "current_version": 3}


def test_unknown_chunk(client):
    resp = client.get("/v1/chunks/d_0000000000000000:v1:c0000")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


def test_document_outline_and_text(client, carryover):
    doc_id = carryover["doc_id"]
    doc = client.get(f"/v1/documents/{doc_id}").json()
    assert doc["title"] == "就業規則" and doc["version"] == 3
    assert "chunks" not in doc  # no text unless asked
    headings = [(n["level"], n["heading"]) for n in doc["outline"]]
    assert headings[:3] == [(1, "第5章 休暇"), (2, "第22条 年次有給休暇"), (2, "第23条 特別休暇")]

    chapter6 = next(n for n in doc["outline"] if n["heading"] == "第6章 服務")
    part = client.get(f"/v1/documents/{doc_id}",
                      params={"include": "text", "section": chapter6["section_id"]}).json()
    assert [c["heading_path"] for c in part["chunks"]] == [["第6章 服務", "第30条 副業"]]

    resp = client.get(f"/v1/documents/{doc_id}", params={"section": "s999"})
    assert resp.status_code == 404


def test_feedback(client, engine, carryover):
    search = client.post("/v1/search", json={"query": "副業"}).json()
    resp = client.post("/v1/feedback", json={"query_id": search["query_id"],
                                             "chunk_id": search["results"][0]["chunk_id"],
                                             "signal": "relevant"})
    assert resp.status_code == 202 and resp.json() == {"accepted": True}
    assert engine.feedback[-1][0].key_id == "agent-1"

    resp = client.post("/v1/feedback", json={"query_id": "q_unknown", "signal": "click"})
    assert resp.status_code == 404


def test_unknown_path_uses_the_error_shape(client):
    resp = client.get("/v1/answer")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"
