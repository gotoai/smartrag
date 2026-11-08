from __future__ import annotations


def _search(client, **body):
    resp = client.post("/v1/search", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_sources_describe_fields_without_uris(client):
    data = client.get("/v1/sources").json()
    ids = {s["source_id"] for s in data["sources"]}
    assert ids == {"nas-kitei", "s3-manuals"}
    assert all("uri" not in s for s in data["sources"])
    fields = {f["name"]: f for f in data["fields"]}
    assert set(fields["file_type"]["values"]) == {"pdf", "docx", "md"}
    assert data["index"]["languages"] == ["ja", "en"]


def test_search_returns_evidence_with_citations(client):
    data = _search(client, query="有給休暇の繰り越しは何日まで？", include=["highlights", "text"])
    top = data["results"][0]
    assert "繰り越す" in top["snippet"]
    assert top["citation"]["page"] == 14
    assert top["citation"]["heading_path"] == ["第5章 休暇", "第22条 年次有給休暇"]
    assert top["citation"]["link"].endswith(f"/v1/documents/{top['doc_id']}?page=14")
    assert top["text"].startswith("年次有給休暇は、翌年度")
    assert top["highlights"] and all(0 <= s < e <= len(top["snippet"])
                                     for s, e in top["highlights"])
    assert data["query_id"].startswith("q_")
    assert "answer" not in data


def test_filters(client):
    only_docx = _search(client, query="申請", filters={"file_type": ["docx"]})
    assert {r["metadata"]["file_type"] for r in only_docx["results"]} == {"docx"}

    by_path = _search(client, query="休暇", filters={"path_prefix": "/就業規則/"})
    assert by_path["results"]
    assert all(r["citation"]["path"].startswith("就業規則/") for r in by_path["results"])

    recent = _search(client, query="申請 休暇", filters={"modified_at": {"gte": "2026-01-01T00:00:00Z"}})
    assert all(r["metadata"]["modified_at"] >= "2026-01-01" for r in recent["results"])


def test_unknown_request_field_is_rejected(client):
    resp = client.post("/v1/search", json={"query": "休暇", "filter": {"file_type": ["pdf"]}})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_request"


def test_empty_query_is_rejected(client):
    resp = client.post("/v1/search", json={"query": ""})
    assert resp.status_code == 400
    assert resp.json()["error"]["details"]["errors"][0]["loc"] == ["body", "query"]


def test_pagination_with_bound_cursors(client):
    first = _search(client, query="休暇", limit=1)
    assert len(first["results"]) == 1 and first["next_cursor"]
    second = _search(client, query="休暇", limit=1, cursor=first["next_cursor"])
    assert second["results"][0]["chunk_id"] != first["results"][0]["chunk_id"]

    resp = client.post("/v1/search", json={"query": "副業", "cursor": first["next_cursor"]})
    assert resp.status_code == 400
    assert "different request" in resp.json()["error"]["message"]

    resp = client.post("/v1/search", json={"query": "休暇", "cursor": "garbage!"})
    assert resp.status_code == 400


def test_group_by_document_and_facets(client):
    data = _search(client, query="休暇 申請", group_by="document", include=["facets"])
    doc_ids = [r["doc_id"] for r in data["results"]]
    assert len(doc_ids) == len(set(doc_ids))
    assert data["facets"]["file_type"] == {"pdf": 1, "docx": 1}


def test_get_search_for_the_search_box(client):
    page1 = client.get("/v1/search", params={"q": "休暇 申請", "limit": 1}).json()
    page2 = client.get("/v1/search", params={"q": "休暇 申請", "limit": 1, "page": 2}).json()
    assert page1["results"][0]["doc_id"] != page2["results"][0]["doc_id"]
    assert page1["results"][0]["highlights"]

    only = client.get("/v1/search", params={"q": "indexing", "source": "s3-manuals"}).json()
    assert {r["citation"]["source_id"] for r in only["results"]} == {"s3-manuals"}


def test_no_match_is_an_empty_list(client):
    data = _search(client, query="量子コンピュータ")
    assert data["results"] == [] and data.get("next_cursor") is None
