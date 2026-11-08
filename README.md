# Smart RAG

A knowledge source any agent can use. **Search returns evidence** (snippets, stable ids,
citations); the agent, or a person at the search box, writes the answer.

Status: **skeleton** (0.0.1). The query-plane Web API is complete as a contract and runs on an
in-memory demo engine. The real index (PostgreSQL + pgvector + PGroonga, embeddings from
OumiGo's `/v1/embeddings`), ingestion and the admin CLI are still to come.

## Run

```sh
uv venv --python 3.12 && uv pip install -e '.[dev]'
.venv/bin/smartrag serve --demo --no-auth          # http://127.0.0.1:8710/docs
curl -s 'http://127.0.0.1:8710/v1/search?q=有給休暇の繰り越し' | python3 -m json.tool
.venv/bin/pytest
```

`--no-auth` is refused unless the server binds a loopback address. Otherwise clients send
`Authorization: Bearer <key>`, checked against the hashed key file `SMARTRAG_API_KEYS_FILE`:

```json
[{"key_id": "ui-search", "sha256": "<sha256 of the key>", "role": "reader", "delegate": true}]
```

## Web API (query plane)

| Method & path | Purpose |
|---|---|
| `GET /v1/sources` | What is indexed, filterable fields, freshness (no URIs, no credentials) |
| `POST /v1/search` | Natural-language search: snippets, ids, citations; filters, cursors, facets |
| `GET /v1/search?q=` | The same for the search box: one result per document, bookmarkable |
| `GET /v1/chunks/{chunk_id}` | Read with `scope=chunk\|neighbors\|section\|document`, paged by `max_chars` |
| `GET /v1/documents/{doc_id}` | Metadata and outline; text in pages with `include=text` |
| `POST /v1/feedback` | `click` / `relevant` / `irrelevant` for a `query_id` |
| `GET /healthz`, `/readyz`, `/version` | Ops (no auth) |

- Errors always look like `{"error": {"code", "message", "details"}}`.
- A chunk from an older document version returns `410 chunk_superseded`, with the current version.
- `X-SmartRAG-On-Behalf-Of: <user>` names the end user (keys with `delegate` only).

Admin operations (sources, sync, jobs, index, query logs, API keys, eval) are **not** part of
the Web API by design: they run through the `smartrag` CLI on the operator host, and the
query server holds no source credentials.

## Layout

```
src/smartrag/
  models.py        request/response schemas (the contract)
  errors.py        errors with stable codes
  paging.py        opaque cursors bound to their request
  engine/          RetrievalEngine protocol; in-memory engine + demo corpus
  api/             FastAPI app: query routes, ops routes, auth, error shape
  cli/             `smartrag serve` (admin commands to come)
```
