# Smart RAG

A knowledge source any agent can use. **Search returns evidence** (snippets, stable ids,
citations); the agent, or a person at the search box, writes the answer.

Status: **skeleton** (0.0.x). Knowledge bases and their sources are managed with
`smartrag run kb|source`, and `kb sync` builds the SQLite index (text, Markdown, HTML, PDF,
DOCX, PPTX, CSV/TSV/Excel; BM25 over character bigrams; vectors from `/v1/embeddings`). The query-plane Web API is complete as a contract and runs on an
in-memory demo engine; it does not yet route by kb. `kb search` and `kb read` query the index,
and `smartrag mcp` serves them to agents. Still to come: the kb-aware Web API,
`smartrag console`, `doc list|show|remove`, and PostgreSQL. Design:
`../Memo/design_SmartRAG.md` §3.6 (CLI) and §3.9 (MCP).

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

## Knowledge bases

```sh
smartrag run kb new --kb legal --title 法務ナレッジベース --description "契約書と社内規程"
smartrag run kb set --kb legal --description "契約書と社内規程。判例は含まない。"
smartrag run source add --kb legal --source kitei ~/Documents/規程 --include '**/*.pdf'
smartrag run source test --kb legal --source kitei      # what a sync would see; reads no contents
smartrag run kb status --kb legal
smartrag run --format json kb list                      # JSON for programs, errors included
```

| Command | Purpose |
|---|---|
| `kb list` | The kbs in the home, each `ok`, `locked` or `config_error` |
| `kb new --kb ID` | Create `<home>/<ID>/` with `config.yaml`, `sink/`, `cache/`, `runs/`. `--remote URL --token-env VAR` makes a remote kb |
| `kb set --kb ID --description TEXT` | Change the description in place (comments kept). At most 300 characters; agents choose a kb by it and do not see a kb without one |
| `kb status --kb ID` | Settings, sources, sink, index, lock, environment variables, warnings. No network calls |
| `kb remove --kb ID` | Delete the kb's folder (config, index, sink copies). Original files are never touched |
| `kb sync --kb ID` | Index new and changed files, remove deleted ones; `--dry-run` writes nothing, `--source SID` one source, `--force` re-extracts. On a terminal, a progress bar redrawn every `--refresh` seconds (default 1). Exit 0 done, 1 failed, 3 done with skipped files, 130 interrupted |
| `kb ingest --kb ID PATH...` | Copy files (folders with `-r`) into the kb's sink and index them; copies are snapshots, the same content twice is a no-op |
| `kb reindex --kb ID --refresh-stale-only\|--rebuild` | Re-process stale documents, or rebuild the index beside the live one and swap it in (also how a kb changes model) |
| `kb search --kb ID "QUERY"` | Hybrid keyword + vector search: ranked chunks with snippets and citations (`--search-mode`, `--group-by`, filters, `--page`/`--cursor`). Evidence only, never an answer |
| `kb read --kb ID --chunk CHUNK_ID` | A search result widened to `--scope neighbors|section|document`, paged by `--max-chars` |
| `source list --kb ID` | Registered sources plus the reserved `sink` |
| `source add --kb ID --source SID LOCATION` | Register a folder or file. Nothing is copied |
| `source remove --kb ID --source SID` | Unregister a source |
| `source test --kb ID --source SID` | Walk it as a sync would: matches, types, skips, unreadable paths |
| `source apply --kb ID --source SID --manifest FILE [--schema FILE]` | Replace the source's document attributes (`title`, `link`, others) from a JSONL manifest keyed by `path`; all or nothing, no re-embedding. `--clear` removes them, `--dry-run` only counts |

- The home is `--home` > `$SMARTRAG_HOME` > `~/.smartrag`; `--kb` is always required.
- Secrets appear in `config.yaml` only as `${ENV_VAR}` references; a literal value is rejected.
- Commands edit `config.yaml` in place: your comments, key order and quoting are kept.
- Supported file types: text, Markdown, HTML, PDF, DOCX, PPTX, and tables (CSV, TSV, XLSX,
  XLSM, XLS, XLSB), which are converted to TSV internally. `source test` lists the rest.

## Extraction

The ingestion pipeline is being built piece by piece; the first piece is the table converter.

```sh
python -m smartrag.extract 見積.xlsx            # each table as TSV, under its sheet and cell range
python -m smartrag.extract 見積.csv --json      # the blocks: header, rows, source row numbers
```

CSV, TSV and Excel files become **table blocks**: Excel's own Table objects as they are,
everything else split at blank rows, with a text-only first row taken as the header and a
lone cell above a table as its title. Each block keeps its header apart from its rows, so
the chunker can break between rows and repeat the header at the top of every chunk.
CSV encodings are detected (cp932 before cp1252) and the delimiter is sniffed. Hidden sheets
are skipped unless asked for. Best effort: side-by-side tables with no blank row between
them become one block, and multi-row headers become a header plus data rows.
- Destructive commands ask first, and need `--yes` when there is no terminal.
- Exit codes: `0` success, `1` failed, `2` usage error (`kb sync` adds `3` and `130`).
- An embedding endpoint without authentication: remove `api_key` from `config.yaml`.

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

## MCP server (agents)

```sh
smartrag mcp [--home DIR] [--allow-kb ID]...     # MCP over stdio; spawned by the MCP host
```

| Tool | Purpose |
|---|---|
| `list_kbs()` | The usable kbs: `kb_id` and description only |
| `describe_sources(kb_id)` | Sources, file types, date range, last sync; the filter values `search` accepts |
| `search(kb_id, query, k, cursor, source, file_type, modified_after, modified_before)` | Hybrid search: passages with snippets and citations, never an answer |
| `read(kb_id, chunk_id, scope, cursor)` | A passage widened to `neighbors`, its `section` or the `document`, in pages |

- **Read-only.** Agents never sync, ingest or change a kb.
- **Agents see only usable kbs:** local, indexed with at least one document, built with the
  configured embedding profile, and with a description. The set is fixed at startup (it is
  in the server instructions and the `kb_id` enum) and re-checked on every call; any other
  kb id is refused as `not_found`. `--allow-kb` narrows it further.
- **Citations carry the document's full path** (`file_path`) and its manifest `link`, unless
  the kb's config sets `agent.show_paths: false`; error messages never carry paths. Titles
  are cut to file names. Manifest attributes reach agents only when `agent.attributes` in
  the kb's config lists them.
- **Embedding endpoint down:** keyword results, with `search_mode` `keyword`. A query waits
  at most 10 s with one retry; after a failure the vector side is skipped for 60 s.
- Always starts, even with no home or no usable kb. stdout carries only the protocol; logs
  go to stderr at warning level, without query text.

## Layout

```
src/smartrag/
  models.py        request/response schemas (the contract)
  errors.py        errors with stable codes
  paging.py        opaque cursors bound to their request
  engine/          RetrievalEngine protocol; in-memory engine + demo corpus
  kb/              knowledge bases: home, config.yaml schema, lock, sources (core library)
  extract/         source files to text; tabular.py: CSV/TSV/Excel to TSV table blocks
  chunking.py      units to chunks; index/: SQLite store and bigram lexify
  embedding.py     the /v1/embeddings client; kb/sync.py: the sync
  search/          kb search (hybrid, RRF, paging) and kb read
  agent/           the agent's view: usable kbs, results without paths, fast-failing embedders
  mcp/             MCP server over agent/ (`smartrag mcp`)
  api/             FastAPI app: query routes, ops routes, auth, error shape
  cli/             `smartrag run` (kb, source), `smartrag serve` and `smartrag mcp`
```
