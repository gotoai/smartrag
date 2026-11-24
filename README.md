# Smart RAG

A knowledge source any agent can use. **Search returns evidence** (snippets, stable ids,
citations); the agent, or a person at the search box, writes the answer.

Status: **skeleton** (0.0.x). Knowledge bases and their sources are managed with
`smartrag run kb|source`, and `kb sync` builds the SQLite index (PDF, Office, HTML, Markdown
and other markup, RTF, e-mail, JSON/YAML/XML, CSV/TSV/Excel, and any other text file; BM25 over character bigrams; vectors from a catalog of embedding
providers). The query-plane Web API is complete as a contract and runs on an
in-memory demo engine; it does not yet route by kb. `kb search` and `kb read` query the index,
and `smartrag mcp` serves them to agents. Still to come: the kb-aware Web API,
`smartrag console`, `doc list|show|remove`, and PostgreSQL.

## Run

```sh
uv venv --python 3.12 && uv pip install -e '.[dev]'
.venv/bin/smartrag serve --demo=true --auth=false         # http://127.0.0.1:8710/docs
curl -s 'http://127.0.0.1:8710/v1/search?q=有給休暇の繰り越し' | python3 -m json.tool
.venv/bin/pytest
```

`--auth=false` is refused unless the server binds a loopback address. Otherwise clients send
`Authorization: Bearer <key>`, checked against the hashed key file `SMARTRAG_API_KEYS_FILE`:

```json
[{"key_id": "ui-search", "sha256": "<sha256 of the key>", "role": "reader", "delegate": true}]
```

## Knowledge bases

A first kb, from creation to search:

```sh
smartrag run kb new --kb legal --title 法務ナレッジベース --description "契約書と社内規程"
smartrag run source add --kb legal --source kitei ~/Documents/規程 --include '**/*.pdf'
smartrag run source test --kb legal --source kitei      # which files a sync would read
smartrag run kb sync --kb legal
smartrag run kb search --kb legal "有給休暇の繰り越し"
```

## Command line

### Reading the grammar

| Notation | Meaning |
|---|---|
| `UPPERCASE` | A value you type, e.g. `ID` → `legal` |
| `[ … ]` | Optional |
| `a\|b` | Exactly one of the choices, typed as shown |
| `--opt X…` | Repeatable: give the whole option again for each value, `--opt a --opt b` |
| `--name=true\|false` | An on/off setting. It always takes a value: `--recursive=false` (or `--recursive false`); a bare `--recursive` is an error |
| `--dry-run`, `--yes`, `--force` | A switch: give it to turn it on, with no value |

Quote any value that contains spaces or `*` (`'**/*.pdf'`), so the shell passes it unchanged.

### General form

```
smartrag run [--home DIR] [--format text|json] SUBJECT VERB [OPTIONS] [ARGUMENTS]
smartrag mcp [--home DIR] [--allow-kb ID]…
smartrag serve [--host HOST] [--port PORT] [--api-keys-file FILE] [--demo=true|false]
               [--auth=true|false] [--log-level LEVEL]
smartrag version
```

- `--home` and `--format` belong to `run` and go **before** the subject:
  `smartrag run --home ./raghome kb list`, not `smartrag run kb list --home ./raghome`.
- `--home DIR` is the folder that holds all kbs. Without it: `$SMARTRAG_HOME`, else
  `~/.smartrag`.
- `--format text` (default) prints for people, errors on stderr. `--format json` prints one
  JSON document on stdout, also for a failure (`{"error": {"code", "message", "details"}}`).
- Everything after the verb (its options and arguments) may come in any order.
- `--kb ID` is required by every command that acts on one kb, even when the home holds only
  one kb.
- **Ids** (kb, source): 1–64 characters of lowercase letters, digits, `-` and `_`, starting
  with a letter or a digit. `sink` is reserved for each kb's own folder of ingested copies.
- **Confirmation:** `kb remove`, `source remove` and `kb reindex --rebuild` ask before
  deleting or rebuilding. With `--format json`, or when stdin is not a terminal, nobody can
  answer, so the command fails unless you pass `--yes`.

**Exit codes**

| Code | Meaning |
|---|---|
| 0 | Done (a search with no results is also 0) |
| 1 | Failed; the message says why. `source test` also exits 1 when the source's location is missing or cannot be read |
| 2 | The command line is wrong: an unknown option, a missing required option, a bad value |
| 3 | `kb sync`, `kb ingest`, `kb reindex` only: done, but some files were skipped or failed |
| 130 | `kb sync`, `kb ingest`, `kb reindex` only: interrupted (Ctrl-C). Running the same command again continues where it stopped |

### Commands at a glance

| Command | What it does |
|---|---|
| [`kb list`](#kb-list) | List the kbs in the home |
| [`kb new`](#kb-new) | Create a kb |
| [`kb set`](#kb-set) | Change a kb's description |
| [`kb verify`](#kb-verify) | Test the kb's embedding provider with one request |
| [`kb status`](#kb-status) | Show a kb's settings, index and warnings |
| [`kb remove`](#kb-remove) | Delete a kb |
| [`kb sync`](#kb-sync) | Index the sources' new and changed files, remove deleted ones |
| [`kb ingest`](#kb-ingest) | Copy files into the kb and index them |
| [`kb reindex`](#kb-reindex) | Re-process old documents, or rebuild the whole index |
| [`kb search`](#kb-search) | Search a kb |
| [`kb read`](#kb-read) | Read a search result with its context |
| [`source list`](#source-list) | List a kb's sources |
| [`source add`](#source-add) | Register a folder or file for `kb sync` to follow |
| [`source remove`](#source-remove) | Unregister a source and remove its documents |
| [`source test`](#source-test) | Show which files a sync of a source would read |
| [`provider list`](#provider-list) | List the embedding providers and models |

### kb commands

#### `kb list`

```
smartrag run kb list
```

Lists the kbs in the home as a table: id, title, kind (`local` or `remote`), number of
sources, and state. The state is `ok`; `locked` (a sync or another writing command is running
on the kb); or `config_error`, followed by what is wrong in its `config.yaml`.

```sh
smartrag run kb list
smartrag run --format json kb list
```

#### `kb new`

```
smartrag run kb new --kb ID [--title TEXT] [--description TEXT]
                    [--provider PROVIDER_ID] [--model MODEL_PROFILE_ID]
smartrag run kb new --kb ID [--title TEXT] [--description TEXT]
                    --remote URL --token-env VAR [--remote-kb REMOTE_ID]
```

Creates `<home>/ID/` with `config.yaml`, `sink/`, `cache/` and `runs/`. The first form makes
a **local** kb: the index is built and kept on this machine.

| Option | Default | Meaning |
|---|---|---|
| `--kb ID` | required | The new kb's id, also its folder name. Fails if the kb exists |
| `--title TEXT` | the id | Display name |
| `--description TEXT` | empty | What the kb contains, at most 300 characters. Agents choose a kb by it and do not see a kb without one; set it later with `kb set` |
| `--provider PROVIDER_ID` | `gotoai` | Local kb: the embedding provider, from `provider list` |
| `--model MODEL_PROFILE_ID` | `ruri-v3-310m@8k` | Local kb: one of that provider's models, from `provider list` |
| `--remote URL` | – | Makes a **remote** kb instead: the `https://` URL of the Smart RAG server that holds it |
| `--token-env VAR` | – | Remote kb, required: the name of the environment variable that holds your token for that server (the name, not the token) |
| `--remote-kb REMOTE_ID` | the `--kb` id | Remote kb: the kb's id on that server |

```sh
smartrag run kb new --kb legal --title 法務ナレッジベース --description "契約書と社内規程"
smartrag run kb new --kb edinet --provider gotoai --model harrier-oss-v1-0.6b@32k
smartrag run kb new --kb hq-legal --remote https://rag.example.com --token-env HQ_RAG_TOKEN --remote-kb legal
```

#### `kb set`

```
smartrag run kb set --kb ID --description TEXT
```

Replaces the kb's description in `config.yaml`, keeping the file's comments and formatting.
`--description` is required, at most 300 characters; `--description ""` empties it, which
hides the kb from agents.

```sh
smartrag run kb set --kb legal --description "契約書と社内規程。判例は含まない。"
```

#### `kb verify`

```
smartrag run kb verify --kb ID
```

Sends one short test text to the kb's embedding provider and checks the URL, the API key,
the model and the vector size. It sends a few tokens, so a cloud provider bills a negligible
amount. Exit 1 says what failed.

```sh
GOTOAI_EMBED_BASE_URL=http://localhost:7012/v1 smartrag run kb verify --kb edinet
```

#### `kb status`

```
smartrag run kb status --kb ID
```

Shows the kb's settings, sources, sink, index size and last sync, whether a command holds its
lock, which environment variables it needs and whether they are set, and warnings. Makes no
network calls.

```sh
smartrag run kb status --kb legal
```

#### `kb remove`

```
smartrag run kb remove --kb ID [--yes]
```

Deletes the folder `<home>/ID/`: its config, its index and the copies in its sink. The
original files of its sources are never touched. Asks first; `--yes` skips the question.

```sh
smartrag run kb remove --kb old-test --yes
```

#### `kb sync`

```
smartrag run kb sync --kb ID [--source SID|sink] [--dry-run] [--force]
                     [--embedding-parallelism N] [--adapt-to-rate-limit=true|false]
                     [--refresh SECONDS] [--exit-with-parent=true|false]
```

Brings the index in line with the sources: indexes files that are new or changed since the
last sync, and removes the documents whose files are gone or no longer match the source's
`include`/`exclude` patterns, and those of sources no longer registered. Unchanged files
are not read again. An interrupted sync (exit 130) continues where it stopped when run again.
Before any file, it reads each source's attributes folder and applies it if it changed (see
[Attributes](#attributes)).

| Option | Default | Meaning |
|---|---|---|
| `--source SID` | all sources and the sink | Sync only this source (`sink`: only the ingested copies). Documents of other sources are left as they are |
| `--dry-run` | off | List, extract and chunk the files, and report documents, chunks and estimated tokens. Calls no embedding endpoint and writes nothing |
| `--force` | off | Re-extract every file, changed or not. Chunks whose text did not change keep their vectors, so only changed text is embedded |
| `--embedding-parallelism N` | the kb's `embedding.parallelism` (1 unless set) | Embedding requests in flight at once, each in its own worker process. Raise it (e.g. to 4) for a remote endpoint, so the GPU does not wait for the network |
| `--adapt-to-rate-limit=true\|false` | `true` | `true`: wait out rate limits (429, or 503 while a model starts), as described under [Embedding providers](#embedding-providers). `false`: stop at the first one |
| `--refresh SECONDS` | `1` | How often the progress bar is redrawn (at least 0.1) |
| `--exit-with-parent=true\|false` | `false` | `true`: stop cleanly when stdin closes. For a parent program that starts the sync and may exit, such as an agent application like Kari |

Output: on a terminal, a progress bar on stderr, then a summary:

```
[######----------------------------------]   15.5%  5600 / 36063  01:12:09 / 06:33:10  52.4M tokens  43.6M/h
```

That is files done / all files, time elapsed / estimated time left, the tokens sent for
embedding so far and their rate per hour over the last stretch. The bar counts files, so a
large file holds it still while its chunks are embedded. The tokens are the provider's own
count; a provider that gives none is counted by Smart RAG's estimate, and the figure starts
with `~`. Reused vectors cost nothing and are not counted. The summary's `tokens` line
gives the total and says whether the provider counted it, and an `attributes` line per
source says what was applied and what was skipped.

Not on a terminal: one line per document, with its tokens
(`[812/36063] data/yuho/…/S100U9OX.md: added, 157 chunks, 58,210 tokens`). With
`--format json`: one JSON object per event, one per line (a `document` event carries
`tokens` and `tokens_estimated`, the part of it Smart RAG estimated), and the report as the
last line, with `tokens_embedded` and `tokens_embedded_estimated`.

```sh
smartrag run kb sync --kb legal
smartrag run kb sync --kb legal --dry-run                     # how much would be embedded
smartrag run kb sync --kb edinet --embedding-parallelism 4    # remote endpoint
smartrag run kb sync --kb legal --source kitei --force        # after changing extraction
```

#### `kb ingest`

```
smartrag run kb ingest --kb ID PATH… [--recursive=true|false] [--dry-run]
                       [--embedding-parallelism N] [--adapt-to-rate-limit=true|false]
                       [--exit-with-parent=true|false]
```

Copies the named files into the kb's sink (`<home>/ID/sink/`), then syncs the sink. The
copies are snapshots: later edits to the originals are not picked up (to follow a folder,
use `source add`). Only file types Smart RAG can read are copied. The same content twice,
under any name, is copied once.

| Option | Default | Meaning |
|---|---|---|
| `PATH…` | required | One or more files or folders |
| `--recursive=true\|false` | `false` | `true`: copy the files inside the named folders, at every depth. With `false`, naming a folder is refused before anything is copied |
| `--dry-run` | off | Report the files, bytes, chunks and tokens; copy and embed nothing |
| others | | As for `kb sync` |

Exit 3 also when a named file was skipped (unsupported type, unreadable).

```sh
smartrag run kb ingest --kb legal ~/Downloads/契約書_2026.pdf ~/Downloads/覚書.docx
smartrag run kb ingest --kb legal ~/Documents/規程 --recursive=true
```

#### `kb reindex`

```
smartrag run kb reindex --kb ID --refresh-stale-only [--dry-run] [COMMON]
smartrag run kb reindex --kb ID --rebuild [--yes] [--dry-run] [COMMON]
```

`COMMON` is `--embedding-parallelism N`, `--adapt-to-rate-limit=true|false` and
`--exit-with-parent=true|false`, as for `kb sync`. Give exactly one of the two modes.

| Option | Meaning |
|---|---|
| `--refresh-stale-only` | Re-process only the documents built by an older Smart RAG pipeline (a new extractor, chunker or chunk setting). New, changed and deleted files are left for the next `kb sync` |
| `--rebuild` | Build a complete new index beside the live one from all files, then swap it in. Search keeps using the old index meanwhile. Vectors are reused where the vector space is unchanged. This is how a kb moves to a model in another vector space. A stopped rebuild is continued by the next `--rebuild`. Asks first; `--yes` skips the question |
| `--dry-run` | Report the work; change nothing |

```sh
smartrag run kb reindex --kb legal --refresh-stale-only
smartrag run kb reindex --kb legal --rebuild --yes
```

#### `kb search`

```
smartrag run kb search --kb ID QUERY [-k|--limit N] [--page N | --cursor TOKEN]
                       [--search-mode hybrid|keyword|vector] [--group-by chunk|document]
                       [--max-per-doc N] [--source SID]… [--type EXT]… [--path PREFIX]
                       [--modified-after DATE] [--modified-before DATE] [--attr FILTER]…
                       [--include highlights|text]… [--snippet-chars N]
```

Returns ranked passages (chunks) with snippets and citations. It returns evidence, never an
answer.

| Option | Default | Meaning |
|---|---|---|
| `QUERY` | required | Natural language. A part in double quotes must appear as an exact phrase on the keyword side; quote the whole query for the shell: `'"年次有給休暇" 繰り越し'` |
| `-k`, `--limit N` | `10` | Results per page, 1–50 |
| `--page N` | `1` | Page number. Use either `--page` or `--cursor`, not both |
| `--cursor TOKEN` | – | The `next_cursor` of a previous `--format json` result: the next page of exactly that search. For programs |
| `--search-mode` | `hybrid` | `hybrid`: keyword and vector results merged; `keyword`: words only; `vector`: meaning only |
| `--group-by` | `chunk` | `chunk`: passages; `document`: one result per document |
| `--max-per-doc N` | `3` | At most N passages from one document (1–50) |
| `--source SID…` | all | Only these sources. Repeat for several: any of them |
| `--type EXT…` | all | Only these file types, by extension without the dot: `pdf`, `md`, `docx`. Repeat for several: any of them |
| `--path PREFIX` | – | Only files whose path inside their source's folder starts with `PREFIX`; end it with `/` to mean a folder. The text output prints `SID/PATH`: give only the `PATH` part. For `kitei/人事/就業規則.pdf`, that is `--path 人事/` |
| `--modified-after DATE` | – | Files last modified at or after `DATE` (local time): `2026-04-01`, `2026-04-01T09:00:00` or `"2026-04-01 09:00:00"` |
| `--modified-before DATE` | – | Files last modified before `DATE` (not including it) |
| `--attr FILTER…` | – | Attributes, see below |
| `--include highlights\|text…` | none | Extra fields: `highlights` (the start and end of each match in the snippet, for programs), `text` (the whole chunk). Repeat for both |
| `--snippet-chars N` | `300` | Snippet length in characters, 50–2000 |

Different filters combine with AND: `--source kitei --type pdf` means PDFs of `kitei`.

`--attr` filters on a document's attributes (see [Attributes](#attributes)). Every
attribute can be used; one the source's `schema.json` does not declare is a string.

| `FILTER` | Matches documents whose attribute |
|---|---|
| `KEY=VALUE` | equals `VALUE`, read as the key's type (`2026-03-31`, `7203`, `true`) |
| `KEY=A --attr KEY=B` | equals `A` or `B`: repeating a key means OR (at most 100 values) |
| `KEY=FROM..TO` | is from `FROM` to `TO`, both included. Integer, number and date keys only; for a string key, `..` is part of the value |
| `KEY=FROM..` / `KEY=..TO` | is at least `FROM` / at most `TO` |

Different keys combine with AND. A range cannot be combined with other values of the same key.

Text output, per result: its rank, title, page or sheet location and outline path (OPath),
and which side found it (`keyword 1 · vector 3` are its ranks there); the snippet; the
source and file path and the `chunk` id (for `kb read`); the link and attributes, if
any. When the embedding endpoint cannot be reached, a `note:` line says that the
results are keyword-only.

```sh
smartrag run kb search --kb legal "有給休暇の繰り越し"
smartrag run kb search --kb legal '"年次有給休暇" 繰り越し' --type pdf -k 5
smartrag run kb search --kb legal "出張旅費" --source kitei --path 人事/ --modified-after 2026-04-01
smartrag run kb search --kb edinet "為替変動リスク" --attr sec_code=7203 --attr fiscal_year_end=2026-03-31
smartrag run kb search --kb edinet "研究開発費" --attr sec_code=6758 --attr sec_code=7751 --attr fiscal_year_end=2025-01-01..2026-12-31
smartrag run kb search --kb legal "経費精算" --group-by document --page 2
```

#### `kb read`

```
smartrag run kb read --kb ID --chunk CHUNK_ID [--scope chunk|neighbors|section|document]
                     [--window N] [--max-chars N] [--page N | --cursor TOKEN]
```

Reads a search result with its context.

| Option | Default | Meaning |
|---|---|---|
| `--chunk CHUNK_ID` | required | A chunk id from `kb search` (`chunk` in the text output, `chunk_id` in JSON) |
| `--scope` | `chunk` | `chunk`: the chunk alone; `neighbors`: with `--window` chunks on each side; `section`: the chunks around it under the same headings (a document without headings: the whole document); `document`: the whole document |
| `--window N` | `1` | For `--scope neighbors`: chunks on each side, 0–20 |
| `--max-chars N` | `4000` | Characters per page, 200–100,000. Pages break between chunks |
| `--page N` / `--cursor TOKEN` | page 1 | As for `kb search` |

```sh
smartrag run kb read --kb edinet --chunk d_ae50b334255c36406537:7a7b5aa3299b3646 --scope section
smartrag run kb read --kb edinet --chunk d_ae50b334255c36406537:7a7b5aa3299b3646 --scope document --page 2
```

### source commands

A **source** is a folder (or one file) that `kb sync` follows. Smart RAG reads the files
where they are and copies nothing. A source has no settings command: to change its
`include`, `exclude` or `recursive` after adding it, edit the kb's `config.yaml`.

#### `source list`

```
smartrag run source list --kb ID
```

Lists the kb's sources and its `sink` as a table: id, kind, location (marked `missing` if
it does not exist now), whether subfolders are included, and the include and exclude
patterns. The last line lists the patterns that are always skipped.

#### `source add`

```
smartrag run source add --kb ID --source SID LOCATION [--include GLOB]… [--exclude GLOB]…
                        [--recursive=true|false] [--allow-missing=true|false]
```

Registers `LOCATION` as a new source. The next `kb sync` indexes it.

| Option | Default | Meaning |
|---|---|---|
| `LOCATION` | required | A folder or a file: a path (`~/Documents/規程`) or a `file://` URI. It cannot be inside the kb's own folder, or already another source |
| `--source SID` | required | The new source's id |
| `--include GLOB…` | all files | Only files matching at least one of these patterns |
| `--exclude GLOB…` | none | Skip files matching any of these patterns, even if included |
| `--recursive=true\|false` | `true` | `true`: files in subfolders too; `false`: only the folder's own files |
| `--allow-missing=true\|false` | `false` | `true`: accept a location that does not exist now, e.g. a share that is not mounted |

**Patterns** are matched against each file's path relative to `LOCATION`, with `/`
separators, ignoring case. They must not start with `/`.

| Pattern part | Matches |
|---|---|
| `*` | Any characters within one folder or file name |
| `?` | One character, not `/` |
| `**/` | Zero or more folders: `**/*.pdf` is every PDF at any depth, including the top folder |
| `**` at the end | Everything below: `drafts/**` is everything in `drafts/`, which is then not even walked |
| `[abc]`, `[!abc]` | One of these characters, or none of them |

Some files are always skipped, whatever the patterns: secrets (`.env`, keys, `.ssh`), tool
folders (`.git`, `node_modules`, `.venv`), lock files and attributes folders (`*.attributes`).
`source list` prints the full list.

```sh
smartrag run source add --kb legal --source kitei ~/Documents/規程 --include '**/*.pdf' --include '**/*.docx'
smartrag run source add --kb edinet --source data /data/EDINET_DATA/rag/data --exclude 'drafts/**'
smartrag run source add --kb legal --source share /mnt/legal-share --allow-missing=true
```

#### `source remove`

```
smartrag run source remove --kb ID --source SID [--yes]
```

Unregisters the source and removes its documents from the index. Its files are not touched.
Asks first; `--yes` skips the question. The `sink` cannot be removed.

```sh
smartrag run source remove --kb legal --source share --yes
```

#### `source test`

```
smartrag run source test --kb ID --source SID|sink [--sample N]
```

Walks the source the way `kb sync` would, without reading file contents or changing
anything, and reports how many files would be read, by type, which were skipped and why,
and the paths that could not be read. `--sample N` (default 10, 0–1000) lists that many of
the matching paths. Run it after `source add`, or after editing a source's patterns.

```sh
smartrag run source test --kb edinet --source filings --sample 20
```

### `provider list`

```
smartrag run provider list
```

Lists the embedding providers and models a kb can use, with each provider's URL, the
environment variable for its key, and its data terms. Makes no network calls. `--format
json` gives the same as data.

### Notes

- Secrets appear in `config.yaml` only as `${ENV_VAR}` references; a literal value is rejected.
- Commands edit `config.yaml` in place: your comments, key order and quoting are kept.
- Supported file types: PDF, DOCX/DOCM, PPTX/PPTM, HTML, Markdown, reStructuredText,
  AsciiDoc, RTF, e-mail (`.eml`), JSON, JSON Lines, YAML and XML (flattened to `path: value`
  records), plain text and logs, and tables (CSV, TSV, XLSX, XLSM, XLS, XLSB), which are
  converted to TSV internally. Tables inside HTML and Markdown (pipe tables) are read as
  tables too: a large one repeats its header row, with its caption (such as `（単位：百万円）`),
  on every chunk, and a small one is kept whole with the text around it. Any other file is
  indexed as plain text when its first 8 KiB are text, so source code and configuration
  files need no list.

### Attributes

Attributes are facts about a document that are not in the file itself, such as a company
code or a fiscal year. Whoever publishes the files (for example the program that downloaded
them) writes them into a folder **beside** the source folder, named after it with
`.attributes` added:

```
rag/
├── data/                  the source folder (source add … rag/data)
└── data.attributes/
    ├── attributes.jsonl   required
    └── schema.json        optional
```

Every `kb sync` and `kb reindex` reads this folder and applies it if it changed; there is no
separate command, and the location cannot be changed. A source without such a folder has no
attributes. The folder is never indexed as documents, even by a source that contains it.
Files ingested into the sink have no attributes.

**Agents see every attribute.** Whatever `attributes.jsonl` holds is shown to agents with
each search result, and can go from there to the model's provider and into answers. Leave
out anything agents should not see.

- **`attributes.jsonl`** is JSON Lines: one JSON object per line, one line per document.
  `path` is required: the file's path relative to the source folder, with `/` separators.
  Every other key is an attribute:
  ```json
  {"path": "yuho/7203_トヨタ自動車/2026-03期_有価証券報告書_S100Y8NY.md", "title": "トヨタ自動車 有価証券報告書 2026年3月期", "link": "https://example.com/S100Y8NY", "sec_code": "7203", "fiscal_year_end": "2026-03-31"}
  ```
  `title` replaces the document's extracted title in search results and citations, and
  `link` (an `http` or `https` URL) becomes the citation's link. Values are a string,
  number, boolean or `null`, at most 500 characters; lists and objects are not accepted.
- **`schema.json`** says what each attribute is, in a subset of JSON Schema: per property
  a `type` (`string`, `integer`, `number` or `boolean`), optionally a `format` for strings
  (`date`, `date-time` or `uri`), an `enum` of allowed values, and a `description` (up to
  200 characters; a longer one is cut). Other JSON Schema keywords are ignored.
- **Every attribute can be a search filter** (`kb search --attr`, the MCP `attributes`
  argument); its type decides whether ranges work. A key the schema does not declare, or
  every key when there is no `schema.json`, is a string: `7203` is read as `"7203"` and
  `true` as `"true"`.
- **Reserved names.** Attribute names Smart RAG uses for its own result fields (`doc_id`,
  `chunk_id`, `kb_id`, `source_id`, `file_path`, `file_type`, `page`, `text` and a few more)
  are renamed with the prefix `_attr_`: `doc_id` becomes `_attr_doc_id`. A name of your own
  must not start with `_attr_`.
- **A mistake skips only the bad part.** A line that is not a JSON object, has no or a bad
  `path`, repeats an earlier path, or names a file the source does not have is skipped. A
  value that is nested, too long, or does not match the schema is dropped, and the rest of
  its line is kept.
- **If `attributes.jsonl` cannot be read at all** (for example while it is being rewritten),
  the attributes applied before are kept. A `schema.json` that cannot be read is ignored on
  its own. Deleting the folder removes the source's attributes at the next sync.
- **Nothing is re-extracted or re-embedded** when only attributes change, so that sync takes
  seconds.

The sync summary has one `attributes` line per source, followed by what was renamed and
skipped:

```
attributes data: 36,060 rows applied
           renamed doc_id to _attr_doc_id: Smart RAG uses the name
           3 lines skipped: not valid JSON (812, 9051, 20113)
```

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

## Embedding providers

A kb picks a provider and one of its models from the catalog (`provider list`,
`src/smartrag/embedding/providers.yaml`); everything else comes from the catalog:

| `provider_id` | Models | URL | Key |
|---|---|---|---|
| `gotoai` | `ruri-v3-310m@8k`, `harrier-oss-v1-0.6b@8k`, `harrier-oss-v1-0.6b@32k` | `${GOTOAI_EMBED_BASE_URL}` | `${GOTOAI_EMBED_API_KEY}` |
| `voyage_ai` | `voyage-4`, `voyage-4-lite`, `voyage-4-large` | fixed | `${VOYAGE_API_KEY}` |
| `openai` | `text-embedding-3-small`, `text-embedding-3-large` | fixed | `${OPENAI_API_KEY}` |
| `google_gemini` | `gemini-embedding-2` | fixed | `${GEMINI_API_KEY}` |
| `cohere` | `embed-v5.0-pro`, `embed-v5.0-fast` (one vector space) | fixed | `${CO_API_KEY}` |
| `mistral` | `mistral-embed` | fixed | `${MISTRAL_API_KEY}` |
| `custom` | any OpenAI-compatible endpoint, described in the config | `${SMARTRAG_EMBED_BASE_URL}` | `${SMARTRAG_EMBED_API_KEY}` |

```yaml
embedding:
  provider_id: voyage_ai
  model_profile_id: voyage-4-lite
  api_key: ${VOYAGE_API_KEY}     # optional: the provider's usual variable is the default
  dimensions: 512                # optional, for a model that allows shorter vectors
```

- The index records its **vector space**: the model's compatibility id, the dimensions, and how
  queries and documents are marked. A model in another space needs `kb reindex --rebuild`; one
  in the same space (harrier `@32k` → `@8k`, `voyage-4` → `voyage-4-lite`) needs nothing.
- Only `gotoai` and `custom` take a `base_url`; a cloud provider's URL is fixed, so its key
  cannot be sent elsewhere. A cloud provider without its key fails before sending anything;
  a GotoAI or custom endpoint without authentication simply gets no Authorization header.
- **Rate limits** (`kb sync`, `ingest`, `reindex`; on by default, `--adapt-to-rate-limit=false`
  turns it off): a 429, or GotoAI's 503 while a model starts, is waited out (`Retry-After`, or
  a backoff doubling from 2 s to 60 s, for up to 10 minutes per request); a request still
  refused after a minute is split in halves, and later ones stay that small; fewer requests
  are kept in flight. The progress line shows each wait, the report counts them. Search does
  not wait: it falls back to keyword results at once.
- `custom` also states `model_canonical_name`, `dimensions`, `max_input_tokens` and how it
  marks queries: `dialect: openai_plain` (default), `openai_input_type`, or `openai_prefix`
  with `prefix: {query: "…{text}", document: "…{text}"}`.

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
| `describe_sources(kb_id)` | Sources, file types, date range, last sync; each source's attributes and what they mean; the filters `search` accepts |
| `search(kb_id, query, k, cursor, source, file_type, modified_after, modified_before, attributes)` | Hybrid search: passages with snippets and citations, never an answer. `attributes` filters on attributes: a value, a list, or `{from, to}` |
| `read(kb_id, chunk_id, scope, cursor)` | A passage widened to `neighbors`, its `section` or the `document`, in pages |

- **Read-only.** Agents never sync, ingest or change a kb.
- **Agents see only usable kbs:** local, indexed with at least one document, built with the
  configured model's vector space, and with a description. The set is fixed at startup (it is
  in the server instructions and the `kb_id` enum) and re-checked on every call; any other
  kb id is refused as `not_found`. `--allow-kb` narrows it further.
- **Citations carry the document's full path** (`file_path`), unless the kb's config sets
  `agent.show_paths: false`, and its `link`; error messages never carry paths. Titles are
  cut to file names.
- **Agents see every attribute** of a document, as the publisher wrote it (see
  [Attributes](#attributes)).
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
  embedding/       provider catalog, resolve, dialects (marking, wire), the HTTP embedder
                   kb/sync.py: the sync
  search/          kb search (hybrid, RRF, paging) and kb read
  agent/           the agent's view: usable kbs, results without paths, fast-failing embedders
  mcp/             MCP server over agent/ (`smartrag mcp`)
  api/             FastAPI app: query routes, ops routes, auth, error shape
  cli/             `smartrag run` (kb, source, provider), `smartrag serve` and `smartrag mcp`
```
