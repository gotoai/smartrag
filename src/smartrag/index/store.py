"""The SQLite index of one kb: `<kb>/index.db`.

One file, WAL mode, one writer (the sync, under the kb lock), any number of readers.

| Table | Holds |
|---|---|
| `meta` | schema version, the embedding profile and dimensions every vector came from |
| `documents` | one row per file seen: ok, skipped or failed, with the reason; skipped and failed files are not retried until they change |
| `contents` | each distinct extracted text once, by content hash (identical files share it) |
| `chunks` | spans of a document's text, their heading path and location, and their vector |
| `chunk_fts` | FTS5, contentless, over bigram-split text: BM25 for the keyword side |

Vectors are float32 BLOBs. A chunk whose embedded text is unchanged keeps its vector
across re-syncs (`vectors_for`), so editing one paragraph re-embeds one or two chunks.
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from smartrag.errors import ConfigError

INDEX_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    doc_id           TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL,
    rel_path         TEXT NOT NULL,
    uri              TEXT NOT NULL,
    file_type        TEXT NOT NULL,
    size             INTEGER NOT NULL,
    mtime_ns         INTEGER NOT NULL,
    content_hash     TEXT,
    pipeline_version TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('ok', 'skipped', 'failed')),
    reason           TEXT,
    title            TEXT,
    pages            INTEGER,
    chunk_count      INTEGER NOT NULL DEFAULT 0,
    indexed_at       TEXT NOT NULL,
    UNIQUE (source_id, rel_path)
);
CREATE TABLE IF NOT EXISTS contents (
    content_hash TEXT PRIMARY KEY,
    text         TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    chunk_id     TEXT NOT NULL UNIQUE,
    doc_id       TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    start        INTEGER NOT NULL,
    end          INTEGER NOT NULL,
    prefix       TEXT NOT NULL DEFAULT '',
    heading_path TEXT NOT NULL DEFAULT '[]',
    page         INTEGER,
    page_end     INTEGER,
    location     TEXT,
    tokens       INTEGER NOT NULL,
    text_hash    TEXT NOT NULL,
    profile      TEXT NOT NULL,
    embedding    BLOB
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id, ordinal);
CREATE INDEX IF NOT EXISTS chunks_text_hash ON chunks (text_hash, profile);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    heading, body, path,
    content = '', contentless_delete = 1,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


@dataclass
class DocumentRow:
    doc_id: str
    source_id: str
    rel_path: str
    uri: str
    file_type: str
    size: int
    mtime_ns: int
    content_hash: str | None
    pipeline_version: str
    status: str
    reason: str | None
    title: str | None
    pages: int | None
    chunk_count: int
    indexed_at: str


@dataclass
class ChunkRow:
    chunk_id: str
    ordinal: int
    start: int
    end: int
    prefix: str
    heading_path: list[str]
    page: int | None
    page_end: int | None
    location: str | None
    tokens: int
    text_hash: str
    profile: str
    embedding: Sequence[float] | None
    fts_heading: str
    fts_body: str
    fts_path: str


class IndexStats(BaseModel):
    documents: int
    ok: int
    skipped: int
    failed: int
    chunks: int
    embedded: int
    profile: str | None
    dimensions: int | None
    last_indexed_at: str | None


def pack_vector(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack_vector(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


class SqliteIndex:
    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        self.db = connection
        self.path = path

    @classmethod
    def open(cls, path: Path, *, readonly: bool = False) -> SqliteIndex:
        if readonly:
            connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30)
        else:
            connection = sqlite3.connect(path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if not readonly:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)
            path.chmod(0o600)  # the index holds the full extracted text of the documents
            index = cls(connection, path)
            version = index.meta("schema_version")
            if version is None:
                index.set_meta("schema_version", str(INDEX_SCHEMA_VERSION))
                connection.commit()
            elif int(version) != INDEX_SCHEMA_VERSION:
                raise ConfigError(f"{path} has index schema {version}; this smartrag "
                                  f"reads {INDEX_SCHEMA_VERSION}")
            return index
        return cls(connection, path)

    def close(self) -> None:
        self.db.close()

    # --- meta ----------------------------------------------------------------------

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                        "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))

    # --- documents -----------------------------------------------------------------

    def documents(self, source_id: str | None = None) -> dict[tuple[str, str], DocumentRow]:
        sql = "SELECT * FROM documents"
        args: tuple[str, ...] = ()
        if source_id is not None:
            sql += " WHERE source_id = ?"
            args = (source_id,)
        return {(row["source_id"], row["rel_path"]): DocumentRow(**dict(row))
                for row in self.db.execute(sql, args)}

    def source_ids(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT DISTINCT source_id FROM documents")}

    def vectors_for(self, text_hashes: Iterable[str], profile: str) -> dict[str, bytes]:
        found: dict[str, bytes] = {}
        hashes = list(set(text_hashes))
        for i in range(0, len(hashes), 500):
            batch = hashes[i:i + 500]
            marks = ",".join("?" * len(batch))
            for row in self.db.execute(
                    f"SELECT text_hash, embedding FROM chunks WHERE profile = ? AND "
                    f"embedding IS NOT NULL AND text_hash IN ({marks})", (profile, *batch)):
                found.setdefault(row["text_hash"], row["embedding"])
        return found

    def _delete_chunks(self, doc_id: str) -> None:
        ids = [row[0] for row in self.db.execute("SELECT id FROM chunks WHERE doc_id = ?",
                                                 (doc_id,))]
        for i in range(0, len(ids), 500):
            batch = ids[i:i + 500]
            marks = ",".join("?" * len(batch))
            self.db.execute(f"DELETE FROM chunk_fts WHERE rowid IN ({marks})", batch)
        self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def write_document(self, doc: DocumentRow, text: str | None,
                       chunks: Sequence[ChunkRow]) -> None:
        """Replace a document, its text and its chunks in one transaction: a document is
        never half-written, so an interrupted sync resumes by simply running again."""
        with self.db:
            self._delete_chunks(doc.doc_id)
            if text is not None and doc.content_hash is not None:
                self.db.execute("INSERT OR IGNORE INTO contents (content_hash, text) "
                                "VALUES (?, ?)", (doc.content_hash, text))
            self.db.execute(
                "INSERT INTO documents (doc_id, source_id, rel_path, uri, file_type, size, "
                "mtime_ns, content_hash, pipeline_version, status, reason, title, pages, "
                "chunk_count, indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (doc_id) DO UPDATE SET uri = excluded.uri, "
                "file_type = excluded.file_type, size = excluded.size, "
                "mtime_ns = excluded.mtime_ns, content_hash = excluded.content_hash, "
                "pipeline_version = excluded.pipeline_version, status = excluded.status, "
                "reason = excluded.reason, title = excluded.title, pages = excluded.pages, "
                "chunk_count = excluded.chunk_count, indexed_at = excluded.indexed_at",
                (doc.doc_id, doc.source_id, doc.rel_path, doc.uri, doc.file_type, doc.size,
                 doc.mtime_ns, doc.content_hash, doc.pipeline_version, doc.status, doc.reason,
                 doc.title, doc.pages, len(chunks), doc.indexed_at))
            for chunk in chunks:
                cursor = self.db.execute(
                    "INSERT INTO chunks (chunk_id, doc_id, ordinal, start, end, prefix, "
                    "heading_path, page, page_end, location, tokens, text_hash, profile, "
                    "embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (chunk.chunk_id, doc.doc_id, chunk.ordinal, chunk.start, chunk.end,
                     chunk.prefix, json.dumps(chunk.heading_path, ensure_ascii=False),
                     chunk.page, chunk.page_end, chunk.location, chunk.tokens,
                     chunk.text_hash, chunk.profile,
                     pack_vector(chunk.embedding) if chunk.embedding is not None else None))
                self.db.execute("INSERT INTO chunk_fts (rowid, heading, body, path) "
                                "VALUES (?, ?, ?, ?)",
                                (cursor.lastrowid, chunk.fts_heading, chunk.fts_body,
                                 chunk.fts_path))

    def touch_document(self, doc_id: str, size: int, mtime_ns: int) -> None:
        """A file whose bytes did not change (only its timestamp did)."""
        with self.db:
            self.db.execute("UPDATE documents SET size = ?, mtime_ns = ? WHERE doc_id = ?",
                            (size, mtime_ns, doc_id))

    def delete_documents(self, doc_ids: Iterable[str]) -> int:
        count = 0
        with self.db:
            for doc_id in doc_ids:
                self._delete_chunks(doc_id)
                count += self.db.execute("DELETE FROM documents WHERE doc_id = ?",
                                         (doc_id,)).rowcount
        return count

    def delete_source(self, source_id: str) -> int:
        return self.delete_documents(row.doc_id for row in self.documents(source_id).values())

    def collect_garbage(self) -> None:
        with self.db:
            self.db.execute("DELETE FROM contents WHERE content_hash NOT IN "
                            "(SELECT content_hash FROM documents WHERE content_hash IS NOT NULL)")

    def optimize(self) -> None:
        """Merge FTS5's index segments after a large sync: faster queries, same results."""
        with self.db:
            self.db.execute("INSERT INTO chunk_fts (chunk_fts) VALUES ('optimize')")

    # --- reading -------------------------------------------------------------------

    def stats(self) -> IndexStats:
        counts = {row["status"]: row["n"] for row in self.db.execute(
            "SELECT status, COUNT(*) AS n FROM documents GROUP BY status")}
        chunks = self.db.execute("SELECT COUNT(*), COUNT(embedding) FROM chunks").fetchone()
        last = self.db.execute("SELECT MAX(indexed_at) FROM documents").fetchone()[0]
        dimensions = self.meta("dimensions")
        return IndexStats(
            documents=sum(counts.values()), ok=counts.get("ok", 0),
            skipped=counts.get("skipped", 0), failed=counts.get("failed", 0),
            chunks=chunks[0], embedded=chunks[1], profile=self.meta("embedding_profile"),
            dimensions=int(dimensions) if dimensions else None, last_indexed_at=last)

    def document_text(self, content_hash: str) -> str | None:
        row = self.db.execute("SELECT text FROM contents WHERE content_hash = ?",
                              (content_hash,)).fetchone()
        return None if row is None else str(row["text"])
