"""The vectors of a kb as one normalised matrix, cached under `cache/`.

Search reads every chunk vector (brute force, design memo §3.3). Pulling thousands of
BLOBs out of SQLite on every query is slow, so the matrix is built once per index
generation, normalised (ruri-v3's vectors are not unit length), and saved as `.npy` files
that later searches memory-map. The SQLite index stays the source of truth; the cache is
disposable and rebuilt whenever the generation or the profile changes. If the cache
cannot be written, the matrix is simply used from memory.

The query-embedding cache keeps one vector per (profile, query) under a hash of the two,
so a second page or a repeated query does not call the endpoint again. It stores no query
text.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartrag.index import SqliteIndex

_CACHE_SCHEMA = 1


@dataclass
class VectorMatrix:
    rowids: np.ndarray
    matrix: np.ndarray

    @property
    def dimensions(self) -> int:
        return int(self.matrix.shape[1]) if self.matrix.ndim == 2 and self.matrix.size else 0


def _build(store: SqliteIndex) -> VectorMatrix:
    rowids: list[int] = []
    vectors: list[np.ndarray] = []
    for row in store.db.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"):
        rowids.append(int(row[0]))
        vectors.append(np.frombuffer(row[1], dtype=np.float32))
    if not vectors:
        return VectorMatrix(np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32))
    matrix = np.vstack(vectors).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return VectorMatrix(np.asarray(rowids, dtype=np.int64), matrix / norms)


def load_vectors(store: SqliteIndex, cache_dir: Path, profile: str) -> VectorMatrix:
    stamp = {"schema": _CACHE_SCHEMA, "generation": store.generation(), "profile": profile}
    meta_path = cache_dir / "vectors.json"
    matrix_path, ids_path = cache_dir / "vectors.npy", cache_dir / "vector_ids.npy"
    try:
        if json.loads(meta_path.read_text(encoding="utf-8")) == stamp:
            return VectorMatrix(np.load(ids_path), np.load(matrix_path, mmap_mode="r"))
    except (OSError, ValueError):
        pass
    built = _build(store)
    try:
        cache_dir.mkdir(mode=0o700, exist_ok=True)
        for path, array in ((matrix_path, built.matrix), (ids_path, built.rowids)):
            tmp = path.with_name(f".{path.stem}-{os.getpid()}.npy")
            np.save(tmp, array)
            os.replace(tmp, path)
        tmp_meta = meta_path.with_name(f".vectors-{os.getpid()}.json")
        tmp_meta.write_text(json.dumps(stamp), encoding="utf-8")
        os.replace(tmp_meta, meta_path)
    except OSError:
        pass  # a read-only home: use the matrix from memory
    return built


class QueryVectorCache:
    def __init__(self, cache_dir: Path) -> None:
        self.path = cache_dir / "query_vectors.db"

    @staticmethod
    def _key(profile: str, query: str) -> str:
        return hashlib.sha256(f"{profile}\0{query}".encode()).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("CREATE TABLE IF NOT EXISTS q (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        return db

    def get(self, profile: str, query: str) -> np.ndarray | None:
        if not self.path.is_file():
            return None
        try:
            db = self._connect()
            try:
                row = db.execute("SELECT vector FROM q WHERE key = ?",
                                 (self._key(profile, query),)).fetchone()
            finally:
                db.close()
        except sqlite3.Error:
            return None
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def put(self, profile: str, query: str, vector: np.ndarray) -> None:
        try:
            self.path.parent.mkdir(mode=0o700, exist_ok=True)
            db = self._connect()
            try:
                with db:
                    db.execute("INSERT OR REPLACE INTO q (key, vector) VALUES (?, ?)",
                               (self._key(profile, query),
                                np.asarray(vector, dtype=np.float32).tobytes()))
            finally:
                db.close()
            self.path.chmod(0o600)
        except (sqlite3.Error, OSError):
            pass
