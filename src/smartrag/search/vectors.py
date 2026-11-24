"""The vectors of a kb, cached under `cache/vectors/` in partitions.

Search scores chunk vectors by brute force. Pulling them out of SQLite on every query is
slow, so they are cached as normalised float32 matrices (ruri-v3's vectors are not unit
length) that searches memory-map. The SQLite index stays the source of truth; the cache is
disposable.

The cache is split by insertion order: each partition is a range of chunk ids holding at
most `PARTITION_ROWS` chunks, so building one needs bounded memory, and new chunks, whose
ids are above every existing one, go to the last partition. Chunks are never updated, only
deleted and inserted with their vectors, so a partition's content is fixed by its rows' ids
and text hashes; its signature is computed from those alone (read from an index, without
touching a vector), and a partition is rebuilt only when its signature changes. A partition
under half full is merged with a neighbour when the two fit in one.

`parts.json` lists the partitions with the vector space and the index generation they were
checked at. A search whose generation matches uses the files as they are; otherwise it
refreshes them first. A writer (sync, under the kb lock) refreshes at its end and removes
files no longer listed; a search never removes any, so a concurrent reader keeps its files.
If the cache cannot be written, the partitions are simply built in memory.

The query-embedding cache keeps one vector per (vector space, query) under a hash of the two,
so a second page or a repeated query does not call the endpoint again. It stores no query
text.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smartrag.index import SqliteIndex

PARTITION_ROWS = 262_144
"""Chunks per partition: 768 MB at 768 dimensions, 1 GB at 1,024."""

GATHER_SHARE = 0.01
"""Under a filter, a partition in which at most this share of rows match has only those
rows scored; above it, scoring the whole partition and masking is faster (measured on a
million rows)."""

_CACHE_SCHEMA = 3
_DIR = "vectors"
_LIST = "parts.json"
_LEGACY = ("vectors.npy", "vector_ids.npy", "vectors.json")
_MIX = np.uint64(0x9E3779B97F4A7C15)
_BATCH = 100_000


@dataclass
class Partition:
    start: int
    """The first chunk id of its range; the range ends where the next partition starts."""
    rows: int
    """Chunks in the range, with or without a vector."""
    signature: str

    @property
    def stem(self) -> str:
        return f"part-{self.start:012d}-{self.signature}"


@dataclass
class _Loaded:
    rowids: np.ndarray
    matrix: np.ndarray


@dataclass
class VectorCache:
    partitions: list[_Loaded]
    dimensions: int

    @property
    def rows(self) -> int:
        return sum(int(p.rowids.size) for p in self.partitions)

    def top(self, query: np.ndarray, pool: int, allowed: np.ndarray | None = None) -> list[int]:
        """The `pool` chunk ids most similar to a unit `query`, best first, ties by id.
        `allowed` (sorted chunk ids) limits the candidates: partitions without any are not
        read, and in one where few match, only those rows are scored."""
        found_ids: list[np.ndarray] = []
        found_scores: list[np.ndarray] = []
        for part in self.partitions:
            ids, scores = _score(part, query, allowed)
            if ids.size > pool:
                keep = np.argpartition(-scores, pool - 1)[:pool]
                ids, scores = ids[keep], scores[keep]
            found_ids.append(ids)
            found_scores.append(scores)
        if not found_ids:
            return []
        ids, scores = np.concatenate(found_ids), np.concatenate(found_scores)
        order = np.lexsort((ids, -scores))[:pool]
        return [int(i) for i in ids[order]]


def _score(part: _Loaded, query: np.ndarray,
           allowed: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    rowids = part.rowids
    if not rowids.size:
        return rowids, np.zeros(0, dtype=np.float32)
    if allowed is None:
        return rowids, part.matrix @ query
    lo, hi = np.searchsorted(allowed, (rowids[0], rowids[-1] + 1))
    wanted = allowed[lo:hi]
    if not wanted.size:
        return rowids[:0], np.zeros(0, dtype=np.float32)  # not read at all
    positions = np.searchsorted(rowids, wanted)
    positions = positions[rowids[np.minimum(positions, rowids.size - 1)] == wanted]
    if positions.size <= GATHER_SHARE * rowids.size:
        return rowids[positions], part.matrix[positions] @ query
    mask = np.zeros(rowids.size, dtype=bool)
    mask[positions] = True
    return rowids[mask], (part.matrix @ query)[mask]


# --- the partitions ----------------------------------------------------------------


def _signatures(store: SqliteIndex) -> tuple[np.ndarray, np.ndarray]:
    """Every chunk's id, sorted, and a number mixing it with its text hash."""
    sql = "SELECT id, text_hash FROM chunks INDEXED BY chunks_text_hash"
    try:
        cursor = store.db.execute(sql)
    except sqlite3.OperationalError:  # an index from before the text-hash index
        cursor = store.db.execute("SELECT id, text_hash FROM chunks")
    ids_parts: list[np.ndarray] = []
    mix_parts: list[np.ndarray] = []
    while batch := cursor.fetchmany(_BATCH):
        ids = np.fromiter((row[0] for row in batch), dtype=np.int64, count=len(batch))
        hashes = np.fromiter((int(row[1][:16], 16) for row in batch), dtype=np.uint64,
                             count=len(batch))
        ids_parts.append(ids)
        mix_parts.append((ids.astype(np.uint64) * _MIX) ^ hashes)
    if not ids_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.uint64)
    ids, mix = np.concatenate(ids_parts), np.concatenate(mix_parts)
    order = np.argsort(ids, kind="stable")
    return ids[order], mix[order]


def _layout(ids: np.ndarray, previous: list[int]) -> list[int]:
    """Partition starts: the previous ones, with neighbours merged where one is under half
    full and the two fit, and any partition over `PARTITION_ROWS` split."""
    if not ids.size:
        return []
    starts = sorted(set(previous) | {0}) if previous else [0]
    starts[0] = 0  # ids below the old first start (a table emptied and refilled)
    bounds = np.searchsorted(ids, np.asarray(starts[1:], dtype=np.int64))
    counts = np.diff(np.concatenate(([0], bounds, [ids.size]))).tolist()
    merged: list[tuple[int, int]] = []
    for start, count in zip(starts, counts, strict=True):
        if merged:
            last_start, last_count = merged[-1]
            small = min(last_count, count) < PARTITION_ROWS // 2
            if small and last_count + count <= PARTITION_ROWS:
                merged[-1] = (last_start, last_count + count)
                continue
        merged.append((start, count))
    layout: list[int] = []
    for start, count in merged:
        layout.append(start)
        first = int(np.searchsorted(ids, start))
        layout += [int(ids[first + k]) for k in range(PARTITION_ROWS, count, PARTITION_ROWS)]
    return layout


def _partitions(ids: np.ndarray, mix: np.ndarray, starts: list[int]) -> list[Partition]:
    bounds = np.searchsorted(ids, np.asarray(starts, dtype=np.int64)).tolist() + [ids.size]
    parts = []
    for start, lo, hi in zip(starts, bounds, bounds[1:], strict=False):
        total = int(np.sum(mix[lo:hi], dtype=np.uint64)) if hi > lo else 0
        parts.append(Partition(start, hi - lo, f"{hi - lo:x}-{total:016x}"))
    return parts


def _build(store: SqliteIndex, start: int, end: int | None, rows: int) -> _Loaded:
    """One partition's vectors, normalised, in id order: memory for its rows only."""
    sql = "SELECT id, embedding FROM chunks WHERE id >= ? AND embedding IS NOT NULL"
    args: list[int] = [start]
    if end is not None:
        sql += " AND id < ?"
        args.append(end)
    rowids = np.zeros(rows, dtype=np.int64)
    matrix: np.ndarray | None = None
    n = 0
    for row in store.db.execute(sql + " ORDER BY id", args):
        vector = np.frombuffer(row[1], dtype=np.float32)
        if matrix is None:
            matrix = np.zeros((rows, vector.size), dtype=np.float32)
        if n == rows:  # cannot happen within one snapshot; never overrun the arrays
            break
        rowids[n], matrix[n] = row[0], vector
        n += 1
    if matrix is None:
        return _Loaded(np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32))
    matrix = matrix[:n]
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    matrix /= norms[:, None]
    return _Loaded(rowids[:n], matrix)


def _ends(parts: list[Partition]) -> Iterator[tuple[Partition, int | None]]:
    for i, part in enumerate(parts):
        yield part, parts[i + 1].start if i + 1 < len(parts) else None


def _save(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}-{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def _read_list(folder: Path) -> dict | None:
    try:
        data = json.loads((folder / _LIST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == _CACHE_SCHEMA else None


def _open(folder: Path, data: dict) -> VectorCache:
    loaded = []
    for entry in data["parts"]:
        stem = Partition(entry["start"], entry["rows"], entry["signature"]).stem
        loaded.append(_Loaded(np.load(folder / f"{stem}.ids.npy"),
                              np.load(folder / f"{stem}.npy", mmap_mode="r")))
    return VectorCache(loaded, int(data.get("dimensions") or 0))


def refresh_vectors(store: SqliteIndex, cache_dir: Path, space: str, *,
                    prune: bool = False) -> VectorCache:
    """Bring the cache up to date with the index: rebuild the partitions whose chunks
    changed, then list them. `prune` (writers, under the kb lock) also removes files no
    longer listed."""
    with _snapshot(store):
        return _refresh(store, cache_dir, space, prune)


@contextmanager
def _snapshot(store: SqliteIndex) -> Iterator[None]:
    """One read transaction, so the signatures and the vectors come from the same state of
    the index even while a sync commits (WAL)."""
    if store.db.in_transaction:
        yield
        return
    store.db.execute("BEGIN")
    try:
        yield
    finally:
        store.db.execute("ROLLBACK")


def _refresh(store: SqliteIndex, cache_dir: Path, space: str, prune: bool) -> VectorCache:
    folder = cache_dir / _DIR
    previous = _read_list(folder)
    starts = [p["start"] for p in previous["parts"]] \
        if previous and previous.get("space") == space else []
    generation = store.generation()
    ids, mix = _signatures(store)
    parts = _partitions(ids, mix, _layout(ids, starts))
    try:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        writable = True
    except OSError:
        writable = False
    loaded: list[_Loaded] = []
    dimensions = 0
    for part, end in _ends(parts):
        matrix_path, ids_path = folder / f"{part.stem}.npy", folder / f"{part.stem}.ids.npy"
        if writable and matrix_path.is_file() and ids_path.is_file():
            loaded.append(_Loaded(np.load(ids_path), np.load(matrix_path, mmap_mode="r")))
        else:
            built = _build(store, part.start, end, part.rows)
            if writable:
                try:
                    _save(matrix_path, built.matrix)
                    _save(ids_path, built.rowids)
                except OSError:
                    writable = False
            loaded.append(built)
        if loaded[-1].matrix.ndim == 2 and loaded[-1].matrix.size:
            dimensions = int(loaded[-1].matrix.shape[1])
    if writable:
        listing = {"schema": _CACHE_SCHEMA, "space": space, "generation": generation,
                   "dimensions": dimensions,
                   "parts": [{"start": p.start, "rows": p.rows, "signature": p.signature}
                             for p in parts]}
        try:
            tmp = folder / f".{_LIST}-{os.getpid()}.tmp"
            tmp.write_text(json.dumps(listing), encoding="utf-8")
            os.replace(tmp, folder / _LIST)
            if prune:
                _prune(cache_dir, {f"{p.stem}{suffix}" for p in parts
                                   for suffix in (".npy", ".ids.npy")})
        except OSError:
            pass
    return VectorCache(loaded, dimensions)


def _prune(cache_dir: Path, keep: set[str]) -> None:
    folder = cache_dir / _DIR
    for path in folder.glob("part-*"):
        if path.name not in keep:
            path.unlink(missing_ok=True)
    for name in _LEGACY:  # the single-matrix cache of schema 2
        (cache_dir / name).unlink(missing_ok=True)


def load_vectors(store: SqliteIndex, cache_dir: Path, space: str) -> VectorCache:
    """The cache as listed, when it was checked at the index's current generation;
    otherwise refreshed first."""
    folder = cache_dir / _DIR
    data = _read_list(folder)
    if data and data.get("space") == space and data.get("generation") == store.generation():
        try:
            return _open(folder, data)
        except (OSError, ValueError):
            pass  # a writer replaced the files since: refresh
    return refresh_vectors(store, cache_dir, space)


def refresh_after_write(index_path: Path, cache_dir: Path, space: str) -> None:
    """A writer's last step, under the kb lock: bring a stale cache up to date and remove
    what it no longer lists, so the first search after a large sync does not build it. A
    failure here only leaves the work to that search."""
    try:
        store = SqliteIndex.open(index_path, readonly=True)
    except (OSError, sqlite3.Error):
        return
    try:
        data = _read_list(cache_dir / _DIR)
        if data and data.get("space") == space and data.get("generation") == store.generation():
            # Current, perhaps refreshed by a search, which leaves old files behind.
            _prune(cache_dir, {Partition(p["start"], p["rows"], p["signature"]).stem + suffix
                               for p in data["parts"] for suffix in (".npy", ".ids.npy")})
        else:
            refresh_vectors(store, cache_dir, space, prune=True)
    except (OSError, sqlite3.Error):
        pass
    finally:
        store.close()


class QueryVectorCache:
    def __init__(self, cache_dir: Path) -> None:
        self.path = cache_dir / "query_vectors.db"

    @staticmethod
    def _key(space: str, query: str) -> str:
        return hashlib.sha256(f"{space}\0{query}".encode()).hexdigest()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("CREATE TABLE IF NOT EXISTS q (key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
        return db

    def get(self, space: str, query: str) -> np.ndarray | None:
        if not self.path.is_file():
            return None
        try:
            db = self._connect()
            try:
                row = db.execute("SELECT vector FROM q WHERE key = ?",
                                 (self._key(space, query),)).fetchone()
            finally:
                db.close()
        except sqlite3.Error:
            return None
        return None if row is None else np.frombuffer(row[0], dtype=np.float32)

    def put(self, space: str, query: str, vector: np.ndarray) -> None:
        try:
            self.path.parent.mkdir(mode=0o700, exist_ok=True)
            db = self._connect()
            try:
                with db:
                    db.execute("INSERT OR REPLACE INTO q (key, vector) VALUES (?, ?)",
                               (self._key(space, query),
                                np.asarray(vector, dtype=np.float32).tobytes()))
            finally:
                db.close()
            self.path.chmod(0o600)
        except (sqlite3.Error, OSError):
            pass
