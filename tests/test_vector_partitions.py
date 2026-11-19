"""The vector cache in partitions (design memo §3.3)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from smartrag.index import SqliteIndex
from smartrag.search import vectors
from smartrag.search.vectors import (
    carry_forward,
    load_vectors,
    refresh_after_write,
    refresh_vectors,
)

SPACE = "test-space"
DIM = 8


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(vectors, "PARTITION_ROWS", 4)


@pytest.fixture
def store(tmp_path):
    index = SqliteIndex.open(tmp_path / "index.db")
    yield index
    index.close()


_rng = np.random.default_rng(7)


def _add(store, doc, count, *, texts=None):
    """A document with `count` chunks and random vectors; returns their ids."""
    with store.db:
        store.db.execute(
            "INSERT OR REPLACE INTO documents (doc_id, source_id, rel_path, uri, file_type, "
            "size, mtime_ns, pipeline_version, status, chunk_count, indexed_at) "
            "VALUES (?, 's', ?, 'file:///x', 'md', 1, 1, 'v', 'ok', ?, 't')",
            (doc, doc, count))
        ids = []
        for n in range(count):
            text = (texts or {}).get(n, f"{doc}-{n}")
            vector = _rng.standard_normal(DIM).astype(np.float32)
            cursor = store.db.execute(
                "INSERT INTO chunks (chunk_id, doc_id, ordinal, start, end, tokens, text_hash, "
                "vector_space, embedding) VALUES (?, ?, ?, 0, 1, 1, ?, ?, ?)",
                (f"{doc}:{n}:{text}", doc, n, _hash(text), SPACE, vector.tobytes()))
            ids.append(cursor.lastrowid)
        store.set_meta("generation", str(store.generation() + 1))
    return ids


def _delete(store, doc):
    with store.db:
        store.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc,))
        store.set_meta("generation", str(store.generation() + 1))


def _hash(text):
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()[:32]


def _brute(store, query, pool, allowed=None):
    rows = store.db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
    ids = np.array([r[0] for r in rows])
    matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    scores = matrix @ query
    if allowed is not None:
        scores = np.where(np.isin(ids, allowed), scores, -np.inf)
    order = np.lexsort((ids, -scores))
    return [int(i) for i in ids[order] if np.isfinite(scores[i == ids][0])][:pool]


def _parts(tmp_path):
    return json.loads((tmp_path / "cache" / "vectors" / "parts.json").read_text())["parts"]


def _builds(monkeypatch):
    built = []
    original = vectors._build

    def spy(store, start, end, rows):
        built.append(start)
        return original(store, start, end, rows)

    monkeypatch.setattr(vectors, "_build", spy)
    return built


# --- results -----------------------------------------------------------------------


@pytest.mark.parametrize("gather", [0.0, 1.0])
def test_partitioned_results_equal_one_matrix(small, store, tmp_path, monkeypatch, gather):
    monkeypatch.setattr(vectors, "GATHER_SHARE", gather)  # mask, or score only the matches
    for doc in "abcde":
        _add(store, doc, 3)
    cache = refresh_vectors(store, tmp_path / "cache", SPACE)
    assert len(cache.partitions) == 4 and cache.rows == 15
    query = _rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)
    assert cache.top(query, 6) == _brute(store, query, 6)
    allowed = np.array([2, 3, 9, 14], dtype=np.int64)
    assert cache.top(query, 3, allowed) == _brute(store, query, 3, allowed)
    assert cache.top(query, 3, np.array([], dtype=np.int64)) == []


def test_a_partition_without_matches_is_not_read(small, store, tmp_path):
    for doc in "abc":
        _add(store, doc, 4)
    cache = refresh_vectors(store, tmp_path / "cache", SPACE)
    first = cache.partitions[0]
    first.matrix = None  # reading it would fail
    query = np.ones(DIM, dtype=np.float32) / np.sqrt(DIM)
    assert set(cache.top(query, 10, np.array([9, 10], dtype=np.int64))) == {9, 10}


# --- what is rebuilt ---------------------------------------------------------------


def test_only_changed_partitions_are_rebuilt(small, store, tmp_path, monkeypatch):
    for doc in "abc":
        _add(store, doc, 4)
    built = _builds(monkeypatch)
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    assert built == [0, 5, 9]  # three full partitions
    built.clear()
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    assert built == []  # nothing changed

    _add(store, "d", 2)  # appended: a new last partition
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    assert built == [13]
    built.clear()
    _add(store, "e", 1)
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    assert built == [13]  # only the last one


def test_a_reused_id_with_new_text_is_caught(small, store, tmp_path, monkeypatch):
    _add(store, "a", 4)
    ids = _add(store, "b", 2)
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    _delete(store, "b")
    again = _add(store, "b", 2, texts={1: "edited"})  # SQLite reuses the top ids
    assert again == ids
    built = _builds(monkeypatch)
    cache = refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    assert built == [5]
    vector = np.frombuffer(store.db.execute("SELECT embedding FROM chunks WHERE id = ?",
                                            (ids[1],)).fetchone()[0], dtype=np.float32)
    assert cache.top(vector / np.linalg.norm(vector), 1) == [ids[1]]


def test_partitions_under_half_full_are_merged(small, store, tmp_path):
    for doc in "abc":
        _add(store, doc, 4)
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    with store.db:  # leave one chunk in each of the first two partitions
        store.db.execute("DELETE FROM chunks WHERE id IN (1, 2, 3, 6, 7, 8)")
        store.set_meta("generation", str(store.generation() + 1))
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    assert [(p["start"], p["rows"]) for p in _parts(tmp_path)] == [(0, 2), (9, 4)]


# --- when it is refreshed ----------------------------------------------------------


def test_search_uses_the_listed_cache_until_the_index_changes(small, store, tmp_path,
                                                              monkeypatch):
    _add(store, "a", 3)
    refresh_vectors(store, tmp_path / "cache", SPACE, prune=True)
    refreshed = []
    original = vectors.refresh_vectors
    monkeypatch.setattr(vectors, "refresh_vectors",
                        lambda *a, **k: refreshed.append(1) or original(*a, **k))
    assert load_vectors(store, tmp_path / "cache", SPACE).rows == 3
    assert refreshed == []
    _add(store, "b", 1)
    assert load_vectors(store, tmp_path / "cache", SPACE).rows == 4
    assert refreshed == [1]


def test_source_apply_carries_the_cache_forward(store, tmp_path):
    _add(store, "a", 3)
    refresh_vectors(store, tmp_path / "cache", SPACE)
    generation = store.generation()
    store.bump_generation()
    carry_forward(tmp_path / "cache", generation, store.generation())
    listing = json.loads((tmp_path / "cache" / "vectors" / "parts.json").read_text())
    assert listing["generation"] == store.generation()


def test_writers_prune_and_searches_do_not(small, store, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "vectors.npy").write_bytes(b"old")  # the single-matrix cache
    _add(store, "a", 4)
    _add(store, "b", 2)
    refresh_vectors(store, cache_dir, SPACE, prune=True)
    _add(store, "c", 1)
    refresh_vectors(store, cache_dir, SPACE)  # a search: the old last partition stays
    files = {p.name for p in (cache_dir / "vectors").glob("part-*.npy")}
    assert len(files) == 6  # 3 partitions' worth (matrix and ids) incl. the stale one
    refresh_after_write(tmp_path / "index.db", cache_dir, SPACE)
    assert {p.name for p in (cache_dir / "vectors").glob("part-*")} != files
    assert len(list((cache_dir / "vectors").glob("part-*"))) == 4
    assert not (cache_dir / "vectors.npy").exists()


def test_an_unwritable_cache_is_built_in_memory(store, tmp_path):
    _add(store, "a", 3)
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the cache folder would be")
    assert load_vectors(store, blocked, SPACE).rows == 3
