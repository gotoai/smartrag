"""`kb reindex`: re-process what an older pipeline built, or rebuild the index.

* `--refresh-stale-only`: re-process exactly the documents whose pipeline version (extractor,
  chunking, normalisation) is older than the current one. New and deleted files are left
  for the next `kb sync`. (A plain sync does this too, along with everything else.)
* `--rebuild`: build a fresh index **beside** the live one, then swap it in. Its
  attributes come from the sources' attributes folders, as in a sync. Search keeps
  working on the old index meanwhile, and a failed or interrupted rebuild leaves it intact.
  **A stopped rebuild is continued** by the next `--rebuild`: its partial
  index (`index.db.rebuild`) is kept, every document in it is saved whole, and the next run
  skips what is already done, so vectors a slow or rate-limited provider was paid for are
  not paid for twice. A partial index in another vector space (the kb's model changed
  since) is discarded instead; one built by an older pipeline has those documents redone.
  It is also how a kb moves to another vector space: the new index
  holds only the new space's vectors. When the space is unchanged, the old index lends its
  vectors, so a rebuild does not pay to embed everything again. An index built by an older
  smartrag (an older schema) is rebuilt the same way.

The rebuilt index continues the old one's generation counter rather than starting at zero,
so no cursor or cached vector matrix from the old index can be mistaken for the new one's.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from smartrag.embedding.api import Embedder
from smartrag.errors import ConfigError
from smartrag.index import SqliteIndex
from smartrag.kb.home import INDEX_FILE, REBUILD_FILE, Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.models import SyncReport
from smartrag.kb.sync import _execute, _finish, _prepare, _Run, sync_into, sync_kb, sync_kb_tables
from smartrag.search.vectors import refresh_after_write

_SIDE_FILES = ("-wal", "-shm")


def reindex_stale(home: Home, kb_id: str, *, dry_run: bool = False,
                  embedder: Embedder | None = None,
                  progress: Callable[[dict[str, Any]], None] | None = None,
                  embedding_parallelism: int | None = None,
                  adapt_to_rate_limit: bool = True) -> SyncReport:
    return sync_kb(home, kb_id, dry_run=dry_run, only_stale=True, embedder=embedder,
                   progress=progress, embedding_parallelism=embedding_parallelism,
                   adapt_to_rate_limit=adapt_to_rate_limit)


def _remove_database(path: Path) -> None:
    for suffix in ("", *_SIDE_FILES):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def reindex_rebuild(home: Home, kb_id: str, *, dry_run: bool = False,
                    embedder: Embedder | None = None,
                    progress: Callable[[dict[str, Any]], None] | None = None,
                    embedding_parallelism: int | None = None,
                    adapt_to_rate_limit: bool = True) -> SyncReport:
    config, targets, report = _prepare(home, kb_id, None, dry_run, "reindex-rebuild",
                                       embedding_parallelism, adapt_to_rate_limit)
    kb_dir = home.kb_dir(kb_id)
    live = kb_dir / INDEX_FILE
    started = time.monotonic()
    if dry_run:
        # As if the index were empty: every file counts as added.
        _execute(_Run(home, config, None, None, report, True, progress), targets, None)
        sync_kb_tables(config, kb_dir, report, dry_run=True, force=True)
        return _finish(report, started, None)

    with kb_lock(kb_dir, "kb reindex --rebuild"):
        fresh = kb_dir / REBUILD_FILE
        resumed = _resumable(fresh, report.vector_space)
        if resumed is None:
            _remove_database(fresh)  # none, or one that cannot be continued
        report.resumed_documents = resumed or 0
        donor: SqliteIndex | None = None
        first_generation = 1
        if live.is_file():
            donor = SqliteIndex.open(live, readonly=True, any_schema=True)
            first_generation = donor.generation() + 1
            if donor.meta("vector_space") != report.vector_space:
                donor.close()
                donor = None
        try:
            # Resuming, the documents already done are unchanged and skipped.
            sync_into(home, config, fresh, report, targets, None, force=resumed is None,
                      embedder=embedder, progress=progress, donor=donor,
                      first_generation=first_generation)
        finally:  # whatever stopped it, the partial index is kept for the next run
            if donor is not None:
                donor.close()
        if report.error or report.interrupted:
            report.error = ((report.error or "interrupted") + "; the old index is still in "
                            "use. Run `kb reindex --rebuild` again to continue where this "
                            "one stopped")
            return _finish(report, started, kb_dir / "runs")
        _swap(live, fresh)
        refresh_after_write(live, kb_dir / "cache", report.vector_space)
        sync_kb_tables(config, kb_dir, report, force=True)
        return _finish(report, started, kb_dir / "runs")


def _resumable(fresh: Path, space: str) -> int | None:
    """The documents a stopped rebuild left in `fresh`, if it can be continued: same schema,
    same vector space. None to start afresh."""
    if not fresh.is_file():
        return None
    try:
        store = SqliteIndex.open(fresh, readonly=True)  # refuses another schema
    except (ConfigError, sqlite3.Error):
        return None
    try:
        if store.meta("vector_space") != space:
            return None
        return int(store.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
    except sqlite3.Error:
        return None
    finally:
        store.close()


def _swap(live: Path, fresh: Path) -> None:
    """Replace the live index with the fresh one. The old index's write-ahead log is
    folded in and removed first, so it can never be replayed against the new file."""
    if live.is_file():
        old = sqlite3.connect(live, timeout=30)  # any schema: an older one is replaced too
        try:
            old.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            old.close()
    for suffix in _SIDE_FILES:
        Path(f"{live}{suffix}").unlink(missing_ok=True)
    fresh.replace(live)
    for suffix in _SIDE_FILES:
        Path(f"{fresh}{suffix}").unlink(missing_ok=True)
