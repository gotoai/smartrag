"""`kb reindex`: re-process what an older pipeline built, or rebuild the index.

* `--refresh-stale-only`: re-process exactly the documents whose pipeline version (extractor,
  chunking, normalisation) is older than the current one. New and deleted files are left
  for the next `kb sync`. (A plain sync does this too, along with everything else.)
* `--rebuild`: build a fresh index **beside** the live one, then swap it in. Search keeps
  working on the old index meanwhile, and a failed or interrupted rebuild leaves it intact.
  It is also how a kb changes its embedding model: the new index holds only the new
  profile's vectors. When the profile is unchanged, the old index lends its vectors, so a
  rebuild does not pay to embed everything again.

The rebuilt index continues the old one's generation counter rather than starting at zero,
so no cursor or cached vector matrix from the old index can be mistaken for the new one's.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from smartrag.embedding import Embedder
from smartrag.index import SqliteIndex
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.models import SyncReport
from smartrag.kb.sync import _execute, _finish, _prepare, _Run, sync_into, sync_kb

REBUILD_SUFFIX = ".rebuild"
_SIDE_FILES = ("-wal", "-shm")


def reindex_stale(home: Home, kb_id: str, *, dry_run: bool = False,
                  embedder: Embedder | None = None,
                  progress: Callable[[dict[str, Any]], None] | None = None) -> SyncReport:
    return sync_kb(home, kb_id, dry_run=dry_run, only_stale=True, embedder=embedder,
                   progress=progress)


def _remove_database(path: Path) -> None:
    for suffix in ("", *_SIDE_FILES):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def reindex_rebuild(home: Home, kb_id: str, *, dry_run: bool = False,
                    embedder: Embedder | None = None,
                    progress: Callable[[dict[str, Any]], None] | None = None) -> SyncReport:
    config, targets, report = _prepare(home, kb_id, None, dry_run, "reindex-rebuild")
    kb_dir = home.kb_dir(kb_id)
    live = kb_dir / INDEX_FILE
    started = time.monotonic()
    if dry_run:
        # As if the index were empty: every file counts as added.
        _execute(_Run(home, config, None, None, report, True, progress), targets, None)
        return _finish(report, started, None)

    with kb_lock(kb_dir, "kb reindex --rebuild"):
        fresh = kb_dir / f"{INDEX_FILE}{REBUILD_SUFFIX}"
        _remove_database(fresh)  # left over from an interrupted rebuild
        donor: SqliteIndex | None = None
        first_generation = 1
        if live.is_file():
            donor = SqliteIndex.open(live, readonly=True)
            first_generation = donor.generation() + 1
            if donor.meta("embedding_profile") != report.profile:
                donor.close()
                donor = None
        try:
            sync_into(home, config, fresh, report, targets, None, force=True,
                      embedder=embedder, progress=progress, donor=donor,
                      first_generation=first_generation)
        except BaseException:
            _remove_database(fresh)
            raise
        finally:
            if donor is not None:
                donor.close()
        if report.error or report.interrupted:
            _remove_database(fresh)
            report.error = (report.error or "interrupted") + "; the old index was kept"
            return _finish(report, started, kb_dir / "runs")
        _swap(live, fresh)
        return _finish(report, started, kb_dir / "runs")


def _swap(live: Path, fresh: Path) -> None:
    """Replace the live index with the fresh one. The old index's write-ahead log is
    folded in and removed first, so it can never be replayed against the new file."""
    if live.is_file():
        old = SqliteIndex.open(live)
        try:
            old.checkpoint()
        finally:
            old.close()
    for suffix in _SIDE_FILES:
        Path(f"{live}{suffix}").unlink(missing_ok=True)
    fresh.replace(live)
    for suffix in _SIDE_FILES:
        Path(f"{fresh}{suffix}").unlink(missing_ok=True)
