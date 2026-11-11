"""`kb sync`: bring a kb's index in line with its sources (design memo §3.3, §3.6).

For each source (and the sink):

1. **List** every file, the way `source test` walks.
2. **Skip what has not changed**: same size and modification time, built by the current
   pipeline version. A file whose timestamp moved but whose bytes did not (same content
   hash) is only re-stamped.
3. **Process** the rest: extract → chunk → embed what is new → write the document, its
   text and its chunks **in one transaction**. A chunk whose embedded text is unchanged
   keeps its vector, so an edit re-embeds only the chunks it touched.
4. **Sweep**: documents whose files are gone are removed — but only when the listing was
   complete. A source that is unmounted, or a folder that could not be read, removes
   nothing; a network blip must never look like the user deleting files.

Files that cannot be indexed are recorded too (`skipped`: unsupported type, no text layer;
`failed`: unreadable), with the reason, and are not retried until they change.

`dry_run` lists, extracts and chunks — so it can report documents, chunks, estimated
tokens and problems — but calls no embedding endpoint and writes nothing.

An interrupted sync loses at most the document in progress; running it again resumes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartrag.chunking import TOKEN_ESTIMATOR, Chunk, chunk_units
from smartrag.embedding import Embedder, HttpEmbedder
from smartrag.errors import ConfigError, EmbedderUnavailable, InvalidRequest, SmartRAGError
from smartrag.extract.tabular import ExtractError
from smartrag.extract.units import EXTRACTOR_VERSION, Extracted, extract
from smartrag.index import LEXIFY_VERSION, SqliteIndex, lexify
from smartrag.index.store import ChunkRow, DocumentRow, unpack_vector
from smartrag.kb.config import SINK_ID, KbConfig
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.models import SourceSync, SyncIssue, SyncReport
from smartrag.kb.sources import (
    SUPPORTED_EXTENSIONS,
    ListedFile,
    SyncTarget,
    list_files,
    sync_targets,
)

Progress = Callable[[dict[str, Any]], None]

_MAX_ISSUES_LISTED = 200
_PATH_SEPARATORS = re.compile(r"[/\\._\-]+")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def pipeline_version(config: KbConfig) -> str:
    """Everything that shapes a document's chunks. When it changes, unchanged files are
    processed again on the next sync."""
    assert config.chunk is not None
    payload = {"extract": EXTRACTOR_VERSION, "tokens": TOKEN_ESTIMATOR,
               "lexify": LEXIFY_VERSION, "chunk": config.chunk.model_dump()}
    return _sha256(json.dumps(payload, sort_keys=True))[:12]


def doc_id_for(source: str, rel_path: str) -> str:
    return "d_" + _sha256(f"{source}\0{rel_path}")[:20]


class _Run:
    def __init__(self, home: Home, config: KbConfig, store: SqliteIndex | None,
                 embedder: Embedder | None, report: SyncReport, force: bool,
                 progress: Progress | None) -> None:
        self.home = home
        self.config = config
        self.store = store
        self.embedder = embedder
        self.report = report
        self.force = force
        self.progress = progress
        self.version = report.pipeline_version
        self.profile = report.profile
        self.timings = {"list": 0, "extract": 0, "embed": 0, "write": 0}

    # --- reporting -----------------------------------------------------------------

    def _emit(self, **event: Any) -> None:
        if self.progress is not None:
            self.progress(event)

    def _issue(self, source: str, rel: str, status: str, reason: str) -> None:
        self.report.issues_total += 1
        if len(self.report.issues) < _MAX_ISSUES_LISTED:
            self.report.issues.append(SyncIssue(source=source, path=rel, status=status,  # type: ignore[arg-type]
                                                reason=reason))

    # --- one source ----------------------------------------------------------------

    def sync_target(self, target: SyncTarget) -> None:
        started = time.monotonic()
        listing = list_files(target, {self.home.root.resolve()})
        self.timings["list"] += int((time.monotonic() - started) * 1000)
        info = SourceSync(source=target.source, root=str(target.root), files=len(listing.files),
                          complete=listing.complete, note=listing.problem)
        self.report.sources.append(info)
        existing = self.store.documents(target.source) if self.store else {}
        self._emit(event="source", source=target.source, files=len(listing.files),
                   complete=listing.complete)

        seen: set[str] = set()
        for number, listed in enumerate(listing.files, 1):
            seen.add(listed.rel)
            old = existing.get((target.source, listed.rel))
            status, chunks = self._file(target, listed, old)
            self._emit(event="document", source=target.source, path=listed.rel,
                       status=status, chunks=chunks, n=number, total=len(listing.files))

        gone = [row.doc_id for (_s, rel), row in existing.items() if rel not in seen]
        if gone and not listing.complete:
            info.note = (info.note or "the listing was incomplete") + \
                f"; {len(gone)} documents not seen were kept"
            return
        if gone:
            if self.store is not None and not self.report.dry_run:
                self.store.delete_documents(gone)
            info.removed = len(gone)
            self.report.counts.removed += len(gone)

    def _file(self, target: SyncTarget, listed: ListedFile,
              old: DocumentRow | None) -> tuple[str, int]:
        counts = self.report.counts
        current = (old is not None and old.pipeline_version == self.version
                   and not self.force)
        if current and old is not None and (old.size, old.mtime_ns) == (listed.size,
                                                                       listed.mtime_ns):
            counts.unchanged += 1
            return "unchanged", old.chunk_count

        ext = listed.path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            reason = f"unsupported file type {ext or '(no extension)'}"
            return self._record(target, listed, old, None, "skipped", reason), 0
        try:
            data = listed.path.read_bytes()
        except OSError as exc:
            reason = f"cannot read: {exc.strerror or exc}"
            return self._record(target, listed, old, None, "failed", reason), 0
        content_hash = _sha256(data)
        if current and old is not None and old.content_hash == content_hash:
            if self.store is not None and not self.report.dry_run:
                self.store.touch_document(old.doc_id, listed.size, listed.mtime_ns)
            counts.unchanged += 1
            return "unchanged", old.chunk_count

        started = time.monotonic()
        try:
            extracted = extract(listed.path)
        except ImportError as exc:
            # A missing library is a broken installation, not a bad file: stop, rather than
            # record every file as failed (and never retry them).
            raise ConfigError(f"a library needed to read {ext} files is missing "
                              f"({exc.name or exc}); reinstall smartrag with its "
                              "dependencies", {"module": exc.name}) from exc
        except ExtractError as exc:
            return self._record(target, listed, old, content_hash, "failed", exc.message), 0
        except SmartRAGError as exc:
            return self._record(target, listed, old, content_hash, "skipped", exc.message), 0
        except Exception as exc:  # noqa: BLE001 - an extractor bug must not end the whole sync
            reason = f"extraction error: {type(exc).__name__}: {exc}"
            return self._record(target, listed, old, content_hash, "failed", reason), 0
        finally:
            self.timings["extract"] += int((time.monotonic() - started) * 1000)
        if not extracted.has_text:
            reason = ("no extractable text (a scanned PDF needs OCR, which v1 does not do)"
                      if ext == ".pdf" else "no extractable text")
            return self._record(target, listed, old, content_hash, "skipped", reason), 0
        return self._index(target, listed, old, content_hash, extracted)

    def _record(self, target: SyncTarget, listed: ListedFile, old: DocumentRow | None,
                content_hash: str | None, status: str, reason: str) -> str:
        """A file that cannot be indexed: remember it with its reason."""
        if status == "skipped":
            self.report.counts.skipped += 1
        else:
            self.report.counts.failed += 1
        self._issue(target.source, listed.rel, status, reason)
        if self.store is not None and not self.report.dry_run:
            self.store.write_document(
                self._row(target, listed, content_hash, status, reason, None, None), None, [])
        return status

    def _row(self, target: SyncTarget, listed: ListedFile, content_hash: str | None,
             status: str, reason: str | None, title: str | None,
             pages: int | None) -> DocumentRow:
        return DocumentRow(
            doc_id=doc_id_for(target.source, listed.rel), source_id=target.source,
            rel_path=listed.rel, uri=f"{target.uri.rstrip('/')}/{listed.rel}"
            if target.root.is_dir() else target.uri,
            file_type=listed.path.suffix.lower().lstrip(".") or "",
            size=listed.size, mtime_ns=listed.mtime_ns, content_hash=content_hash,
            pipeline_version=self.version, status=status, reason=reason, title=title,
            pages=pages, chunk_count=0, indexed_at=_now())

    def _index(self, target: SyncTarget, listed: ListedFile, old: DocumentRow | None,
               content_hash: str, extracted: Extracted) -> tuple[str, int]:
        assert self.config.chunk is not None
        text, chunks = chunk_units(extracted.units, self.config.chunk)
        doc_id = doc_id_for(target.source, listed.rel)
        rows = self._chunk_rows(doc_id, listed.rel, chunks)
        self.report.tokens_estimated += sum(chunk.tokens for chunk in chunks)
        status = "updated" if old is not None else "added"
        if old is not None:
            self.report.counts.updated += 1
        else:
            self.report.counts.added += 1
        self.report.chunks_written += len(rows)
        if self.report.dry_run or self.store is None:
            return status, len(rows)

        started = time.monotonic()
        known = self.store.vectors_for((r.text_hash for r in rows), self.profile)
        needed = [r for r in rows if r.text_hash not in known]
        for row in rows:
            if row.text_hash in known:
                row.embedding = unpack_vector(known[row.text_hash])
        if needed:
            assert self.embedder is not None
            texts = {r.text_hash: e for r, e in zip(rows, (c.embed_text() for c in chunks))}
            vectors = self.embedder.embed([texts[r.text_hash] for r in needed], "document")
            for row, vector in zip(needed, vectors):
                row.embedding = vector
        self.report.chunks_embedded += len(needed)
        self.report.chunks_reused += len(rows) - len(needed)
        self.timings["embed"] += int((time.monotonic() - started) * 1000)

        started = time.monotonic()
        document = self._row(target, listed, content_hash, "ok", None, extracted.title,
                             extracted.pages)
        self.store.write_document(document, text, rows)
        self.timings["write"] += int((time.monotonic() - started) * 1000)
        return status, len(rows)

    def _chunk_rows(self, doc_id: str, rel: str, chunks: list[Chunk]) -> list[ChunkRow]:
        rows: list[ChunkRow] = []
        occurrences: dict[str, int] = {}
        path_text = lexify(_PATH_SEPARATORS.sub(" ", rel))
        for chunk in chunks:
            embed_text = chunk.embed_text()
            text_hash = _sha256(f"{self.profile}\0{embed_text}")[:32]
            seen = occurrences.get(text_hash, 0)
            occurrences[text_hash] = seen + 1
            chunk_id = f"{doc_id}:{text_hash[:16]}" + (f"-{seen}" if seen else "")
            body = f"{chunk.prefix}\n{chunk.text}" if chunk.prefix else chunk.text
            rows.append(ChunkRow(
                chunk_id=chunk_id, ordinal=chunk.ordinal, start=chunk.start, end=chunk.end,
                prefix=chunk.prefix, heading_path=chunk.heading_path, page=chunk.page,
                page_end=chunk.page_end, location=chunk.location, tokens=chunk.tokens,
                text_hash=text_hash, profile=self.profile, embedding=None,
                fts_heading=lexify(" ".join(chunk.heading_path)), fts_body=lexify(body),
                fts_path=path_text))
        return rows

    # --- documents of sources no longer configured ---------------------------------

    def sweep_removed_sources(self, configured: set[str]) -> None:
        if self.store is None:
            return
        for source in sorted(self.store.source_ids() - configured):
            rows = self.store.documents(source)
            if not self.report.dry_run:
                self.store.delete_documents(row.doc_id for row in rows.values())
            self.report.sources.append(SourceSync(
                source=source, root="", files=0, removed=len(rows),
                note="no longer a source of this kb; its documents were removed"))
            self.report.counts.removed += len(rows)


def sync_kb(home: Home, kb_id: str, *, source: str | None = None, dry_run: bool = False,
            force: bool = False, embedder: Embedder | None = None,
            progress: Progress | None = None) -> SyncReport:
    config = home.load(kb_id)
    if config.kind == "remote":
        raise InvalidRequest(f"kb {kb_id!r} is remote: its sources are managed by its server",
                             {"kb": kb_id})
    assert config.embedding is not None
    targets = sync_targets(home, config, source)
    report = SyncReport(kb=kb_id, dry_run=dry_run, started_at=_now(),
                        profile=config.embedding.profile, pipeline_version=pipeline_version(config))
    kb_dir = home.kb_dir(kb_id)
    index_path = kb_dir / INDEX_FILE
    started = time.monotonic()

    if dry_run:
        store = SqliteIndex.open(index_path, readonly=True) if index_path.is_file() else None
        try:
            _execute(home, config, store, None, report, targets, source, force, progress)
        finally:
            if store is not None:
                store.close()
        return _finish(report, started, None)

    with kb_lock(kb_dir, "kb sync"):
        if embedder is None:
            embedder = HttpEmbedder(config.embedding)
        dimensions = embedder.check()
        (kb_dir / "sink").mkdir(mode=0o700, exist_ok=True)
        store = SqliteIndex.open(index_path)
        try:
            _check_profile(store, report.profile, dimensions)
            _execute(home, config, store, embedder, report, targets, source, force, progress)
            dims = dimensions or getattr(embedder, "dimensions", None)
            if dims and store.meta("dimensions") is None:
                store.set_meta("dimensions", str(dims))
            store.set_meta("last_sync_at", _now())
            store.db.commit()
            store.collect_garbage()
            counts = report.counts
            if counts.added + counts.updated + counts.removed:
                store.optimize()
        finally:
            store.close()
        return _finish(report, started, kb_dir / "runs")


def _check_profile(store: SqliteIndex, profile: str, dimensions: int | None) -> None:
    held = store.meta("embedding_profile")
    if held is not None and held != profile:
        raise ConfigError(
            f"the index holds vectors from {held!r} but the config says {profile!r}; "
            "vectors from different models cannot be mixed. Restore the profile, or rebuild "
            "the index (`kb reindex --rebuild`, to come).",
            {"index_profile": held, "config_profile": profile})
    held_dims = store.meta("dimensions")
    if held_dims and dimensions and int(held_dims) != dimensions:
        raise ConfigError(f"the index holds {held_dims}-dimensional vectors but {profile!r} "
                          f"now returns {dimensions}", {"profile": profile})
    if held is None:
        store.set_meta("embedding_profile", profile)
        store.db.commit()


def _execute(home: Home, config: KbConfig, store: SqliteIndex | None,
             embedder: Embedder | None, report: SyncReport, targets: list[SyncTarget],
             source: str | None, force: bool, progress: Progress | None) -> None:
    run = _Run(home, config, store, embedder, report, force, progress)
    try:
        for target in targets:
            run.sync_target(target)
        if source is None:
            run.sweep_removed_sources({s.id for s in config.sources} | {SINK_ID})
    except EmbedderUnavailable as exc:
        report.error = exc.message
    except KeyboardInterrupt:
        report.interrupted = True
    finally:
        report.timings_ms = run.timings


def _finish(report: SyncReport, started: float, runs_dir: Path | None) -> SyncReport:
    report.finished_at = _now()
    report.took_ms = int((time.monotonic() - started) * 1000)
    if runs_dir is not None:
        runs_dir.mkdir(mode=0o700, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = runs_dir / f"{stamp}-sync.json"
        report.report_path = str(path)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
