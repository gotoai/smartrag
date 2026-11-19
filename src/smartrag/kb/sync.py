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
`failed`: unreadable), with the reason, and are not retried until they change or the
pipeline version does. A file whose extension names no extractor is indexed as plain text
when its first 8 KiB are text (design memo §3.4).

`dry_run` lists, extracts and chunks — so it can report documents, chunks, estimated
tokens and problems — but calls no embedding endpoint and writes nothing.

With `embedding.parallelism` above 1 (§3.11), step 3's embedding requests go to worker
processes, up to that many at once, while this process goes on extracting the next files;
a document is written when all its requests are back, so documents finish out of listing
order. Everything else stays in this process, which alone touches the index.

`adapt_to_rate_limit` (on by default) waits out rate limits instead of stopping: requests
back off and are split when too big for a per-minute limit (`embedding.transport`), fewer
are kept in flight while the provider pushes back, and every wait is reported as a
`wait` progress event and counted in the report.

An interrupted sync loses at most the documents in progress; running it again resumes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smartrag.chunking import CHUNKER_VERSION, TOKEN_ESTIMATOR, Chunk, chunk_units
from smartrag.embedding.api import Embedder, EmbeddingSpec, open_embedder, resolve
from smartrag.errors import ConfigError, EmbedderUnavailable, InvalidRequest, SmartRAGError
from smartrag.extract.tabular import ExtractError
from smartrag.extract.units import EXTRACTOR_VERSION, Extracted, extract
from smartrag.index import LEXIFY_VERSION, SqliteIndex, lexify
from smartrag.index.store import ChunkRow, DocumentRow, unpack_vector
from smartrag.kb.config import SINK_ID, KbConfig
from smartrag.kb.embed_pool import EmbedPool
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.models import SourceSync, SyncIssue, SyncReport
from smartrag.kb.sources import (
    SINK_SIDECAR,
    ListedFile,
    Listing,
    SyncTarget,
    list_files,
    sync_targets,
    unsupported_reason,
)
from smartrag.search.vectors import refresh_after_write

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
               "lexify": LEXIFY_VERSION, "chunker": CHUNKER_VERSION,
               "chunk": config.chunk.model_dump()}
    return _sha256(json.dumps(payload, sort_keys=True))[:12]


def doc_id_for(source: str, rel_path: str) -> str:
    return "d_" + _sha256(f"{source}\0{rel_path}")[:20]


@dataclass
class _Pending:
    """A document whose embedding requests are in flight."""

    document: DocumentRow
    text: str
    rows: list[ChunkRow]
    status: str
    embedded: int
    requests: int = 0


class _Run:
    def __init__(self, home: Home, config: KbConfig, store: SqliteIndex | None,
                 embedder: Embedder | None, report: SyncReport, force: bool,
                 progress: Progress | None, *, only_stale: bool = False,
                 donor: SqliteIndex | None = None) -> None:
        self.home = home
        self.config = config
        self.store = store
        self.embedder = embedder
        self.report = report
        self.force = force
        self.progress = progress
        self.only_stale = only_stale
        self.donor = donor
        """Another index whose vectors may be reused (a rebuild's old index)."""
        self.version = report.pipeline_version
        self.space = report.vector_space
        self.timings = {"list": 0, "extract": 0, "embed": 0, "write": 0}
        self.pool: EmbedPool | None = None
        adapting = embedder is not None and report.adapt_to_rate_limit
        if embedder is not None and report.embedding_parallelism > 1:
            self.pool = EmbedPool(embedder, report.embedding_parallelism,
                                  on_wait=self._on_wait if adapting else None)
        elif adapting and hasattr(embedder, "on_wait"):
            embedder.on_wait = self._on_wait  # type: ignore[union-attr]
        self.dimensions: int | None = None
        """The length of the vectors returned so far."""
        self._done = 0
        self._total = 0

    # --- reporting -----------------------------------------------------------------

    def _emit(self, **event: Any) -> None:
        if self.progress is not None:
            self.progress(event)

    def _on_wait(self, seconds: float, kind: str) -> None:
        self.report.rate_limit_waits += 1
        self.report.rate_limit_wait_ms += int(seconds * 1000)
        self._emit(event="wait", seconds=seconds, kind=kind)

    def _issue(self, source: str, rel: str, status: str, reason: str) -> None:
        self.report.issues_total += 1
        if len(self.report.issues) < _MAX_ISSUES_LISTED:
            self.report.issues.append(SyncIssue(source=source, path=rel, status=status,  # type: ignore[arg-type]
                                                reason=reason))

    # --- one source ----------------------------------------------------------------

    def list_target(self, target: SyncTarget) -> Listing:
        started = time.monotonic()
        listing = list_files(target, {self.home.root.resolve()})
        self.timings["list"] += int((time.monotonic() - started) * 1000)
        self._emit(event="source", source=target.source, files=len(listing.files),
                   complete=listing.complete)
        return listing

    def sync_target(self, target: SyncTarget, listing: Listing) -> None:
        info = SourceSync(source=target.source, root=str(target.root), files=len(listing.files),
                          complete=listing.complete, note=listing.problem)
        self.report.sources.append(info)
        existing = self.store.documents(target.source) if self.store else {}

        seen: set[str] = set()
        self._done, self._total = 0, len(listing.files)
        for listed in listing.files:
            seen.add(listed.rel)
            old = existing.get((target.source, listed.rel))
            result = self._file(target, listed, old)
            if result is not None:  # else its embedding requests are in flight
                self._document_done(target.source, listed.rel, *result)
        while self.pool is not None and self.pool.in_flight:
            self._collect(block=True)

        if self.only_stale:
            return  # removals are the next sync's job
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

    def _document_done(self, source: str, path: str, status: str, chunks: int) -> None:
        self._done += 1
        self._emit(event="document", source=source, path=path, status=status, chunks=chunks,
                   n=self._done, total=self._total)

    def _file(self, target: SyncTarget, listed: ListedFile,
              old: DocumentRow | None) -> tuple[str, int] | None:
        """The file's status and chunk count, or None while its embedding requests are in
        flight."""
        counts = self.report.counts
        if self.only_stale and (old is None or old.pipeline_version == self.version):
            if old is not None and (old.size, old.mtime_ns) == (listed.size, listed.mtime_ns):
                counts.unchanged += 1
                return "unchanged", old.chunk_count
            counts.deferred += 1
            return "deferred", 0
        current = (old is not None and old.pipeline_version == self.version
                   and not self.force)
        if current and old is not None and (old.size, old.mtime_ns) == (listed.size,
                                                                       listed.mtime_ns):
            counts.unchanged += 1
            return "unchanged", old.chunk_count

        ext = listed.path.suffix.lower()
        try:
            # A file whose extension names no extractor is sniffed before it is read, so
            # a large binary costs one small read, not a whole one.
            unsupported = unsupported_reason(listed.path)
            if unsupported:
                return self._record(target, listed, old, None, "skipped", unsupported), 0
            data = listed.path.read_bytes()
        except OSError as exc:
            reason = f"cannot read: {exc.strerror or exc}"
            return self._record(target, listed, old, None, "failed", reason), 0
        # The extracted text depends on the bytes *and* on the pipeline that read them, so
        # both make up the key it is stored under: identical files share it, and a new
        # extractor version never inherits the text an older one produced.
        content_hash = f"{_sha256(data)}:{self.version}"
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
            pages=pages, chunk_count=0, indexed_at=_now(),
            origin_path=_origin_of(listed.path) if target.source == SINK_ID else None)

    def _index(self, target: SyncTarget, listed: ListedFile, old: DocumentRow | None,
               content_hash: str, extracted: Extracted) -> tuple[str, int] | None:
        assert self.config.chunk is not None
        text, chunks = chunk_units(extracted.units, self.config.chunk)
        doc_id = doc_id_for(target.source, listed.rel)
        origin = _origin_of(listed.path) if target.source == SINK_ID else None
        rows = self._chunk_rows(doc_id, origin or listed.rel, chunks)
        self.report.tokens_estimated += sum(chunk.tokens for chunk in chunks)
        status = "updated" if old is not None else "added"
        document = self._row(target, listed, content_hash, "ok", None, extracted.title,
                             extracted.pages)
        if self.report.dry_run or self.store is None:
            self._count(status, len(rows))
            return status, len(rows)

        started = time.monotonic()
        known = self.store.vectors_for((r.text_hash for r in rows), self.space)
        if self.donor is not None:
            missing = [r.text_hash for r in rows if r.text_hash not in known]
            known.update(self.donor.vectors_for(missing, self.space))
        needed = [r for r in rows if r.text_hash not in known]
        for row in rows:
            if row.text_hash in known:
                row.embedding = unpack_vector(known[row.text_hash])
        if needed:
            texts = {r.text_hash: e for r, e in zip(rows, (c.embed_text() for c in chunks))}
            needed_texts = [texts[r.text_hash] for r in needed]
            if self.pool is not None:
                self.timings["embed"] += int((time.monotonic() - started) * 1000)
                self._submit(_Pending(document, text, rows, status, len(needed)),
                             needed, needed_texts)
                return None
            assert self.embedder is not None
            try:
                self._take(needed, self.embedder.embed(needed_texts, "document"))
            finally:  # a failed request's time counts too
                self.timings["embed"] += int((time.monotonic() - started) * 1000)
        else:
            self.timings["embed"] += int((time.monotonic() - started) * 1000)
        self._write(_Pending(document, text, rows, status, len(needed)))
        return status, len(rows)

    def _write(self, pending: _Pending) -> None:
        assert self.store is not None
        started = time.monotonic()
        self.store.write_document(pending.document, pending.text, pending.rows)
        self.timings["write"] += int((time.monotonic() - started) * 1000)
        self._count(pending.status, len(pending.rows))
        self.report.chunks_embedded += pending.embedded
        self.report.chunks_reused += len(pending.rows) - pending.embedded

    def _count(self, status: str, chunks: int) -> None:
        if status == "updated":
            self.report.counts.updated += 1
        else:
            self.report.counts.added += 1
        self.report.chunks_written += chunks

    # --- embedding in worker processes (§3.11) --------------------------------------

    def _submit(self, pending: _Pending, needed: list[ChunkRow], texts: list[str]) -> None:
        """Send the document's chunks as requests of `batch_size` texts, waiting whenever
        `parallelism` requests are already in flight."""
        assert self.pool is not None and self.config.embedding is not None
        size = self.config.embedding.batch_size
        starts = range(0, len(needed), size)
        pending.requests = len(starts)  # all of them, before any can come back
        for start in starts:
            while self.pool.in_flight >= self.pool.limit:
                self._collect(block=True)
            self.pool.submit(texts[start:start + size], (pending, needed[start:start + size]))
        self._collect(block=False)

    def _collect(self, *, block: bool) -> None:
        """Take in the requests that have come back, and write each document whose
        requests are all back."""
        assert self.pool is not None
        try:
            finished = self.pool.wait(block=block)
        except EmbedderUnavailable as exc:  # a failed request's time counts too
            self.timings["embed"] += int(exc.details.get("took_ms", 0))
            raise
        for (pending, rows), vectors, took_ms in finished:
            self.timings["embed"] += took_ms
            self._take(rows, vectors)
            pending.requests -= 1
            if not pending.requests:
                self._write(pending)
                self._document_done(pending.document.source_id, pending.document.rel_path,
                                    pending.status, len(pending.rows))

    def _take(self, rows: list[ChunkRow], vectors: list[list[float]]) -> None:
        """Give each row its vector, after checking they all have one length."""
        if len(vectors) != len(rows):
            raise EmbedderUnavailable(f"embedding endpoint returned {len(vectors)} "
                                      f"vectors for {len(rows)} inputs")
        for row, vector in zip(rows, vectors):
            if self.dimensions is None:
                self.dimensions = len(vector)
            elif len(vector) != self.dimensions:
                raise EmbedderUnavailable(f"embedding endpoint returned {len(vector)} "
                                          f"dimensions, expected {self.dimensions}")
            row.embedding = vector

    def _chunk_rows(self, doc_id: str, searchable_path: str,
                    chunks: list[Chunk]) -> list[ChunkRow]:
        """`searchable_path` is the relative path, or for an ingested copy the place it was
        copied from: folder names carry the user's own taxonomy."""
        rows: list[ChunkRow] = []
        occurrences: dict[str, int] = {}
        path_text = lexify(_PATH_SEPARATORS.sub(" ", searchable_path))
        for chunk in chunks:
            embed_text = chunk.embed_text()
            text_hash = _sha256(f"{self.space}\0{embed_text}")[:32]
            seen = occurrences.get(text_hash, 0)
            occurrences[text_hash] = seen + 1
            chunk_id = f"{doc_id}:{text_hash[:16]}" + (f"-{seen}" if seen else "")
            body = f"{chunk.prefix}\n{chunk.text}" if chunk.prefix else chunk.text
            rows.append(ChunkRow(
                chunk_id=chunk_id, ordinal=chunk.ordinal, start=chunk.start, end=chunk.end,
                prefix=chunk.prefix, heading_path=chunk.heading_path, page=chunk.page,
                page_end=chunk.page_end, location=chunk.location, tokens=chunk.tokens,
                text_hash=text_hash, vector_space=self.space, embedding=None,
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
            force: bool = False, only_stale: bool = False, embedder: Embedder | None = None,
            progress: Progress | None = None, operation: str | None = None,
            embedding_parallelism: int | None = None,
            adapt_to_rate_limit: bool = True) -> SyncReport:
    """Sync a kb. `only_stale` (for `kb reindex --refresh-stale-only`) re-processes only documents built
    by an older pipeline version: no new files, no removals. `operation` labels the run's
    report when another command runs the sync (`kb ingest`). `embedding_parallelism`
    overrides the config's `embedding.parallelism` for this run. `adapt_to_rate_limit`
    waits out rate limits instead of stopping the run."""
    config, targets, report = _prepare(home, kb_id, source, dry_run,
                                       operation or ("reindex-stale" if only_stale else "sync"),
                                       embedding_parallelism, adapt_to_rate_limit)
    kb_dir = home.kb_dir(kb_id)
    index_path = kb_dir / INDEX_FILE
    started = time.monotonic()
    if dry_run:
        store = SqliteIndex.open(index_path, readonly=True) if index_path.is_file() else None
        try:
            _execute(_Run(home, config, store, None, report, force, progress,
                          only_stale=only_stale), targets, source)
        finally:
            if store is not None:
                store.close()
        return _finish(report, started, None)
    with kb_lock(kb_dir, "kb reindex --refresh-stale-only" if only_stale else "kb sync"):
        sync_into(home, config, index_path, report, targets, source, force=force,
                  only_stale=only_stale, embedder=embedder, progress=progress)
        refresh_after_write(index_path, kb_dir / "cache", report.vector_space)
        return _finish(report, started, kb_dir / "runs")


def _prepare(home: Home, kb_id: str, source: str | None, dry_run: bool, operation: str,
             embedding_parallelism: int | None = None, adapt_to_rate_limit: bool = True,
             ) -> tuple[KbConfig, list[SyncTarget], SyncReport]:
    if embedding_parallelism is not None and embedding_parallelism < 1:
        raise InvalidRequest("the embedding parallelism must be at least 1",
                             {"embedding_parallelism": embedding_parallelism})
    config = home.load(kb_id)
    if config.kind == "remote":
        raise InvalidRequest(f"kb {kb_id!r} is remote: its sources are managed by its server",
                             {"kb": kb_id})
    assert config.embedding is not None
    spec = resolve(config.embedding)
    report = SyncReport(kb=kb_id, operation=operation, dry_run=dry_run,  # type: ignore[arg-type]
                        started_at=_now(), provider_id=spec.provider_id,
                        model_profile_id=spec.model_profile_id, vector_space=spec.space.key,
                        pipeline_version=pipeline_version(config),
                        embedding_parallelism=embedding_parallelism
                        or config.embedding.parallelism,
                        adapt_to_rate_limit=adapt_to_rate_limit)
    return config, sync_targets(home, config, source), report


def sync_into(home: Home, config: KbConfig, index_path: Path, report: SyncReport,
              targets: list[SyncTarget], source: str | None, *, force: bool = False,
              only_stale: bool = False, embedder: Embedder | None = None,
              progress: Progress | None = None, donor: SqliteIndex | None = None,
              first_generation: int | None = None) -> None:
    """The sync itself, into `index_path`, by a caller already holding the kb lock.
    `donor` lends vectors (a rebuild's old index); `first_generation` starts a new
    index's generation counter after the one it replaces."""
    assert config.embedding is not None
    spec = resolve(config.embedding)
    if embedder is None:
        embedder = open_embedder(config.embedding, adaptive=report.adapt_to_rate_limit)
    home.sink_dir(config).mkdir(mode=0o700, parents=True, exist_ok=True)
    store = SqliteIndex.open(index_path)
    try:
        if first_generation is not None and store.generation() < first_generation:
            with store.db:
                store.set_meta("generation", str(first_generation))
        _check_space(store, spec)
        run = _Run(home, config, store, embedder, report, force, progress,
                   only_stale=only_stale, donor=donor)
        _execute(run, targets, source)
        store.set_meta("last_sync_at", _now())
        store.db.commit()
        store.collect_garbage()
        counts = report.counts
        if counts.added + counts.updated + counts.removed:
            store.optimize()
        if counts.added + counts.updated + counts.removed + counts.skipped + counts.failed:
            store.bump_generation()
    finally:
        store.close()


def _check_space(store: SqliteIndex, spec: EmbeddingSpec) -> None:
    """Refuse to mix vector spaces (design memo §3.2); then record the space, and the
    provider and model now filling it, which may change within the space."""
    held = store.meta("vector_space")
    if held is not None and held != spec.space.key:
        raise ConfigError(
            f"the index holds vectors in the space {held!r}, but {spec.model_profile_id!r} "
            f"gives {spec.space.key!r}; vectors from different spaces cannot be mixed. "
            "Go back to a model in the index's space, or rebuild the index with "
            "`smartrag run kb reindex --kb <id> --rebuild`.",
            {"index_space": held, "config_space": spec.space.key})
    with store.db:
        for key, value in (("vector_space", spec.space.key),
                           ("model_embedding_compatibility_id", spec.space.compatibility_id),
                           ("dimensions", str(spec.space.dimensions)),
                           ("marking_id", spec.space.marking_id),
                           ("provider_id", spec.provider_id),
                           ("model_profile_id", spec.model_profile_id),
                           ("model_canonical_name", spec.model_canonical_name)):
            store.set_meta(key, value)


def _execute(run: _Run, targets: list[SyncTarget], source: str | None) -> None:
    report = run.report
    try:
        # Every source is listed before any is processed, so `plan` can give the total.
        listings = [(target, run.list_target(target)) for target in targets]
        run._emit(event="plan", files=sum(len(listing.files) for _t, listing in listings))
        for target, listing in listings:
            run.sync_target(target, listing)
        if source is None and not run.only_stale:
            run.sweep_removed_sources({s.id for s in run.config.sources} | {SINK_ID})
    except EmbedderUnavailable as exc:
        report.error = exc.message
    except KeyboardInterrupt:
        report.interrupted = True
    finally:
        if run.pool is not None:
            run.pool.close()  # requests still in flight are abandoned
        report.timings_ms = run.timings


def _origin_of(path: Path) -> str | None:
    """Where an ingested copy came from, as recorded beside it by `kb ingest`."""
    try:
        data = json.loads((path.parent / SINK_SIDECAR).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    origin = data.get("origin_path") if isinstance(data, dict) else None
    return origin if isinstance(origin, str) else None


def _finish(report: SyncReport, started: float, runs_dir: Path | None) -> SyncReport:
    report.finished_at = _now()
    report.took_ms = int((time.monotonic() - started) * 1000)
    if runs_dir is not None:
        runs_dir.mkdir(mode=0o700, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = runs_dir / f"{stamp}-{report.operation}.json"
        report.report_path = str(path)
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report
