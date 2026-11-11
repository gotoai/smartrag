"""`kb ingest`: copy files into the kb's sink, then sync the sink (design memo §3.6).

The sink is the kb's own source. A copy lives at `sink/<content-hash>/<original name>`:

* **The same content twice is a no-op**, whatever it is called; two different files with
  the same name cannot collide.
* `.origin.json` beside the copy records where it came from. Sync indexes that original
  path as the document's searchable path, since folder names carry the user's own
  taxonomy, and citations show it.
* **A copy is a snapshot.** Later edits to the original are not picked up; to follow a
  folder, `source add` it instead.

Only supported file types are copied, so the sink never accumulates files that sync can
only record as skipped. A folder needs `--recursive`: without it the command refuses before
copying anything, rather than guess.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from smartrag.chunking import chunk_units
from smartrag.embedding import Embedder
from smartrag.errors import InvalidRequest, SmartRAGError
from smartrag.extract.units import extract
from smartrag.kb.config import SINK_ID
from smartrag.kb.home import Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.models import SyncReport
from smartrag.kb.sources import SINK_SIDECAR, SUPPORTED_EXTENSIONS, SyncTarget, list_files
from smartrag.kb.uris import path_to_uri

_HASH_CHARS = 32
_READ_BLOCK = 1 << 20


class IngestFile(BaseModel):
    path: str
    status: Literal["copied", "would_copy", "already", "skipped"]
    reason: str | None = None
    sink_path: str | None = None
    bytes: int = 0


class IngestReport(BaseModel):
    kb: str
    dry_run: bool
    files: list[IngestFile] = Field(default_factory=list)
    copied: int = 0
    already: int = 0
    skipped: int = 0
    bytes_copied: int = 0
    chunks_estimated: int = 0
    tokens_estimated: int = 0
    sync: SyncReport | None = None
    note: str = ("Copies are snapshots: later changes to the originals are not picked up. "
                 "To follow a folder, use `source add` instead.")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()[:_HASH_CHARS]


def _collect(paths: list[str], recursive: bool, kb_dir: Path,
             home_root: Path) -> list[tuple[Path, str | None]]:
    """Every file the arguments name, with a reason when one must be skipped. Raises for
    an argument that makes the whole command wrong, before anything is copied."""
    if not paths:
        raise InvalidRequest("name at least one file or folder to ingest")
    found: list[tuple[Path, str | None]] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path == kb_dir or kb_dir in path.parents:
            raise InvalidRequest(f"{path} is inside the kb's own folder")
        if path.is_dir():
            if not recursive:
                raise InvalidRequest(f"{path} is a folder: pass --recursive to ingest the "
                                     "files in it, or name the files")
            target = SyncTarget("ingest", path_to_uri(path), path, True, [], [])
            listing = list_files(target, {home_root})
            if not listing.available:
                raise InvalidRequest(listing.problem or f"{path} cannot be read")
            found += [(f.path, None) for f in listing.files]
        elif path.is_file():
            found.append((path, None))
        else:
            found.append((path, "does not exist"))
    return found


def ingest_files(home: Home, kb_id: str, paths: list[str], *, recursive: bool = False,
                 dry_run: bool = False, embedder: Embedder | None = None,
                 progress: Callable[[dict[str, Any]], None] | None = None) -> IngestReport:
    from smartrag.kb.sync import sync_kb

    config = home.load(kb_id)
    if config.kind == "remote":
        raise InvalidRequest(f"kb {kb_id!r} is remote: its sources are managed by its server",
                             {"kb": kb_id})
    assert config.chunk is not None
    kb_dir = home.kb_dir(kb_id).resolve()
    sink = home.sink_dir(config)
    candidates = _collect(paths, recursive, kb_dir, home.root.resolve())
    report = IngestReport(kb=kb_id, dry_run=dry_run)

    def skip(path: Path, reason: str) -> None:
        report.skipped += 1
        report.files.append(IngestFile(path=str(path), status="skipped", reason=reason))

    planned: list[tuple[Path, str, int, IngestFile]] = []
    seen: dict[str, Path] = {}
    for path, problem in candidates:
        if problem:
            skip(path, problem)
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            skip(path, f"unsupported file type {ext or '(no extension)'}")
            continue
        try:
            digest = _file_hash(path)
            size = path.stat().st_size
        except OSError as exc:
            skip(path, f"cannot read: {exc.strerror or exc}")
            continue
        existing = sorted(p.name for p in (sink / digest).glob("*")
                          if p.is_file() and p.name != SINK_SIDECAR)
        if existing or digest in seen:
            name = existing[0] if existing else seen[digest].name
            report.already += 1
            report.files.append(IngestFile(path=str(path), status="already", bytes=size,
                                           sink_path=f"{digest}/{name}",
                                           reason=f"the same content is already in the sink "
                                                  f"as {name}"))
            continue
        seen[digest] = path
        item = IngestFile(path=str(path), status="would_copy", bytes=size,
                          sink_path=f"{digest}/{path.name}")
        report.files.append(item)  # in argument order; the status is settled below
        planned.append((path, digest, size, item))

    if dry_run:
        for path, _digest, size, _item in planned:
            report.bytes_copied += size
            try:
                _text, chunks = chunk_units(extract(path).units, config.chunk)
            except SmartRAGError:
                continue
            report.chunks_estimated += len(chunks)
            report.tokens_estimated += sum(c.tokens for c in chunks)
        return report

    with kb_lock(kb_dir, "kb ingest"):
        sink.mkdir(mode=0o700, parents=True, exist_ok=True)
        for path, digest, size, item in planned:
            folder = sink / digest
            try:
                _copy(path, folder)
            except OSError as exc:
                shutil.rmtree(folder, ignore_errors=True)
                item.status, item.reason, item.sink_path = (
                    "skipped", f"cannot copy: {exc.strerror or exc}", None)
                report.skipped += 1
                continue
            item.status = "copied"
            report.copied += 1
            report.bytes_copied += size
    report.sync = sync_kb(home, kb_id, source=SINK_ID, embedder=embedder, progress=progress,
                          operation="ingest")
    return report


def _copy(source: Path, folder: Path) -> None:
    """Copy under a temporary name, then rename: a half-copied file never appears."""
    folder.mkdir(mode=0o700)
    tmp = folder / f".{source.name}.part"
    shutil.copy2(source, tmp)
    os.replace(tmp, folder / source.name)
    sidecar = {"origin_path": str(source), "ingested_at":
               datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "size": source.stat().st_size}
    (folder / SINK_SIDECAR).write_text(json.dumps(sidecar, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
