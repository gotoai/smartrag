"""A kb's sources: the registered roots `sync` follows, plus the reserved sink.

`test` walks a source the way a sync would — same recursion, same include and exclude
rules, same built-in excludes — and reports what it would see, without reading any
file's contents and without touching the index.

Patterns are globs over the path relative to the source root, with `/` as separator:
`*` stays within one folder, `**/` spans any number of folders (including none), `?` is
one character. Matching ignores case, because macOS and Windows file systems do, and
`REPORT.PDF` should not escape `**/*.pdf`.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from smartrag.errors import Conflict, InvalidRequest, NotFound, Unsupported
from smartrag.extract.tabular import TABULAR_EXTENSIONS
from smartrag.kb.config import (
    SINK_ID,
    KbConfig,
    SourceConfig,
    check_id,
    describe_validation_error,
)
from smartrag.kb.home import INDEX_FILE, Home
from smartrag.kb.lock import kb_lock
from smartrag.kb.models import (
    PathError,
    SourceAdded,
    SourceList,
    SourceRemoved,
    SourceTestReport,
    SourceView,
)
from smartrag.kb.uris import location_to_uri, path_to_uri, uri_to_path

#: Always skipped, whatever a source's own patterns say: editor and OS litter, and
#: folders that are never documents.
BUILTIN_EXCLUDE = [
    "**/~$*",            # Office lock files
    "**/.~lock.*#",      # LibreOffice lock files
    "**/.DS_Store",
    "**/._*",            # macOS AppleDouble files
    "**/Thumbs.db",
    "**/desktop.ini",
    "**/.git/**",
    "**/.svn/**",
    "**/node_modules/**",
    "**/__pycache__/**",
]

#: What v1 extracts text from (design memo §3.4). Tabular files — CSV, TSV and Excel —
#: are converted to TSV blocks by `smartrag.extract.tabular` and chunked by rows.
SUPPORTED_EXTENSIONS = frozenset({
    ".txt", ".text", ".md", ".markdown", ".html", ".htm", ".xhtml",
    ".pdf", ".docx", ".pptx",
}) | TABULAR_EXTENSIONS

_MAX_ERRORS_LISTED = 20


def compile_glob(pattern: str) -> re.Pattern[str]:
    if not pattern or pattern.startswith("/"):
        raise ValueError(f"invalid pattern {pattern!r}: patterns are relative to the "
                         "source root and must not start with '/'")
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            end = pattern.find("]", i + 2)
            if end == -1:
                out.append(re.escape("["))
                i += 1
            else:
                body = pattern[i + 1:end].replace("\\", "\\\\")
                if body.startswith("!"):
                    body = "^" + body[1:]
                out.append(f"[{body}]")
                i = end + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$", re.IGNORECASE)


class _Matcher:
    def __init__(self, include: list[str], exclude: list[str]) -> None:
        self.builtin = [compile_glob(p) for p in BUILTIN_EXCLUDE]
        self.exclude = [compile_glob(p) for p in exclude]
        self.include = [compile_glob(p) for p in include]
        # A folder is pruned when an exclude pattern ending in "/**" names it, so a sync
        # never descends into node_modules or a .git folder at all.
        self.prune = [compile_glob(p[:-3]) for p in [*BUILTIN_EXCLUDE, *exclude]
                      if p.endswith("/**") and len(p) > 3]

    def prunes(self, rel_dir: str) -> bool:
        return any(p.match(rel_dir) for p in self.prune)

    def classify(self, rel: str) -> str | None:
        """Why the file is left out, or None if a sync would read it."""
        if any(p.match(rel) for p in self.builtin):
            return "builtin"
        if any(p.match(rel) for p in self.exclude):
            return "pattern"
        if self.include and not any(p.match(rel) for p in self.include):
            return "not_included"
        return None


def _validate_patterns(patterns: list[str], option: str) -> None:
    for pattern in patterns:
        try:
            compile_glob(pattern)
        except (ValueError, re.error) as exc:
            raise InvalidRequest(f"{option}: {exc}", {"pattern": pattern}) from exc


def _require_local(config: KbConfig) -> KbConfig:
    if config.kind == "remote":
        raise InvalidRequest(f"kb {config.id!r} is remote: its sources are managed by its "
                             "server", {"kb": config.id})
    return config


def _local(home: Home, kb_id: str) -> KbConfig:
    return _require_local(home.load(kb_id))


def _find(config: KbConfig, source_id: str) -> SourceConfig:
    for source in config.sources:
        if source.id == source_id:
            return source
    raise NotFound(f"kb {config.id!r} has no source {source_id!r}",
                   {"kb": config.id, "source": source_id})


def _view(source: SourceConfig) -> SourceView:
    root = uri_to_path(source.uri)
    return SourceView(id=source.id, kind="source", uri=source.uri,
                      root=str(root) if root else source.uri,
                      root_exists=bool(root and root.exists()), recursive=source.recursive,
                      include=list(source.include), exclude=list(source.exclude))


def _sink_view(home: Home, config: KbConfig) -> SourceView:
    sink = home.sink_dir(config)
    return SourceView(id=SINK_ID, kind="sink", uri=path_to_uri(sink), root=str(sink),
                      root_exists=sink.is_dir())


# --- listing for sync --------------------------------------------------------------


@dataclass
class ListedFile:
    rel: str
    path: Path
    size: int
    mtime_ns: int


@dataclass
class Listing:
    files: list[ListedFile]
    available: bool
    """False if the root is missing or unreadable: then nothing may be swept."""
    errors: int = 0
    """Paths that could not be read. Any error also means nothing may be swept."""
    problem: str | None = None

    @property
    def complete(self) -> bool:
        return self.available and self.errors == 0


@dataclass
class SyncTarget:
    source: str
    uri: str
    root: Path
    recursive: bool
    include: list[str]
    exclude: list[str]


def sync_targets(home: Home, config: KbConfig, source_id: str | None) -> list[SyncTarget]:
    """The kb's sources plus its sink, or just the one asked for."""
    targets: list[SyncTarget] = []
    for source in config.sources:
        root = uri_to_path(source.uri)
        if root is None:
            raise Unsupported(f"source {source.id!r}: only file:// sources can be synced in v1")
        targets.append(SyncTarget(source.id, source.uri, root, source.recursive,
                                  list(source.include), list(source.exclude)))
    sink = home.sink_dir(config)
    targets.append(SyncTarget(SINK_ID, path_to_uri(sink), sink, True, [], []))
    if source_id is None:
        return targets
    chosen = [t for t in targets if t.source == source_id]
    if not chosen:
        raise NotFound(f"kb {config.id!r} has no source {source_id!r}",
                       {"kb": config.id, "source": source_id})
    return chosen


def list_files(target: SyncTarget, skip: set[Path]) -> Listing:
    """Every file a sync of `target` should consider, walked like `source test` walks."""
    root = target.root
    matcher = _Matcher(target.include, target.exclude)
    if not root.exists():
        return Listing([], available=False, problem=f"{root} does not exist (not mounted?)")
    if root.is_file():
        if matcher.classify(root.name) is not None:
            return Listing([], available=True)
        info = root.stat()
        return Listing([ListedFile(root.name, root, info.st_size, info.st_mtime_ns)], True)
    listing = Listing([], available=True)
    pending: list[tuple[Path, str]] = [(root, "")]
    while pending:
        folder, rel_folder = pending.pop()
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError as exc:
            listing.errors += 1
            if not rel_folder:
                listing.available = False
                listing.problem = f"{root} cannot be read: {exc.strerror or exc}"
                return listing
            continue
        for entry in entries:
            rel = f"{rel_folder}/{entry.name}" if rel_folder else entry.name
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    path = Path(entry.path)
                    if (target.recursive and not matcher.prunes(rel)
                            and path.resolve() not in skip):
                        pending.append((path, rel))
                elif entry.is_file(follow_symlinks=False) and matcher.classify(rel) is None:
                    info = entry.stat(follow_symlinks=False)
                    listing.files.append(ListedFile(rel, Path(entry.path), info.st_size,
                                                    info.st_mtime_ns))
            except OSError:
                listing.errors += 1
    listing.files.sort(key=lambda f: f.rel)
    return listing


# --- source list -------------------------------------------------------------------


def list_sources(home: Home, kb_id: str) -> SourceList:
    config = _local(home, kb_id)
    views = [_view(s) for s in config.sources]
    views.append(_sink_view(home, config))
    return SourceList(kb=kb_id, sources=views, builtin_exclude=list(BUILTIN_EXCLUDE))


# --- source add --------------------------------------------------------------------


def add_source(home: Home, kb_id: str, source_id: str, location: str, *,
               recursive: bool = True, include: list[str] | None = None,
               exclude: list[str] | None = None, allow_missing: bool = False) -> SourceAdded:
    include = list(include or [])
    exclude = list(exclude or [])
    try:
        check_id(source_id, "source id")
    except ValueError as exc:
        raise InvalidRequest(str(exc), {"source": source_id}) from exc
    if source_id == SINK_ID:
        raise InvalidRequest(f"source id {SINK_ID!r} is reserved for the kb's sink")
    _validate_patterns(include, "--include")
    _validate_patterns(exclude, "--exclude")

    _local(home, kb_id)
    uri, path = location_to_uri(location)
    kb_dir = home.kb_dir(kb_id).resolve()
    if path == kb_dir or kb_dir in path.parents:
        raise InvalidRequest("a source cannot be inside the kb's own directory; "
                             "use `kb ingest` to copy files into the kb")
    if not allow_missing and not path.exists():
        raise InvalidRequest(f"{path} does not exist; check the path, or pass "
                             "--allow-missing for a share that is not mounted right now",
                             {"path": str(path)})

    warnings: list[str] = []
    with kb_lock(home.kb_dir(kb_id), "source add"):
        document = home.document(kb_id)
        config = _require_local(document.model)
        for existing in config.sources:
            if existing.id == source_id:
                raise Conflict(f"kb {kb_id!r} already has a source {source_id!r}",
                               {"kb": kb_id, "source": source_id})
            other = uri_to_path(existing.uri)
            if other is None:
                continue
            if other == path:
                raise Conflict(f"{path} is already source {existing.id!r}",
                               {"kb": kb_id, "source": existing.id})
            if other in path.parents or path in other.parents:
                warnings.append(f"overlaps source {existing.id!r} ({other}): files in both "
                                "are indexed twice unless one excludes them")
        home_root = home.root.resolve()
        if home_root == path or path in home_root.parents:
            warnings.append(f"the Smart RAG home {home_root} is inside this source; "
                            "it is always skipped")
        try:
            source = SourceConfig(id=source_id, uri=uri, recursive=recursive,
                                  include=include, exclude=exclude)
        except ValidationError as exc:
            raise InvalidRequest(describe_validation_error(exc)) from exc
        document.add_source(source)
        document.save()
    return SourceAdded(kb=kb_id, source=_view(source), warnings=warnings)


# --- source remove -----------------------------------------------------------------


def remove_source(home: Home, kb_id: str, source_id: str) -> SourceRemoved:
    if source_id == SINK_ID:
        raise InvalidRequest("the sink cannot be removed; remove documents from it with "
                             "`doc remove`")
    _find(_local(home, kb_id), source_id)
    removed: int | None = None
    with kb_lock(home.kb_dir(kb_id), "source remove"):
        document = home.document(kb_id)
        source = _find(_require_local(document.model), source_id)
        index_path = home.kb_dir(kb_id) / INDEX_FILE
        if index_path.is_file():
            from smartrag.index import SqliteIndex

            store = SqliteIndex.open(index_path)
            try:
                removed = store.delete_source(source_id)
                store.collect_garbage()
            finally:
                store.close()
        document.remove_source(source_id)
        document.save()
    note = None if removed is not None else "the kb has no index yet"
    return SourceRemoved(kb=kb_id, source=source_id, uri=source.uri,
                         documents_removed=removed, note=note)


# --- source test -------------------------------------------------------------------


def test_source(home: Home, kb_id: str, source_id: str, *, sample: int = 10) -> SourceTestReport:
    config = _local(home, kb_id)
    root: Path | None
    if source_id == SINK_ID:
        sink = home.sink_dir(config)
        root, uri, recursive, include, exclude = sink, path_to_uri(sink), True, [], []
    else:
        source = _find(config, source_id)
        root = uri_to_path(source.uri)
        uri, recursive = source.uri, source.recursive
        include, exclude = list(source.include), list(source.exclude)
    if root is None:
        raise Unsupported(f"only file:// sources can be tested in v1, not {uri}")

    started = time.monotonic()
    report = SourceTestReport(kb=kb_id, source=source_id, uri=uri, root=str(root), ok=False,
                              root_kind="missing", recursive=recursive)
    if not root.exists():
        report.problems.append(f"{root} does not exist (not mounted?)")
    elif root.is_file():
        report.root_kind = "file"
        _consider(report, _Matcher(include, exclude), root.name, root, sample)
        report.ok = True
    elif root.is_dir():
        report.root_kind = "directory"
        skip = {home.root.resolve()}
        report.ok = _walk(report, root, recursive, _Matcher(include, exclude), skip, sample)
    else:
        report.root_kind = "other"
        report.problems.append(f"{root} is neither a folder nor a file")

    if report.errors_total:
        report.problems.append(f"{_count(report.errors_total, 'path')} could not be read")
    if report.ok and report.files_matched == 0:
        report.problems.append("no files match: check the include and exclude patterns")
    if report.unsupported_files:
        kinds = sorted(ext for ext in report.by_extension if ext not in SUPPORTED_EXTENSIONS)
        report.problems.append(f"{_count(report.unsupported_files, 'file')} of types v1 "
                               f"cannot extract ({', '.join(kinds)}); a sync will skip them")
    report.took_ms = int((time.monotonic() - started) * 1000)
    return report


def _count(n: int, noun: str) -> str:
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


test_source.__test__ = False  # type: ignore[attr-defined]  # a library function, not a pytest test


def _error(report: SourceTestReport, rel: str, exc: OSError) -> None:
    report.errors_total += 1
    if len(report.errors) < _MAX_ERRORS_LISTED:
        report.errors.append(PathError(path=rel or ".", error=exc.strerror or str(exc)))


def _consider(report: SourceTestReport, matcher: _Matcher, rel: str, path: Path,
              sample: int, size: int | None = None) -> None:
    reason = matcher.classify(rel)
    if reason == "builtin":
        report.excluded_builtin += 1
        return
    if reason == "pattern":
        report.excluded_by_pattern += 1
        return
    if reason == "not_included":
        report.not_included += 1
        return
    if size is None:
        try:
            size = path.stat().st_size
        except OSError as exc:
            _error(report, rel, exc)
            return
    ext = path.suffix.lower() or "(none)"
    report.files_matched += 1
    report.bytes_matched += size
    report.by_extension[ext] = report.by_extension.get(ext, 0) + 1
    if ext in SUPPORTED_EXTENSIONS:
        report.supported_files += 1
    else:
        report.unsupported_files += 1
    if len(report.sample) < sample:
        report.sample.append(rel)


def _walk(report: SourceTestReport, root: Path, recursive: bool, matcher: _Matcher,
          skip: set[Path], sample: int) -> bool:
    """Walk without following symbolic links (no cycles, no escaping the root).
    Returns False only if the root itself cannot be read."""
    pending: list[tuple[Path, str]] = [(root, "")]
    while pending:
        folder, rel_folder = pending.pop()
        try:
            entries = sorted(os.scandir(folder), key=lambda e: e.name)
        except OSError as exc:
            _error(report, rel_folder, exc)
            if not rel_folder:
                return False
            continue
        for entry in entries:
            rel = f"{rel_folder}/{entry.name}" if rel_folder else entry.name
            try:
                if entry.is_symlink():
                    report.symlinks_skipped += 1
                elif entry.is_dir(follow_symlinks=False):
                    path = Path(entry.path)
                    if recursive and not matcher.prunes(rel) and path.resolve() not in skip:
                        pending.append((path, rel))
                elif entry.is_file(follow_symlinks=False):
                    size = entry.stat(follow_symlinks=False).st_size
                    _consider(report, matcher, rel, Path(entry.path), sample, size)
            except OSError as exc:
                _error(report, rel, exc)
    return True
