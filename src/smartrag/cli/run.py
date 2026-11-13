"""`smartrag run`: one command, non-interactive, then exit.

    smartrag run [--home DIR] [--format text|json] <subject> <verb> --kb ID [options]

Every verb maps to one function of the core library (`smartrag.kb`); this module only
parses options and renders the result. `--kb` is required wherever a command acts on a
kb, with no default even when only one exists.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import typer

from smartrag.cli.output import (
    OutputFormat,
    RunState,
    bullets,
    confirm,
    emit_error,
    human_bytes,
    invoke,
    table,
)
from smartrag.cli.progress import ProgressBar, progress_lines
from smartrag.errors import NotFound, SmartRAGError
from smartrag.kb import Home, add_source, list_sources, remove_source, resolve_home, test_source
from smartrag.kb.config import CONFIG_FILE, DEFAULT_PROFILE
from smartrag.kb.ingest import IngestReport, ingest_files
from smartrag.kb.models import (
    KbCreated,
    KbList,
    KbRemoved,
    KbStatus,
    KbUpdated,
    SourceAdded,
    SourceList,
    SourceRemoved,
    SourceTestReport,
    SyncReport,
)
from smartrag.kb.reindex import reindex_rebuild, reindex_stale
from smartrag.kb.sync import sync_kb
from smartrag.search import (
    KbReadResponse,
    KbSearchResponse,
    SearchFilter,
    read_chunk,
    search_kb,
)

run_app = typer.Typer(help="Run one command and exit (non-interactive).",
                      no_args_is_help=True, add_completion=False)
kb_app = typer.Typer(help="Knowledge bases: list, create, inspect, remove.",
                     no_args_is_help=True)
source_app = typer.Typer(help="A kb's sources: the folders `sync` follows.",
                         no_args_is_help=True)
run_app.add_typer(kb_app, name="kb")
run_app.add_typer(source_app, name="source")


@run_app.callback()
def run_options(
    ctx: typer.Context,
    home: Path = typer.Option(None, "--home", help="Folder holding all kbs "
                                                   "[default: $SMARTRAG_HOME or ~/.smartrag]"),
    output: OutputFormat = typer.Option(OutputFormat.text, "--format",
                                        help="text for people, json for programs."),
) -> None:
    ctx.obj = RunState(home=Home(resolve_home(home)), format=output)


def _state(ctx: typer.Context) -> RunState:
    state = ctx.find_object(RunState)
    assert state is not None
    return state


# --- kb ------------------------------------------------------------------------------


@kb_app.command("list")
def kb_list(ctx: typer.Context) -> None:
    """List the kbs in the home."""
    state = _state(ctx)
    invoke(state, state.home.list, _render_kb_list)


@kb_app.command("new")
def kb_new(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="Id of the new kb: lowercase letters, digits, "
                                             "'-', '_'. Also its folder name."),
    title: str = typer.Option(None, "--title", help="Display name [default: the id]."),
    description: str = typer.Option("", "--description",
                                    help="What the kb contains; agents choose kbs by it."),
    profile: str = typer.Option(DEFAULT_PROFILE, "--profile",
                                help="Embedding profile id (local kb)."),
    remote: str = typer.Option(None, "--remote",
                               help="Make a remote kb: the https:// URL of its Smart RAG server."),
    remote_kb: str = typer.Option(None, "--remote-kb",
                                  help="The kb's id on the remote server [default: --kb]."),
    token_env: str = typer.Option(None, "--token-env",
                                  help="Environment variable holding your token for --remote."),
) -> None:
    """Create a kb: its folder and config.yaml."""
    state = _state(ctx)
    invoke(state, lambda: state.home.new(kb, title=title, description=description,
                                         profile=profile, remote=remote, remote_kb=remote_kb,
                                         token_env=token_env),
           _render_kb_created)


@kb_app.command("set")
def kb_set(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    description: str = typer.Option(None, "--description",
                                    help="What the kb contains, and what it does not; at most "
                                         "300 characters. Agents choose a kb by it, and do not "
                                         "see a kb without one."),
) -> None:
    """Change a kb's description in config.yaml, keeping your comments and formatting."""
    state = _state(ctx)
    if description is None:
        raise typer.BadParameter("nothing to set: give --description", param_hint="--description")
    invoke(state, lambda: state.home.set(kb, description=description), _render_kb_updated)


@kb_app.command("status")
def kb_status(ctx: typer.Context, kb: str = typer.Option(..., "--kb", help="The kb's id.")) -> None:
    """Show a kb's settings, sources, sink, index and warnings. Makes no network calls."""
    state = _state(ctx)
    invoke(state, lambda: state.home.status(kb), _render_kb_status)


@kb_app.command("remove")
def kb_remove(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation."),
) -> None:
    """Delete a kb: its config, index and the sink's copies. Original files are untouched."""
    state = _state(ctx)

    def action() -> KbRemoved:
        kb_dir = state.home.kb_dir(kb)
        if not (kb_dir / CONFIG_FILE).is_file():
            raise NotFound(f"no kb {kb!r} in {state.home.root}", {"kb": kb})
        confirm(state, yes, f"Delete kb {kb!r} at {kb_dir}, including its index and the "
                            "files copied into its sink? Original files are not touched.")
        return state.home.remove(kb)

    invoke(state, action, _render_kb_removed)


@kb_app.command("sync")
def kb_sync(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    source: str = typer.Option(None, "--source",
                               help="Sync only this source (or 'sink') [default: all]."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="List, extract and chunk; embed and write nothing."),
    force: bool = typer.Option(False, "--force",
                               help="Re-extract files even if unchanged. Unchanged chunks "
                                    "keep their vectors."),
    exit_with_parent: bool = typer.Option(
        False, "--exit-with-parent",
        help="Stop cleanly when stdin closes: how a parent app (Kari) says it has gone."),
    refresh: float = typer.Option(1.0, "--refresh", min=0.1, metavar="SECONDS",
                                  help="How often the progress bar is redrawn."),
) -> None:
    """Bring the kb's index in line with its sources.

    Exit codes: 0 done; 1 failed; 3 done, but some files were skipped or failed;
    130 interrupted. Running it again resumes where an interrupted sync stopped."""
    state = _state(ctx)
    if exit_with_parent:
        _stop_when_stdin_closes()
    try:
        with _progress(state, refresh) as progress:
            report = sync_kb(state.home, kb, source=source, dry_run=dry_run, force=force,
                             progress=progress)
    except SmartRAGError as err:
        emit_error(state.format, err)
        raise typer.Exit(1) from None
    _print_report(state, report, _render_sync)
    raise typer.Exit(_sync_exit_code(report))


def _progress(state: RunState,
              refresh: float = 1.0) -> AbstractContextManager[Callable[[dict[str, object]], None]]:
    """JSON lines for programs; a progress bar on a terminal; else a line per document."""
    if state.format is OutputFormat.json:
        return nullcontext(lambda event: typer.echo(json.dumps(event, ensure_ascii=False)))
    if sys.stderr.isatty():
        return ProgressBar(refresh)
    return nullcontext(progress_lines)


def _print_report(state: RunState, report: SyncReport | IngestReport,
                  render: Callable[[Any], str]) -> None:
    if state.format is OutputFormat.json:
        typer.echo(report.model_dump_json())  # one line: the last line of the output
    else:
        typer.echo(render(report))


@kb_app.command("ingest")
def kb_ingest(
    ctx: typer.Context,
    paths: list[str] = typer.Argument(..., help="Files, or folders with --recursive."),
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    recursive: bool = typer.Option(False, "--recursive", "-r",
                                   help="Ingest the files inside the folders named."),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="Report what would be copied and embedded; copy nothing."),
    exit_with_parent: bool = typer.Option(False, "--exit-with-parent",
                                          help="Stop cleanly when stdin closes."),
) -> None:
    """Copy files into the kb's sink and index them. The copies are snapshots: later
    changes to the originals are not picked up (use `source add` to follow a folder).

    Exit codes: as `kb sync`; 3 also when a file was skipped (unsupported, unreadable)."""
    state = _state(ctx)
    if exit_with_parent:
        _stop_when_stdin_closes()
    try:
        with _progress(state) as progress:
            report = ingest_files(state.home, kb, paths, recursive=recursive, dry_run=dry_run,
                                  progress=progress)
    except SmartRAGError as err:
        emit_error(state.format, err)
        raise typer.Exit(1) from None
    _print_report(state, report, _render_ingest)
    code = _sync_exit_code(report.sync) if report.sync is not None else 0
    raise typer.Exit(code if code else (3 if report.skipped else 0))


@kb_app.command("reindex")
def kb_reindex(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    refresh_stale_only: bool = typer.Option(
        False, "--refresh-stale-only",
        help="Re-process only the documents built by an older pipeline version; new, changed "
             "and deleted files are left for the next sync."),
    rebuild: bool = typer.Option(False, "--rebuild",
                                 help="Build a fresh index beside the live one, then swap it in. "
                                      "Also how a kb changes its embedding model."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report the work; change nothing."),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation (--rebuild)."),
    exit_with_parent: bool = typer.Option(False, "--exit-with-parent",
                                          help="Stop cleanly when stdin closes."),
) -> None:
    """Re-process stale documents, or rebuild the index. Give exactly one of --refresh-stale-only and
    --rebuild. Exit codes as `kb sync`."""
    state = _state(ctx)
    if refresh_stale_only == rebuild:
        raise typer.BadParameter("give exactly one of --refresh-stale-only and --rebuild",
                                 param_hint="--refresh-stale-only / --rebuild")
    if exit_with_parent:
        _stop_when_stdin_closes()
    try:
        if not refresh_stale_only and not dry_run:
            confirm(state, yes, f"Rebuild the index of kb {kb!r}? Every file is read and "
                                "processed again; vectors are reused where the embedding "
                                "profile is unchanged. Search keeps working meanwhile.")
        with _progress(state) as progress:
            if refresh_stale_only:
                report = reindex_stale(state.home, kb, dry_run=dry_run, progress=progress)
            else:
                report = reindex_rebuild(state.home, kb, dry_run=dry_run, progress=progress)
    except SmartRAGError as err:
        emit_error(state.format, err)
        raise typer.Exit(1) from None
    _print_report(state, report, _render_sync)
    raise typer.Exit(_sync_exit_code(report))


def _sync_exit_code(report: SyncReport) -> int:
    if report.interrupted:
        return 130
    if report.error:
        return 1
    if report.counts.skipped or report.counts.failed:
        return 3
    return 0


def _stop_when_stdin_closes() -> None:
    """Interrupt the main thread when stdin reaches end-of-file, so the sync stops
    between documents, releases its lock and writes its report."""
    import _thread

    def watch() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except (OSError, ValueError):
            pass
        _thread.interrupt_main()

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


class SearchModeOption(str, Enum):
    hybrid = "hybrid"
    lexical = "lexical"
    vector = "vector"


class GroupByOption(str, Enum):
    chunk = "chunk"
    document = "document"


class IncludeOption(str, Enum):
    highlights = "highlights"
    text = "text"


class ScopeOption(str, Enum):
    chunk = "chunk"
    neighbors = "neighbors"
    section = "section"
    document = "document"


_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]


@kb_app.command("search")
def kb_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Natural language; a part in double quotes is an "
                                          "exact phrase on the keyword side."),
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    limit: int = typer.Option(10, "-k", "--limit", min=1, max=50, help="Results per page."),
    page: int = typer.Option(None, "--page", min=1, help="Page number (for people)."),
    cursor: str = typer.Option(None, "--cursor",
                               help="The next_cursor of a previous JSON result (for programs)."),
    mode: SearchModeOption = typer.Option(SearchModeOption.hybrid, "--mode",
                                          help="hybrid, lexical (keyword only) or vector."),
    group_by: GroupByOption = typer.Option(GroupByOption.chunk, "--group-by",
                                           help="chunk: passages; document: one per file."),
    max_per_doc: int = typer.Option(3, "--max-per-doc", min=1, max=50,
                                    help="At most this many chunks from one document."),
    source: list[str] = typer.Option(None, "--source", help="Only this source. Repeatable."),
    file_type: list[str] = typer.Option(None, "--type",
                                        help="Only this file type, e.g. pdf. Repeatable."),
    path: str = typer.Option(None, "--path", help="Only files under this folder of the source."),
    modified_after: datetime = typer.Option(None, "--modified-after", formats=_DATE_FORMATS,
                                            help="Files modified on or after this date."),
    modified_before: datetime = typer.Option(None, "--modified-before", formats=_DATE_FORMATS,
                                             help="Files modified before this date."),
    include: list[IncludeOption] = typer.Option(None, "--include",
                                                help="Extra fields: highlights, text. "
                                                     "Repeatable."),
    snippet_chars: int = typer.Option(300, "--snippet-chars", min=50, max=2000,
                                      help="Snippet length."),
) -> None:
    """Search the kb for evidence: ranked chunks with snippets and citations. An empty
    result list is a success (exit 0)."""
    state = _state(ctx)
    if page is not None and cursor:
        raise typer.BadParameter("give --page or --cursor, not both", param_hint="--page")
    filters = SearchFilter(source=source or [], file_type=file_type or [], path_prefix=path,
                           modified_after=modified_after, modified_before=modified_before)
    invoke(state, lambda: search_kb(
        state.home, kb, query, limit=limit, page=page, cursor=cursor, mode=mode.value,
        group_by=group_by.value, max_per_doc=max_per_doc, filters=filters,
        include=[i.value for i in include or []], snippet_chars=snippet_chars),
        _render_search)


@kb_app.command("read")
def kb_read(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    chunk: str = typer.Option(..., "--chunk", help="A chunk_id from a search result."),
    scope: ScopeOption = typer.Option(ScopeOption.chunk, "--scope",
                                      help="chunk, neighbors, section or document."),
    window: int = typer.Option(1, "--window", min=0, max=20,
                               help="For --scope neighbors: chunks on each side."),
    max_chars: int = typer.Option(4000, "--max-chars", min=200, max=100_000,
                                  help="Characters per page; pages break between chunks."),
    page: int = typer.Option(None, "--page", min=1, help="Page number (for people)."),
    cursor: str = typer.Option(None, "--cursor",
                               help="The next_cursor of a previous JSON result (for programs)."),
) -> None:
    """Read a chunk, widened to its neighbours, its section or its whole document."""
    state = _state(ctx)
    if page is not None and cursor:
        raise typer.BadParameter("give --page or --cursor, not both", param_hint="--page")
    invoke(state, lambda: read_chunk(state.home, kb, chunk, scope=scope.value, window=window,
                                     max_chars=max_chars, page=page, cursor=cursor),
           _render_read)


# --- source --------------------------------------------------------------------------


@source_app.command("list")
def source_list(ctx: typer.Context,
                kb: str = typer.Option(..., "--kb", help="The kb's id.")) -> None:
    """List a kb's sources, including its sink."""
    state = _state(ctx)
    invoke(state, lambda: list_sources(state.home, kb), _render_source_list)


@source_app.command("add")
def source_add(
    ctx: typer.Context,
    location: str = typer.Argument(..., help="A folder or file: a path or a file:// URI."),
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    source: str = typer.Option(..., "--source", help="Id for the new source."),
    include: list[str] = typer.Option(None, "--include",
                                      help="Only files matching this glob, e.g. '**/*.pdf'. "
                                           "Repeatable."),
    exclude: list[str] = typer.Option(None, "--exclude",
                                      help="Skip files matching this glob. Repeatable."),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive",
                                   help="Include subfolders."),
    allow_missing: bool = typer.Option(False, "--allow-missing",
                                       help="Accept a path that does not exist now "
                                            "(e.g. a share that is not mounted)."),
) -> None:
    """Register a folder (or file) that `sync` will follow. Nothing is copied."""
    state = _state(ctx)
    invoke(state, lambda: add_source(state.home, kb, source, location, recursive=recursive,
                                     include=include or [], exclude=exclude or [],
                                     allow_missing=allow_missing),
           _render_source_added)


@source_app.command("remove")
def source_remove(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    source: str = typer.Option(..., "--source", help="The source's id."),
    yes: bool = typer.Option(False, "--yes", help="Do not ask for confirmation."),
) -> None:
    """Unregister a source and remove its documents from the index."""
    state = _state(ctx)

    def action() -> SourceRemoved:
        listed = list_sources(state.home, kb)
        match = next((s for s in listed.sources if s.id == source), None)
        if match is None:
            raise NotFound(f"kb {kb!r} has no source {source!r}", {"kb": kb, "source": source})
        if match.kind == "source":
            confirm(state, yes, f"Remove source {source!r} ({match.root}) from kb {kb!r}, "
                                "with its documents in the index? Its files are not touched.")
        return remove_source(state.home, kb, source)

    invoke(state, action, _render_source_removed)


@source_app.command("test")
def source_test(
    ctx: typer.Context,
    kb: str = typer.Option(..., "--kb", help="The kb's id."),
    source: str = typer.Option(..., "--source", help="The source's id (or 'sink')."),
    sample: int = typer.Option(10, "--sample", min=0, max=1000,
                               help="How many matching paths to list."),
) -> None:
    """Show what a sync of this source would see. Reads no file contents; changes nothing."""
    state = _state(ctx)
    invoke(state, lambda: test_source(state.home, kb, source, sample=sample),
           _render_source_test, failed=lambda report: not report.ok)


# --- text rendering ------------------------------------------------------------------


def _render_kb_list(result: KbList) -> str:
    if not result.kbs:
        where = result.home if result.home_exists else f"{result.home} (does not exist yet)"
        return f"No kbs in {where}.\nCreate one: smartrag run kb new --kb <id>"
    rows = []
    for item in result.kbs:
        status = item.status if item.error is None else f"{item.status}: {item.error}"
        rows.append([item.kb, item.title, item.kind, item.sources, status])
    return f"home: {result.home}\n\n" + table(["KB", "TITLE", "KIND", "SOURCES", "STATUS"], rows)


def _render_kb_created(result: KbCreated) -> str:
    return (f"Created {result.kind} kb {result.kb!r} at {result.path}\n"
            f"Config: {result.config}\n\nNext:\n{bullets(result.next_steps)}")


def _render_kb_updated(result: KbUpdated) -> str:
    head = (f"Updated kb {result.kb!r}: {', '.join(result.changed)}" if result.changed
            else f"kb {result.kb!r} already has this description; nothing changed")
    text = head + "\n\ndescription\n" + ("\n".join(
        f"  {line}" for line in result.description.splitlines()) or "  (none)")
    return text + ("\n\n" + "\n".join(result.notes) if result.notes else "")


def _render_kb_status(result: KbStatus) -> str:
    env = ", ".join(f"{name} ({'missing' if name in result.env.missing else 'set'})"
                    for name in result.env.referenced) or "-"
    if result.lock is None:
        lock = "free"
    elif result.lock.stale:
        lock = f"stale (left by pid {result.lock.pid})"
    else:
        lock = f"held by pid {result.lock.pid} ({result.lock.command or 'unknown'})"
    rows: list[list[object]] = [
        ["kb", result.kb], ["title", result.title], ["kind", result.kind],
        ["path", result.path],
    ]
    if result.remote is not None:
        rows += [["endpoint", result.remote.endpoint], ["remote kb", result.remote.kb]]
    else:
        index = result.index
        rows += [
            ["profile", result.embedding_profile],
            ["sources", result.sources],
            ["sink", f"{result.sink_files or 0} files, {human_bytes(result.sink_bytes or 0)}"],
            ["index", _index_line(index)],
        ]
    rows += [["lock", lock], ["env", env]]
    text = "\n".join(f"{label:<11}{value}" for label, value in rows)
    description = result.description.strip()
    text += "\n\ndescription\n" + ("\n".join(f"  {line}" for line in description.splitlines())
                                   if description else "  (none)")
    if result.warnings:
        text += "\n\nwarnings\n" + bullets(result.warnings)
    return text


def _index_line(index: object) -> str:
    from smartrag.kb.models import IndexInfo

    if not isinstance(index, IndexInfo) or not index.exists:
        return "not built"
    if index.documents is None:
        return human_bytes(index.bytes)
    return (f"{index.ok} documents, {index.chunks:,} chunks, {human_bytes(index.bytes)}"
            + (f" ({index.skipped} skipped, {index.failed} failed)"
               if index.skipped or index.failed else "")
            + (f"; last sync {index.last_sync_at}" if index.last_sync_at else ""))


def _render_kb_removed(result: KbRemoved) -> str:
    parts = [f"Deleted kb {result.kb!r} ({result.path})."]
    details = []
    if result.index_deleted:
        details.append("its index")
    if result.sink_files_deleted:
        details.append(f"{result.sink_files_deleted} files copied into its sink")
    if details:
        parts.append("Removed " + " and ".join(details) + ".")
    if result.kept_sink:
        parts.append(f"Its sink outside the kb folder was kept: {result.kept_sink}")
    parts.append("Original files were not touched.")
    return " ".join(parts)


_OPERATION_TITLES = {"sync": "Synced", "reindex-stale": "Re-processed stale documents of",
                     "reindex-rebuild": "Rebuilt the index of", "ingest": "Synced the sink of"}


def _render_sync(r: SyncReport) -> str:
    c = r.counts
    head = "Dry run:" if r.dry_run else _OPERATION_TITLES.get(r.operation, "Synced")
    lines = [f"{head} kb {r.kb!r} in {r.took_ms / 1000:.1f} s"]
    if r.error:
        lines.append(f"stopped: {r.error}")
    if r.interrupted:
        lines.append("interrupted: run it again to resume")
    lines.append(f"documents  {c.added} added, {c.updated} updated, {c.unchanged} unchanged, "
                 f"{c.removed} removed, {c.skipped} skipped, {c.failed} failed"
                 + (f", {c.deferred} left for the next sync" if c.deferred else ""))
    if r.dry_run:
        lines.append(f"chunks     {r.chunks_written:,} (about {r.tokens_estimated:,} tokens "
                     f"to embed)")
    else:
        lines.append(f"chunks     {r.chunks_written:,} written: {r.chunks_embedded:,} embedded, "
                     f"{r.chunks_reused:,} reused (about {r.tokens_estimated:,} tokens)")
    t = r.timings_ms
    lines.append("time       " + ", ".join(f"{k} {v / 1000:.1f} s" for k, v in t.items()))
    for s in r.sources:
        state = "" if s.complete else " (incomplete listing)"
        note = f": {s.note}" if s.note else ""
        lines.append(f"source     {s.source}: {s.files} files{state}"
                     + (f", {s.removed} removed" if s.removed else "") + note)
    if r.issues:
        lines.append("problems")
        lines += [f"  {i.status}: {i.source}/{i.path}: {i.reason}" for i in r.issues[:20]]
        if r.issues_total > 20:
            where = "listed in the report" if r.report_path else "run with --format json"
            lines.append(f"  ... and {r.issues_total - 20} more ({where})")
    if r.report_path:
        lines.append(f"report     {r.report_path}")
    return "\n".join(lines)


def _pages(page: int | None, page_end: int | None) -> str:
    if page is None:
        return ""
    return f"p.{page}" if page_end in (None, page) else f"p.{page}–{page_end}"


def _render_search(r: KbSearchResponse) -> str:
    lines: list[str] = []
    if r.degraded:
        lines.append(f"note: {r.degraded}")
    if not r.results:
        lines.append(f"No results for {r.query!r} ({r.mode}, {r.took_ms} ms).")
        return "\n".join(lines)
    for hit in r.results:
        c = hit.citation
        where = "  ".join(part for part in (
            _pages(c.page, c.page_end), c.location or "", " > ".join(c.heading_path)) if part)
        sides = " · ".join(f"{side} {rank}" for side, rank in
                           (("lexical", hit.ranks.lexical), ("vector", hit.ranks.vector))
                           if rank is not None)
        lines.append(f"{hit.rank}. {hit.title}  {where}".rstrip() + f"    [{sides}]")
        lines.append(f"   {hit.snippet}")
        where_from = f"sink, from {c.origin_path}" if c.origin_path else f"{c.source_id}/{c.path}"
        lines.append(f"   {where_from} · chunk {hit.chunk_id}")
        if hit.text:
            lines.append("   " + hit.text.replace("\n", "\n   "))
    more = f" · more: --page {r.page + 1}" if r.next_cursor else ""
    lines.append(f"page {r.page} · {len(r.results)} results · {r.mode} · {r.took_ms} ms{more}")
    return "\n".join(lines)


def _render_read(r: KbReadResponse) -> str:
    c = r.citation
    head = f"{r.title}  {c.source_id}/{c.path}  ({r.scope}, {r.total_chunks} chunks)"
    lines = [head]
    path: list[str] = []
    for chunk in r.chunks:
        if chunk.heading_path != path:
            path = chunk.heading_path
            if path:
                lines.append(f"── {' > '.join(path)}")
        marker = "  ".join(part for part in (_pages(chunk.page, chunk.page_end),
                                              chunk.location or "") if part)
        lines.append(f"[{chunk.ordinal}] {marker}".rstrip() + (" ◀" if chunk.chunk_id
                                                                == r.chunk_id else ""))
        if chunk.prefix:
            lines.append(chunk.prefix)
        lines.append(chunk.text)
    if r.truncated:
        lines.append("(a chunk longer than --max-chars was cut)")
    more = f" · more: --page {r.page + 1}" if r.next_cursor else ""
    lines.append(f"page {r.page}{more}")
    return "\n".join(lines)


def _render_ingest(r: IngestReport) -> str:
    verb = {"copied": "copied", "would_copy": "would copy", "already": "already in the sink",
            "skipped": "skipped"}
    lines = [f"{'Dry run: ' if r.dry_run else ''}ingest into kb {r.kb!r}"]
    for item in r.files[:50]:
        reason = f" ({item.reason})" if item.reason and item.status == "skipped" else ""
        lines.append(f"  {verb[item.status]}: {item.path}{reason}")
    if len(r.files) > 50:
        lines.append(f"  ... and {len(r.files) - 50} more")
    to_copy = len(r.files) - r.already - r.skipped if r.dry_run else r.copied
    lines.append(f"files      {to_copy} {'to copy' if r.dry_run else 'copied'} "
                 f"({human_bytes(r.bytes_copied)}), {r.already} already in the sink, "
                 f"{r.skipped} skipped")
    if r.dry_run:
        lines.append(f"chunks     about {r.chunks_estimated:,} "
                     f"(about {r.tokens_estimated:,} tokens to embed)")
    if r.sync is not None:
        lines.append(_render_sync(r.sync))
    if r.copied or r.dry_run:
        lines.append(r.note)
    return "\n".join(lines)


def _render_source_list(result: SourceList) -> str:
    rows = []
    for item in result.sources:
        root = item.root if item.root_exists else f"{item.root} (missing)"
        rows.append([item.id, item.kind, root, "yes" if item.recursive else "no",
                     ", ".join(item.include), ", ".join(item.exclude)])
    return (table(["ID", "KIND", "ROOT", "RECURSIVE", "INCLUDE", "EXCLUDE"], rows)
            + "\n\nAlways skipped: " + ", ".join(result.builtin_exclude))


def _render_source_added(result: SourceAdded) -> str:
    item = result.source
    text = (f"Added source {item.id!r} to kb {result.kb!r}: {item.root}"
            f"{' (recursive)' if item.recursive else ''}")
    if not item.root_exists:
        text += "\n(the path does not exist right now)"
    if result.warnings:
        text += "\n\nwarnings\n" + bullets(result.warnings)
    return text + (f"\n\nNext: smartrag run source test --kb {result.kb} --source {item.id}")


def _render_source_removed(result: SourceRemoved) -> str:
    text = f"Removed source {result.source!r} ({result.uri}) from kb {result.kb!r}."
    if result.documents_removed is not None:
        text += f" {result.documents_removed} documents were removed from the index."
    return f"{text}\n{result.note}" if result.note else text


def _render_source_test(r: SourceTestReport) -> str:
    shape = f"{r.root_kind}{', recursive' if r.recursive and r.root_kind == 'directory' else ''}"
    lines = [f"source   {r.source} -> {r.root} ({shape})"]
    if r.root_kind in ("directory", "file"):
        lines.append(f"matched  {r.files_matched:,} files, {human_bytes(r.bytes_matched)} "
                     f"(supported {r.supported_files:,}, unsupported {r.unsupported_files:,})")
        if r.by_extension:
            kinds = sorted(r.by_extension.items(), key=lambda kv: (-kv[1], kv[0]))
            lines.append("by type  " + ", ".join(f"{ext} {count:,}" for ext, count in kinds))
        lines.append(f"skipped  built-in {r.excluded_builtin:,}, excluded {r.excluded_by_pattern:,}"
                     f", not included {r.not_included:,}, symlinks {r.symlinks_skipped:,}")
    if r.errors:
        lines.append("errors")
        lines += [f"  {e.path}: {e.error}" for e in r.errors]
        if r.errors_total > len(r.errors):
            lines.append(f"  ... and {r.errors_total - len(r.errors)} more")
    if r.sample:
        lines.append("sample")
        lines += [f"  {path}" for path in r.sample]
    if r.problems:
        lines.append("problems")
        lines.append(bullets(r.problems))
    lines.append(f"took {r.took_ms} ms")
    return "\n".join(lines)
