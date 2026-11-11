"""What the kb and source operations return.

The CLI prints these as text or as JSON (`--format json`); later adapters return the
same models, so each operation's result is defined once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from smartrag.kb.lock import LockHolder

KbKind = Literal["local", "remote"]


class KbSummary(BaseModel):
    kb: str
    title: str | None = None
    kind: KbKind | None = None
    status: Literal["ok", "config_error", "locked"]
    sources: int | None = None
    error: str | None = None


class KbList(BaseModel):
    home: str
    home_exists: bool
    kbs: list[KbSummary]


class KbCreated(BaseModel):
    kb: str
    kind: KbKind
    path: str
    config: str
    next_steps: list[str]


class IndexInfo(BaseModel):
    exists: bool
    path: str
    bytes: int | None = None
    documents: int | None = None
    ok: int | None = None
    skipped: int | None = None
    failed: int | None = None
    chunks: int | None = None
    profile: str | None = None
    last_sync_at: str | None = None


class EnvInfo(BaseModel):
    referenced: list[str]
    missing: list[str]


class RemoteInfo(BaseModel):
    endpoint: str
    kb: str


class KbStatus(BaseModel):
    kb: str
    title: str
    kind: KbKind
    path: str
    description: str
    embedding_profile: str | None = None
    remote: RemoteInfo | None = None
    sources: int
    sink_files: int | None = None
    sink_bytes: int | None = None
    index: IndexInfo | None = None
    lock: LockHolder | None = None
    env: EnvInfo
    warnings: list[str] = Field(default_factory=list)


class KbRemoved(BaseModel):
    kb: str
    path: str
    sink_files_deleted: int
    index_deleted: bool
    kept_sink: str | None = Field(
        default=None, description="A sink outside the kb's directory, left in place.")


class SourceView(BaseModel):
    id: str
    kind: Literal["source", "sink"]
    uri: str
    root: str
    root_exists: bool
    recursive: bool = True
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class SourceList(BaseModel):
    kb: str
    sources: list[SourceView]
    builtin_exclude: list[str]


class SourceAdded(BaseModel):
    kb: str
    source: SourceView
    warnings: list[str] = Field(default_factory=list)


class SourceRemoved(BaseModel):
    kb: str
    source: str
    uri: str
    documents_removed: int | None = Field(
        default=None, description="None until the kb has an index.")
    note: str | None = None


class PathError(BaseModel):
    path: str
    error: str


class SourceTestReport(BaseModel):
    """What a sync of this source would see, without reading any file's contents."""

    kb: str
    source: str
    uri: str
    root: str
    ok: bool
    root_kind: Literal["directory", "file", "missing", "other"]
    recursive: bool
    files_matched: int = 0
    bytes_matched: int = 0
    supported_files: int = 0
    unsupported_files: int = 0
    by_extension: dict[str, int] = Field(default_factory=dict)
    excluded_builtin: int = 0
    excluded_by_pattern: int = 0
    not_included: int = 0
    symlinks_skipped: int = 0
    errors: list[PathError] = Field(default_factory=list)
    errors_total: int = 0
    sample: list[str] = Field(default_factory=list)
    problems: list[str] = Field(default_factory=list)
    took_ms: int = 0


class SyncCounts(BaseModel):
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    failed: int = 0


class SyncIssue(BaseModel):
    source: str
    path: str
    status: Literal["skipped", "failed"]
    reason: str


class SourceSync(BaseModel):
    source: str
    root: str
    files: int = 0
    complete: bool = True
    removed: int = 0
    note: str | None = None


class SyncReport(BaseModel):
    """One sync run. Counts cover this run only: a file skipped in an earlier run and
    unchanged since is `unchanged`, not `skipped` again."""

    kb: str
    dry_run: bool
    started_at: str
    finished_at: str | None = None
    took_ms: int = 0
    profile: str
    pipeline_version: str
    sources: list[SourceSync] = Field(default_factory=list)
    counts: SyncCounts = Field(default_factory=SyncCounts)
    chunks_written: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    tokens_estimated: int = 0
    timings_ms: dict[str, int] = Field(default_factory=dict)
    issues: list[SyncIssue] = Field(default_factory=list)
    issues_total: int = 0
    interrupted: bool = False
    error: str | None = None
    report_path: str | None = None
