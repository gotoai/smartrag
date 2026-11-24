"""What the kb and source operations return.

The CLI prints these as text or as JSON (`--format json`); later adapters return the
same models, so each operation's result is defined once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr

from smartrag.kb.lock import LockHolder
from smartrag.tables.models import TableSync

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
    vector_space: str | None = None
    last_sync_at: str | None = None


class EmbeddingInfo(BaseModel):
    """A local kb's embedding provider and model, and the vector space they give."""

    provider_id: str
    model_profile_id: str
    model_canonical_name: str
    vector_space: str


class KbVerified(BaseModel):
    """One test embedding through a kb's provider and model succeeded."""

    kb: str
    provider_id: str
    model_profile_id: str
    vector_space: str
    url: str
    took_ms: int


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
    embedding: EmbeddingInfo | None = None
    remote: RemoteInfo | None = None
    sources: int
    sink_files: int | None = None
    sink_bytes: int | None = None
    index: IndexInfo | None = None
    lock: LockHolder | None = None
    env: EnvInfo
    warnings: list[str] = Field(default_factory=list)


class KbUpdated(BaseModel):
    kb: str
    changed: list[str] = Field(description="The settings that changed; empty if none did.")
    description: str
    notes: list[str] = Field(default_factory=list)


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


class AttributesSync(BaseModel):
    """What `sync` (or `reindex`) did with one source's attributes folder."""

    source: str
    folder: str
    status: Literal["applied", "unchanged", "removed", "kept", "none"] = Field(
        description="applied: read and written (a dry run: would be); unchanged: same as "
                    "applied before; removed: the folder is gone, so were the attributes; "
                    "kept: the rows file could not be read, or the listing was incomplete, "
                    "so the attributes applied before stay; none: no folder.")
    rows: int = Field(default=0, description="Rows read and matched to a file.")
    keys: list[str] = Field(default_factory=list, description="The attribute names, as "
                                                                "stored (escaped).")
    has_schema: bool = False
    renamed: dict[str, str] = Field(default_factory=dict, description="Names that clash with "
                                    "Smart RAG's own fields, and what they became.")
    untyped: list[str] = Field(default_factory=list, description="Keys the schema does not "
                               "declare (all, without one): read as strings.")
    skipped_lines: int = 0
    dropped_values: int = 0
    documents_without_row: int = 0
    warnings: list[str] = Field(default_factory=list)


class PathError(BaseModel):
    path: str
    error: str


class SourceTestReport(BaseModel):
    """What a sync of this source would see. File contents are not read, except the first
    8 KiB of a file whose extension names no extractor, to tell whether it is text."""

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
    _unsupported_kinds: set[str] = PrivateAttr(default_factory=set)


class SyncCounts(BaseModel):
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    skipped: int = 0
    failed: int = 0
    deferred: int = 0
    """`reindex --refresh-stale-only` only: new or changed files left for the next sync."""


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
    operation: Literal["sync", "reindex-stale", "reindex-rebuild", "ingest"] = "sync"
    dry_run: bool
    started_at: str
    finished_at: str | None = None
    took_ms: int = 0
    provider_id: str
    model_profile_id: str
    vector_space: str
    pipeline_version: str
    embedding_parallelism: int = 1
    """The most embedding requests in flight at once during this run."""
    resumed_documents: int = 0
    """`reindex --rebuild`: documents a stopped rebuild had done, kept and not redone."""
    adapt_to_rate_limit: bool = True
    """Rate limits were waited out rather than stopping the run."""
    rate_limit_waits: int = 0
    rate_limit_wait_ms: int = 0
    """Time spent waiting out rate limits, summed over the requests in flight."""
    sources: list[SourceSync] = Field(default_factory=list)
    attributes: list[AttributesSync] = Field(default_factory=list)
    tables: list[TableSync] = Field(default_factory=list)
    """What the sync did with each table (`smartrag.tables`); none with `--source`."""
    """Per source, its attributes folder; the sink has none."""
    counts: SyncCounts = Field(default_factory=SyncCounts)
    chunks_written: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    tokens_estimated: int = 0
    """Smart RAG's estimate of the chunk text in the files processed this run, embedded or
    reused; for a dry run, what would be embedded."""
    tokens_embedded: int = 0
    """Tokens sent for embedding: the provider's count where its answers gave one, else
    Smart RAG's estimate. Reused vectors cost nothing and are not counted."""
    tokens_embedded_estimated: int = 0
    """The part of `tokens_embedded` that Smart RAG estimated."""
    timings_ms: dict[str, int] = Field(default_factory=dict)
    issues: list[SyncIssue] = Field(default_factory=list)
    issues_total: int = 0
    interrupted: bool = False
    error: str | None = None
    report_path: str | None = None
