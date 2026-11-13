"""The Smart RAG home: one directory per knowledge base.

```
<home>/
  legal/
    config.yaml        this kb's settings (secrets only as ${ENV} references)
    index.db           SQLite index, created by the first sync
    sink/              files copied in by `kb ingest`
    cache/  runs/      derived vector cache; per-run sync reports
    smartrag.lock      held by a sync or a config change
  factory/
    config.yaml        kind: remote — endpoint and token reference only
```

`--home` > `$SMARTRAG_HOME` > `~/.smartrag`. A kb's id is its directory name.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from pydantic import ValidationError

from smartrag.errors import ConfigError, Conflict, InvalidRequest, NotFound
from smartrag.kb.config import (
    CONFIG_FILE,
    DEFAULT_EMBED_KEY_ENV,
    DEFAULT_PROFILE,
    ENV_REF,
    ID_PATTERN,
    MAX_DESCRIPTION,
    SINK_ID,
    ConfigDocument,
    EmbeddingConfig,
    KbConfig,
    RemoteConfig,
    check_description,
    check_id,
    describe_validation_error,
    description_length,
)
from smartrag.kb.lock import kb_lock, read_lock
from smartrag.kb.models import (
    EnvInfo,
    IndexInfo,
    KbCreated,
    KbList,
    KbRemoved,
    KbStatus,
    KbSummary,
    KbUpdated,
    RemoteInfo,
)
from smartrag.kb.uris import uri_to_path

HOME_ENV = "SMARTRAG_HOME"
DEFAULT_HOME = "~/.smartrag"
INDEX_FILE = "index.db"
LOCAL_SUBDIRS = ("cache", "runs")


def resolve_home(explicit: str | Path | None = None) -> Path:
    raw = explicit or os.environ.get(HOME_ENV) or DEFAULT_HOME
    return Path(raw).expanduser().absolute()


def tree_stats(root: Path) -> tuple[int, int]:
    """(files, bytes) under `root`, not following symlinks."""
    files = size = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                info = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            files += 1
            size += info.st_size
    return files, size


class Home:
    def __init__(self, root: Path) -> None:
        self.root = root

    # --- locating ------------------------------------------------------------------

    def kb_dir(self, kb_id: str) -> Path:
        try:
            check_id(kb_id, "kb id")
        except ValueError as exc:
            raise InvalidRequest(str(exc), {"kb": kb_id}) from exc
        return self.root / kb_id

    def document(self, kb_id: str) -> ConfigDocument:
        """The kb's config.yaml, editable in place."""
        kb_dir = self.kb_dir(kb_id)
        if not (kb_dir / CONFIG_FILE).is_file():
            raise NotFound(f"no kb {kb_id!r} in {self.root}",
                           {"kb": kb_id, "home": str(self.root)})
        document = ConfigDocument.load(kb_dir / CONFIG_FILE)
        if document.model.id != kb_id:
            raise ConfigError(
                f"{kb_dir / CONFIG_FILE}: id {document.model.id!r} does not match its "
                f"directory name {kb_id!r}", {"kb": kb_id})
        return document

    def load(self, kb_id: str) -> KbConfig:
        return self.document(kb_id).model

    def sink_dir(self, config: KbConfig) -> Path:
        sink = Path(config.sink or SINK_ID).expanduser()
        return sink if sink.is_absolute() else self.kb_dir(config.id) / sink

    # --- kb list -------------------------------------------------------------------

    def list(self) -> KbList:
        kbs: list[KbSummary] = []
        if self.root.is_dir():
            for entry in sorted(self.root.iterdir()):
                if not (entry.is_dir() and ID_PATTERN.match(entry.name)
                        and (entry / CONFIG_FILE).is_file()):
                    continue
                kbs.append(self._summary(entry.name))
        return KbList(home=str(self.root), home_exists=self.root.is_dir(), kbs=kbs)

    def _summary(self, kb_id: str) -> KbSummary:
        try:
            config = self.load(kb_id)
        except ConfigError as exc:
            return KbSummary(kb=kb_id, status="config_error", error=exc.message)
        holder = read_lock(self.kb_dir(kb_id))
        return KbSummary(
            kb=kb_id, title=config.title, kind=config.kind,
            status="locked" if holder is not None and not holder.stale else "ok",
            sources=len(config.sources) if config.kind == "local" else None)

    # --- kb new --------------------------------------------------------------------

    def new(self, kb_id: str, *, title: str | None = None, description: str = "",
            profile: str = DEFAULT_PROFILE, remote: str | None = None,
            remote_kb: str | None = None, token_env: str | None = None) -> KbCreated:
        kb_dir = self.kb_dir(kb_id)
        if kb_dir.exists():
            raise Conflict(f"kb {kb_id!r} already exists in {self.root}", {"kb": kb_id})
        description = _checked_description(description)
        config = self._new_config(kb_id, title, description, profile, remote, remote_kb,
                                  token_env)

        created_root = not self.root.exists()
        self.root.mkdir(parents=True, exist_ok=True)
        if created_root:
            os.chmod(self.root, 0o700)
        try:
            kb_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise Conflict(f"kb {kb_id!r} already exists in {self.root}", {"kb": kb_id}) from exc
        try:
            if config.kind == "local":
                for name in (SINK_ID, *LOCAL_SUBDIRS):
                    (kb_dir / name).mkdir(mode=0o700)
            ConfigDocument.create(kb_dir / CONFIG_FILE, config).save()
        except BaseException:
            shutil.rmtree(kb_dir, ignore_errors=True)
            raise
        return KbCreated(kb=kb_id, kind=config.kind, path=str(kb_dir),
                         config=str(kb_dir / CONFIG_FILE), next_steps=_next_steps(config))

    @staticmethod
    def _new_config(kb_id: str, title: str | None, description: str, profile: str,
                    remote: str | None, remote_kb: str | None,
                    token_env: str | None) -> KbConfig:
        if token_env is not None and not ENV_REF.match(f"${{{token_env}}}"):
            raise InvalidRequest(f"--token-env must be an environment variable name, "
                                 f"not {token_env!r}")
        try:
            if remote is not None:
                if not token_env:
                    raise InvalidRequest(
                        "a remote kb needs --token-env: the environment variable that holds "
                        "your token for that server (a remote kb without credentials is a "
                        "configuration error)")
                return KbConfig(
                    id=kb_id, title=title or kb_id, description=description, kind="remote",
                    remote=RemoteConfig(endpoint=remote, kb=remote_kb or kb_id,
                                        token=f"${{{token_env}}}"))
            if remote_kb is not None or token_env is not None:
                raise InvalidRequest("--remote-kb and --token-env apply only with --remote")
            return KbConfig(id=kb_id, title=title or kb_id, description=description,
                            embedding=EmbeddingConfig(profile=profile,
                                                      api_key=f"${{{DEFAULT_EMBED_KEY_ENV}}}"))
        except ValidationError as exc:
            raise InvalidRequest(describe_validation_error(exc), {"kb": kb_id}) from exc

    # --- kb status -----------------------------------------------------------------

    def status(self, kb_id: str) -> KbStatus:
        config = self.load(kb_id)
        kb_dir = self.kb_dir(kb_id)
        missing = config.missing_env()
        warnings: list[str] = []
        if not config.description.strip():
            warnings.append("no description: agents do not see this kb until it has one; "
                            f"set it with `smartrag run kb set --kb {kb_id} --description`")
        elif description_length(config.description) > MAX_DESCRIPTION:
            warnings.append(f"the description is {description_length(config.description)} "
                            f"characters; agents see only the first {MAX_DESCRIPTION}")
        if missing:
            warnings.append(f"environment variables not set: {', '.join(missing)}")
        holder = read_lock(kb_dir)
        if holder is not None and holder.stale:
            warnings.append(f"stale lock left by pid {holder.pid}; the next writer takes "
                            "it over")

        status = KbStatus(
            kb=kb_id, title=config.title, kind=config.kind, path=str(kb_dir),
            description=config.description, sources=len(config.sources), lock=holder,
            env=EnvInfo(referenced=config.env_refs(), missing=missing), warnings=warnings)
        warnings = status.warnings  # the model holds a copy; keep appending to that one

        if config.kind == "remote":
            assert config.remote is not None
            status.remote = RemoteInfo(endpoint=config.remote.endpoint, kb=config.remote.kb)
            return status

        assert config.embedding is not None
        status.embedding_profile = config.embedding.profile
        sink = self.sink_dir(config)
        if sink.is_dir():
            status.sink_files, status.sink_bytes = tree_stats(sink)
        else:
            warnings.append(f"sink directory {sink} is missing")
        index = kb_dir / INDEX_FILE
        status.index = IndexInfo(exists=index.is_file(), path=str(index),
                                 bytes=index.stat().st_size if index.is_file() else None)
        if not index.is_file():
            warnings.append("no index yet: run `smartrag run kb sync`")
        else:
            _index_status(status.index, config, warnings)
        if not config.sources:
            warnings.append("no sources: add one with `smartrag run source add`")
        for source in config.sources:
            root = uri_to_path(source.uri)
            if root is not None and not root.exists():
                warnings.append(f"source {source.id!r}: {root} does not exist (not mounted?)")
        return status

    # --- kb set --------------------------------------------------------------------

    def set(self, kb_id: str, *, description: str | None = None) -> KbUpdated:
        """Change the settings people write: v1 the description (`--title` later).
        Edits config.yaml in place, under the kb lock."""
        if description is None:
            raise InvalidRequest("nothing to set: give --description")
        text = _checked_description(description)
        self.document(kb_id)  # NotFound before taking a lock in a folder that is not a kb
        changed: list[str] = []
        with kb_lock(self.kb_dir(kb_id), "kb set"):
            document = self.document(kb_id)
            if document.model.description != text:
                document.set_description(text)
                document.save()
                changed.append("description")
        notes = []
        if not text:
            notes.append("With no description, agents do not see this kb.")
        elif changed:
            notes.append("A running `smartrag mcp` lists the new text at once, and puts it in "
                         "its instructions when it restarts.")
        return KbUpdated(kb=kb_id, changed=changed, description=text, notes=notes)

    # --- kb remove -----------------------------------------------------------------

    def remove(self, kb_id: str) -> KbRemoved:
        """Delete the kb's directory: config, index, cache, reports and the sink's copies.

        The user's original files are never touched. A sink configured outside the kb's
        directory is left in place, since it may be a folder the user also uses."""
        kb_dir = self.kb_dir(kb_id)
        if kb_dir.is_symlink():
            raise InvalidRequest(f"{kb_dir} is a symbolic link; remove it by hand")
        if not (kb_dir / CONFIG_FILE).is_file():
            raise NotFound(f"no kb {kb_id!r} in {self.root}",
                           {"kb": kb_id, "home": str(self.root)})
        try:
            config: KbConfig | None = self.load(kb_id)
        except ConfigError:
            config = None  # a broken kb can still be removed
        with kb_lock(kb_dir, "kb remove"):
            if config is None:
                sink: Path | None = kb_dir / SINK_ID  # unreadable config: assume the default
            else:
                sink = self.sink_dir(config) if config.kind == "local" else None
            inside = sink is not None and kb_dir in sink.parents
            sink_files = tree_stats(sink)[0] if sink is not None and inside and sink.is_dir() \
                else 0
            index_existed = (kb_dir / INDEX_FILE).is_file()
            shutil.rmtree(kb_dir)
        return KbRemoved(kb=kb_id, path=str(kb_dir), sink_files_deleted=sink_files,
                         index_deleted=index_existed,
                         kept_sink=str(sink) if sink is not None and not inside else None)


def _checked_description(text: str) -> str:
    try:
        return check_description(text)
    except ValueError as exc:
        raise InvalidRequest(str(exc), {"limit": MAX_DESCRIPTION}) from exc


def _next_steps(config: KbConfig) -> list[str]:
    missing = config.missing_env()
    steps: list[str] = []
    if missing:
        steps.append(f"Set {', '.join(missing)} in the environment.")
    if config.kind == "local":
        steps.append(f"Add a folder: smartrag run source add --kb {config.id} "
                     f"--source <id> <folder>")
    if not config.description.strip():
        steps.append(f"Describe it: smartrag run kb set --kb {config.id} --description "
                     "\"...\" (agents choose a kb by it, and do not see one without it)")
    steps.append(f"Check it: smartrag run kb status --kb {config.id}")
    return steps


def _index_status(info: IndexInfo, config: KbConfig, warnings: list[str]) -> None:
    from smartrag.index import SqliteIndex

    try:
        store = SqliteIndex.open(Path(info.path), readonly=True)
    except Exception as exc:  # noqa: BLE001 - status must report, not fail
        warnings.append(f"the index cannot be read: {exc}")
        return
    try:
        stats = store.stats()
        info.last_sync_at = store.meta("last_sync_at")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"the index cannot be read: {exc}")
        return
    finally:
        store.close()
    info.documents, info.ok, info.chunks = stats.documents, stats.ok, stats.chunks
    info.skipped, info.failed, info.profile = stats.skipped, stats.failed, stats.profile
    assert config.embedding is not None
    if stats.profile and stats.profile != config.embedding.profile:
        warnings.append(f"the index holds vectors from {stats.profile!r} but the config "
                        f"says {config.embedding.profile!r}")
    if stats.skipped or stats.failed:
        warnings.append(f"{stats.skipped} documents skipped and {stats.failed} failed; "
                        "the last sync report in runs/ lists them")
