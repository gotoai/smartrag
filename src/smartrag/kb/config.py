"""A knowledge base's `config.yaml`: its schema, and how it is read and written.

One config per kb (design memo §3.6). The schema enforces three rules from the design
rather than leaving them to whoever edits the file:

* **Secrets appear only as `${ENV_VAR}` references.** `embedding.api_key` and
  `remote.token` are rejected if they hold anything else, so a key pasted into the file
  fails loudly instead of sitting in plain text in the user's home.
* **A remote kb is one hop and nothing more.** `kind: remote` carries an endpoint, the
  kb's id on that server and a token reference; it has no sources, no sink, no store and
  no field through which a further hop could be expressed. Unknown keys are refused, so
  such a field cannot be added by hand either.
* **Remote kbs use TLS**, except on a loopback address, where plain HTTP is allowed for
  development.

The file is edited in place (`ConfigDocument`): a command changes only the keys it owns,
and the user's comments, key order and quoting survive.
"""

from __future__ import annotations

import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError
from ruamel.yaml.scalarstring import LiteralScalarString

from smartrag.errors import ConfigError

CONFIG_FILE = "config.yaml"
CONFIG_VERSION = 1
SINK_ID = "sink"
DEFAULT_PROFILE = "ruri-v3-310m@8k"
DEFAULT_EMBED_URL_ENV = "SMARTRAG_EMBED_BASE_URL"
DEFAULT_EMBED_KEY_ENV = "SMARTRAG_EMBED_API_KEY"
MAX_DESCRIPTION = 300
"""Characters of a kb's description, counted as agents see it (one paragraph). Agents
choose a kb by it, and the MCP server puts it in every session's instructions."""

#: Lowercase only: a kb id is a directory name, and macOS and Windows file systems
#: ignore case, so `Legal` and `legal` would be two ids for one directory.
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_ENV_REF_ANYWHERE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_HEADER = (
    "Smart RAG knowledge base. `smartrag run` edits this file in place and keeps comments.\n"
    "Secrets appear only as ${ENV_VAR} references, never as values."
)


def check_id(value: str, what: str) -> str:
    if not ID_PATTERN.match(value):
        raise ValueError(
            f"invalid {what} {value!r}: use 1-64 lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit")
    return value


def description_length(text: str) -> int:
    """Its length as agents see it: whitespace runs, line breaks included, count as one."""
    return len(" ".join(text.split()))


def check_description(text: str) -> str:
    """The description, stripped. Enforced where it is written (`kb new`, `kb set`), not on
    load: a longer hand-edited one still loads, and `kb status` warns."""
    text = text.strip()
    length = description_length(text)
    if length > MAX_DESCRIPTION:
        raise ValueError(f"the description is {length} characters; the limit is "
                         f"{MAX_DESCRIPTION}, since agents read it in every session")
    return text


def _secret_ref(value: str | None) -> str | None:
    if value is not None and not ENV_REF.match(value):
        raise ValueError(
            f"must be an environment variable reference such as "
            f"${{{DEFAULT_EMBED_KEY_ENV}}}, never the secret itself")
    return value


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingConfig(_Strict):
    """Where this kb's vectors come from. The profile id is recorded with every vector:
    a different model is a different vector space."""

    base_url: str = f"${{{DEFAULT_EMBED_URL_ENV}}}"
    """The endpoint's base (`…/v1`) or its full `…/v1/embeddings` URL."""
    api_key: str | None = None
    """Omit (or `null`) for an endpoint without authentication; no header is sent."""
    profile: str = DEFAULT_PROFILE
    batch_size: int = Field(default=32, ge=1, le=2048)
    """Texts per request. Keep within the endpoint's per-request input cap."""

    @field_validator("api_key")
    @classmethod
    def _api_key_is_ref(cls, value: str | None) -> str | None:
        return _secret_ref(value)


class ChunkConfig(_Strict):
    """Chunking budget, counted in the embedding model's own tokenizer."""

    target_tokens: int = Field(default=350, ge=16, le=8192)
    max_tokens: int = Field(default=512, ge=16, le=8192)
    min_tokens: int = Field(default=80, ge=1, le=8192)
    overlap: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> ChunkConfig:
        if not self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("chunk sizes must satisfy min_tokens <= target_tokens <= max_tokens")
        if self.overlap >= self.min_tokens:
            raise ValueError("chunk.overlap must be smaller than chunk.min_tokens")
        return self


class SourceConfig(_Strict):
    """A registered root that `sync` follows. v1 supports local paths (`file://`)."""

    id: str
    uri: str
    recursive: bool = True
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        check_id(value, "source id")
        if value == SINK_ID:
            raise ValueError(f"source id {SINK_ID!r} is reserved for the kb's sink")
        return value


class RemoteConfig(_Strict):
    """The single hop to a Smart RAG server. The server embeds queries with its own
    credentials; only the user's token for that server is held here."""

    endpoint: str
    kb: str = Field(description="The kb's id on the remote server.")
    token: str

    @field_validator("kb")
    @classmethod
    def _valid_kb(cls, value: str) -> str:
        return check_id(value, "remote kb id")

    @field_validator("token")
    @classmethod
    def _token_is_ref(cls, value: str) -> str:
        _secret_ref(value)
        return value

    @field_validator("endpoint")
    @classmethod
    def _tls(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme == "https" and parsed.hostname:
            return value
        if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
            return value
        raise ValueError("remote.endpoint must be an https:// URL "
                         "(plain http:// is allowed only on a loopback address)")


class KbConfig(_Strict):
    version: int = CONFIG_VERSION
    id: str
    title: str
    description: str = ""
    kind: Literal["local", "remote"] = "local"
    embedding: EmbeddingConfig | None = None
    chunk: ChunkConfig | None = None
    sink: str | None = None
    sources: list[SourceConfig] = Field(default_factory=list)
    remote: RemoteConfig | None = None

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        return check_id(value, "kb id")

    @field_validator("version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != CONFIG_VERSION:
            raise ValueError(f"config version {value} is not supported "
                             f"(this smartrag reads version {CONFIG_VERSION})")
        return value

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> KbConfig:
        if self.kind == "local":
            if self.remote is not None:
                raise ValueError("a local kb has no `remote` section")
            if self.embedding is None:
                raise ValueError("a local kb needs an `embedding` section")
            if self.chunk is None:
                self.chunk = ChunkConfig()
            if self.sink is None:
                self.sink = SINK_ID
            seen: set[str] = set()
            for source in self.sources:
                if source.id in seen:
                    raise ValueError(f"source id {source.id!r} appears twice")
                seen.add(source.id)
        else:
            if self.remote is None:
                raise ValueError("a remote kb needs a `remote` section")
            extra = [name for name in ("embedding", "chunk", "sink")
                     if getattr(self, name) is not None]
            if self.sources:
                extra.append("sources")
            if extra:
                raise ValueError(
                    f"a remote kb's sources, sink and embedding are managed by its server; "
                    f"remove: {', '.join(extra)}")
        return self

    def env_refs(self) -> list[str]:
        """Every `${VAR}` this config references, in order of first appearance."""
        names: list[str] = []
        for value in _strings(self.model_dump(mode="json")):
            for name in _ENV_REF_ANYWHERE.findall(value):
                if name not in names:
                    names.append(name)
        return names

    def missing_env(self, environ: dict[str, str] | None = None) -> list[str]:
        env = os.environ if environ is None else environ
        return [name for name in self.env_refs() if not env.get(name)]


def resolve_refs(value: str, environ: dict[str, str] | None = None) -> str:
    """`value` with every `${VAR}` replaced from the environment."""
    env = os.environ if environ is None else environ
    missing = [name for name in _ENV_REF_ANYWHERE.findall(value) if not env.get(name)]
    if missing:
        raise ConfigError(f"environment variable(s) not set: {', '.join(missing)}",
                          {"missing": missing})
    return _ENV_REF_ANYWHERE.sub(lambda m: env[m.group(1)], value)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item)]
    return []


def describe_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "config"
        message = str(item["msg"]).removeprefix("Value error, ")
        lines.append(f"{where}: {message}")
    return "; ".join(lines)


def parse_config(data: Any, origin: str) -> KbConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"{origin}: expected a mapping at the top level")
    try:
        return KbConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{origin}: {describe_validation_error(exc)}",
                          {"path": origin}) from exc


def _yaml() -> YAML:
    yaml = YAML()  # round-trip: keeps comments, key order and quoting
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096  # never fold a long path onto a second line
    return yaml


def _plain(value: Any) -> Any:
    """Round-trip containers and string subclasses as plain Python, for validation."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, str):
        return str(value)
    return value


def _block_seq(items: list[Any]) -> CommentedSeq:
    seq = CommentedSeq(items)
    seq.fa.set_block_style()
    return seq


def _source_map(source: SourceConfig) -> CommentedMap:
    node = CommentedMap([("id", source.id), ("uri", source.uri), ("recursive", source.recursive)])
    if source.include:
        node["include"] = _block_seq(list(source.include))
    if source.exclude:
        node["exclude"] = _block_seq(list(source.exclude))
    return node


class ConfigDocument:
    """A kb's `config.yaml` as the user sees it, plus its validated model.

    Edits go to the YAML tree, not to a fresh dump of the model, so anything a command
    does not own — comments, key order, quoting, blank lines — is written back as it was.
    Every edit is re-validated before it can be saved."""

    def __init__(self, path: Path, data: CommentedMap, model: KbConfig) -> None:
        self.path = path
        self.data = data
        self.model = model

    @classmethod
    def load(cls, path: Path) -> ConfigDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ConfigError(f"{path} does not exist", {"path": str(path)}) from exc
        except OSError as exc:
            raise ConfigError(f"{path} cannot be read: {exc}", {"path": str(path)}) from exc
        try:
            data = _yaml().load(text)
        except YAMLError as exc:
            raise ConfigError(f"{path} is not valid YAML: {exc}", {"path": str(path)}) from exc
        model = parse_config(_plain(data), str(path))
        assert isinstance(data, CommentedMap)
        return cls(path, data, model)

    @classmethod
    def create(cls, path: Path, model: KbConfig) -> ConfigDocument:
        """A new file, with a header and a short guide comment on the keys people edit."""
        data = CommentedMap()
        data["version"] = model.version
        data["id"] = model.id
        data["title"] = model.title
        description = model.description
        data["description"] = (LiteralScalarString(description) if "\n" in description
                               else description)
        data["kind"] = model.kind
        data.yaml_set_start_comment(_HEADER)
        data.yaml_set_comment_before_after_key(
            "description", before="What this kb contains, and what it does not. Agents choose "
                                  "a kb by this text.")
        if model.kind == "remote":
            assert model.remote is not None
            data["remote"] = CommentedMap(model.remote.model_dump(mode="json"))
            data.yaml_set_comment_before_after_key(
                "remote", before="The one hop to a Smart RAG server, which embeds and searches. "
                                 "Only the name of your token's variable is stored.")
        else:
            assert model.embedding is not None and model.chunk is not None
            data["embedding"] = CommentedMap(model.embedding.model_dump(mode="json",
                                                                        exclude_none=True))
            data.yaml_set_comment_before_after_key(
                "embedding", before="Where vectors come from. The profile id is stored with "
                                    "every vector; changing it means `kb reindex`.")
            data["chunk"] = CommentedMap(model.chunk.model_dump(mode="json"))
            data["sink"] = model.sink
            data.yaml_set_comment_before_after_key(
                "sink", before="Files copied in by `kb ingest`; the kb owns them.")
            data["sources"] = _block_seq([_source_map(s) for s in model.sources])
            data.yaml_set_comment_before_after_key(
                "sources", before="Folders `kb sync` follows. Edit with `smartrag run source "
                                  "add|remove`, or by hand.")
        document = cls(path, data, model)
        document._revalidate()
        return document

    def add_source(self, source: SourceConfig) -> None:
        sources = self.data.get("sources")
        if not isinstance(sources, CommentedSeq):
            sources = _block_seq([])
            self.data["sources"] = sources
        elif not sources:
            sources.fa.set_block_style()  # `sources: []` would otherwise stay inline
        sources.append(_source_map(source))
        self._revalidate()

    def set_description(self, text: str) -> None:
        value = LiteralScalarString(text) if "\n" in text else text
        if "description" in self.data:
            self.data["description"] = value
        else:
            keys = list(self.data)
            at = keys.index("title") + 1 if "title" in keys else len(keys)
            self.data.insert(at, "description", value)
        self._revalidate()

    def remove_source(self, source_id: str) -> None:
        sources = self.data.get("sources")
        if isinstance(sources, CommentedSeq):
            for index, item in enumerate(sources):
                if isinstance(item, dict) and item.get("id") == source_id:
                    del sources[index]
                    break
        self._revalidate()

    def _revalidate(self) -> None:
        self.model = parse_config(_plain(self.data), str(self.path))

    def save(self) -> None:
        """Write atomically, readable by the owner only: the file names env variables, and
        the directory beside it holds the index of the user's documents."""
        tmp = self.path.with_name(f".{self.path.name}.tmp-{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as handle:
                _yaml().dump(self.data, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)


def load_config(path: Path) -> KbConfig:
    return ConfigDocument.load(path).model
