"""A kb's embedding settings plus the catalog → one frozen `EmbeddingSpec` (design memo §3.2).

`resolve` is the only place the two are merged, and it checks the result: the provider and
model exist, a fixed URL is not overridden, `dimensions` is one the model allows,
`batch_size` is within the provider's limit, and a `custom` kb states what the catalog
would otherwise give. The engine, `kb status` and `provider list` all read the same spec.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from smartrag.embedding.catalog import CUSTOM, Dialect, Marking, catalog
from smartrag.errors import ConfigError

if TYPE_CHECKING:
    from smartrag.kb.config import EmbeddingConfig

_CUSTOM_ONLY = ("model_canonical_name", "model_embedding_compatibility_id",
                "max_input_tokens", "dialect", "prefix")


@dataclass(frozen=True)
class VectorSpace:
    """Which vectors an index holds. Vectors from two spaces cannot be mixed; a change of
    model, provider or profile that keeps the space needs no reindex."""

    compatibility_id: str
    dimensions: int
    marking_id: str

    @property
    def key(self) -> str:
        return f"{self.compatibility_id}|{self.dimensions}|{self.marking_id}"


@dataclass(frozen=True)
class EmbeddingSpec:
    provider_id: str
    model_profile_id: str
    """Sent as `model` in each request."""
    model_canonical_name: str
    space: VectorSpace
    dialect: Dialect
    """The provider's dialect with the model's marking in place."""
    base_url: str
    """As configured: may hold `${VAR}` references, resolved when a request is made."""
    api_key: str | None
    """A `${VAR}` reference, or None for `auth: none`."""
    key_required: bool
    request_dimensions: int | None
    """Sent in `dialect.dimensions_field` when the kb asks for shorter vectors."""
    batch_size: int
    max_input_tokens: int
    max_request_tokens: int | None

    @property
    def marking(self) -> Marking:
        return self.dialect.marking


def resolve(config: EmbeddingConfig) -> EmbeddingSpec:
    providers = catalog()
    provider = providers.get(config.provider_id)
    if provider is None:
        raise ConfigError(f"unknown embedding provider {config.provider_id!r}; known: "
                          f"{', '.join(providers)}", {"provider_id": config.provider_id})
    where = f"provider {config.provider_id!r}"
    if config.base_url is not None and provider.fixed_url:
        raise ConfigError(f"{where} has a fixed URL; remove embedding.base_url",
                          {"provider_id": config.provider_id})
    if config.batch_size > provider.max_inputs:
        raise ConfigError(f"embedding.batch_size {config.batch_size} is over {where}'s limit "
                          f"of {provider.max_inputs} inputs per request",
                          {"limit": provider.max_inputs})
    base_url = config.base_url or provider.base_url or f"${{{provider.base_url_env}}}"
    api_key = None
    if provider.dialect.auth != "none":
        api_key = config.api_key or (f"${{{provider.api_key_env}}}"
                                     if provider.api_key_env else None)

    if config.provider_id == CUSTOM:
        return _custom(config, provider.dialect, base_url, api_key)
    extra = [name for name in _CUSTOM_ONLY if getattr(config, name) is not None]
    if extra:
        raise ConfigError(f"{where} sets these from its catalog; remove embedding."
                          f"{', embedding.'.join(extra)}", {"fields": extra})
    model = provider.models.get(config.model_profile_id)
    if model is None:
        raise ConfigError(f"{where} has no model {config.model_profile_id!r}; its models: "
                          f"{', '.join(provider.models)}",
                          {"model_profile_id": config.model_profile_id})
    dimensions = config.dimensions or model.dimensions
    if not model.allows(dimensions):
        raise ConfigError(f"{config.model_profile_id!r} cannot return {dimensions} "
                          "dimensions", {"dimensions": dimensions})
    if dimensions != model.dimensions and provider.dialect.dimensions_field is None:
        raise ConfigError(f"{where} cannot shorten vectors; remove embedding.dimensions")
    dialect = provider.dialect
    if model.marking is not None:
        dialect = dialect.model_copy(update={"marking": model.marking})
    return EmbeddingSpec(
        provider_id=config.provider_id, model_profile_id=config.model_profile_id,
        model_canonical_name=model.canonical_name,
        space=VectorSpace(model.compatibility, dimensions, dialect.marking.id),
        dialect=dialect, base_url=base_url, api_key=api_key,
        key_required=provider.key_required and api_key is not None,
        request_dimensions=dimensions if dimensions != model.dimensions else None,
        batch_size=config.batch_size, max_input_tokens=model.max_input_tokens,
        max_request_tokens=model.max_request_tokens or provider.max_request_tokens)


def _custom(config: EmbeddingConfig, dialect: Dialect, base_url: str,
            api_key: str | None) -> EmbeddingSpec:
    missing = [name for name in ("model_canonical_name", "dimensions", "max_input_tokens")
               if getattr(config, name) is None]
    if missing:
        raise ConfigError(f"a custom provider needs embedding.{', embedding.'.join(missing)}",
                          {"fields": missing})
    preset = config.dialect or "openai_plain"
    if (preset == "openai_prefix") != (config.prefix is not None):
        raise ConfigError("embedding.prefix goes with dialect openai_prefix, and only with it")
    if preset == "openai_input_type":
        marking = Marking(kind="field", field="input_type", query="query", document="document")
    elif preset == "openai_prefix":
        assert config.prefix is not None
        digest = hashlib.sha256(f"{config.prefix.query}\0{config.prefix.document}".encode())
        try:
            marking = Marking(kind="prefix", name=f"custom-{digest.hexdigest()[:8]}", version=1,
                              query=config.prefix.query, document=config.prefix.document)
        except ValueError as exc:
            raise ConfigError(f"embedding.prefix: {exc}") from exc
    else:
        marking = Marking(kind="none")
    assert config.model_canonical_name and config.dimensions and config.max_input_tokens
    compatibility = config.model_embedding_compatibility_id or config.model_canonical_name
    return EmbeddingSpec(
        provider_id=CUSTOM, model_profile_id=config.model_profile_id,
        model_canonical_name=config.model_canonical_name,
        space=VectorSpace(compatibility, config.dimensions, marking.id),
        dialect=dialect.model_copy(update={"marking": marking}), base_url=base_url,
        api_key=api_key, key_required=False, request_dimensions=None,
        batch_size=config.batch_size,
        max_input_tokens=config.max_input_tokens, max_request_tokens=None)
