"""The embedding provider catalog: `providers.yaml`, shipped in the package (design memo §3.2).

Each provider has a URL (fixed, or per deployment through an environment variable), the
variable its key is usually kept in, a **dialect**, request limits, data terms and its
models. A model inherits its provider's dialect and may override the marking.

A dialect is data in four parts, each from a fixed set:

* `wire`: the request and response format; `openai` only, for now.
* `auth`: `bearer`, or `none`.
* `marking`: how a query and a document are told apart: `none`; `field` (a request field
  with one value for each side); `prefix` (a template around the text for each side).
* `dimensions_field`: the request field that asks for shorter vectors, if any.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from ruamel.yaml import YAML

CATALOG_FILE = "providers.yaml"
CUSTOM = "custom"
"""The provider whose endpoint, model and dialect the kb's config states itself."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Marking(_Strict):
    kind: Literal["none", "field", "prefix"]
    field: str | None = None
    """`field`: the request field, e.g. `input_type`."""
    query: str | None = None
    """`field`: the field's value for a query. `prefix`: the template, with `{text}`."""
    document: str | None = None
    name: str | None = None
    """`prefix`: the templates' name, part of `marking_id`."""
    version: int | None = None
    """`prefix`: bumped whenever a template changes, since that changes every vector."""

    @model_validator(mode="after")
    def _complete(self) -> Marking:
        if self.kind == "field" and not (self.field and self.query and self.document):
            raise ValueError("a field marking needs field, query and document")
        if self.kind == "prefix":
            if not (self.query and self.document and self.name and self.version):
                raise ValueError("a prefix marking needs query, document, name and version")
            for side in (self.query, self.document):
                if "{text}" not in side:
                    raise ValueError(f"a prefix template must contain {{text}}: {side!r}")
        return self

    @property
    def id(self) -> str:
        """Part of the vector space: the same model marked differently gives other vectors."""
        if self.kind == "field":
            return f"field:{self.field}"
        if self.kind == "prefix":
            return f"prefix:{self.name}/v{self.version}"
        return "none"


class Dialect(_Strict):
    wire: Literal["openai"] = "openai"
    auth: Literal["bearer", "none"] = "bearer"
    marking: Marking = Marking(kind="none")
    dimensions_field: Literal["dimensions", "output_dimension"] | None = None


class ModelEntry(_Strict):
    canonical_name: str
    """The Hugging Face repo id of an open model; `@<vendor>/<model>` for a closed one."""
    compatibility_id: str | None = None
    """Set when the vendor declares a vector space shared by several models."""
    dimensions: int = Field(ge=1)
    allowed_dimensions: list[int] = Field(default_factory=list)
    dimension_range: tuple[int, int] | None = None
    max_input_tokens: int = Field(ge=1)
    max_request_tokens: int | None = Field(default=None, ge=1)
    price: float | None = None
    marking: Marking | None = None
    """Overrides the provider's marking."""

    @property
    def compatibility(self) -> str:
        return self.compatibility_id or self.canonical_name

    def allows(self, dimensions: int) -> bool:
        if dimensions == self.dimensions or dimensions in self.allowed_dimensions:
            return True
        low_high = self.dimension_range
        return low_high is not None and low_high[0] <= dimensions <= low_high[1]


class ProviderEntry(_Strict):
    title: str
    base_url: str | None = None
    """Fixed by the catalog. Absent for a provider reached per deployment."""
    base_url_env: str | None = None
    api_key_env: str | None = None
    key_required: bool = False
    """A cloud API: without a key, fail before sending anything. A self-hosted endpoint
    may run without authentication, so no key is not an error there."""
    dialect: Dialect
    max_inputs: int = Field(ge=1)
    max_request_tokens: int | None = Field(default=None, ge=1)
    data_terms: str
    checked: str
    models: dict[str, ModelEntry] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _one_url(self) -> ProviderEntry:
        if (self.base_url is None) == (self.base_url_env is None):
            raise ValueError("a provider has either base_url or base_url_env")
        return self

    @property
    def fixed_url(self) -> bool:
        return self.base_url is not None


@lru_cache(maxsize=1)
def catalog() -> dict[str, ProviderEntry]:
    text = resources.files(__package__).joinpath(CATALOG_FILE).read_text(encoding="utf-8")
    data = YAML(typ="safe").load(text)
    try:
        return {provider_id: ProviderEntry.model_validate(entry)
                for provider_id, entry in data.items()}
    except ValidationError as exc:  # a packaging bug, not the user's
        raise RuntimeError(f"{CATALOG_FILE} is invalid: {exc}") from exc


# --- `provider list`: the catalog as Kari and the CLI show it ---------------------------


class ModelInfo(BaseModel):
    model_profile_id: str
    model_canonical_name: str
    model_embedding_compatibility_id: str
    dimensions: int
    max_input_tokens: int
    marking_id: str
    price_per_mtok_usd: float | None = None


class ProviderInfo(BaseModel):
    provider_id: str
    title: str
    base_url: str | None = Field(description="Fixed by the catalog; None when per deployment.")
    base_url_env: str | None
    api_key_env: str | None
    data_terms: str
    checked: str
    models: list[ModelInfo]


class ProviderList(BaseModel):
    providers: list[ProviderInfo]


def provider_list() -> ProviderList:
    providers = []
    for provider_id, entry in catalog().items():
        models = [ModelInfo(
            model_profile_id=model_id, model_canonical_name=model.canonical_name,
            model_embedding_compatibility_id=model.compatibility,
            dimensions=model.dimensions, max_input_tokens=model.max_input_tokens,
            marking_id=(model.marking or entry.dialect.marking).id,
            price_per_mtok_usd=model.price) for model_id, model in entry.models.items()]
        providers.append(ProviderInfo(
            provider_id=provider_id, title=entry.title, base_url=entry.base_url,
            base_url_env=entry.base_url_env, api_key_env=entry.api_key_env,
            data_terms=entry.data_terms, checked=entry.checked, models=models))
    return ProviderList(providers=providers)
