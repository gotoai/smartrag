"""Request and response formats. Only this part of a dialect is code.

* `openai`: `POST {base}/embeddings` with `{"model", "input": [...]}` plus the marking's
  and the dimensions' fields; the answer's `data[].embedding`, ordered by `index`.
* `cohere_v2`: `POST {base}/embed` with `{"model", "texts": [...], "embedding_types":
  ["float"], "truncate": "NONE"}` plus the marking's field (`input_type`, which Cohere
  requires) and `output_dimension`; the answer's `embeddings.float`, in input order. Only
  Cohere's native API marks queries and documents: its OpenAI-compatible one drops the
  field. `truncate: NONE` makes an over-long input an error, as everywhere else in Smart
  RAG, instead of Cohere's default of silently cutting its end.

Both read errors from the usual shapes (`detail`, `error.message`, `message`), and the
tokens the provider counted for the request: `usage.prompt_tokens` (or `total_tokens`) on
the `openai` wire, `meta.billed_units.input_tokens` on `cohere_v2`. A provider that gives
no count gets `None`, and the caller estimates instead.
"""

from __future__ import annotations

import json
from typing import Any

from smartrag.embedding.resolve import EmbeddingSpec
from smartrag.errors import EmbedderUnavailable

MAX_DETAIL = 1000
"""Characters of a server's error message kept. Providers explain limits and billing in
full sentences (Voyage's reduced-rate notice is about 400); an HTML error page is cut."""


def _dimensions(spec: EmbeddingSpec, body: dict[str, Any]) -> dict[str, Any]:
    if spec.request_dimensions is not None:
        assert spec.dialect.dimensions_field is not None
        body[spec.dialect.dimensions_field] = spec.request_dimensions
    return body


def _count(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 \
        else None


def _checked(vectors: Any, count: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        got = len(vectors) if isinstance(vectors, list) else 0
        raise EmbedderUnavailable(f"embedding endpoint returned {got} vectors for {count} "
                                  "inputs")
    checked: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise EmbedderUnavailable("embedding endpoint returned an empty vector")
        checked.append([float(x) for x in vector])
    return checked


class OpenAIWire:
    path = "/embeddings"

    @staticmethod
    def body(spec: EmbeddingSpec, texts: list[str], fields: dict[str, str]) -> dict[str, Any]:
        return _dimensions(spec, {"model": spec.model_profile_id, "input": texts, **fields})

    @staticmethod
    def vectors(answer: Any, count: int) -> list[list[float]]:
        data = answer.get("data") if isinstance(answer, dict) else None
        if not isinstance(data, list):
            raise EmbedderUnavailable("the embedding endpoint's answer has no `data` list")
        ordered = sorted(data, key=lambda item: item.get("index", 0)
                         if isinstance(item, dict) else 0)
        return _checked([item.get("embedding") if isinstance(item, dict) else None
                         for item in ordered], count)

    @staticmethod
    def tokens(answer: Any) -> int | None:
        usage = answer.get("usage") if isinstance(answer, dict) else None
        if not isinstance(usage, dict):
            return None
        counted = _count(usage.get("prompt_tokens"))
        return counted if counted is not None else _count(usage.get("total_tokens"))

    @staticmethod
    def error_detail(raw: str) -> str:
        return error_detail(raw)


class CohereV2Wire:
    path = "/embed"

    @staticmethod
    def body(spec: EmbeddingSpec, texts: list[str], fields: dict[str, str]) -> dict[str, Any]:
        return _dimensions(spec, {"model": spec.model_profile_id, "texts": texts,
                                  "embedding_types": ["float"], "truncate": "NONE", **fields})

    @staticmethod
    def vectors(answer: Any, count: int) -> list[list[float]]:
        embeddings = answer.get("embeddings") if isinstance(answer, dict) else None
        if not isinstance(embeddings, dict) or "float" not in embeddings:
            raise EmbedderUnavailable("the embedding endpoint's answer has no "
                                      "`embeddings.float` list")
        return _checked(embeddings["float"], count)

    @staticmethod
    def tokens(answer: Any) -> int | None:
        meta = answer.get("meta") if isinstance(answer, dict) else None
        billed = meta.get("billed_units") if isinstance(meta, dict) else None
        return _count(billed.get("input_tokens")) if isinstance(billed, dict) else None

    @staticmethod
    def error_detail(raw: str) -> str:
        return error_detail(raw)


def error_detail(raw: str) -> str:
    """The server's own message, from the usual error shapes."""
    try:
        body = json.loads(raw)
    except ValueError:
        return raw.strip()[:MAX_DETAIL]
    if isinstance(body, list) and body and isinstance(body[0], dict):
        body = body[0]  # Google's compatible endpoint wraps the error in a list
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body.get("message") or body
        if isinstance(detail, dict):
            detail = detail.get("message") or detail
        return str(detail)[:MAX_DETAIL]
    return raw[:MAX_DETAIL]


WIRES: dict[str, type[OpenAIWire | CohereV2Wire]] = {
    "openai": OpenAIWire, "cohere_v2": CohereV2Wire}
