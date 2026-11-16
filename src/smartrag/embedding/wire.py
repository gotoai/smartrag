"""Request and response formats. Only this part of a dialect is code (design memo §3.2).

`openai`: `POST {base}/embeddings` with `{"model", "input": [...]}` plus the marking's and
the dimensions' fields; the answer's `data[].embedding`, ordered by `index`.
"""

from __future__ import annotations

import json
from typing import Any

from smartrag.embedding.resolve import EmbeddingSpec
from smartrag.errors import EmbedderUnavailable

MAX_DETAIL = 1000
"""Characters of a server's error message kept. Providers explain limits and billing in
full sentences (Voyage's reduced-rate notice is about 400); an HTML error page is cut."""


class OpenAIWire:
    path = "/embeddings"

    @staticmethod
    def body(spec: EmbeddingSpec, texts: list[str], fields: dict[str, str]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": spec.model_profile_id, "input": texts, **fields}
        if spec.request_dimensions is not None:
            assert spec.dialect.dimensions_field is not None
            body[spec.dialect.dimensions_field] = spec.request_dimensions
        return body

    @staticmethod
    def vectors(answer: Any, count: int) -> list[list[float]]:
        data = answer.get("data") if isinstance(answer, dict) else None
        if not isinstance(data, list):
            raise EmbedderUnavailable("the embedding endpoint's answer has no `data` list")
        if len(data) != count:
            raise EmbedderUnavailable(f"embedding endpoint returned {len(data)} vectors "
                                      f"for {count} inputs")
        ordered = sorted(data, key=lambda item: item.get("index", 0)
                         if isinstance(item, dict) else 0)
        vectors: list[list[float]] = []
        for item in ordered:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list) or not vector:
                raise EmbedderUnavailable("embedding endpoint returned an empty vector")
            vectors.append([float(x) for x in vector])
        return vectors

    @staticmethod
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


WIRES = {"openai": OpenAIWire}
