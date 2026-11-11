"""The one outbound call: OpenAI-compatible `POST /v1/embeddings` (design memo §3.2).

`HttpEmbedder` talks to OumiGo or any compatible endpoint, with the standard library only.

* `base_url` may be the base (`http://host/v1`) or the full `…/v1/embeddings` URL.
* `check()` asks `/v1/models` whether the profile is served and is an embedding model,
  and learns its dimensions and whether it takes OumiGo's `input_type` (`document` /
  `query`), which applies the model's own prefixes server-side. An endpoint without
  `/v1/models` is used as configured.
* Requests are batched (`embedding.batch_size`); 5xx answers and connection failures are
  retried with backoff; when it gives up, the error carries the server's own message.
* Credentials come from the environment through `${VAR}` references and never appear in
  errors or logs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from smartrag.errors import ConfigError, EmbedderUnavailable
from smartrag.kb.config import EmbeddingConfig, resolve_refs

InputType = Literal["document", "query"]

_RETRYABLE = frozenset({429, 500, 502, 503, 504})


class Embedder(Protocol):
    profile: str

    def check(self) -> int | None:
        """Verify the profile is served; return its dimensions if the endpoint says."""
        ...

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]: ...


class HttpEmbedder:
    def __init__(self, config: EmbeddingConfig, *, environ: dict[str, str] | None = None,
                 timeout: float = 120.0, attempts: int = 3) -> None:
        self.profile = config.profile
        self.batch_size = config.batch_size
        url = resolve_refs(config.base_url, environ).rstrip("/")
        try:
            self.api_key = resolve_refs(config.api_key, environ) if config.api_key else None
        except ConfigError as exc:
            raise ConfigError(f"{exc.message}. embedding.api_key refers to it; if the endpoint "
                              "needs no key, remove `api_key` from config.yaml",
                              exc.details) from exc
        if url.endswith("/embeddings"):
            self.url, self.models_url = url, url[: -len("/embeddings")] + "/models"
        else:
            self.url, self.models_url = url + "/embeddings", url + "/models"
        if not self.url.startswith(("http://", "https://")):
            raise ConfigError(f"embedding.base_url must be an http(s) URL, not {self.url!r}")
        self.timeout = timeout
        self.attempts = attempts
        self.input_types: list[str] = []
        self.dimensions: int | None = None

    # --- HTTP ----------------------------------------------------------------------

    def _request(self, url: str, body: dict[str, Any] | None) -> Any:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        last = ""
        for attempt in range(self.attempts):
            if attempt:
                time.sleep(min(8.0, 1.0 * 2 ** (attempt - 1)))
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {_detail(exc)}"
                if exc.code not in _RETRYABLE:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                reason = getattr(exc, "reason", exc)
                last = f"cannot connect: {reason}"
        raise EmbedderUnavailable(f"embedding endpoint {self._safe(url)} failed: {last}",
                                  details={"url": self._safe(url)})

    @staticmethod
    def _safe(url: str) -> str:
        return url.split("?", 1)[0]

    # --- API -----------------------------------------------------------------------

    def check(self) -> int | None:
        try:
            listing = self._request(self.models_url, None)
        except EmbedderUnavailable as exc:
            if "HTTP 404" in exc.message:
                return None  # an endpoint without /v1/models: use it as configured
            raise
        models = {m.get("id"): m for m in listing.get("data", []) if isinstance(m, dict)}
        model = models.get(self.profile)
        if model is None:
            served = ", ".join(sorted(str(k) for k in models)) or "none"
            raise EmbedderUnavailable(
                f"profile {self.profile!r} is not served at {self.models_url} "
                f"(served: {served})", details={"served": sorted(map(str, models))})
        nested = model.get("oumigo")
        info: dict[str, Any] = nested if isinstance(nested, dict) else model
        task = info.get("task")
        if task not in (None, "embed"):
            raise EmbedderUnavailable(f"profile {self.profile!r} is a {task!r} model, "
                                      "not an embedding model")
        self.input_types = list(info.get("input_types") or [])
        dims = info.get("dimensions")
        self.dimensions = int(dims) if isinstance(dims, int) else None
        return self.dimensions

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            body: dict[str, Any] = {"model": self.profile, "input": batch}
            if input_type in self.input_types:
                body["input_type"] = input_type
            answer = self._request(self.url, body)
            data = sorted(answer.get("data", []), key=lambda item: item.get("index", 0))
            if len(data) != len(batch):
                raise EmbedderUnavailable(f"embedding endpoint returned {len(data)} vectors "
                                          f"for {len(batch)} inputs")
            for item in data:
                vector = item.get("embedding")
                if not isinstance(vector, list) or not vector:
                    raise EmbedderUnavailable("embedding endpoint returned an empty vector")
                if self.dimensions is not None and len(vector) != self.dimensions:
                    raise EmbedderUnavailable(f"embedding endpoint returned {len(vector)} "
                                              f"dimensions, expected {self.dimensions}")
                vectors.append([float(x) for x in vector])
            if self.dimensions is None and vectors:
                self.dimensions = len(vectors[0])
        return vectors


def _detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except OSError:
        return exc.reason if isinstance(exc.reason, str) else str(exc.reason)
    try:
        body = json.loads(raw)
    except ValueError:
        return raw.strip()[:300] or str(exc.reason)
    if isinstance(body, dict):
        detail = body.get("detail") or body.get("error") or body
        if isinstance(detail, dict):
            detail = detail.get("message") or detail
        return str(detail)[:300]
    return raw[:300]
