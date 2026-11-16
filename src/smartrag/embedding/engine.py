"""The one generic embedder: the same five steps for every provider (design memo §3.2).

resolve (done before) → mark → batch → send → read back. Batches respect the provider's
inputs per request and the model's tokens per request; token counts are Smart RAG's
estimates (`chunking.estimate_tokens`), so the token limit keeps a margin.

With `adaptive` (sync's `--adapt-to-rate-limit`), rate limits are waited out rather than
failing (`transport`), and a batch refused for a full minute is split in halves; later
batches then stay at the smaller size, since the next one would most likely be refused
too. `on_wait` is told of every wait, so a progress display can show it.
"""

from __future__ import annotations

from collections.abc import Sequence

from smartrag.chunking import estimate_tokens
from smartrag.embedding.marking import Side, mark
from smartrag.embedding.resolve import EmbeddingSpec
from smartrag.embedding.transport import OnWait, SplitRequest, Transport
from smartrag.embedding.wire import WIRES
from smartrag.errors import ConfigError, EmbedderUnavailable
from smartrag.kb.config import resolve_optional_ref, resolve_refs

TOKEN_MARGIN = 0.8
"""The share of a request's token limit a batch fills, since counts are estimates."""
VERIFY_TEXT = "Smart RAG embedding check"


class HttpEmbedder:
    def __init__(self, spec: EmbeddingSpec, *, environ: dict[str, str] | None = None,
                 timeout: float = 120.0, attempts: int = 3, adaptive: bool = False) -> None:
        self.spec = spec
        self.adaptive = adaptive
        self.batch_cap = spec.batch_size
        """Texts per request: `batch_size`, or less after a split."""
        self.wire = WIRES[spec.dialect.wire]
        url = resolve_refs(spec.base_url, environ).rstrip("/")
        if not url.startswith(("http://", "https://")):
            raise ConfigError(f"embedding.base_url must be an http(s) URL, not {url!r}")
        self.url = url if url.endswith(self.wire.path) else url + self.wire.path
        api_key = resolve_optional_ref(spec.api_key, environ)
        if api_key is None and spec.key_required:
            raise ConfigError(f"{spec.api_key} is empty or not set; {spec.provider_id} needs "
                              "an API key", {"provider_id": spec.provider_id})
        self.transport = Transport(api_key, spec.api_key, timeout=timeout, attempts=attempts,
                                   detail=self.wire.error_detail, adaptive=adaptive)

    @property
    def on_wait(self) -> OnWait | None:
        return self.transport.on_wait

    @on_wait.setter
    def on_wait(self, callback: OnWait | None) -> None:
        self.transport.on_wait = callback

    def embed(self, texts: Sequence[str], side: Side) -> list[list[float]]:
        marked, fields = mark(self.spec.marking, texts, side)
        vectors: list[list[float]] = []
        for batch in self._batches(marked):
            vectors += self._send(batch, fields)
        return vectors

    def _send(self, batch: list[str], fields: dict[str, str]) -> list[list[float]]:
        try:
            answer = self.transport.post(self.url, self.wire.body(self.spec, batch, fields),
                                         can_split=self.adaptive and len(batch) > 1)
        except SplitRequest:
            half = len(batch) // 2
            self.batch_cap = min(self.batch_cap, half)
            return self._send(batch[:half], fields) + self._send(batch[half:], fields)
        vectors = self.wire.vectors(answer, len(batch))
        for vector in vectors:
            if len(vector) != self.spec.space.dimensions:
                raise EmbedderUnavailable(
                    f"{self.spec.model_profile_id!r} returned {len(vector)} dimensions; "
                    f"expected {self.spec.space.dimensions}")
        return vectors

    def verify(self) -> None:
        """One short embedding: checks the URL, the key, the model and the vector size."""
        self.embed([VERIFY_TEXT], "document")

    def _batches(self, texts: list[str]) -> list[list[str]]:
        budget = (int(self.spec.max_request_tokens * TOKEN_MARGIN)
                  if self.spec.max_request_tokens else None)
        batches: list[list[str]] = []
        batch: list[str] = []
        tokens = 0
        for text in texts:
            size = estimate_tokens(text)
            if batch and (len(batch) >= self.batch_cap
                          or (budget is not None and tokens + size > budget)):
                batches.append(batch)
                batch, tokens = [], 0
            batch.append(text)
            tokens += size
        if batch:
            batches.append(batch)
        return batches
