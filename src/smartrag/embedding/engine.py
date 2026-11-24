"""The one generic embedder: the same five steps for every provider.

resolve (done before) → mark → batch → send → read back. Batches respect the provider's
inputs per request and the model's tokens per request; token counts are Smart RAG's
estimates (`chunking.estimate_tokens`), so the token limit keeps a margin.

With `adaptive` (sync's `--adapt-to-rate-limit`), rate limits are waited out rather than
failing (`transport`), and a batch refused for a full minute is split in halves; later
batches then stay at the smaller size, since the next one would most likely be refused
too. `on_wait` is told of every wait, so a progress display can show it.

`embed_counted` also returns the tokens the requests used: the provider's own count where
its answer gives one, else Smart RAG's estimate of the texts sent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Tokens:
    """Tokens sent for embedding: `counted` by the provider, plus `estimated` by Smart RAG
    for requests whose answer gave no count."""

    counted: int = 0
    estimated: int = 0

    @property
    def total(self) -> int:
        return self.counted + self.estimated

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(self.counted + other.counted, self.estimated + other.estimated)


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
        return self.embed_counted(texts, side)[0]

    def embed_counted(self, texts: Sequence[str],
                      side: Side) -> tuple[list[list[float]], Tokens]:
        marked, fields = mark(self.spec.marking, texts, side)
        vectors: list[list[float]] = []
        tokens = Tokens()
        for batch in self._batches(marked):
            batch_vectors, batch_tokens = self._send(batch, fields)
            vectors += batch_vectors
            tokens += batch_tokens
        return vectors, tokens

    def _send(self, batch: list[str],
              fields: dict[str, str]) -> tuple[list[list[float]], Tokens]:
        try:
            answer = self.transport.post(self.url, self.wire.body(self.spec, batch, fields),
                                         can_split=self.adaptive and len(batch) > 1)
        except SplitRequest:
            half = len(batch) // 2
            self.batch_cap = min(self.batch_cap, half)
            first, first_tokens = self._send(batch[:half], fields)
            second, second_tokens = self._send(batch[half:], fields)
            return first + second, first_tokens + second_tokens
        vectors = self.wire.vectors(answer, len(batch))
        for vector in vectors:
            if len(vector) != self.spec.space.dimensions:
                raise EmbedderUnavailable(
                    f"{self.spec.model_profile_id!r} returned {len(vector)} dimensions; "
                    f"expected {self.spec.space.dimensions}")
        counted = self.wire.tokens(answer)
        tokens = (Tokens(counted=counted) if counted is not None
                  else Tokens(estimated=sum(estimate_tokens(text) for text in batch)))
        return vectors, tokens

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
