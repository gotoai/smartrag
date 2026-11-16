"""Query embedders for a long-lived server: built once, failing fast (design memo §3.9).

`kb search` builds an embedder per search, and the sync's embedder waits up to 120 s × 3
attempts, about 6 minutes, before giving up. Both suit a one-shot command, not an agent
waiting on a tool call (Kari's extensions time out after 300 s). Here:

* each kb keeps one embedder for the life of the process, built on first use;
* a query waits `QUERY_TIMEOUT` seconds with one retry: a healthy endpoint embeds one
  query in well under a second;
* after a failure, the vector side is skipped for `COOLDOWN` seconds, so each search
  falls back to keywords at once instead of waiting out the timeout again.

An endpoint that cannot be configured (its base URL's `${VAR}` unset) counts as
unavailable: search still returns keyword results, with `search_mode` `keyword`, rather
than failing. An unset or empty API key is not an error: no key is sent.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

from smartrag.embedding.api import Embedder, Side, open_embedder
from smartrag.errors import ConfigError, EmbedderUnavailable
from smartrag.kb.config import EmbeddingConfig

QUERY_TIMEOUT = 10.0
QUERY_ATTEMPTS = 2
COOLDOWN = 60.0

EmbedderFactory = Callable[[EmbeddingConfig], Embedder]


def _http(config: EmbeddingConfig) -> Embedder:
    return open_embedder(config, timeout=QUERY_TIMEOUT, attempts=QUERY_ATTEMPTS)


class GuardedEmbedder:
    """One kb's query embedder: built on first use, skipped while down."""

    def __init__(self, config: EmbeddingConfig, factory: EmbedderFactory, cooldown: float,
                 clock: Callable[[], float]) -> None:
        self._config = config
        self._factory = factory
        self._cooldown = cooldown
        self._clock = clock
        self._inner: Embedder | None = None
        self._down_until = 0.0
        self._lock = threading.Lock()

    def _ready(self) -> Embedder:
        if self._clock() < self._down_until:
            raise EmbedderUnavailable("the endpoint failed recently; skipped")
        with self._lock:
            try:
                if self._inner is None:
                    self._inner = self._factory(self._config)
            except ConfigError as exc:
                self._mark_down()
                raise EmbedderUnavailable(exc.message) from exc
            return self._inner

    def _mark_down(self) -> None:
        self._down_until = self._clock() + self._cooldown

    def embed(self, texts: Sequence[str], side: Side) -> list[list[float]]:
        inner = self._ready()
        try:
            return inner.embed(texts, side)
        except EmbedderUnavailable:
            self._mark_down()
            raise


class QueryEmbedders:
    def __init__(self, factory: EmbedderFactory = _http, *, cooldown: float = COOLDOWN,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._factory = factory
        self._cooldown = cooldown
        self._clock = clock
        self._embedders: dict[str, tuple[str, GuardedEmbedder]] = {}
        self._lock = threading.Lock()

    def get(self, kb_id: str, config: EmbeddingConfig) -> GuardedEmbedder:
        """The kb's embedder; a new one if its embedding settings changed."""
        key = config.model_dump_json()
        with self._lock:
            held = self._embedders.get(kb_id)
            if held is None or held[0] != key:
                held = (key, GuardedEmbedder(config, self._factory, self._cooldown, self._clock))
                self._embedders[kb_id] = held
            return held[1]
