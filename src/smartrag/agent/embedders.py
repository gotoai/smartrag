"""Query embedders for a long-lived server: checked once, failing fast (design memo §3.9).

`kb search` builds an `HttpEmbedder` per search and asks `/v1/models` each time; and the
sync's embedder waits up to 120 s × 3 attempts, about 6 minutes, before giving up. Both
suit a one-shot command, not an agent waiting on a tool call (Kari's extensions time out
after 300 s). Here:

* each kb keeps one embedder for the life of the process, checked on first use;
* a query waits `QUERY_TIMEOUT` seconds with one retry: a healthy endpoint embeds one
  query in well under a second;
* after a failure, the vector side is skipped for `COOLDOWN` seconds, so each search
  falls back to keywords at once instead of waiting out the timeout again.

A missing credential (an unset `${VAR}`) counts as an unavailable endpoint: search still
returns keyword results, flagged `degraded`, rather than failing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence

from smartrag.embedding import Embedder, HttpEmbedder, InputType
from smartrag.errors import ConfigError, EmbedderUnavailable
from smartrag.kb.config import EmbeddingConfig

QUERY_TIMEOUT = 10.0
QUERY_ATTEMPTS = 2
COOLDOWN = 60.0

EmbedderFactory = Callable[[EmbeddingConfig], Embedder]


def _http(config: EmbeddingConfig) -> Embedder:
    return HttpEmbedder(config, timeout=QUERY_TIMEOUT, attempts=QUERY_ATTEMPTS)


class GuardedEmbedder:
    """One kb's query embedder: built and checked on first use, skipped while down."""

    def __init__(self, config: EmbeddingConfig, factory: EmbedderFactory, cooldown: float,
                 clock: Callable[[], float]) -> None:
        self.profile = config.profile
        self._config = config
        self._factory = factory
        self._cooldown = cooldown
        self._clock = clock
        self._inner: Embedder | None = None
        self._checked = False
        self._down_until = 0.0
        self._lock = threading.Lock()

    def _ready(self) -> Embedder:
        if self._clock() < self._down_until:
            raise EmbedderUnavailable("the endpoint failed recently; skipped")
        with self._lock:
            try:
                if self._inner is None:
                    self._inner = self._factory(self._config)
                if not self._checked:
                    self._inner.check()  # learns whether the model takes input_type
                    self._checked = True
            except ConfigError as exc:
                self._mark_down()
                raise EmbedderUnavailable(exc.message) from exc
            except EmbedderUnavailable:
                self._mark_down()
                raise
            return self._inner

    def _mark_down(self) -> None:
        self._down_until = self._clock() + self._cooldown

    def check(self) -> int | None:
        return getattr(self._ready(), "dimensions", None)

    def embed(self, texts: Sequence[str], input_type: InputType) -> list[list[float]]:
        inner = self._ready()
        try:
            return inner.embed(texts, input_type)
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
