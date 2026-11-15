"""Embedding requests in worker processes, for `embedding.parallelism` above 1 (design
memo §3.11).

Only the requests run in parallel. The main process still lists, extracts and chunks
serially, and alone reads and writes the index; a worker gets a batch of texts and
returns their vectors, with no index connection and no shared state.

* Processes, not threads, started with `spawn` on every OS, so Linux, macOS and Windows
  behave alike.
* The pool starts at the first request: a dry run, or a sync with nothing to embed,
  starts no processes.
* Workers ignore SIGINT: Ctrl-C is the main process's to handle, and it stops the pool.
* A worker exits when the main process does, even one killed outright; otherwise it would
  be left running. That watch is the only thread in a worker, and it does nothing else.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time
from concurrent import futures
from concurrent.futures.process import BrokenProcessPool
from typing import Any

from smartrag.embedding import Embedder
from smartrag.errors import EmbedderUnavailable

Vectors = list[list[float]]


class EmbedPool:
    """At most `parallelism` embedding requests in flight. Each request carries a `tag`,
    which `wait` returns with the request's vectors."""

    def __init__(self, embedder: Embedder, parallelism: int) -> None:
        self.embedder = embedder
        self.parallelism = parallelism
        self._executor: futures.ProcessPoolExecutor | None = None
        self._requests: dict[futures.Future[tuple[Vectors | None, str | None, int]], Any] = {}

    @property
    def in_flight(self) -> int:
        return len(self._requests)

    def submit(self, texts: list[str], tag: Any) -> None:
        if self._executor is None:
            self._executor = futures.ProcessPoolExecutor(
                self.parallelism, mp_context=multiprocessing.get_context("spawn"),
                initializer=_start_worker, initargs=(self.embedder,))
        self._requests[self._executor.submit(_embed, texts)] = tag

    def wait(self, *, block: bool) -> list[tuple[Any, Vectors, int]]:
        """The requests that have finished, as (tag, vectors, milliseconds); with `block`,
        waits for at least one. A failed request raises `EmbedderUnavailable`."""
        if not self._requests:
            return []
        done, _ = futures.wait(self._requests, timeout=None if block else 0,
                               return_when=futures.FIRST_COMPLETED)
        finished: list[tuple[Any, Vectors, int]] = []
        for request in done:
            tag = self._requests.pop(request)
            try:
                vectors, error, took_ms = request.result()
            except BrokenProcessPool:
                raise EmbedderUnavailable("an embedding worker process stopped "
                                          "unexpectedly") from None
            if error is not None:
                raise EmbedderUnavailable(error)
            assert vectors is not None
            finished.append((tag, vectors, took_ms))
        return finished

    def close(self) -> None:
        """Shut the pool down: at once, abandoning them, if requests are still in flight
        (an error or an interrupt), else after the idle workers exit."""
        executor, self._executor = self._executor, None
        if executor is None:
            return
        if not self._requests:
            executor.shutdown(wait=True)
            return
        self._requests.clear()
        terminate = getattr(executor, "terminate_workers", None)  # Python 3.14+
        if terminate is not None:
            terminate()
            return
        # Before 3.14 the executor has no public way to stop a busy worker.
        processes = list((getattr(executor, "_processes", None) or {}).values())
        executor.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()
        for process in processes:
            process.join()


# --- in the worker processes -----------------------------------------------------------

_embedder: Embedder | None = None


def _start_worker(embedder: Embedder) -> None:
    global _embedder
    _embedder = embedder
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = multiprocessing.parent_process()
    if parent is not None:
        threading.Thread(target=_exit_with, args=(parent,), name="exit-with-parent",
                         daemon=True).start()


def _exit_with(parent: multiprocessing.process.BaseProcess) -> None:
    parent.join()  # returns when the main process has exited, however it ended
    os._exit(1)


def _embed(texts: list[str]) -> tuple[Vectors | None, str | None, int]:
    """One request. A failure comes back as its message: Smart RAG's errors do not keep
    their details through pickling."""
    assert _embedder is not None
    started = time.monotonic()
    try:
        vectors = _embedder.embed(texts, "document")
        error = None
    except EmbedderUnavailable as exc:
        vectors, error = None, exc.message
    return vectors, error, int((time.monotonic() - started) * 1000)
