"""The progress of `kb sync` (and of `ingest` and `reindex`, which run a sync) in text mode.

On a terminal: one line on stderr, redrawn every `interval` seconds even while a single
large document is being embedded, so the clock never looks stuck:

    [#####-------------------------]  15.5%  1000 / 330000  00:15:39 / 23:45:41  52.4M tokens  43.6M/h

processed / total files, elapsed / estimated remaining, then the tokens sent for embedding
so far and their recent rate per hour. The tokens are the provider's count; when some came
from a provider that gives none, the figure is Smart RAG's estimate for those and starts
with `~`. Reused vectors cost no tokens. The estimate is worked out
every 30 s and counts down in between. It uses the rate of the last 30 s, not of the whole
run: a resumed sync races through its unchanged files first, and an average including them
would promise far too early a finish. A rate needs at least 10 finished files, so when
files are slow the window reaches back further. Until 30 s have passed and 10 files are
done, the estimate shows `--:--:--` rather than a guess.

While a rate limit is being waited out, the line ends with the wait counting down
(`rate limited, retrying in 37 s`). Not on a terminal (a log file, a pipe), a bar redrawn
with carriage returns is noise, so each processed document, and each wait, gets its own
line instead. Problems are not printed as they
happen in either form: the report at the end lists them.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from collections import deque
from types import TracebackType
from typing import IO, final

import typer

_ESTIMATE_SECONDS = 30.0
"""How often the remaining time is worked out, and the shortest window it is measured over."""
_ESTIMATE_MIN_FILES = 10
"""The fewest finished files a rate may rest on."""
_BAR_MIN, _BAR_MAX = 10, 40


def clock(seconds: float) -> str:
    """HH:MM:SS; the hours grow past 99 rather than wrap."""
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def amount(count: float) -> str:
    """812, 52.4k, 52.4M, 1.65B."""
    for limit, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if count >= limit:
            value = count / limit
            return f"{value:.2f}{unit}" if value < 10 else f"{value:.1f}{unit}"
    return f"{count:.0f}"


def render(done: int, total: int, elapsed: float, remaining: float | None, columns: int,
           waiting: float | None = None, tokens: int | None = None,
           approximate: bool = False, per_hour: float | None = None) -> str:
    fraction = done / total if total else 1.0
    head = (f"  {fraction * 100:5.1f}%  {done} / {total}  {clock(elapsed)} / "
            + ("--:--:--" if remaining is None else clock(remaining)))
    extras = []  # left out from the end when the terminal is too narrow for them
    if tokens is not None:
        extras.append(f"  {'~' if approximate else ''}{amount(tokens)} tokens")
        if per_hour is not None:
            extras.append(f"  {amount(per_hour)}/h")
    wait = f"  rate limited, retrying in {waiting:.0f} s" if waiting is not None else ""
    while extras and len(head) + sum(map(len, extras)) + len(wait) + _BAR_MIN + 2 > columns:
        extras.pop()
    tail = head + "".join(extras) + wait
    width = max(_BAR_MIN, min(_BAR_MAX, columns - len(tail) - 3))
    filled = int(width * fraction)
    return f"[{'#' * filled}{'-' * (width - filled)}]{tail}"


@final
class ProgressBar:
    """A sync progress callback that draws the bar. Use it as a context manager, so the
    bar's line is finished before anything else is printed."""

    def __init__(self, interval: float = 1.0, stream: IO[str] | None = None) -> None:
        self.interval = interval
        self.stream = stream or sys.stderr
        self.total: int | None = None
        self.done = 0
        self.tokens = 0
        self.approximate = False
        """Some of `tokens` is Smart RAG's estimate."""
        self.started = 0.0
        self._samples: deque[tuple[float, int, int]] = deque()
        """(when, files done, tokens) at each redraw."""
        self._per_hour: float | None = None
        self._estimate: tuple[float, float] | None = None
        """(when, seconds remaining then) of the last reliable estimate."""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._drawn = 0
        self._wait_until = 0.0

    def __call__(self, event: dict[str, object]) -> None:
        kind = event.get("event")
        if kind == "source":
            typer.echo(f"source {event['source']}: {event['files']} files", err=True)
        elif kind == "plan":
            with self._lock:
                self.total = int(event["files"])  # type: ignore[call-overload]
                self.started = time.monotonic()
                self._samples.append((self.started, 0, 0))
                self._draw()
            self._ticker = threading.Thread(target=self._tick, name="progress", daemon=True)
            self._ticker.start()
        elif kind == "document":
            with self._lock:
                self.done += 1
                self.tokens += int(event.get("tokens") or 0)  # type: ignore[call-overload]
                self.approximate = self.approximate or bool(event.get("tokens_estimated"))
        elif kind == "wait":
            with self._lock:
                until = time.monotonic() + float(event["seconds"])  # type: ignore[arg-type]
                self._wait_until = max(self._wait_until, until)
                self._draw()

    def _tick(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                self._draw()

    def remaining(self, now: float) -> float | None:
        """Seconds left, or None while there is no reliable estimate yet. Worked out every
        `_ESTIMATE_SECONDS` (at every redraw until the first one succeeds), counted down
        in between."""
        assert self.total is not None
        if self.done >= self.total:
            return 0.0
        if self._estimate is not None and now - self._estimate[0] < _ESTIMATE_SECONDS:
            when, seconds = self._estimate
            return max(0.0, seconds - (now - when))
        rate = self._rate(now)
        self._estimate = None if rate is None else (now, (self.total - self.done) / rate)
        return None if self._estimate is None else self._estimate[1]

    def per_hour(self, now: float) -> float | None:
        """Tokens per hour over the same window as the time estimate, worked out when it
        is; None until the first one."""
        if self._estimate is not None and self._estimate[0] == now:
            then, _done, tokens_then = self._samples[0]
            self._per_hour = (self.tokens - tokens_then) / (now - then) * 3600
        return self._per_hour

    def _rate(self, now: float) -> float | None:
        """Files per second since the newest sample that is at least `_ESTIMATE_SECONDS`
        old and at least `_ESTIMATE_MIN_FILES` behind; None if no sample is both."""
        samples = self._samples
        while (len(samples) > 1 and now - samples[1][0] >= _ESTIMATE_SECONDS
               and self.done - samples[1][1] >= _ESTIMATE_MIN_FILES):
            samples.popleft()
        then, done_then, _tokens = samples[0]
        if now - then < _ESTIMATE_SECONDS or self.done - done_then < _ESTIMATE_MIN_FILES:
            return None
        return (self.done - done_then) / (now - then)

    def _draw(self) -> None:
        if self.total is None:
            return
        now = time.monotonic()
        self._samples.append((now, self.done, self.tokens))
        columns = shutil.get_terminal_size().columns
        waiting = self._wait_until - now if self._wait_until > now else None
        remaining = self.remaining(now)
        line = render(self.done, self.total, now - self.started, remaining, columns,
                      waiting, self.tokens, self.approximate, self.per_hour(now))
        # Pad over the previous line instead of an ANSI erase, which old Windows
        # consoles print literally.
        self.stream.write("\r" + line + " " * max(0, self._drawn - len(line)))
        self.stream.flush()
        self._drawn = len(line)

    def close(self) -> None:
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join()
        with self._lock:
            if self.total is not None:
                self._draw()
                self.stream.write("\n")
                self.stream.flush()

    def __enter__(self) -> ProgressBar:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()


def progress_lines(event: dict[str, object]) -> None:
    """The callback for text mode without a terminal: a line per processed document."""
    if event.get("event") == "document" and event.get("status") not in ("unchanged",
                                                                       "deferred"):
        tokens = event.get("tokens")
        typer.echo(f"[{event['n']}/{event['total']}] {event['source']}/{event['path']}: "
                   f"{event['status']}"
                   + (f", {event['chunks']} chunks" if event.get("chunks") else "")
                   + (f", {'about ' if event.get('tokens_estimated') else ''}{tokens:,} tokens"
                      if tokens else ""),
                   err=True)
    elif event.get("event") == "source":
        typer.echo(f"source {event['source']}: {event['files']} files", err=True)
    elif event.get("event") == "wait":
        typer.echo(f"rate limited ({event['kind']}): retrying in {event['seconds']:.0f} s",
                   err=True)
