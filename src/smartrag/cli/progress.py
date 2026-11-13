"""The progress of `kb sync` (and of `ingest` and `reindex`, which run a sync) in text mode.

On a terminal: one line on stderr, redrawn every `interval` seconds even while a single
large document is being embedded, so the clock never looks stuck:

    [#####-------------------------]  15.5%  1000 / 330000  00:15:39 / 23:45:41

processed / total files, then elapsed / estimated remaining. The estimate is worked out
every 30 s and counts down in between. It uses the rate of the last 30 s, not of the whole
run: a resumed sync races through its unchanged files first, and an average including them
would promise far too early a finish. A rate needs at least 10 finished files, so when
files are slow the window reaches back further. Until 30 s have passed and 10 files are
done, the estimate shows `--:--:--` rather than a guess.

Not on a terminal (a log file, a pipe), a bar redrawn with carriage returns is noise, so
each processed document gets its own line instead. Problems are not printed as they
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


def render(done: int, total: int, elapsed: float, remaining: float | None, columns: int) -> str:
    fraction = done / total if total else 1.0
    tail = (f"  {fraction * 100:5.1f}%  {done} / {total}  {clock(elapsed)} / "
            + ("--:--:--" if remaining is None else clock(remaining)))
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
        self.started = 0.0
        self._samples: deque[tuple[float, int]] = deque()
        self._estimate: tuple[float, float] | None = None
        """(when, seconds remaining then) of the last reliable estimate."""
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None
        self._drawn = 0

    def __call__(self, event: dict[str, object]) -> None:
        kind = event.get("event")
        if kind == "source":
            typer.echo(f"source {event['source']}: {event['files']} files", err=True)
        elif kind == "plan":
            with self._lock:
                self.total = int(event["files"])  # type: ignore[call-overload]
                self.started = time.monotonic()
                self._samples.append((self.started, 0))
                self._draw()
            self._ticker = threading.Thread(target=self._tick, name="progress", daemon=True)
            self._ticker.start()
        elif kind == "document":
            with self._lock:
                self.done += 1

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

    def _rate(self, now: float) -> float | None:
        """Files per second since the newest sample that is at least `_ESTIMATE_SECONDS`
        old and at least `_ESTIMATE_MIN_FILES` behind; None if no sample is both."""
        samples = self._samples
        while (len(samples) > 1 and now - samples[1][0] >= _ESTIMATE_SECONDS
               and self.done - samples[1][1] >= _ESTIMATE_MIN_FILES):
            samples.popleft()
        then, done_then = samples[0]
        if now - then < _ESTIMATE_SECONDS or self.done - done_then < _ESTIMATE_MIN_FILES:
            return None
        return (self.done - done_then) / (now - then)

    def _draw(self) -> None:
        if self.total is None:
            return
        now = time.monotonic()
        self._samples.append((now, self.done))
        columns = shutil.get_terminal_size().columns
        line = render(self.done, self.total, now - self.started, self.remaining(now), columns)
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
        typer.echo(f"[{event['n']}/{event['total']}] {event['source']}/{event['path']}: "
                   f"{event['status']}"
                   + (f", {event['chunks']} chunks" if event.get("chunks") else ""),
                   err=True)
    elif event.get("event") == "source":
        typer.echo(f"source {event['source']}: {event['files']} files", err=True)
