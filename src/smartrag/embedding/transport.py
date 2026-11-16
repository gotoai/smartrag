"""HTTP for every provider, with the standard library only (design memo §3.2).

* 429 and 5xx answers and connection failures are retried with backoff, waiting as long as
  the server's `Retry-After` asks (up to `MAX_RETRY_AFTER` seconds), `attempts` times.
* **Adapting to rate limits** (`adaptive`, for sync; search stays fail-fast): a 429 or a
  503 is waited out instead of counting as an attempt: `Retry-After` when the server sends
  one, else a backoff doubling from `BACKOFF_START` to `BACKOFF_MAX` seconds, for up to
  `WAIT_BUDGET` seconds per request. A request still refused with 429 after
  `SPLIT_AFTER` seconds is probably too big for a per-minute token limit on its own:
  `SplitRequest` asks the caller to send it in halves. Only standard HTTP is read, so this
  works with any provider; reading a provider's own quota headers would come on top.
* When it gives up, the error carries the server's own message and one **kind** in its
  details: `auth`, `model_not_found`, `input_too_large`, `rejected`, `rate_limited`,
  `server` or `unreachable`.
* The key travels only in the Authorization header: never in a URL, an error or a log.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from smartrag.errors import EmbedderUnavailable

MAX_RETRY_AFTER = 30.0
_RETRYABLE = frozenset({429, 500, 502, 503, 504})

_WAITED_OUT = frozenset({429, 503})
"""Rate limited, or (OumiGo) a model still starting: worth waiting for, not a fault."""
BACKOFF_START = 2.0
BACKOFF_MAX = 60.0
"""Rate limits are counted per minute, so a minute always reaches a fresh window."""
MAX_ADAPTIVE_RETRY_AFTER = 120.0
MIN_WAIT = 1.0
"""Even after `Retry-After: 0`: a server refusing every request is not hammered, and the
budget below is always used up in the end."""
WAIT_BUDGET = 600.0
"""How long one request may wait in all. An OumiGo pod may take minutes to start its
model (it answers 503 with `Retry-After: 30` meanwhile); beyond ten minutes something is
wrong, and the sync stops so it can be resumed later."""
SPLIT_AFTER = 60.0
"""A full per-minute window: refused still, the request alone is over the limit."""

OnWait = Callable[[float, str], None]


class SplitRequest(Exception):
    """A request refused with 429 for a full minute: send it in halves."""


def error_kind(status: int) -> str:
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "model_not_found"
    if status == 413:
        return "input_too_large"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "server"
    return "rejected"


def _retry_after(exc: urllib.error.HTTPError, cap: float) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return min(cap, max(0.0, float(value))) if value else None
    except ValueError:
        return None  # an HTTP date: fall back to the backoff


class Transport:
    def __init__(self, api_key: str | None, key_ref: str | None, *, timeout: float,
                 attempts: int, detail: Callable[[str], str], adaptive: bool = False) -> None:
        self.api_key = api_key
        self.key_ref = key_ref
        self.timeout = timeout
        self.attempts = attempts
        self.detail = detail
        self.adaptive = adaptive
        self.on_wait: OnWait | None = None
        """Told of each rate-limit wait (seconds, kind) before it starts."""

    def post(self, url: str, body: dict[str, Any], *, can_split: bool = False) -> Any:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(body, ensure_ascii=False).encode()
        last, kind = "", "unreachable"
        failures = 0
        waited, streak = 0.0, 0
        while True:
            pause: float | None = None
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                kind = error_kind(exc.code)
                last = f"HTTP {exc.code}: {self._read_detail(exc)}"
                if kind == "auth" and not self.api_key:
                    last += (f"; no API key was sent ({self.key_ref} is empty or not set)"
                             if self.key_ref else "; no API key was sent")
                if self.adaptive and exc.code in _WAITED_OUT:
                    if can_split and exc.code == 429 and waited >= SPLIT_AFTER:
                        raise SplitRequest(last) from None
                    hint = _retry_after(exc, MAX_ADAPTIVE_RETRY_AFTER)
                    delay = max(MIN_WAIT, hint if hint is not None
                                else min(BACKOFF_MAX, BACKOFF_START * 2 ** streak))
                    streak += 1
                    if waited + delay > WAIT_BUDGET:
                        last += f" (waited {waited:.0f} s)"
                        break
                    if self.on_wait is not None:
                        self.on_wait(delay, kind)
                    time.sleep(delay)
                    waited += delay
                    continue
                if exc.code not in _RETRYABLE:
                    break
                pause = _retry_after(exc, MAX_RETRY_AFTER)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                kind = "unreachable"
                last = f"cannot connect: {getattr(exc, 'reason', exc)}"
            failures += 1
            if failures >= self.attempts:
                break
            time.sleep(pause if pause is not None else min(8.0, 2.0 ** (failures - 1)))
        safe = url.split("?", 1)[0]
        raise EmbedderUnavailable(f"embedding endpoint {safe} failed: {last}",
                                  details={"url": safe, "kind": kind})

    def _read_detail(self, exc: urllib.error.HTTPError) -> str:
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except OSError:
            return str(exc.reason)
        return self.detail(raw) or str(exc.reason)
