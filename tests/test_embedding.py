"""The HTTP embedder's credential: sent when set, never sent empty."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from smartrag.embedding import HttpEmbedder
from smartrag.errors import EmbedderUnavailable
from smartrag.kb.config import EmbeddingConfig


@pytest.fixture
def endpoint():
    """A local OpenAI-compatible endpoint that records the Authorization header it gets.
    With `require` set, it answers 401 to any other header, as OumiGo's router does."""
    seen: list[str | None] = []
    state = {"require": None}

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization")
            seen.append(header)
            if state["require"] and header != f"Bearer {state['require']}":
                self._reply(401, {"detail": "missing or invalid API key"})
                return False
            return True

        def do_GET(self) -> None:
            if self._authorized():
                self._reply(200, {"data": [{"id": "p", "task": "embed", "dimensions": 2}]})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            texts = json.loads(self.rfile.read(length))["input"]
            if self._authorized():
                self._reply(200, {"data": [{"index": i, "embedding": [1.0, 0.0]}
                                           for i in range(len(texts))]})

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen, state
    finally:
        server.shutdown()


def _embedder(url: str, environ: dict[str, str]) -> HttpEmbedder:
    config = EmbeddingConfig(base_url=url, api_key="${KEY}", profile="p")
    return HttpEmbedder(config, environ=environ, timeout=5, attempts=1)


@pytest.mark.parametrize("environ", [{}, {"KEY": ""}, {"KEY": "   "}],
                         ids=["unset", "empty", "whitespace"])
def test_an_empty_key_sends_no_authorization_header(endpoint, environ):
    url, seen, _state = endpoint
    embedder = _embedder(url, environ)
    assert embedder.api_key is None
    embedder.check()
    assert embedder.embed(["相続税"], "query") == [[1.0, 0.0]]
    assert seen and all(header is None for header in seen)


def test_a_set_key_is_sent_trimmed(endpoint):
    url, seen, state = endpoint
    state["require"] = "oumigo-key"
    embedder = _embedder(url, {"KEY": " oumigo-key\n"})
    embedder.check()
    assert seen == ["Bearer oumigo-key"]


def test_a_401_without_a_key_says_no_key_was_sent(endpoint):
    url, _seen, state = endpoint
    state["require"] = "oumigo-key"
    with pytest.raises(EmbedderUnavailable, match=r"no API key was sent \(\$\{KEY\} is empty"):
        _embedder(url, {"KEY": ""}).check()
