"""The embedding layer: catalog, resolve, dialects, and the HTTP engine."""

from __future__ import annotations

import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from smartrag.embedding import transport
from smartrag.embedding.api import open_embedder, resolve
from smartrag.embedding.catalog import catalog, provider_list
from smartrag.embedding.engine import HttpEmbedder
from smartrag.embedding.wire import CohereV2Wire, OpenAIWire
from smartrag.errors import ConfigError, EmbedderUnavailable
from smartrag.kb.config import EmbeddingConfig, KbConfig


@pytest.fixture
def endpoint():
    """A local OpenAI-compatible endpoint that records each request's Authorization header
    and body. `require` makes it answer 401 to any other key, as OumiGo's router does;
    `fail` lists statuses to answer, in turn, before succeeding; `dims` sets the vector
    length; `max_inputs` answers 429 to bigger requests; `echo` makes each vector start
    with its text's length."""
    seen: list[str | None] = []
    bodies: list[dict] = []
    state: dict = {"require": None, "fail": [], "retry_after": None, "dims": 2,
                   "max_inputs": None, "echo": False}

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: object, headers: dict | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reply(404, {"detail": "no such route"})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            header = self.headers.get("Authorization")
            seen.append(header)
            bodies.append(body)
            if state["require"] and header != f"Bearer {state['require']}":
                self._reply(401, {"detail": "missing or invalid API key"})
                return
            if state["fail"]:
                status = state["fail"].pop(0)
                extra = ({"Retry-After": state["retry_after"]} if state["retry_after"]
                         else None)
                self._reply(status, {"error": {"message": f"status {status}"}}, extra)
                return
            if self.path.endswith("/embed"):  # Cohere's native API
                self._reply(200, {"id": "e", "embeddings": {"float": [
                    [float(len(t))] + [1.0] * (state["dims"] - 1) for t in body["texts"]]}})
                return
            texts = body["input"]
            if state["max_inputs"] and len(texts) > state["max_inputs"]:
                self._reply(429, {"detail": "too many tokens per minute"},
                            {"Retry-After": state["retry_after"]} if state["retry_after"]
                            else None)
                return
            self._reply(200, {"data": [
                {"index": i, "embedding": ([float(len(texts[i]))] if state["echo"] else [1.0])
                 + [1.0] * (state["dims"] - 1)} for i in reversed(range(len(texts)))]})

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen, bodies, state
    finally:
        server.shutdown()


def _custom(url: str, **extra) -> EmbeddingConfig:
    fields = {"provider_id": "custom", "model_profile_id": "p", "base_url": url,
              "api_key": "${KEY}", "model_canonical_name": "org/p", "dimensions": 2,
              "max_input_tokens": 512, **extra}
    return EmbeddingConfig(**fields)


# --- catalog and resolve ------------------------------------------------------------


def test_the_first_wave_is_in_the_catalog():
    providers = catalog()
    assert set(providers) == {"gotoai", "voyage_ai", "openai", "google_gemini", "cohere",
                              "mistral", "custom"}
    assert set(providers["gotoai"].models) == {
        "ruri-v3-310m@8k", "harrier-oss-v1-0.6b@8k", "harrier-oss-v1-0.6b@32k"}
    listed = {p.provider_id: p for p in provider_list().providers}
    assert listed["openai"].api_key_env == "OPENAI_API_KEY"
    assert listed["gotoai"].base_url is None and listed["gotoai"].base_url_env


def test_profiles_of_one_model_share_a_vector_space():
    eight = resolve(EmbeddingConfig(provider_id="gotoai",
                                    model_profile_id="harrier-oss-v1-0.6b@8k"))
    long = resolve(EmbeddingConfig(provider_id="gotoai",
                                   model_profile_id="harrier-oss-v1-0.6b@32k"))
    ruri = resolve(EmbeddingConfig(provider_id="gotoai", model_profile_id="ruri-v3-310m@8k"))
    assert eight.space == long.space
    assert eight.space.key == "microsoft/harrier-oss-v1-0.6b|1024|field:input_type"
    assert ruri.space != eight.space
    # Mirrors of the vendor repos: named after the mirror, in the vendor model's space.
    assert eight.model_canonical_name == "GotoAI-Inc/harrier-oss-v1-0.6b-mirror"
    assert ruri.space.key == "cl-nagoya/ruri-v3-310m|768|field:input_type"


def test_a_vendor_suite_shares_one_space_and_dimensions_split_it():
    lite = resolve(EmbeddingConfig(provider_id="voyage_ai", model_profile_id="voyage-4-lite"))
    large = resolve(EmbeddingConfig(provider_id="voyage_ai", model_profile_id="voyage-4-large"))
    short = resolve(EmbeddingConfig(provider_id="voyage_ai", model_profile_id="voyage-4",
                                    dimensions=512))
    assert lite.space == large.space
    assert lite.model_canonical_name != large.model_canonical_name
    assert short.space.dimensions == 512 and short.space != lite.space
    assert OpenAIWire.body(short, ["a"], {}) == {"model": "voyage-4", "input": ["a"],
                                                 "output_dimension": 512}
    assert "output_dimension" not in OpenAIWire.body(lite, ["a"], {})


def test_defaults_come_from_the_catalog():
    spec = resolve(EmbeddingConfig(provider_id="openai",
                                   model_profile_id="text-embedding-3-small"))
    assert spec.base_url == "https://api.openai.com/v1"
    assert spec.api_key == "${OPENAI_API_KEY}"
    assert spec.space.key == "@openai/text-embedding-3-small|1536|none"
    gotoai = resolve(EmbeddingConfig(provider_id="gotoai", model_profile_id="ruri-v3-310m@8k"))
    assert gotoai.base_url == "${GOTOAI_EMBED_BASE_URL}"


@pytest.mark.parametrize(("fields", "message"), [
    ({"provider_id": "nope", "model_profile_id": "m"}, "unknown embedding provider"),
    ({"provider_id": "openai", "model_profile_id": "gpt-5"}, "has no model 'gpt-5'"),
    ({"provider_id": "openai", "model_profile_id": "text-embedding-3-small",
      "base_url": "https://evil.example/v1"}, "fixed URL"),
    ({"provider_id": "voyage_ai", "model_profile_id": "voyage-4", "dimensions": 300},
     "cannot return 300"),
    ({"provider_id": "voyage_ai", "model_profile_id": "voyage-4", "batch_size": 1001},
     "over provider 'voyage_ai'"),
    ({"provider_id": "openai", "model_profile_id": "text-embedding-3-small",
      "max_input_tokens": 100}, "remove embedding.max_input_tokens"),
    ({"provider_id": "custom", "model_profile_id": "m"}, "needs embedding.model_canonical_name"),
    ({"provider_id": "custom", "model_profile_id": "m", "model_canonical_name": "o/m",
      "dimensions": 8, "max_input_tokens": 512, "dialect": "openai_prefix"},
     "goes with dialect openai_prefix"),
])
def test_resolve_refuses_what_the_catalog_does_not_allow(fields, message):
    with pytest.raises(ValueError, match=message):
        EmbeddingConfig(**fields)


def test_chunks_must_fit_the_model():
    with pytest.raises(ValueError, match="over the embedding model's input limit"):
        KbConfig(id="k", title="k", chunk={"max_tokens": 600, "target_tokens": 350},
                 embedding=_custom("http://127.0.0.1/v1"))


def test_a_custom_prefix_names_its_templates():
    spec = resolve(_custom("http://x/v1", dialect="openai_prefix",
                           prefix={"query": "検索クエリ: {text}", "document": "検索文書: {text}"}))
    other = resolve(_custom("http://x/v1", dialect="openai_prefix",
                            prefix={"query": "Q: {text}", "document": "D: {text}"}))
    assert spec.space.marking_id.startswith("prefix:custom-")
    assert spec.space != other.space


# --- the engine ---------------------------------------------------------------------


def test_queries_and_documents_are_marked_by_the_dialect(endpoint):
    url, _seen, bodies, _state = endpoint
    field = open_embedder(_custom(url, dialect="openai_input_type"), environ={})
    field.embed(["相続税"], "query")
    field.embed(["相続税"], "document")
    assert [b.get("input_type") for b in bodies] == ["query", "document"]
    prefix = open_embedder(_custom(url, dialect="openai_prefix",
                                   prefix={"query": "Q: {text}", "document": "D: {text}"}),
                           environ={})
    prefix.embed(["相続税"], "query")
    assert bodies[-1] == {"model": "p", "input": ["Q: 相続税"]}
    plain = open_embedder(_custom(url), environ={})
    plain.embed(["相続税"], "query")
    assert bodies[-1] == {"model": "p", "input": ["相続税"]}


def test_vectors_come_back_in_input_order_and_of_the_promised_length(endpoint):
    url, _seen, _bodies, state = endpoint
    embedder = open_embedder(_custom(url), environ={})
    assert embedder.embed(["a", "b"], "document") == [[1.0, 1.0], [1.0, 1.0]]
    state["dims"] = 3
    with pytest.raises(EmbedderUnavailable, match="returned 3 dimensions; expected 2"):
        embedder.verify()


def test_batches_respect_inputs_and_tokens(endpoint):
    url, _seen, bodies, _state = endpoint
    embedder = open_embedder(_custom(url, batch_size=3), environ={})
    embedder.embed([f"text {i}" for i in range(7)], "document")
    assert [len(b["input"]) for b in bodies] == [3, 3, 1]
    bodies.clear()
    embedder.spec = dataclasses.replace(embedder.spec, max_request_tokens=100)
    embedder.embed(["あ" * 60] * 3, "document")  # 40 estimated tokens each, 80 per request
    assert [len(b["input"]) for b in bodies] == [2, 1]


def test_retries_honour_retry_after_and_report_a_kind(endpoint):
    url, _seen, bodies, state = endpoint
    state["fail"], state["retry_after"] = [429], "0"
    embedder = open_embedder(_custom(url), environ={}, attempts=2, timeout=5)
    assert embedder.embed(["a"], "query") == [[1.0, 1.0]]
    assert len(bodies) == 2
    state["fail"] = [400]
    with pytest.raises(EmbedderUnavailable, match="HTTP 400: status 400") as caught:
        embedder.embed(["a"], "query")
    assert caught.value.details["kind"] == "rejected"


def test_a_cloud_provider_without_its_key_sends_nothing():
    config = EmbeddingConfig(provider_id="voyage_ai", model_profile_id="voyage-4")
    with pytest.raises(ConfigError, match=r"\$\{VOYAGE_API_KEY\} is empty or not set"):
        open_embedder(config, environ={"VOYAGE_API_KEY": " "})


def test_a_long_server_message_is_kept_whole():
    notice = "You have not yet added your payment method. " * 9  # about 400 characters
    assert OpenAIWire.error_detail(json.dumps({"detail": notice})) == notice
    assert len(OpenAIWire.error_detail("x" * 5000)) == 1000


def test_an_unset_base_url_variable_is_a_config_error():
    with pytest.raises(ConfigError, match="GOTOAI_EMBED_BASE_URL"):
        open_embedder(EmbeddingConfig(provider_id="gotoai", model_profile_id="ruri-v3-310m@8k"),
                      environ={})


# --- adapting to rate limits ---------------------------------------------------------


@pytest.fixture
def slept(monkeypatch):
    """The transport's waits, recorded instead of slept."""
    waits: list[float] = []
    monkeypatch.setattr(transport.time, "sleep", waits.append)
    return waits


def test_adapting_waits_out_429_and_503_without_using_up_attempts(endpoint, slept):
    url, _seen, bodies, state = endpoint
    state["fail"], state["retry_after"] = [429, 429, 503], "5"
    embedder = open_embedder(_custom(url), environ={}, attempts=1, adaptive=True)
    told: list[tuple[float, str]] = []
    embedder.on_wait = lambda seconds, kind: told.append((seconds, kind))
    assert embedder.embed(["a"], "document") == [[1.0, 1.0]]
    assert slept == [5.0, 5.0, 5.0] and len(bodies) == 4
    assert told == [(5.0, "rate_limited"), (5.0, "rate_limited"), (5.0, "server")]
    state["fail"] = [429]
    with pytest.raises(EmbedderUnavailable, match="HTTP 429"):
        open_embedder(_custom(url), environ={}, attempts=1).embed(["a"], "document")


def test_without_retry_after_the_backoff_doubles_within_a_budget(endpoint, slept, monkeypatch):
    url, _seen, _bodies, state = endpoint
    monkeypatch.setattr(transport, "WAIT_BUDGET", 20.0)
    state["fail"] = [429] * 50
    embedder = open_embedder(_custom(url), environ={}, adaptive=True)
    with pytest.raises(EmbedderUnavailable, match=r"waited 14 s") as caught:
        embedder.embed(["a"], "document")
    assert slept == [2.0, 4.0, 8.0] and caught.value.details["kind"] == "rate_limited"
    state["fail"], state["retry_after"] = [429], "0"
    slept.clear()
    embedder.embed(["a"], "document")
    assert slept == [transport.MIN_WAIT]  # `Retry-After: 0` still pauses


def test_a_request_refused_for_a_minute_is_split_and_later_batches_stay_small(
        endpoint, slept, monkeypatch):
    url, _seen, bodies, state = endpoint
    monkeypatch.setattr(transport, "SPLIT_AFTER", 3.0)
    state.update(max_inputs=2, retry_after="2", echo=True)
    embedder = open_embedder(_custom(url, batch_size=8), environ={}, adaptive=True)
    vectors = embedder.embed(["a", "bb", "ccc", "dddd", "eeeee"], "document")
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert embedder.batch_cap == 1
    sent = len(bodies)
    embedder.embed(["f", "g"], "document")
    assert [len(b["input"]) for b in bodies[sent:]] == [1, 1]


# --- Cohere's native API, and Mistral ------------------------------------------------


def test_cohere_marks_queries_and_documents_on_its_native_api(endpoint):
    url, _seen, bodies, state = endpoint
    state["dims"] = 512
    spec = resolve(EmbeddingConfig(provider_id="cohere", model_profile_id="embed-v5.0-fast",
                                   dimensions=512))
    assert spec.base_url == "https://api.cohere.com/v2" and spec.api_key == "${CO_API_KEY}"
    embedder = HttpEmbedder(dataclasses.replace(spec, base_url=url),
                            environ={"CO_API_KEY": "k"})
    assert embedder.url.endswith("/v1/embed")
    vectors = embedder.embed(["相続", "税"], "query")
    assert [v[0] for v in vectors] == [2.0, 1.0] and len(vectors[0]) == 512
    embedder.embed(["相続"], "document")
    assert bodies[0] == {"model": "embed-v5.0-fast", "texts": ["相続", "税"],
                         "embedding_types": ["float"], "truncate": "NONE",
                         "input_type": "search_query", "output_dimension": 512}
    assert bodies[1]["input_type"] == "search_document"


def test_cohere_pro_and_fast_share_a_space_and_mistral_marks_nothing():
    pro = resolve(EmbeddingConfig(provider_id="cohere", model_profile_id="embed-v5.0-pro"))
    fast = resolve(EmbeddingConfig(provider_id="cohere", model_profile_id="embed-v5.0-fast"))
    assert pro.space == fast.space
    assert pro.space.key == "@cohere/embed-v5.0-suite|2048|field:input_type"
    mistral = resolve(EmbeddingConfig(provider_id="mistral", model_profile_id="mistral-embed"))
    assert mistral.space.key == "@mistralai/mistral-embed|1024|none"
    with pytest.raises(ValueError, match="over provider 'cohere'"):
        EmbeddingConfig(provider_id="cohere", model_profile_id="embed-v5.0-pro", batch_size=97)
    with pytest.raises(ValueError, match="cannot return 300"):
        EmbeddingConfig(provider_id="cohere", model_profile_id="embed-v5.0-pro", dimensions=300)


def test_a_cohere_answer_without_float_vectors_is_an_error():
    with pytest.raises(EmbedderUnavailable, match="embeddings.float"):
        CohereV2Wire.vectors({"embeddings": {"int8": [[1]]}}, 1)
    with pytest.raises(EmbedderUnavailable, match="returned 1 vectors for 2"):
        CohereV2Wire.vectors({"embeddings": {"float": [[0.1]]}}, 2)


# --- credentials ----------------------------------------------------------------------


@pytest.mark.parametrize("environ", [{}, {"KEY": ""}, {"KEY": "   "}],
                         ids=["unset", "empty", "whitespace"])
def test_an_empty_key_sends_no_authorization_header(endpoint, environ):
    url, seen, _bodies, _state = endpoint
    embedder = open_embedder(_custom(url), environ=environ, attempts=1, timeout=5)
    assert embedder.transport.api_key is None
    assert embedder.embed(["相続税"], "query") == [[1.0, 1.0]]
    assert seen == [None]


def test_a_set_key_is_sent_trimmed(endpoint):
    url, seen, _bodies, state = endpoint
    state["require"] = "gotoai-key"
    open_embedder(_custom(url), environ={"KEY": " gotoai-key\n"}).verify()
    assert seen == ["Bearer gotoai-key"]


def test_a_401_without_a_key_says_no_key_was_sent(endpoint):
    url, _seen, _bodies, state = endpoint
    state["require"] = "gotoai-key"
    with pytest.raises(EmbedderUnavailable,
                       match=r"no API key was sent \(\$\{KEY\} is empty") as caught:
        open_embedder(_custom(url), environ={"KEY": ""}, attempts=1).verify()
    assert caught.value.details["kind"] == "auth"
