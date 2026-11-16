"""Embedding requests in worker processes: `embedding.parallelism` (design memo §3.11)."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from typer.testing import CliRunner

from smartrag.errors import InvalidRequest
from smartrag.index import SqliteIndex
from smartrag.kb import Home, add_source
from smartrag.kb.embed_pool import EmbedPool
from smartrag.kb.sync import sync_kb


def _vector(text: str) -> list[float]:
    """768 values, as ruri-v3-310m@8k (the kbs' model) returns."""
    return [b / 255 for b in hashlib.sha256(text.encode()).digest()[:4]] * 192


class _Endpoint:
    """A local OpenAI-compatible endpoint that takes `delay` seconds per request and
    records the most requests it had in flight at once. A text containing `fail` is
    answered 400; the first `throttle` requests are answered 429 with `Retry-After: 1`."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.fail: str | None = None
        self.throttle = 0
        self.requests = 0
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()

    def handler(self) -> type[BaseHTTPRequestHandler]:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, status: int, body: dict, headers: dict | None = None) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self) -> None:
                texts = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["input"]
                with endpoint._lock:
                    throttled = endpoint.throttle > 0
                    endpoint.throttle -= throttled
                if throttled:
                    self._reply(429, {"detail": "rate limited"}, {"Retry-After": "1"})
                    return
                with endpoint._lock:
                    endpoint.requests += 1
                    endpoint._active += 1
                    endpoint.peak = max(endpoint.peak, endpoint._active)
                try:
                    time.sleep(endpoint.delay)
                finally:
                    with endpoint._lock:
                        endpoint._active -= 1
                if endpoint.fail and any(endpoint.fail in t for t in texts):
                    self._reply(400, {"detail": "rejected"})
                    return
                self._reply(200, {"data": [{"index": i, "embedding": _vector(t)}
                                           for i, t in enumerate(texts)]})

            def log_message(self, *args: object) -> None:
                pass

        return Handler


@pytest.fixture
def endpoint(monkeypatch):
    state = _Endpoint(delay=0.2)
    server = ThreadingHTTPServer(("127.0.0.1", 0), state.handler())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OUMIGO_EMBED_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.delenv("OUMIGO_EMBED_API_KEY", raising=False)
    try:
        yield state
    finally:
        server.shutdown()


def _new_kb(home: Home, kb: str, docs, parallelism: int) -> None:
    """A kb sending one chunk per request, so each document makes several requests."""
    home.new(kb)
    path = home.kb_dir(kb) / "config.yaml"
    text = path.read_text(encoding="utf-8")
    text = text.replace("batch_size: 32", "batch_size: 1")
    text = text.replace("parallelism: 1", f"parallelism: {parallelism}")
    path.write_text(text, encoding="utf-8")
    add_source(home, kb, "docs", str(docs))


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    for n in range(4):
        sections = "\n\n".join(f"## 第{s}条\n\n" + f"規程{n}の第{s}条の本文。" * 60
                               for s in range(1, 4))
        (root / f"規程{n}.md").write_text(f"# 規程{n}\n\n{sections}\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG")
    return root


def _vectors(home: Home, kb: str) -> dict[str, bytes]:
    store = SqliteIndex.open(home.kb_dir(kb) / "index.db", readonly=True)
    try:
        rows = store.db.execute("SELECT text_hash, embedding FROM chunks").fetchall()
    finally:
        store.close()
    return {text_hash: embedding for text_hash, embedding in rows}


def test_parallel_sync_builds_the_same_index_as_serial(tmp_path, endpoint, docs):
    home = Home(tmp_path / "home")
    _new_kb(home, "serial", docs, 1)
    _new_kb(home, "parallel", docs, 3)

    serial = sync_kb(home, "serial")
    assert endpoint.peak == 1
    endpoint.peak = 0
    events: list[dict] = []
    parallel = sync_kb(home, "parallel", progress=events.append)

    assert parallel.embedding_parallelism == 3 and parallel.error is None
    assert endpoint.peak == 3  # never more than configured, and the pool was used
    assert parallel.counts == serial.counts and parallel.counts.added == 4
    assert parallel.chunks_embedded == serial.chunks_embedded > 4
    assert _vectors(home, "parallel") == _vectors(home, "serial")
    # every file reported once, in completion order, counting up to the total
    documents = [e for e in events if e["event"] == "document" and e["source"] == "docs"]
    assert sorted(e["path"] for e in documents) == sorted(p.name for p in docs.iterdir())
    assert [e["n"] for e in documents] == list(range(1, len(documents) + 1))

    endpoint.requests = 0
    again = sync_kb(home, "parallel")
    assert again.counts.unchanged == 5 and endpoint.requests == 0


def test_parallel_sync_is_faster_against_a_slow_endpoint(tmp_path, endpoint, docs):
    endpoint.delay = 0.3
    home = Home(tmp_path / "home")
    _new_kb(home, "serial", docs, 1)
    _new_kb(home, "parallel", docs, 4)
    serial = sync_kb(home, "serial")
    parallel = sync_kb(home, "parallel")
    # Worker start-up costs a little; the requests themselves overlap.
    assert parallel.took_ms < serial.took_ms * 0.6
    assert parallel.timings_ms["embed"] >= parallel.took_ms * 0.9  # summed over workers


def test_the_command_line_overrides_the_config(tmp_path, endpoint, docs):
    from smartrag.cli.main import app

    home = Home(tmp_path / "home")
    _new_kb(home, "legal", docs, 3)

    def sync(*options: str) -> dict:
        result = CliRunner().invoke(app, ["run", "--home", str(home.root), "--format", "json",
                                          "kb", "sync", "--kb", "legal", *options])
        return json.loads(result.stdout.splitlines()[-1])

    assert sync("--dry-run")["embedding_parallelism"] == 3
    report = sync("--embedding-parallelism", "2")
    assert report["embedding_parallelism"] == 2 and report["counts"]["added"] == 4
    assert endpoint.peak == 2
    result = CliRunner().invoke(app, ["run", "--home", str(home.root), "kb", "sync",
                                      "--kb", "legal", "--embedding-parallelism", "0"])
    assert result.exit_code == 2


def test_a_parallelism_below_one_is_refused(tmp_path, endpoint, docs):
    home = Home(tmp_path / "home")
    _new_kb(home, "legal", docs, 2)
    with pytest.raises(InvalidRequest, match="at least 1"):
        sync_kb(home, "legal", embedding_parallelism=0)
    path = home.kb_dir("legal") / "config.yaml"
    path.write_text(path.read_text(encoding="utf-8").replace("parallelism: 2", "parallelism: 0"),
                    encoding="utf-8")
    with pytest.raises(Exception, match="parallelism"):
        home.load("legal")


def test_a_failed_request_stops_the_sync_and_the_next_one_resumes(tmp_path, endpoint, docs):
    home = Home(tmp_path / "home")
    _new_kb(home, "legal", docs, 3)
    endpoint.fail = "規程3の"
    report = sync_kb(home, "legal")
    assert report.error and "HTTP 400" in report.error and "rejected" in report.error
    assert report.counts.added < 4
    assert report.timings_ms["embed"] >= 200  # the failed request's 0.2 s included

    endpoint.fail = None
    again = sync_kb(home, "legal")
    assert again.error is None
    assert again.counts.added + again.counts.unchanged - 1 == 4  # image.png is unchanged
    assert again.counts.added == 4 - report.counts.added


def test_parallel_sync_waits_out_rate_limits(tmp_path, endpoint, docs):
    from smartrag.cli.main import app

    home = Home(tmp_path / "home")
    _new_kb(home, "legal", docs, 3)
    endpoint.throttle = 2
    events: list[dict] = []
    report = sync_kb(home, "legal", progress=events.append)
    assert report.error is None and report.counts.added == 4
    assert report.rate_limit_waits == 2 == sum(e["event"] == "wait" for e in events)

    _new_kb(home, "strict", docs, 3)
    endpoint.throttle = 100
    result = CliRunner().invoke(app, ["run", "--home", str(home.root), "--format", "json",
                                      "kb", "sync", "--kb", "strict",
                                      "--adapt-to-rate-limit=false"])
    report_json = json.loads(result.stdout.splitlines()[-1])  # after the progress events
    assert result.exit_code == 1 and "HTTP 429" in report_json["error"]
    assert report_json["adapt_to_rate_limit"] is False


def test_closing_with_requests_in_flight_stops_the_workers_at_once():
    pool = EmbedPool(_Sleeper(60), 2)
    pool.submit(["a"], "a")
    pool.submit(["b"], "b")
    time.sleep(1.0)  # both workers started and busy
    executor = pool._executor
    assert executor is not None
    processes = list(executor._processes.values())
    assert len(processes) == 2
    started = time.monotonic()
    pool.close()
    assert time.monotonic() - started < 10
    assert not any(process.is_alive() for process in processes)


class _Sleeper:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def embed(self, texts, input_type):
        time.sleep(self.seconds)
        return [_vector(t) for t in texts]


@pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is POSIX")
def test_workers_exit_when_the_main_process_is_killed(tmp_path):
    script = textwrap.dedent("""
        import os, sys, time
        from smartrag.kb.embed_pool import EmbedPool
        from tests.test_parallel_embedding import _Sleeper

        if __name__ == "__main__":
            pool = EmbedPool(_Sleeper(600), 2)
            pool.submit(["a"], "a")
            pool.submit(["b"], "b")
            time.sleep(1.5)
            print(" ".join(str(p.pid) for p in pool._executor._processes.values()), flush=True)
            time.sleep(600)
    """)
    path = tmp_path / "main.py"
    path.write_text(script, encoding="utf-8")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([root, *sys.path])}
    # The killed process's resource tracker reports the semaphores it cleans up; keep that
    # out of the test output.
    errors = tmp_path / "stderr.txt"
    with errors.open("w") as stderr:
        main = subprocess.Popen([sys.executable, str(path)], stdout=subprocess.PIPE,
                                stderr=stderr, text=True, env=env)
    workers: list[int] = []
    try:
        assert main.stdout is not None
        workers = [int(pid) for pid in main.stdout.readline().split()]
        assert len(workers) == 2, errors.read_text()
        main.send_signal(signal.SIGKILL)
        main.wait()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(_alive(pid) for pid in workers):
            time.sleep(0.1)
        assert not any(_alive(pid) for pid in workers)
    finally:
        if main.poll() is None:
            main.kill()
        for pid in workers:  # only if the test failed: leave nothing running
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A worker that has exited but not been reaped (its parent is gone) is a zombie only
    # until init reaps it; /proc tells them apart where it exists.
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as stat:
            return stat.read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return True
