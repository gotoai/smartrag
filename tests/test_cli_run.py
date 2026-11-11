"""`smartrag run`: the CLI contract — options, JSON output, exit codes."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from smartrag.cli.main import app

runner = CliRunner()


@pytest.fixture
def run(tmp_path):
    home = tmp_path / "home"

    def invoke(*args: str, fmt: str = "json"):
        return runner.invoke(app, ["run", "--home", str(home), "--format", fmt, *args])

    return invoke


def _json(result):
    return json.loads(result.stdout)


def test_bare_smartrag_prints_help():
    result = runner.invoke(app, [])
    assert "Usage" in result.output and "run" in result.output


def test_kb_lifecycle(run, tmp_path):
    created = _json(run("kb", "new", "--kb", "legal", "--title", "法務"))
    assert created["kb"] == "legal" and created["kind"] == "local"
    listed = _json(run("kb", "list"))
    assert [(k["kb"], k["title"], k["status"]) for k in listed["kbs"]] == [("legal", "法務", "ok")]
    status = _json(run("kb", "status", "--kb", "legal"))
    assert status["title"] == "法務" and status["index"]["exists"] is False
    removed = run("kb", "remove", "--kb", "legal", "--yes")
    assert removed.exit_code == 0 and _json(removed)["kb"] == "legal"
    assert _json(run("kb", "list"))["kbs"] == []


def test_kb_is_always_required(run):
    for args in (["kb", "status"], ["kb", "new"], ["source", "list"]):
        result = run(*args)
        assert result.exit_code == 2, args


def test_errors_are_json_on_stdout_with_exit_1(run):
    result = run("kb", "status", "--kb", "nope")
    assert result.exit_code == 1
    assert _json(result)["error"]["code"] == "not_found"


def test_text_errors_go_to_stderr(run):
    result = run("kb", "status", "--kb", "nope", fmt="text")
    assert result.exit_code == 1
    assert result.stdout == "" and "no kb 'nope'" in result.stderr


def test_destructive_commands_need_yes_without_a_terminal(run, tmp_path):
    run("kb", "new", "--kb", "legal")
    result = run("kb", "remove", "--kb", "legal", fmt="text")
    assert result.exit_code == 1 and "--yes" in result.stderr
    assert _json(run("kb", "list"))["kbs"]


def test_source_commands(run, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("x", encoding="utf-8")
    run("kb", "new", "--kb", "legal")
    added = run("source", "add", "--kb", "legal", "--source", "kitei", str(docs),
                "--include", "**/*.txt", "--exclude", "**/draft/**")
    assert added.exit_code == 0, added.output
    assert _json(added)["source"]["include"] == ["**/*.txt"]
    ids = [s["id"] for s in _json(run("source", "list", "--kb", "legal"))["sources"]]
    assert ids == ["kitei", "sink"]
    report = run("source", "test", "--kb", "legal", "--source", "kitei")
    assert report.exit_code == 0 and _json(report)["files_matched"] == 1
    removed = run("source", "remove", "--kb", "legal", "--source", "kitei", "--yes")
    assert removed.exit_code == 0


def test_source_test_exit_code_reflects_an_unreadable_root(run, tmp_path):
    run("kb", "new", "--kb", "legal")
    run("source", "add", "--kb", "legal", "--source", "nas", str(tmp_path / "gone"),
        "--allow-missing")
    result = run("source", "test", "--kb", "legal", "--source", "nas")
    assert result.exit_code == 1 and _json(result)["ok"] is False


def test_text_output_is_readable(run, tmp_path):
    run("kb", "new", "--kb", "legal", "--title", "法務ナレッジベース")
    text = run("kb", "list", fmt="text").stdout
    assert "法務ナレッジベース" in text and "KB" in text
    status = run("kb", "status", "--kb", "legal", fmt="text").stdout
    assert "not built" in status and "warnings" in status
