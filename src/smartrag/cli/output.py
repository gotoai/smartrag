"""How `smartrag run` prints: text for people, JSON for programs (`--format json`).

In JSON mode every outcome is one JSON document on stdout, failures included
(`{"error": {"code", "message", "details"}}`, the Web API's error shape), so a caller
such as Kari reads stdout either way and checks the exit code. In text mode errors go
to stderr.

Exit codes: 0 success, 1 failed, 2 usage error (bad options; from the CLI parser).
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import typer
from pydantic import BaseModel

from smartrag.errors import InvalidRequest, SmartRAGError
from smartrag.kb import Home

M = TypeVar("M", bound=BaseModel)


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


@dataclass
class RunState:
    home: Home
    format: OutputFormat


def invoke(state: RunState, action: Callable[[], M], render: Callable[[M], str],
           failed: Callable[[M], bool] = lambda _result: False) -> None:
    try:
        result = action()
    except SmartRAGError as err:
        emit_error(state.format, err)
        raise typer.Exit(1) from None
    if state.format is OutputFormat.json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(render(result))
    if failed(result):
        raise typer.Exit(1)


def emit_error(fmt: OutputFormat, err: SmartRAGError) -> None:
    if fmt is OutputFormat.json:
        body = {"error": {"code": err.code, "message": err.message, "details": err.details}}
        typer.echo(json.dumps(body, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"smartrag: {err.message}", err=True)


def confirm(state: RunState, yes: bool, question: str) -> None:
    """Ask before a destructive step. Without a terminal there is nobody to ask, so
    `--yes` is required rather than assumed."""
    if yes:
        return
    if state.format is OutputFormat.json or not sys.stdin.isatty():
        raise InvalidRequest("this command deletes data: pass --yes to confirm when it is "
                             "not run interactively")
    if not typer.confirm(question, default=False):
        typer.echo("cancelled", err=True)
        raise typer.Exit(1)


def display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cells = [[str(h) for h in headers]] + [["-" if c in (None, "") else str(c) for c in row]
                                          for row in rows]
    widths = [max(display_width(row[i]) for row in cells) for i in range(len(headers))]
    lines = []
    for row in cells:
        padded = [cell + " " * (widths[i] - display_width(cell)) for i, cell in enumerate(row)]
        lines.append("  ".join(padded).rstrip())
    return "\n".join(lines)


def human_bytes(size: int | None) -> str:
    if size is None:
        return "-"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def bullets(items: Sequence[str], indent: str = "  - ") -> str:
    return "\n".join(f"{indent}{item}" for item in items)
