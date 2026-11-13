"""The `smartrag` CLI.

    smartrag                      help
    smartrag run ...              one command, non-interactive (see `smartrag.cli.run`)
    smartrag serve ...            the Web API
    smartrag mcp ...              the MCP server over stdio, for agents (read-only)
    smartrag version

`console` (interactive session) is to come. Admin commands run through `smartrag run` on
the host that owns the kbs, never through the Web API or MCP.
"""

from __future__ import annotations

import ipaddress
import logging
from pathlib import Path

import typer

from smartrag.__about__ import __version__
from smartrag.cli.run import run_app

app = typer.Typer(help="Smart RAG: a knowledge source any agent can use.",
                  no_args_is_help=True, add_completion=False)
app.add_typer(run_app, name="run")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address [default: $SMARTRAG_HOST or 127.0.0.1]"),
    port: int = typer.Option(None, help="Port [default: $SMARTRAG_PORT or 8710]"),
    api_keys_file: Path = typer.Option(None, help="Hashed API key file "
                                                  "[default: $SMARTRAG_API_KEYS_FILE]"),
    demo: bool = typer.Option(False, help="Serve the built-in in-memory demo corpus."),
    no_auth: bool = typer.Option(False, help="Disable API keys (loopback only, development)."),
    log_level: str = typer.Option("info"),
) -> None:
    """Run the query-plane Web API."""
    import uvicorn

    from smartrag.api import create_app
    from smartrag.engine import demo_engine
    from smartrag.settings import Settings

    overrides = {k: v for k, v in {"host": host, "port": port,
                                   "api_keys_file": api_keys_file}.items() if v is not None}
    if no_auth:
        overrides["auth_disabled"] = True
    settings = Settings(**overrides)
    if settings.auth_disabled and not _is_loopback(settings.host):
        raise typer.BadParameter(f"--no-auth is only allowed on a loopback address, not "
                                 f"{settings.host}", param_hint="--no-auth")
    if not demo:
        # TODO: build the PostgreSQL engine from settings once the index exists.
        typer.echo("smartrag: no index backend is implemented yet; run with --demo", err=True)
        raise typer.Exit(2)

    logging.basicConfig(level=log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        web_app = create_app(demo_engine(), settings)
    except ValueError as exc:
        typer.echo(f"smartrag: {exc}", err=True)
        raise typer.Exit(2) from exc
    uvicorn.run(web_app, host=settings.host, port=settings.port, log_level=log_level)


@app.command("mcp")
def mcp_command(
    home: Path = typer.Option(None, "--home", help="Folder holding all kbs "
                                                   "[default: $SMARTRAG_HOME or ~/.smartrag]"),
    allow_kb: list[str] = typer.Option(None, "--allow-kb",
                                       help="Offer only this kb. Repeatable [default: all]."),
) -> None:
    """Run the MCP server over stdio for agents: list_kbs, describe_sources, search, read.

    Always starts, even with no home or no usable kb: an MCP host shows a server that
    fails to start as broken, not as unconfigured. stdout carries only the protocol."""
    from smartrag.kb import Home, resolve_home
    from smartrag.mcp import run_stdio

    run_stdio(Home(resolve_home(home)), allow_kb or None)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


def main() -> None:
    app()
