"""`smartrag mcp`: the agent-facing MCP server over stdio (design memo §3.9).

A thin adapter over `smartrag.agent.AgentService`: four read-only tools, the usable kbs
listed in the server instructions, `kb_id` an enum of them in every schema. Each result
carries the model as structured content and as compact JSON text (Japanese unescaped);
each failure is a tool result with `isError` and `{"error": {code, message, hint}}`, so
the model reads it. Tool functions are synchronous: the SDK runs each call in a worker
thread, so a slow search never stalls the session.

stdout carries only the protocol. Logs go to stderr at warning level and never contain
query text (no query log in v1, design memo §6 #15).

No `from __future__ import annotations` here: the SDK reads the tools' signatures at
registration, and `KbId` is built per server, so the annotations must be real objects.
"""

import json
import logging
from collections.abc import Callable, Collection
from typing import Annotated, Literal

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import BaseModel, Field

from smartrag.__about__ import __version__
from smartrag.agent import AgentService, QueryEmbedders, agent_error
from smartrag.agent.models import (
    AgentKb,
    AgentKbList,
    AgentReadResponse,
    AgentSearchResponse,
    KbDescription,
)
from smartrag.errors import SmartRAGError
from smartrag.kb.home import Home

log = logging.getLogger("smartrag.mcp")

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                            open_world_hint=False)


def instructions(kbs: list[AgentKb]) -> str:
    """The server instructions: the usable kbs, one line each, and how to use them."""
    lines = [("Smart RAG searches knowledge bases for evidence: passages with citations. "
              "It never writes the answer; you do, from the evidence."), ""]
    if kbs:
        lines.append("Knowledge bases:")
        lines += [f"- {kb.kb_id}: {kb.description}" for kb in kbs]
    else:
        lines.append("No knowledge bases are available yet.")
    lines += [
        "",
        "How to use them:",
        ("- Choose the knowledge base by its description; describe_sources tells you its "
         "sources, file types and dates."),
        "- search returns passages; read widens one when you need more of it.",
        "- Cite what you use as (kb_id, doc_id, title, page).",
        ("- When a search result has `degraded` set, only keyword search was possible: "
         "tell the user."),
        ("- An empty result means nothing in that knowledge base matched. It is not proof "
         "that something does not exist, and when `degraded` is set, not even that."),
    ]
    return "\n".join(lines)


def _result(call: Callable[[], BaseModel], kb_id: str | None = None) -> CallToolResult:
    try:
        model = call()
    except Exception as exc:  # every failure becomes a result the model reads
        if not isinstance(exc, SmartRAGError):
            log.exception("tool call failed")
        error = agent_error(exc, kb_id).model_dump(exclude_defaults=True)
        text = json.dumps({"error": error}, ensure_ascii=False)
        return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
    return CallToolResult(
        content=[TextContent(type="text", text=model.model_dump_json(exclude_none=True))],
        structured_content=model.model_dump(mode="json"))


def build_server(home: Home, allow: Collection[str] | None = None, *,
                 embedders: QueryEmbedders | None = None) -> MCPServer:
    service = AgentService(home, allow, embedders)
    visible = list(service.visible)
    KbId = Annotated[str, Field(  # a type, built per server
        description="A kb_id from list_kbs.",
        json_schema_extra={"enum": visible} if visible else None)]

    server = MCPServer(name="smartrag", title="Smart RAG", version=__version__,
                       instructions=instructions(service.list_kbs().kbs), log_level="WARNING")

    def list_kbs() -> Annotated[CallToolResult, AgentKbList]:
        return _result(service.list_kbs)

    def describe_sources(kb_id: KbId) -> Annotated[CallToolResult, KbDescription]:
        return _result(lambda: service.describe_sources(kb_id), kb_id)

    def search(
        kb_id: KbId,
        query: Annotated[str, Field(description="Natural language, up to 2,000 characters. "
                                                "A part in double quotes is an exact phrase.")],
        k: Annotated[int, Field(description="Results per page, 1-50.")] = 10,
        cursor: Annotated[str | None, Field(
            description="next_cursor from the previous page of the same search.")] = None,
        source: Annotated[list[str] | None, Field(
            description="Only these sources (values from describe_sources).")] = None,
        file_type: Annotated[list[str] | None, Field(
            description="Only these file types, e.g. pdf, docx, xlsx.")] = None,
        modified_after: Annotated[str | None, Field(
            description="Only files modified on or after this date (YYYY-MM-DD).")] = None,
        modified_before: Annotated[str | None, Field(
            description="Only files modified before this date (YYYY-MM-DD).")] = None,
    ) -> Annotated[CallToolResult, AgentSearchResponse]:
        return _result(lambda: service.search(
            kb_id, query, k=k, cursor=cursor, source=source or (), file_type=file_type or (),
            modified_after=modified_after, modified_before=modified_before), kb_id)

    def read(
        kb_id: KbId,
        chunk_id: Annotated[str, Field(description="A chunk_id from a search result.")],
        scope: Annotated[Literal["chunk", "neighbors", "section", "document"], Field(
            description="chunk: the passage; neighbors: with the passages on each side; "
                        "section: its whole section; document: the whole document.")]
        = "chunk",
        cursor: Annotated[str | None, Field(
            description="next_cursor from the previous page of the same read.")] = None,
    ) -> Annotated[CallToolResult, AgentReadResponse]:
        return _result(lambda: service.read(kb_id, chunk_id, scope=scope, cursor=cursor), kb_id)

    server.add_tool(list_kbs, title="List knowledge bases", annotations=READ_ONLY,
                    description="List the knowledge bases you can search, each with what it "
                                "holds. Choose one by its description.")
    server.add_tool(describe_sources, title="Describe a knowledge base", annotations=READ_ONLY,
                    description="Describe one knowledge base: its sources, file types, date "
                                "range and last sync, and the filter values search accepts.")
    server.add_tool(search, title="Search a knowledge base", annotations=READ_ONLY,
                    description="Search one knowledge base for evidence: ranked passages with "
                                "snippets and citations, never an answer. If `degraded` is set, "
                                "only keyword search was possible: say so. read widens a "
                                "passage; next_cursor gives more results.")
    server.add_tool(read, title="Read a passage", annotations=READ_ONLY,
                    description="Read a passage from a search result, widened as far as you "
                                "need. Long reads come in pages: pass next_cursor to continue.")
    return server


def run_stdio(home: Home, allow: Collection[str] | None = None) -> None:
    build_server(home, allow).run("stdio")
