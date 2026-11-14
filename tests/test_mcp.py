"""`smartrag mcp`: the MCP adapter over the agent's view (design memo §3.9)."""

from __future__ import annotations

import json
import sys

from mcp import Client, StdioServerParameters

from smartrag.agent import QueryEmbedders
from smartrag.kb import Home
from smartrag.mcp import build_server
from tests.test_search_read import CharEmbedder

TOOLS = {"list_kbs", "describe_sources", "search", "read"}


def _server(home):
    return build_server(home, embedders=QueryEmbedders(lambda _config: CharEmbedder()))


async def test_four_read_only_tools_with_a_kb_id_enum(agent_home):
    async with Client(_server(agent_home)) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == TOOLS
    for tool in tools.values():
        assert tool.annotations.read_only_hint and tool.annotations.idempotent_hint
        assert tool.annotations.destructive_hint is False and tool.output_schema
        if tool.name != "list_kbs":
            assert tool.input_schema["properties"]["kb_id"]["enum"] == ["legal"]
    search = tools["search"].input_schema["properties"]
    assert set(search) == {"kb_id", "query", "k", "cursor", "source", "file_type",
                           "modified_after", "modified_before"}
    assert set(tools["read"].input_schema["properties"]) == {"kb_id", "chunk_id", "scope",
                                                              "cursor"}


async def test_instructions_list_only_usable_kbs(agent_home):
    async with Client(_server(agent_home)) as client:
        text = client.instructions or ""
    assert "- legal: 税務と人事の資料: 相続税、電子申告、休暇と賃金の規程。" in text
    for hidden in ("nodesc", "empty", "later", "factory"):
        assert f"- {hidden}:" not in text
    assert "search_mode is `keyword`" in text and "never writes the answer" in text


async def test_search_then_read(agent_home):
    async with Client(_server(agent_home)) as client:
        found = await client.call_tool("search", {"kb_id": "legal", "query": "相続税の評価"})
        assert not found.is_error
        data = found.structured_content
        top = data["results"][0]
        assert top["title"] == "相続.docx" and "path" not in top["citation"]
        assert top["citation"]["file_path"].endswith("相続.md")
        text = found.content[0].text
        assert "相続" in text and "\\u" not in text  # Japanese stays readable, and cheap
        assert json.loads(text)["results"][0]["chunk_id"] == top["chunk_id"]

        read = await client.call_tool("read", {"kb_id": "legal", "chunk_id": top["chunk_id"],
                                               "scope": "section"})
        assert not read.is_error and read.structured_content["chunks"]

        kbs = await client.call_tool("list_kbs", {})
        assert kbs.structured_content == {"kbs": [
            {"kb_id": "legal", "description": "税務と人事の資料: 相続税、電子申告、休暇と賃金の規程。"}]}
        described = await client.call_tool("describe_sources", {"kb_id": "legal"})
        assert described.structured_content["documents"] == 3


async def test_failures_are_results_the_model_reads(agent_home):
    async with Client(_server(agent_home)) as client:
        hidden = await client.call_tool("search", {"kb_id": "nodesc", "query": "賃金"})
        bad = await client.call_tool("search", {"kb_id": "legal", "query": "賃金", "k": 99})
        stale = await client.call_tool("read", {"kb_id": "legal", "chunk_id": "d_x:y"})
    assert hidden.is_error and bad.is_error and stale.is_error
    error = json.loads(hidden.content[0].text)["error"]
    assert error == {"code": "not_found", "message": "No knowledge base 'nodesc' is available.",
                     "hint": "Use a kb_id from list_kbs."}
    assert json.loads(bad.content[0].text)["error"]["code"] == "invalid_request"
    assert "chunk_id" in json.loads(stale.content[0].text)["error"]["message"]
    for result in (hidden, bad, stale):
        assert str(agent_home.root) not in result.content[0].text


async def test_starts_with_no_home(tmp_path):
    async with Client(_server(Home(tmp_path / "missing"))) as client:
        assert "No knowledge bases are available yet." in (client.instructions or "")
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert "enum" not in tools["search"].input_schema["properties"]["kb_id"]
        listed = await client.call_tool("list_kbs", {})
    assert listed.structured_content == {"kbs": []}


async def test_stdio_carries_only_the_protocol(agent_home):
    """The real command, as an MCP host spawns it: any stray print would break this. The
    child gets the SDK's minimal environment, so no embedding endpoint is configured and
    the search falls back to keywords, as it would with the endpoint down."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "smartrag", "mcp", "--home", str(agent_home.root), "--allow-kb", "legal"])
    async with Client(params) as client:
        assert "- legal:" in (client.instructions or "")
        assert {t.name for t in (await client.list_tools()).tools} == TOOLS
        result = await client.call_tool("search", {"kb_id": "legal", "query": '"賃金"'})
    assert not result.is_error
    assert result.structured_content["search_mode"] == "keyword"
    assert result.structured_content["results"][0]["file_type"] == "txt"
