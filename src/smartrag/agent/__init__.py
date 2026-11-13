"""The agent's view of a home: usable kbs, and search and read without paths.

Core library, no MCP dependency: the MCP server (`smartrag.mcp`) is a thin adapter over
`AgentService`, and the kb-aware Web API can return the same models (design memo §3.9).
"""

from smartrag.agent.catalog import MAX_DESCRIPTION, AgentCatalog, agent_description
from smartrag.agent.embedders import QueryEmbedders
from smartrag.agent.errors import agent_error
from smartrag.agent.service import DEGRADED, AgentService, safe_title

__all__ = ["DEGRADED", "MAX_DESCRIPTION", "AgentCatalog", "AgentService", "QueryEmbedders",
           "agent_description", "agent_error", "safe_title"]
