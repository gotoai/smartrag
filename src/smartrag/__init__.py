"""Smart RAG — a knowledge source any agent can use.

Search returns evidence (snippets, stable ids, citations); the agent, or a person at the
search box, writes the answer. See `smartrag.api` for the query-plane Web API and
`smartrag.engine` for the retrieval-engine seam behind it.
"""

from smartrag.__about__ import __version__

__all__ = ["__version__"]
