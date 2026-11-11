"""Knowledge bases and their sources: the core library behind `smartrag run kb|source`.

No CLI or web dependencies: the CLI, and later the console, Web API and MCP adapters,
call these functions and render the models they return.
"""

from smartrag.kb.home import Home, resolve_home
from smartrag.kb.sources import add_source, list_sources, remove_source, test_source

__all__ = [
           "Home",
           "add_source",
           "list_sources",
           "remove_source",
           "resolve_home",
           "test_source",
]
