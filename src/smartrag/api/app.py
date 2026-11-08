"""Build the query-plane FastAPI app around a `RetrievalEngine`."""

from __future__ import annotations

from fastapi import FastAPI

from smartrag.__about__ import __version__
from smartrag.api import ops, query
from smartrag.api.auth import FileKeyStore, KeyStore
from smartrag.api.errors import install_error_handlers
from smartrag.engine.base import RetrievalEngine
from smartrag.settings import Settings


def create_app(
    engine: RetrievalEngine,
    settings: Settings | None = None,
    key_store: KeyStore | None = None,
) -> FastAPI:
    """The Smart RAG Web API (query plane only).

    Auth: `key_store` if given, else the hashed key file `settings.api_keys_file`. With
    neither, the app refuses to start unless `settings.auth_disabled` is set — an
    unauthenticated server is never the silent default.
    """
    settings = settings or Settings()
    if key_store is None and not settings.auth_disabled:
        if settings.api_keys_file is None:
            raise ValueError("no API keys configured: set SMARTRAG_API_KEYS_FILE "
                             "(or auth_disabled for local development)")
        key_store = FileKeyStore(settings.api_keys_file)

    app = FastAPI(title="Smart RAG", version=__version__,
                  description="Search returns evidence with citations; the agent writes "
                              "the answer.")
    app.state.engine = engine
    app.state.settings = settings
    app.state.key_store = None if settings.auth_disabled else key_store
    install_error_handlers(app)
    app.include_router(query.router)
    app.include_router(ops.router)
    return app
