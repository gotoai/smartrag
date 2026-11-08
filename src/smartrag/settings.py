"""Server settings, from `SMARTRAG_*` environment variables (or explicit values).

Only query-plane settings live here. The query server deliberately has no source
credentials and no admin settings: those belong to the CLI and the ingestion worker,
which run on the operator host (see the design memo, "admin plane").
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SMARTRAG_")

    host: str = "127.0.0.1"
    port: int = 8710
    api_keys_file: Path | None = None
    """JSON list of hashed API keys, written by `smartrag apikey create` (to come)."""
    auth_disabled: bool = False
    """Development only: `smartrag serve` refuses it unless bound to loopback."""
    public_base_url: str | None = None
    """Base for links in citations (e.g. https://rag.example.jp); default: the request's."""
