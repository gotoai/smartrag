"""The index: SQLite in v1 (design memo §3.3)."""

from smartrag.index.lexical import LEXIFY_VERSION, fts_query, lexify
from smartrag.index.store import INDEX_SCHEMA_VERSION, DocumentRow, IndexStats, SqliteIndex

__all__ = ["INDEX_SCHEMA_VERSION", "LEXIFY_VERSION", "DocumentRow", "IndexStats",
           "SqliteIndex", "fts_query", "lexify"]
