"""Tables: files of rows a kb imports into DuckDB, for SQL beside search.

A kb's `tables:` locations (files or folders of CSV, TSV, Excel and SQLite) are imported
by `kb sync` (`store`), and read through `list`, `describe` and read-only SQL (`query`).
Import from the submodules (`manage`, `store`, `query`): `smartrag.kb` imports them too.
Tables meet documents through attributes: a column named like a document attribute links
its rows to the documents with that value.
"""
