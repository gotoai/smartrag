"""What the table operations record and return."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Value = str | int | float | bool | None


class ColumnInfo(BaseModel):
    name: str
    original: str = Field(description="As the file's header or database names it.")
    type: str = Field(description="integer, number, boolean, date, timestamp or string.")
    unit: str | None = None
    description: str | None = None
    nulls: int = 0
    distinct: int | None = Field(default=None, description="Approximate.")
    min: str | None = None
    max: str | None = None
    values: list[Value] | None = Field(default=None, description="The distinct values, when "
                                       "there are at most 10.")


class TableMeta(BaseModel):
    """One imported table, as `tables.json` records it."""

    name: str
    entry: str = Field(description="The `tables:` location it came through.")
    path: str = Field(description="The file it was read from.")
    file_name: str
    sheet: str | None = Field(default=None, description="The sheet or SQLite table.")
    original: str
    description: str | None = None
    rows: int
    columns: list[ColumnInfo]
    warnings: list[str] = Field(default_factory=list)
    imported_at: str
    importer: str
    size: int
    mtime_ns: int
    digest: str
    schema_stamp: str | None = Field(default=None, description="The schema file's size and "
                                     "time when imported; None without one.")


class TablesFile(BaseModel):
    version: int = 1
    tables: list[TableMeta] = Field(default_factory=list)


class TableSync(BaseModel):
    """What a sync did with one table."""

    name: str
    original: str
    status: Literal["added", "updated", "unchanged", "removed", "kept", "skipped"] = Field(
        description="kept: its file could not be read, or its location is missing, so the "
                    "table from the last import stays; skipped: not imported (its name "
                    "clashes with another table's).")
    rows: int = 0
    columns: int = 0
    warnings: list[str] = Field(default_factory=list)


class TableEntryView(BaseModel):
    uri: str
    exists: bool
    tables: list[str] = Field(default_factory=list, description="The tables its last sync "
                              "imported.")


class TableList(BaseModel):
    kb: str
    entries: list[TableEntryView]
    tables: list[TableMeta]


class TableAdded(BaseModel):
    kb: str
    uri: str
    note: str = "imported at the next `kb sync`"


class TableRemoved(BaseModel):
    kb: str
    uri: str
    tables_removed: list[str] = Field(default_factory=list)


class QueryColumn(BaseModel):
    name: str
    type: str


class TableCitation(BaseModel):
    name: str
    file: str = Field(description="The file's full path; its name only when paths are "
                                  "hidden.")
    sheet: str | None = None
    imported_at: str


class QueryResult(BaseModel):
    kb: str
    sql: str
    columns: list[QueryColumn]
    rows: list[list[Value]]
    row_count: int = Field(description="Rows the query produced, of which `rows` holds the "
                                       "first `max_rows`.")
    truncated: bool
    tables: list[TableCitation] = Field(description="The tables the query read: cite them.")
    took_ms: int


class ColumnLink(BaseModel):
    column: str
    attribute: str


class TableDescription(BaseModel):
    kb: str
    name: str
    original: str
    file: str
    sheet: str | None = None
    description: str | None = None
    rows: int
    columns: list[ColumnInfo]
    links: list[ColumnLink] = Field(
        default_factory=list, description="Columns named like a document attribute of this "
        "kb: pass their values to search's `attributes` filter to reach the documents.")
    sample: list[list[Value]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    imported_at: str
