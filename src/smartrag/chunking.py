"""Packing extracted units into chunks (design: SmartRAG_design §3.4, chunking notes).

* The document's **canonical text** is its units joined by newlines; a chunk is a span
  `[start, end)` of it, so snippets, highlights and neighbour expansion need no copy.
* Units are packed greedily up to `target_tokens`, never past `max_tokens`.
* A **heading** ends the current chunk if it already holds `min_tokens`; otherwise the
  heading joins it. Every chunk records the heading path in force where it starts, which
  is prepended to the text that gets embedded (a heading is never a chunk of its own).
* **Table rows** share a chunk only with rows of the same table, and every such chunk
  repeats the table's header row as its `prefix`.
* A unit longer than `max_tokens` is split at sentence ends (`。！？.!?` and line breaks);
  a single sentence that is still too long is cut into windows that overlap slightly.
* A small last chunk is merged into the one before it when they belong together.

**Token counts are estimates** until the embedding profile's own tokenizer is available
(OumiGo prerequisite #8): Japanese characters count 1/1.5 token each and other characters
1/4, which errs towards smaller chunks. `TOKEN_ESTIMATOR` is part of the pipeline version.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from smartrag.extract.units import Unit
from smartrag.kb.config import ChunkConfig

TOKEN_ESTIMATOR = "chars-v1"
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")
_SENTENCE_END = re.compile(r"(?<=[。！？!?])|(?<=\.)\s+|\n")
_MAX_HEADING_CHARS = 80
_WINDOW_OVERLAP_TOKENS = 32


def estimate_tokens(text: str) -> int:
    cjk = len(_CJK.findall(text))
    return max(1, math.ceil(cjk / 1.5 + (len(text) - cjk) / 4))


@dataclass
class Chunk:
    ordinal: int
    start: int
    end: int
    text: str
    tokens: int
    heading_path: list[str] = field(default_factory=list)
    prefix: str = ""
    """A table's header row, for row chunks."""
    page: int | None = None
    page_end: int | None = None
    location: str | None = None

    def embed_text(self) -> str:
        """What is embedded: heading path, then the table header, then the text."""
        parts = []
        if self.heading_path:
            parts.append(" > ".join(self.heading_path))
        if self.prefix:
            parts.append(self.prefix)
        parts.append(self.text)
        return "\n".join(parts)


@dataclass
class _Piece:
    unit: Unit
    start: int
    end: int
    tokens: int


def canonical_text(units: list[Unit]) -> tuple[str, list[tuple[int, int]]]:
    spans = []
    position = 0
    for unit in units:
        spans.append((position, position + len(unit.text)))
        position += len(unit.text) + 1
    return "\n".join(unit.text for unit in units), spans


def chunk_units(units: list[Unit], config: ChunkConfig) -> tuple[str, list[Chunk]]:
    text, spans = canonical_text(units)
    headers = {u.group: u.text for u in units if u.kind == "header" and u.group is not None}
    pieces: list[_Piece] = []
    for unit, (start, end) in zip(units, spans):
        tokens = estimate_tokens(unit.text)
        if unit.kind not in ("heading", "header") and tokens > config.max_tokens:
            pieces += _split(unit, start, end, text, config.max_tokens)
        else:
            pieces.append(_Piece(unit, start, end, tokens))

    chunks: list[Chunk] = []
    path: list[tuple[int, str]] = []
    current: list[_Piece] = []
    current_path: list[str] = []
    current_group: int | None = None
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        body = [p for p in current if p.unit.kind != "heading"]
        if body:
            first, last = current[0], current[-1]
            pages = [p.unit.page for p in current if p.unit.page is not None]
            prefix = headers.get(current_group, "") if current_group is not None else ""
            chunks.append(Chunk(
                ordinal=len(chunks), start=first.start, end=last.end,
                text=text[first.start:last.end], tokens=current_tokens,
                heading_path=list(current_path), prefix=prefix,
                page=min(pages) if pages else None, page_end=max(pages) if pages else None,
                location=_location(body)))
        current, current_tokens = [], 0

    for piece in pieces:
        unit = piece.unit
        if unit.kind == "header":
            continue
        if unit.kind == "heading":
            if current_tokens >= config.min_tokens:
                flush()
            heading = unit.text.strip()[:_MAX_HEADING_CHARS]
            path = [(level, h) for level, h in path if level < unit.level] + [(unit.level, heading)]
            if current:
                current.append(piece)
            continue
        group = unit.group if unit.kind == "row" else None
        if current and (group != current_group or current_tokens + piece.tokens
                        > config.max_tokens):
            flush()
        if not current:
            current_path = [h for _level, h in path]
            current_group = group
            if group is not None and group in headers:
                current_tokens = estimate_tokens(headers[group])
        current.append(piece)
        current_tokens += piece.tokens
        if current_tokens >= config.target_tokens:
            flush()
    flush()
    return text, _merge_small_tail(chunks, text, config)


def _location(body: list[_Piece]) -> str | None:
    rows = [p.unit.row for p in body if p.unit.row is not None]
    where = body[0].unit.location
    if rows and where:
        first, last = min(rows), max(rows)
        return f"{where}, row {first}" if first == last else f"{where}, rows {first}-{last}"
    return where


def _merge_small_tail(chunks: list[Chunk], text: str, config: ChunkConfig) -> list[Chunk]:
    if len(chunks) < 2:
        return chunks
    last, before = chunks[-1], chunks[-2]
    same_context = (last.prefix == before.prefix and last.heading_path == before.heading_path)
    if (last.tokens < config.min_tokens and same_context
            and before.tokens + last.tokens <= config.max_tokens):
        before.end = last.end
        before.text = text[before.start:before.end]
        before.tokens += last.tokens
        before.page_end = max(p for p in (before.page_end, last.page_end) if p is not None) \
            if (before.page_end or last.page_end) else None
        return chunks[:-1]
    return chunks


def _split(unit: Unit, start: int, end: int, text: str, max_tokens: int) -> list[_Piece]:
    """Sentences packed up to `max_tokens`; an over-long sentence in overlapping windows."""
    spans: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        if match.end() > cursor and match.start() >= cursor:
            spans.append((cursor, match.end()))
            cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))

    pieces: list[_Piece] = []
    group_start: int | None = None
    group_end = start
    group_tokens = 0
    for s, e in spans:
        tokens = estimate_tokens(text[s:e])
        if tokens > max_tokens:
            if group_start is not None:
                pieces.append(_Piece(unit, group_start, group_end, group_tokens))
                group_start, group_tokens = None, 0
            pieces += _windows(unit, s, e, text, max_tokens)
            continue
        if group_start is not None and group_tokens + tokens > max_tokens:
            pieces.append(_Piece(unit, group_start, group_end, group_tokens))
            group_start, group_tokens = None, 0
        if group_start is None:
            group_start = s
        group_end, group_tokens = e, group_tokens + tokens
    if group_start is not None:
        pieces.append(_Piece(unit, group_start, group_end, group_tokens))
    return [p for p in pieces if text[p.start:p.end].strip()]


def _windows(unit: Unit, start: int, end: int, text: str, max_tokens: int) -> list[_Piece]:
    # Characters per window at the densest rate (1.5 Japanese characters per token), so a
    # window never exceeds the budget whatever the script.
    size = max(1, int(max_tokens * 1.5))
    step = max(1, size - int(_WINDOW_OVERLAP_TOKENS * 1.5))
    pieces = []
    position = start
    while position < end:
        stop = min(end, position + size)
        pieces.append(_Piece(unit, position, stop, estimate_tokens(text[position:stop])))
        if stop == end:
            break
        position += step
    return pieces
