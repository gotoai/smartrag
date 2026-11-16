"""Telling a query from a document, as the model's dialect says (design memo §3.2).

* `none`: the text as it is.
* `field`: the text as it is, plus one request field, e.g. `input_type: query`.
* `prefix`: the text inside the side's template, e.g. `task: search result | query: {text}`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from smartrag.embedding.catalog import Marking

Side = Literal["query", "document"]


def mark(marking: Marking, texts: Sequence[str], side: Side) -> tuple[list[str], dict[str, str]]:
    """The texts to send, and the request fields to add."""
    value = marking.query if side == "query" else marking.document
    if marking.kind == "field":
        assert marking.field is not None and value is not None
        return list(texts), {marking.field: value}
    if marking.kind == "prefix":
        assert value is not None
        return [value.replace("{text}", text) for text in texts], {}
    return list(texts), {}
