"""Text for the keyword index: NFKC, lowercase, Japanese split into character bigrams.

FTS5's trigram tokenizer cannot match two-character words, and Japanese is full of them
(休暇, 契約, 規程); tested in Kari's bundled SQLite 3.53.1 (design memo §3.3). So each run
of Japanese characters becomes overlapping bigrams, separated by spaces, and FTS5's
`unicode61` tokenizer splits on those spaces. Latin words and numbers stay whole.

The same function runs on documents and on queries; changing it changes `LEXIFY_VERSION`,
which is part of the pipeline version.
"""

from __future__ import annotations

import re
import unicodedata

LEXIFY_VERSION = "bigram-v1"

_CJK = "々぀-ヿ㐀-䶿一-鿿豈-﫿"
_RUN = re.compile(f"([{_CJK}]+)|([^\\s{_CJK}]+)")


def lexify(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    for cjk, other in _RUN.findall(text):
        if cjk:
            tokens += [cjk] if len(cjk) == 1 else [cjk[i:i + 2] for i in range(len(cjk) - 1)]
        else:
            tokens.append(other)
    return " ".join(tokens)


def fts_query(query: str) -> str:
    """An FTS5 MATCH expression: quoted parts as exact phrases, the rest as an OR of
    bigrams and words, ranked by BM25. Everything is quoted, so user text can never be
    read as FTS5 query syntax."""
    parts: list[str] = []
    for term in re.findall(r'"[^"]+"|\S+', query):
        tokens = lexify(term.strip('"')).split()
        if not tokens:
            continue
        if term.startswith('"') and term.endswith('"') and len(term) > 1:
            parts.append('"' + " ".join(t.replace('"', '""') for t in tokens) + '"')
        else:
            parts += ['"' + t.replace('"', '""') + '"' for t in tokens]
    return " OR ".join(parts)
