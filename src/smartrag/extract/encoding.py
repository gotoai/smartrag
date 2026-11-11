"""Decoding text files: UTF-8 when it decodes, then detection, then Japanese-first fallbacks."""

from __future__ import annotations

SAMPLE_BYTES = 256 * 1024

#: Tried in order when a file is not UTF-8 and detection declines to answer: Japanese
#: exports are usually cp932, so it comes before cp1252.
_FALLBACKS = ("cp932", "cp1252", "latin-1")


def detect_encoding(sample: bytes, complete: bool) -> str:
    """UTF-8 first, since it wins whenever it decodes; then charset-normalizer; then the
    fallbacks. A sample can cut a multi-byte character in half at its end, so a decode
    error in the last few bytes of an incomplete sample is not evidence against UTF-8."""
    if not sample:
        return "utf-8-sig"
    try:
        sample.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError as error:
        if not complete and error.start >= len(sample) - 4:
            return "utf-8-sig"
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(sample).best()
        if best is not None and best.encoding:
            return best.encoding
    except ImportError:  # pragma: no cover - a declared dependency
        pass
    for candidate in _FALLBACKS:
        try:
            sample.decode(candidate)
            return candidate
        except UnicodeDecodeError:
            continue
    return "latin-1"


def decode_bytes(data: bytes) -> tuple[str, str]:
    """(text, encoding). Undecodable bytes become U+FFFD rather than an error."""
    encoding = detect_encoding(data[:SAMPLE_BYTES], complete=len(data) <= SAMPLE_BYTES)
    return data.decode(encoding, errors="replace"), encoding
