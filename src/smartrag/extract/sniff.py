"""Whether a file is text, for files no extractor claims by extension (design memo §3.4).

Source code, configuration files, logs and READMEs come in more extensions than any list
would cover, so for them the bytes decide, the way Kari's viewer decides the same
question: a NUL byte in the first 8 KiB means binary, and so does a sample dense with
control characters. A UTF-16 file is half NUL bytes by construction, so one that starts
with a byte-order mark is judged on its characters instead.

Extensions in `NEVER_TEXT` are not read at all: formats known to be binary, so a share
full of photos costs no reads, and formats that are text on disk but markup or encoded
data rather than anything a reader would search (`.svg`, `.ipynb` with its base64
outputs, source maps).
"""

from __future__ import annotations

from pathlib import Path

#: What is read to decide; Kari's viewer reads the same amount.
SNIFF_BYTES = 8192

#: Control bytes that appear freely in text: backspace, tab, newlines, form feed, escape.
_TEXT_CONTROL_BYTES = frozenset({0x08, 0x09, 0x0A, 0x0C, 0x0D, 0x1B})

#: Above this share of other control bytes, a sample cannot be prose or code.
_MAX_CONTROL_SHARE = 0.1

NEVER_TEXT = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif",
    ".avif", ".ico", ".psd", ".ai", ".raw", ".cr2", ".nef", ".dng",
    # Audio and video
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff",
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpg", ".mpeg",
    ".3gp", ".vob",
    # Archives and disk images
    ".zip", ".7z", ".rar", ".gz", ".tgz", ".bz2", ".xz", ".zst", ".tar", ".lzh", ".cab",
    ".iso", ".dmg",
    # Documents in formats no extractor reads yet
    ".doc", ".ppt", ".odt", ".ods", ".odp", ".pages", ".numbers", ".key", ".epub",
    ".msg", ".one", ".vsd", ".vsdx", ".xps", ".mdi",
    # Programs, libraries and compiled code
    ".exe", ".dll", ".so", ".dylib", ".bin", ".dat", ".o", ".obj", ".a", ".lib", ".class",
    ".jar", ".war", ".pyc", ".pyo", ".wasm", ".msi", ".apk", ".deb", ".rpm",
    # Databases and fonts
    ".db", ".sqlite", ".sqlite3", ".mdb", ".accdb", ".ttf", ".otf", ".woff", ".woff2",
    # Text on disk, but markup or encoded data rather than prose
    ".svg", ".eps", ".ps", ".ipynb", ".map",
})


def looks_binary(sample: bytes) -> bool:
    """Whether a file's leading bytes read as binary rather than text."""
    if not sample:
        return False
    for bom, codec in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be")):
        if sample.startswith(bom):
            # A sample can end inside a character; drop the odd byte.
            body = sample[2:len(sample) - (len(sample) % 2)]
            sample = body.decode(codec, errors="replace").encode("utf-8")
            break
    if b"\0" in sample:
        return True
    control = sum(1 for byte in sample if byte < 0x20 and byte not in _TEXT_CONTROL_BYTES)
    return control / len(sample) > _MAX_CONTROL_SHARE


def is_text_file(path: Path) -> bool:
    """Reads at most `SNIFF_BYTES`. Raises `OSError` when the file cannot be read."""
    if path.suffix.lower() in NEVER_TEXT:
        return False
    with path.open("rb") as handle:
        return not looks_binary(handle.read(SNIFF_BYTES))
