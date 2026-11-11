"""Source locations: a path or a URI in, a readable `file://` URI out.

Config files store `file://` URIs **unencoded** (`file:///Users/me/規程`, not
`%E8%A6%8F...`), because people read and edit them. A `file://` URI typed by the user is
percent-decoded once on the way in; stored URIs are never decoded again, so a folder
name that happens to contain `%41` survives.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from smartrag.errors import InvalidRequest, Unsupported

#: Schemes the design plans (fsspec connectors, then document platforms), not built yet.
_PLANNED_SCHEMES = frozenset({
    "s3", "gs", "gcs", "az", "abfs", "abfss", "smb", "sftp", "webdav", "http", "https",
})
_WINDOWS_DRIVE = re.compile(r"^/[A-Za-z]:/")


def path_to_uri(path: Path) -> str:
    posix = path.as_posix()
    if not posix.startswith("/"):
        posix = "/" + posix  # a Windows drive path: file:///C:/Users/...
    return "file://" + posix


def uri_to_path(uri: str) -> Path | None:
    """The local path of a stored `file://` URI; None for any other scheme."""
    if not uri.startswith("file://"):
        return None
    rest = uri[len("file://"):]
    if rest.startswith("localhost/"):
        rest = rest[len("localhost"):]
    if _WINDOWS_DRIVE.match(rest):
        rest = rest[1:]
    return Path(rest)


def location_to_uri(location: str) -> tuple[str, Path]:
    """Normalise what the user typed into a stored URI and an absolute local path."""
    if not location.strip():
        raise InvalidRequest("the source location is empty")
    if "://" in location:
        scheme = location.split("://", 1)[0].lower()
        if scheme in _PLANNED_SCHEMES:
            raise Unsupported(
                f"{scheme}:// sources are not supported yet; v1 reads local paths",
                {"scheme": scheme})
        if scheme != "file":
            raise InvalidRequest(f"unknown source scheme {scheme!r}", {"scheme": scheme})
        local = uri_to_path(unquote(location))
        assert local is not None
        path = local.expanduser()
    else:
        path = Path(location).expanduser()
    path = path.resolve(strict=False)
    return path_to_uri(path), path
