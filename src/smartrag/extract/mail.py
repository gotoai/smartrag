"""E-mail messages (.eml), read with the standard library's `email` package.

The subject is the document's title and its top heading, so every chunk of a long thread
says which message it came from. Then come From, To, Cc and Date, then the body: the
plain-text part when there is one, else the HTML part, read like an HTML file. Encoded
headers and bodies (ISO-2022-JP, Shift_JIS, quoted-printable, base64) are decoded.
Attachments are listed by name only; their contents are not extracted.
"""

from __future__ import annotations

import email
from email import policy
from email.message import EmailMessage
from pathlib import Path

from smartrag.extract.encoding import decode_bytes
from smartrag.extract.text import html_units, paragraphs
from smartrag.extract.units import Extracted, Unit

_HEADERS = ("From", "To", "Cc", "Date")


def _header(message: EmailMessage, name: str) -> str:
    try:
        return " ".join(str(message.get(name) or "").split())
    except (ValueError, LookupError):  # a malformed or wrongly-encoded header
        return ""


def _body_text(part: EmailMessage) -> str:
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
    except (LookupError, ValueError):  # an unknown or wrong charset
        pass
    payload = part.get_payload(decode=True)
    return decode_bytes(payload)[0] if isinstance(payload, bytes) else ""


def extract_eml(path: Path) -> Extracted:
    message = email.message_from_bytes(path.read_bytes(), policy=policy.default)
    assert isinstance(message, EmailMessage)
    subject = _header(message, "Subject") or None
    units: list[Unit] = [Unit(subject, "heading", level=1)] if subject else []
    headers = [f"{name}: {value}" for name in _HEADERS if (value := _header(message, name))]
    if headers:
        units.append(Unit("\n".join(headers)))

    body = message.get_body(preferencelist=("plain", "html"))
    if isinstance(body, EmailMessage):
        text = _body_text(body)
        if body.get_content_subtype() == "html":
            units += html_units(text)[0]
        else:
            units += paragraphs(text)

    names = [part.get_filename() for part in message.iter_attachments()]
    attached = [name for name in names if name]
    if attached:
        units.append(Unit("Attachments: " + ", ".join(attached)))
    return Extracted(units=units, title=subject)
