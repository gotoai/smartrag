"""One writer per kb: the `smartrag.lock` file.

A sync, and every change to a kb's config or directory, holds the lock. Readers (search,
status, the MCP server) never take it.

The lock records its holder's PID, so a lock left behind by a process that crashed or was
killed is recognised as stale and taken over, rather than leaving the kb unusable until
someone deletes a file by hand. Kari's Agent Desks met exactly that failure: "without it
an orphaned desk keeps its lock and the next launch cannot start it".
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from smartrag.errors import KbLocked

LOCK_FILE = "smartrag.lock"

#: A lock file that is empty or unreadable for this long is a holder that died between
#: creating the file and writing it, not one still starting up.
_UNREADABLE_GRACE_SECONDS = 60


class LockHolder(BaseModel):
    pid: int | None = None
    host: str | None = None
    command: str | None = None
    started_at: datetime | None = None
    stale: bool = False


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        # os.kill(pid, 0) on Windows sends CTRL_C_EVENT (which is 0), so ask the kernel.
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_lock(kb_dir: Path) -> LockHolder | None:
    """Who holds the kb's lock, or None. A stale holder is reported with `stale=True`."""
    path = kb_dir / LOCK_FILE
    try:
        raw = path.read_text(encoding="utf-8")
        age = time.time() - path.stat().st_mtime
    except FileNotFoundError:
        return None
    except OSError:
        return LockHolder()
    try:
        holder = LockHolder.model_validate(json.loads(raw))
    except ValueError:
        return LockHolder(stale=age > _UNREADABLE_GRACE_SECONDS)
    if holder.pid is None:
        holder.stale = age > _UNREADABLE_GRACE_SECONDS
    elif holder.host in (None, socket.gethostname()):
        holder.stale = not _pid_alive(holder.pid)
    return holder


def _locked_error(kb_dir: Path, holder: LockHolder | None) -> KbLocked:
    details = holder.model_dump(mode="json", exclude_none=True) if holder else {}
    who = f"pid {holder.pid}" if holder and holder.pid else "another process"
    what = f" ({holder.command})" if holder and holder.command else ""
    return KbLocked(f"kb {kb_dir.name!r} is busy: {who}{what} holds its lock", details)


@contextmanager
def kb_lock(kb_dir: Path, command: str) -> Iterator[None]:
    path = kb_dir / LOCK_FILE
    payload = json.dumps({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            holder = read_lock(kb_dir)
            if holder is not None and holder.stale:
                path.unlink(missing_ok=True)
                continue
            raise _locked_error(kb_dir, holder) from None
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        break
    else:
        raise _locked_error(kb_dir, read_lock(kb_dir))
    try:
        yield
    finally:
        holder = read_lock(kb_dir)
        if holder is not None and holder.pid == os.getpid():
            path.unlink(missing_ok=True)
