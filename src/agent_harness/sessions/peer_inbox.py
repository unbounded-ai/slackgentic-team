"""Deliver Slack replies into a live Claude session through its peer inbox.

Every interactive Claude Code session, including the ones the Claude desktop
app runs, registers ``~/.claude/sessions/<pid>.json`` naming the Unix socket it
listens on for messages from other local sessions. A user frame written to that
socket becomes the session's next prompt. None of this is a public interface,
so every step is best effort and the bridge falls back to a background resume
when delivery cannot be confirmed.
"""

from __future__ import annotations

import json
import os
import socket
import stat
from dataclasses import dataclass
from pathlib import Path

CLAUDE_SESSIONS_DIR = Path(".claude") / "sessions"
PEER_MESSAGE_TAG = "cross-session-message"
PEER_SENDER_ADDRESS = "uds:slackgentic"
PEER_SENDER_NAME = "Slackgentic"


@dataclass(frozen=True)
class ClaudePeerInbox:
    pid: int
    session_id: str
    socket_path: Path


def find_claude_peer_inbox(session_id: str, home: Path | None = None) -> ClaudePeerInbox | None:
    """The inbox of the running process that owns ``session_id``, if any.

    The desktop app keeps a session's process only while a turn runs and for a
    while after, so a missing inbox is the normal idle state, not an error.
    """
    root = (home or Path.home()) / CLAUDE_SESSIONS_DIR
    try:
        paths = sorted(root.glob("*.json"))
    except OSError:
        return None
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict) or record.get("sessionId") != session_id:
            continue
        pid = record.get("pid")
        socket_path = record.get("messagingSocketPath")
        if not isinstance(pid, int) or isinstance(pid, bool):
            continue
        if not isinstance(socket_path, str) or not socket_path:
            continue
        if not _process_alive(pid):
            continue
        candidate = Path(socket_path)
        try:
            if not stat.S_ISSOCK(candidate.stat().st_mode):
                continue
        except OSError:
            continue
        return ClaudePeerInbox(pid=pid, session_id=session_id, socket_path=candidate)
    return None


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def peer_message_line(
    text: str,
    *,
    sender_name: str = PEER_SENDER_NAME,
    sender_address: str = PEER_SENDER_ADDRESS,
) -> str:
    """One newline-terminated frame that the session queues as a prompt.

    ``now`` priority surfaces the message inside a running turn instead of
    waiting for the turn to end, which the app may not outlive.
    """
    body = (
        f'<{PEER_MESSAGE_TAG} from="{sender_address}" from-name="{sender_name}">\n'
        f"{text}\n"
        f"</{PEER_MESSAGE_TAG}>"
    )
    frame = {
        "type": "user",
        "message": {"role": "user", "content": body},
        "priority": "now",
        "from": sender_address,
    }
    return json.dumps(frame, separators=(",", ":")) + "\n"


def send_peer_message(socket_path: Path, line: str, *, timeout: float = 3.0) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(socket_path))
        sock.sendall(line.encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)


def transcript_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def transcript_shows_prompt(path: Path, offset: int, prompt: str) -> bool:
    """Whether a record carrying ``prompt`` was appended after ``offset``.

    A message that starts a turn is written as a ``user`` record; one that
    arrives mid-turn shows up first as the queue operation that parked it.
    """
    needle = prompt.strip()
    if not needle:
        return False
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except OSError:
        return False
    for raw in data.splitlines():
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        if record.get("type") == "queue-operation":
            content = record.get("content")
            if isinstance(content, str) and needle in content:
                return True
            continue
        if record.get("type") != "user":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if needle in content:
                return True
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and needle in text:
                return True
    return False
