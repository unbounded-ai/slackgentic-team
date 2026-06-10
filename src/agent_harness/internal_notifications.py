from __future__ import annotations

import html
import re

_AUTHOR_PREFIX_RE = re.compile(r"^[^\n:]{1,100}:\s*")
_TASK_NOTIFICATION_REQUIRED_TAGS = (
    "task-id",
    "tool-use-id",
    "output-file",
    "status",
    "summary",
)


def is_internal_task_notification_text(text: str) -> bool:
    candidate = _without_optional_author_prefix(html.unescape(text).strip())
    if not (
        candidate.startswith("<task-notification>") and candidate.endswith("</task-notification>")
    ):
        return False
    return all(
        f"<{tag}>" in candidate and f"</{tag}>" in candidate
        for tag in _TASK_NOTIFICATION_REQUIRED_TAGS
    )


def filter_internal_task_notifications(text: str) -> str:
    lines: list[str] = []
    pending: list[str] = []
    for line in text.splitlines():
        if pending:
            pending.append(line)
            if _task_notification_end_line(line):
                if not is_internal_task_notification_text("\n".join(pending)):
                    lines.extend(pending)
                pending = []
            continue
        if _task_notification_start_line(line):
            pending = [line]
            if _task_notification_end_line(line):
                if not is_internal_task_notification_text(line):
                    lines.append(line)
                pending = []
            continue
        lines.append(line)
    lines.extend(pending)
    return "\n".join(lines).strip()


def _without_optional_author_prefix(text: str) -> str:
    return _AUTHOR_PREFIX_RE.sub("", text, count=1).strip()


def _task_notification_start_line(line: str) -> bool:
    return _without_optional_author_prefix(html.unescape(line).strip()).startswith(
        "<task-notification>"
    )


def _task_notification_end_line(line: str) -> bool:
    return "</task-notification>" in html.unescape(line)
