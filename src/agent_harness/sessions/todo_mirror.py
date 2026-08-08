from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_harness.models import AgentSession, Provider, SlackThreadRef
from agent_harness.slack import SLACK_MAX_UPDATE_TEXT_CHARS, SLACK_UPDATE_TEXT_SAFETY_MARGIN
from agent_harness.slack.client import SlackGateway
from agent_harness.storage.jsonl import iter_jsonl
from agent_harness.storage.store import Store

LOGGER = logging.getLogger(__name__)

TODO_MESSAGE_TS_PREFIX = "session_todo_ts."

_STATUS_GLYPHS = {
    "completed": "✅",
    "in_progress": "◼",
    "cancelled": "✖",
    "pending": "◻",
}

# Slack rewrites these glyphs to their :shortcode: form before enforcing the
# chat.update text limit, so each one costs its shortcode length rather than a
# single character. A long task list overruns the limit well before the raw
# rendered text looks anywhere near it.
_GLYPH_SHORTCODES = {
    "✅": ":white_check_mark:",
    "◼": ":black_medium_square:",
    "✖": ":heavy_multiplication_x:",
    "◻": ":white_medium_square:",
}

_DEFAULT_STATUS_GLYPH = _STATUS_GLYPHS["pending"]

_HEADER = "*Tasks*"


@dataclass(frozen=True)
class TodoItem:
    title: str
    status: str
    detail: str | None = None


@dataclass(frozen=True)
class TodoSnapshot:
    items: tuple[TodoItem, ...]


class TodoMirror:
    """Posts and edits a single per-session Slack message that mirrors the
    agent's task tracker.

    Sources:
      * Claude: files in ``~/.claude/tasks/<sessionId>/<taskId>.json``
      * Codex: ``update_plan`` ``function_call`` payloads in the session JSONL
    """

    def __init__(
        self,
        store: Store,
        gateway: SlackGateway,
        *,
        home: Path | None = None,
    ) -> None:
        self.store = store
        self.gateway = gateway
        self.home = home or Path.home()
        self._last_rendered: dict[tuple[Provider, str], str] = {}
        self._codex_cursors: dict[tuple[Provider, str], int] = {}
        self._codex_last_plan: dict[tuple[Provider, str], TodoSnapshot] = {}

    def sync_session(self, session: AgentSession, thread: SlackThreadRef) -> None:
        snapshot = self._snapshot(session)
        if snapshot is None or not snapshot.items:
            return
        text = render_snapshot(snapshot)
        key = (session.provider, session.session_id)
        if self._last_rendered.get(key) == text:
            return
        setting_key = todo_message_ts_key(session)
        existing_ts = self.store.get_setting(setting_key)
        if existing_ts:
            try:
                self.gateway.update_message(thread.channel_id, existing_ts, text)
            except Exception as exc:
                LOGGER.exception(
                    "failed to update todo mirror for %s session %s",
                    session.provider.value,
                    session.session_id,
                )
                if _slack_rejected_payload(exc):
                    # Slack will never accept this render, so remember it instead
                    # of resending the identical payload on every sync tick.
                    self._last_rendered[key] = text
                return
        else:
            try:
                posted = self.gateway.post_thread_reply(thread, text)
            except Exception:
                LOGGER.exception(
                    "failed to post todo mirror for %s session %s",
                    session.provider.value,
                    session.session_id,
                )
                return
            self.store.set_setting(setting_key, posted.ts)
        self._last_rendered[key] = text

    def _snapshot(self, session: AgentSession) -> TodoSnapshot | None:
        if session.provider == Provider.CLAUDE:
            return claude_todo_snapshot(self.home, session.session_id)
        if session.provider == Provider.CODEX:
            key = (session.provider, session.session_id)
            cursor = self._codex_cursors.get(key, 0)
            new_plan, new_cursor = codex_latest_plan_after(session.transcript_path, cursor)
            if new_cursor > cursor:
                self._codex_cursors[key] = new_cursor
            if new_plan is not None:
                self._codex_last_plan[key] = new_plan
            return self._codex_last_plan.get(key)
        return None


def todo_message_ts_key(session: AgentSession) -> str:
    return f"{TODO_MESSAGE_TS_PREFIX}{session.provider.value}.{session.session_id}"


def _slack_rejected_payload(exc: Exception) -> bool:
    """True when Slack rejected the message itself rather than failing transiently."""

    response = getattr(exc, "response", None)
    getter = getattr(response, "get", None)
    if not callable(getter):
        return False
    return getter("error") in {"msg_too_long", "invalid_blocks"}


def render_snapshot(
    snapshot: TodoSnapshot,
    *,
    limit: int = SLACK_MAX_UPDATE_TEXT_CHARS - SLACK_UPDATE_TEXT_SAFETY_MARGIN,
) -> str:
    total = len(snapshot.items)
    lines = [_HEADER]
    shown = 0
    for item in snapshot.items:
        # Reserve room for the widest "more" note so breaking out later never
        # pushes the finished message back over the limit.
        trial = [*lines, _render_item(item)]
        if shown + 1 < total:
            trial.append(_hidden_note(total))
        if slack_text_length("\n".join(trial)) > limit:
            break
        lines.append(_render_item(item))
        shown += 1
    hidden = total - shown
    if hidden:
        lines.append(_hidden_note(hidden))
    return "\n".join(lines)


def slack_text_length(text: str) -> int:
    """Length Slack charges for ``text``, approximating its normalized form."""

    length = len(text)
    for glyph, shortcode in _GLYPH_SHORTCODES.items():
        length += text.count(glyph) * (len(shortcode) - len(glyph))
    for char, entity in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")):
        length += text.count(char) * (len(entity) - 1)
    return length


def _render_item(item: TodoItem) -> str:
    glyph = _STATUS_GLYPHS.get(item.status, _DEFAULT_STATUS_GLYPH)
    title = item.detail if item.status == "in_progress" and item.detail else item.title
    return f"{glyph} {title}"


def _hidden_note(hidden: int) -> str:
    noun = "task" if hidden == 1 else "tasks"
    return f"_+{hidden} more {noun} not shown_"


def claude_todo_snapshot(home: Path, session_id: str) -> TodoSnapshot | None:
    tasks_dir = home / ".claude" / "tasks" / session_id
    if not tasks_dir.is_dir():
        return None
    items_by_id: list[tuple[int, str, TodoItem]] = []
    for entry in tasks_dir.iterdir():
        if not entry.is_file() or entry.suffix != ".json":
            continue
        if entry.name.startswith("."):
            continue
        try:
            data = json.loads(entry.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        title = _first_non_empty_string(data.get("subject"), data.get("activeForm"))
        if not title:
            continue
        status = _normalize_status(data.get("status"))
        detail = data.get("activeForm")
        detail_text = detail.strip() if isinstance(detail, str) and detail.strip() else None
        sort_index, sort_id = _claude_task_sort_key(entry.stem, data.get("id"))
        items_by_id.append(
            (
                sort_index,
                sort_id,
                TodoItem(title=title.strip(), status=status, detail=detail_text),
            )
        )
    items_by_id.sort(key=lambda row: (row[0], row[1]))
    items = tuple(row[2] for row in items_by_id)
    return TodoSnapshot(items=items)


def codex_latest_plan_after(
    transcript_path: Path,
    cursor: int,
) -> tuple[TodoSnapshot | None, int]:
    if not transcript_path.exists():
        return None, cursor
    new_cursor = cursor
    last_plan: list[dict[str, Any]] | None = None
    try:
        for line_number, record in iter_jsonl(transcript_path, after_line=cursor):
            new_cursor = line_number
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if payload.get("type") != "function_call":
                continue
            if payload.get("name") != "update_plan":
                continue
            plan = _codex_plan_from_arguments(payload.get("arguments"))
            if plan is not None:
                last_plan = plan
    except OSError:
        return None, cursor
    if last_plan is None:
        return None, new_cursor
    items: list[TodoItem] = []
    for raw in last_plan:
        if not isinstance(raw, dict):
            continue
        step = raw.get("step")
        if not isinstance(step, str) or not step.strip():
            continue
        items.append(
            TodoItem(
                title=step.strip(),
                status=_normalize_status(raw.get("status")),
            )
        )
    return TodoSnapshot(items=tuple(items)), new_cursor


def _codex_plan_from_arguments(arguments: Any) -> list[dict[str, Any]] | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    plan = arguments.get("plan")
    if not isinstance(plan, list):
        return None
    return plan


def _normalize_status(value: Any) -> str:
    if not isinstance(value, str):
        return "pending"
    normalized = value.strip().lower()
    if normalized in _STATUS_GLYPHS:
        return normalized
    return "pending"


def _first_non_empty_string(*candidates: Any) -> str | None:
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _claude_task_sort_key(stem: str, raw_id: Any) -> tuple[int, str]:
    candidates: list[str] = []
    if isinstance(raw_id, (str, int)):
        candidates.append(str(raw_id))
    candidates.append(stem)
    for candidate in candidates:
        try:
            return (int(candidate), candidate)
        except (TypeError, ValueError):
            continue
    return (10**9, str(raw_id) if raw_id is not None else stem)
