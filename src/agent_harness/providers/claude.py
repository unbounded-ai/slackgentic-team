from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_harness.jsonl import iter_jsonl, last_jsonl_records
from agent_harness.models import (
    AgentEvent,
    AgentSession,
    ControlMode,
    Provider,
    SessionStatus,
    TokenUsage,
    UsageSnapshot,
    parse_timestamp,
)


class ClaudeProvider:
    provider = Provider.CLAUDE

    def __init__(self, home: Path | None = None, active_within_seconds: int = 900):
        self.home = home or Path.home()
        self.active_within_seconds = active_within_seconds

    @property
    def projects_root(self) -> Path:
        return self.home / ".claude" / "projects"

    def discover(self) -> list[AgentSession]:
        if not self.projects_root.exists():
            return []
        sessions: list[AgentSession] = []
        for path in sorted(self.projects_root.rglob("*.jsonl")):
            session = self._session_from_path(path)
            if session:
                sessions.append(session)
        return sorted(
            sessions,
            key=lambda item: item.last_seen_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )

    def _session_from_path(self, path: Path) -> AgentSession | None:
        session_id = path.stem
        stat = path.stat()
        last_seen = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        age = (datetime.now(UTC) - last_seen).total_seconds()
        status = SessionStatus.ACTIVE if age <= self.active_within_seconds else SessionStatus.IDLE
        metadata: dict[str, Any] = {}
        started = _first_timestamp(path)
        cwd: Path | None = None
        model: str | None = None
        git_branch: str | None = None
        permission_mode: str | None = None
        for _, record in last_jsonl_records(path, limit=50):
            session_id = str(record.get("sessionId") or session_id)
            if isinstance(record.get("cwd"), str):
                cwd = Path(record["cwd"])
            if isinstance(record.get("gitBranch"), str):
                git_branch = record["gitBranch"]
            if isinstance(record.get("permissionMode"), str):
                permission_mode = record["permissionMode"]
            message = record.get("message")
            if isinstance(message, dict) and isinstance(message.get("model"), str):
                model = message["model"]
            if record.get("type") == "permission-mode" and isinstance(
                record.get("permissionMode"), str
            ):
                permission_mode = record["permissionMode"]
        return AgentSession(
            provider=self.provider,
            session_id=session_id,
            transcript_path=path,
            cwd=cwd,
            started_at=started,
            last_seen_at=last_seen,
            status=status,
            control_mode=ControlMode.OBSERVED,
            model=model,
            git_branch=git_branch,
            permission_mode=permission_mode,
            metadata=metadata,
        )

    def iter_events(self, transcript_path: Path) -> Iterator[AgentEvent]:
        session_id = transcript_path.stem
        for line_number, record in iter_jsonl(transcript_path):
            session_id = str(record.get("sessionId") or session_id)
            timestamp = parse_timestamp(record.get("timestamp"))
            record_type = str(record.get("type", "unknown"))
            text = _record_text(record)
            yield AgentEvent(
                provider=self.provider,
                session_id=session_id,
                timestamp=timestamp,
                event_type=record_type,
                text=text,
                source_path=transcript_path,
                line_number=line_number,
                metadata=record,
            )

    def usage_for_day(self, transcript_paths: Iterable[Path], day: str) -> list[UsageSnapshot]:
        totals: dict[str, TokenUsage] = {}
        latest: dict[str, datetime] = {}
        seen_message_ids: set[tuple[str, str]] = set()
        for path in transcript_paths:
            session_id = path.stem
            for _, record in iter_jsonl(path):
                timestamp = parse_timestamp(record.get("timestamp"))
                if not timestamp or timestamp.date().isoformat() != day:
                    continue
                session_id = str(record.get("sessionId") or session_id)
                usage = claude_usage_from_record(record)
                if not usage:
                    continue
                message_id = _usage_dedupe_id(record)
                dedupe_key = (session_id, message_id)
                if dedupe_key in seen_message_ids:
                    continue
                seen_message_ids.add(dedupe_key)
                totals[session_id] = totals.get(session_id, TokenUsage()).plus(usage)
                latest[session_id] = max(latest.get(session_id, timestamp), timestamp)
        return [
            UsageSnapshot(
                provider=self.provider,
                session_id=session_id,
                as_of=latest[session_id],
                usage=usage,
                remaining_description="remaining quota unavailable from local Claude transcripts",
            )
            for session_id, usage in sorted(totals.items())
        ]


def claude_usage_from_record(record: dict[str, Any]) -> TokenUsage | None:
    message = record.get("message")
    if not isinstance(message, dict):
        return None
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    return TokenUsage(
        input_tokens=input_tokens,
        cached_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
        output_tokens=output_tokens,
        total_tokens=input_tokens + cache_creation + cache_read + output_tokens,
    )


def _first_timestamp(path: Path) -> datetime | None:
    for _, record in iter_jsonl(path):
        timestamp = parse_timestamp(record.get("timestamp"))
        if timestamp is not None:
            return timestamp
    return None


def _usage_dedupe_id(record: dict[str, Any]) -> str:
    message = record.get("message")
    if isinstance(message, dict) and message.get("id"):
        return str(message["id"])
    if record.get("requestId"):
        return str(record["requestId"])
    if record.get("uuid"):
        return str(record["uuid"])
    return repr(record)


def _record_text(record: dict[str, Any]) -> str | None:
    message = record.get("message")
    if not isinstance(message, dict):
        if isinstance(record.get("type"), str):
            return record["type"]
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            text_parts.append(item["text"])
        elif item.get("type") == "tool_use":
            text_parts.append(f"tool call: {item.get('name', 'unknown')}")
    return "\n".join(text_parts) if text_parts else None
