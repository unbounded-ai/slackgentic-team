from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent_harness.models import (
    AgentEvent,
    AgentSession,
    ControlMode,
    Provider,
    RateLimitWindow,
    SessionStatus,
    TokenUsage,
    UsageSnapshot,
    parse_timestamp,
)
from agent_harness.storage.jsonl import first_jsonl_record, iter_jsonl


class CodexProvider:
    provider = Provider.CODEX

    def __init__(self, home: Path | None = None, active_within_seconds: int = 900):
        self.home = home or Path.home()
        self.active_within_seconds = active_within_seconds

    @property
    def sessions_root(self) -> Path:
        return self.home / ".codex" / "sessions"

    def discover(self) -> list[AgentSession]:
        if not self.sessions_root.exists():
            return []
        sessions: list[AgentSession] = []
        for path in sorted(self.sessions_root.rglob("*.jsonl")):
            session = self._session_from_path(path)
            if session:
                sessions.append(session)
        return sorted(
            sessions,
            key=lambda item: item.last_seen_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )

    def _session_from_path(self, path: Path) -> AgentSession | None:
        first = first_jsonl_record(path)
        record = first[1] if first else {}
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        session_id = payload.get("id") or _session_id_from_filename(path)
        if not session_id:
            return None
        stat = path.stat()
        last_seen = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        started = parse_timestamp(payload.get("timestamp")) or _timestamp_from_path(path)
        age = (datetime.now(UTC) - last_seen).total_seconds()
        status = SessionStatus.ACTIVE if age <= self.active_within_seconds else SessionStatus.IDLE
        cwd = Path(payload["cwd"]) if isinstance(payload.get("cwd"), str) else None
        return AgentSession(
            provider=self.provider,
            session_id=session_id,
            transcript_path=path,
            cwd=cwd,
            started_at=started,
            last_seen_at=last_seen,
            status=status,
            control_mode=ControlMode.OBSERVED,
            model=payload.get("model"),
            metadata={
                "originator": payload.get("originator"),
                "cli_version": payload.get("cli_version"),
                "source": payload.get("source"),
            },
        )

    def iter_events(self, transcript_path: Path) -> Iterator[AgentEvent]:
        session_id = _session_id_from_filename(transcript_path) or "unknown"
        for line_number, record in iter_jsonl(transcript_path):
            timestamp = parse_timestamp(record.get("timestamp"))
            record_type = str(record.get("type", "unknown"))
            if record_type == "session_meta":
                payload = record.get("payload", {})
                if isinstance(payload, dict):
                    session_id = str(payload.get("id") or session_id)
                yield AgentEvent(
                    provider=self.provider,
                    session_id=session_id,
                    timestamp=timestamp,
                    event_type="session_meta",
                    source_path=transcript_path,
                    line_number=line_number,
                    metadata=payload if isinstance(payload, dict) else {},
                )
                continue
            payload = record.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "token_count":
                snapshot = parse_token_count(record, session_id)
                yield AgentEvent(
                    provider=self.provider,
                    session_id=session_id,
                    timestamp=timestamp,
                    event_type="usage",
                    source_path=transcript_path,
                    line_number=line_number,
                    metadata={"usage_snapshot": snapshot},
                )
                continue
            text = _event_text(record)
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
        snapshots: list[UsageSnapshot] = []
        for path in transcript_paths:
            session_id = _session_id_from_filename(path) or None
            latest_for_session: UsageSnapshot | None = None
            for _, record in iter_jsonl(path):
                timestamp = parse_timestamp(record.get("timestamp"))
                if not timestamp or timestamp.date().isoformat() != day:
                    continue
                payload = record.get("payload")
                if isinstance(payload, dict) and payload.get("type") == "token_count":
                    latest_for_session = parse_token_count(record, session_id)
            if latest_for_session:
                snapshots.append(latest_for_session)
        return snapshots


def parse_token_count(record: dict[str, Any], session_id: str | None = None) -> UsageSnapshot:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
    usage = info.get("total_token_usage") if isinstance(info.get("total_token_usage"), dict) else {}
    rate_limits = payload.get("rate_limits") if isinstance(payload.get("rate_limits"), dict) else {}
    timestamp = parse_timestamp(record.get("timestamp")) or datetime.now(UTC)
    primary = _parse_rate_window(rate_limits.get("primary"))
    secondary = _parse_rate_window(rate_limits.get("secondary"))
    remaining = None
    if primary and primary.used_percent is not None:
        remaining = f"{max(0.0, 100.0 - primary.used_percent):.1f}% primary window remaining"
    return UsageSnapshot(
        provider=Provider.CODEX,
        session_id=session_id,
        as_of=timestamp,
        usage=TokenUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            reasoning_output_tokens=int(usage.get("reasoning_output_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        ),
        context_window=info.get("model_context_window"),
        primary_limit=primary,
        secondary_limit=secondary,
        plan_type=rate_limits.get("plan_type"),
        remaining_description=remaining,
    )


def _parse_rate_window(value: Any) -> RateLimitWindow | None:
    if not isinstance(value, dict):
        return None
    resets_at = value.get("resets_at")
    reset_dt = parse_timestamp(resets_at) if resets_at is not None else None
    return RateLimitWindow(
        used_percent=float(value["used_percent"])
        if value.get("used_percent") is not None
        else None,
        window_minutes=int(value["window_minutes"])
        if value.get("window_minutes") is not None
        else None,
        resets_at=reset_dt,
    )


def _session_id_from_filename(path: Path) -> str | None:
    stem = path.stem
    marker = "rollout-"
    if marker not in stem:
        return None
    parts = stem.split("-")
    if len(parts) < 8:
        return None
    return "-".join(parts[-5:])


def _timestamp_from_path(path: Path) -> datetime | None:
    # rollout-2026-04-26T12-59-42-<uuid>.jsonl
    prefix = "rollout-"
    if not path.name.startswith(prefix):
        return None
    text = path.name[len(prefix) : len(prefix) + 19]
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _event_text(record: dict[str, Any]) -> str | None:
    payload = record.get("payload")
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            return payload["text"]
        if payload.get("type") == "function_call":
            return f"tool call: {payload.get('name', 'unknown')}"
        if payload.get("type") == "function_call_output":
            return "tool result"
    return None
