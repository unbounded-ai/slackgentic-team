from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
from agent_harness.providers.path_index import TranscriptPathIndex
from agent_harness.storage.jsonl import iter_jsonl, last_jsonl_line_number, tail_jsonl_records

CLAUDE_LOCAL_EXIT_MARKERS = (
    "<command-name>/exit</command-name>",
    "<local-command-stdout>Bye!</local-command-stdout>",
)


class ClaudeProvider:
    provider = Provider.CLAUDE

    def __init__(
        self,
        home: Path | None = None,
        active_within_seconds: int = 900,
        *,
        full_discovery_interval_seconds: float = 300.0,
        hot_path_retention_seconds: float | None = None,
    ):
        self.home = home or Path.home()
        self.active_within_seconds = active_within_seconds
        self._session_cache: dict[Path, tuple[tuple[int, int], AgentSession | None]] = {}
        self._project_dir_signatures: dict[Path, tuple[int, int]] = {}
        self._hot_path_retention_seconds = (
            hot_path_retention_seconds
            if hot_path_retention_seconds is not None
            else active_within_seconds + full_discovery_interval_seconds + 60.0
        )
        self._path_index = TranscriptPathIndex(
            lambda: self.projects_root,
            full_scan_interval_seconds=full_discovery_interval_seconds,
        )

    @property
    def projects_root(self) -> Path:
        return self.home / ".claude" / "projects"

    def discover(self) -> list[AgentSession]:
        if not self.projects_root.exists():
            return []
        sessions_by_id: dict[str, AgentSession] = {}
        now = datetime.now(UTC)
        scan_roots = () if self._path_index.full_scan_due() else self._changed_project_roots()
        discovery = self._path_index.discover(
            hot_paths=self._hot_session_paths(now),
            scan_roots=scan_roots,
        )
        if discovery.full_scan:
            self._remember_project_roots()
        for path in discovery.paths:
            session = self._cached_session_from_path(path, now)
            if not session:
                continue
            existing = sessions_by_id.get(session.session_id)
            if existing is None or _prefer_session_discovery(session, existing):
                sessions_by_id[session.session_id] = session
        if discovery.full_scan:
            self._drop_deleted_cache_entries(set(discovery.paths))
        return sorted(
            sessions_by_id.values(),
            key=lambda item: item.last_seen_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )

    def _hot_session_paths(self, now: datetime) -> list[Path]:
        paths: list[Path] = []
        for path, (_, session) in self._session_cache.items():
            if session is None or session.last_seen_at is None:
                continue
            age = (now - session.last_seen_at).total_seconds()
            if age <= self._hot_path_retention_seconds:
                paths.append(path)
        return paths

    def _changed_project_roots(self) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()
        for path in self._iter_project_roots():
            seen.add(path)
            try:
                signature = _stat_signature(path.stat())
            except OSError:
                continue
            if self._project_dir_signatures.get(path) != signature:
                roots.append(path)
            self._project_dir_signatures[path] = signature
        for path in list(self._project_dir_signatures):
            if path not in seen:
                self._project_dir_signatures.pop(path, None)
        return roots

    def _remember_project_roots(self) -> None:
        self._project_dir_signatures = {}
        for path in self._iter_project_roots():
            try:
                self._project_dir_signatures[path] = _stat_signature(path.stat())
            except OSError:
                continue

    def _iter_project_roots(self) -> Iterator[Path]:
        try:
            children = self.projects_root.iterdir()
        except OSError:
            return
        for child in children:
            try:
                if child.is_dir():
                    yield child
            except OSError:
                continue

    def _cached_session_from_path(self, path: Path, now: datetime) -> AgentSession | None:
        stat = path.stat()
        signature = _stat_signature(stat)
        cached = self._session_cache.get(path)
        if cached and cached[0] == signature:
            return _refresh_observed_status(cached[1], self.active_within_seconds, now)
        session = self._session_from_path(path, stat=stat, now=now)
        self._session_cache[path] = (signature, session)
        return session

    def _drop_deleted_cache_entries(self, seen_paths: set[Path]) -> None:
        for path in list(self._session_cache):
            if path not in seen_paths:
                self._session_cache.pop(path, None)

    def _session_from_path(
        self,
        path: Path,
        *,
        stat: os.stat_result | None = None,
        now: datetime | None = None,
    ) -> AgentSession | None:
        session_id = path.stem
        stat = stat or path.stat()
        now = now or datetime.now(UTC)
        last_seen = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        age = (now - last_seen).total_seconds()
        status = SessionStatus.ACTIVE if age <= self.active_within_seconds else SessionStatus.IDLE
        tail_records = tail_jsonl_records(path, limit=80)
        if _session_ended_by_exit(tail_records):
            status = SessionStatus.DONE
        metadata: dict[str, Any] = {}
        started = _first_timestamp(path)
        cwd: Path | None = None
        model: str | None = None
        git_branch: str | None = None
        permission_mode: str | None = None
        entrypoint: str | None = None
        for record in tail_records[-50:]:
            session_id = str(record.get("sessionId") or session_id)
            if isinstance(record.get("cwd"), str):
                cwd = Path(record["cwd"])
            if isinstance(record.get("gitBranch"), str):
                git_branch = record["gitBranch"]
            if isinstance(record.get("permissionMode"), str):
                permission_mode = record["permissionMode"]
            if isinstance(record.get("entrypoint"), str):
                entrypoint = record["entrypoint"]
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
            metadata={**metadata, **({"entrypoint": entrypoint} if entrypoint else {})},
        )

    def iter_events(self, transcript_path: Path) -> Iterator[AgentEvent]:
        yield from self.iter_events_after(transcript_path, 0)

    def iter_events_after(
        self,
        transcript_path: Path,
        line_number: int,
    ) -> Iterator[AgentEvent]:
        session_id = transcript_path.stem
        for event_line_number, record in iter_jsonl(transcript_path, after_line=line_number):
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
                line_number=event_line_number,
                metadata=record,
            )

    def last_event_line_number(self, transcript_path: Path) -> int:
        return last_jsonl_line_number(transcript_path)

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
    usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        usage = record.get("usage")
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


def _first_timestamp(path: Path, *, max_records: int = 200) -> datetime | None:
    for index, (_, record) in enumerate(iter_jsonl(path), start=1):
        timestamp = parse_timestamp(record.get("timestamp"))
        if timestamp is not None:
            return timestamp
        if index >= max_records:
            return None
    return None


def _prefer_session_discovery(candidate: AgentSession, existing: AgentSession) -> bool:
    candidate_primary = candidate.transcript_path.stem == candidate.session_id
    existing_primary = existing.transcript_path.stem == existing.session_id
    if candidate_primary != existing_primary:
        return candidate_primary
    oldest = datetime.min.replace(tzinfo=UTC)
    return (candidate.last_seen_at or oldest) > (existing.last_seen_at or oldest)


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
        content = record.get("content")
        if isinstance(content, str):
            return content
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


def _session_ended_by_exit(records: Iterable[dict[str, Any]]) -> bool:
    exited = False
    for record in records:
        text = _record_text(record) or ""
        if any(marker in text for marker in CLAUDE_LOCAL_EXIT_MARKERS) or text.strip() == "/exit":
            exited = True
            continue
        if exited and _is_normal_conversation_record_after_exit(record, text):
            exited = False
    return exited


def _is_normal_conversation_record_after_exit(record: dict[str, Any], text: str) -> bool:
    record_type = record.get("type")
    if record_type == "assistant":
        return not is_synthetic_claude_assistant_record(record)
    if record_type != "user":
        return False
    if record.get("isMeta") is True:
        return False
    return not _is_local_command_text(text)


def is_synthetic_claude_assistant_record(record: dict[str, Any]) -> bool:
    # Claude's CLI inserts these on a resume that has nothing new to say
    # (e.g. when a tool result completes a prior turn). The text would
    # otherwise leak to Slack as if the agent spoke up unprompted.
    message = record.get("message")
    if not isinstance(message, dict):
        return False
    if message.get("model") == "<synthetic>":
        return True
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict)
        and item.get("type") == "text"
        and str(item.get("text") or "").strip() == "No response requested."
        for item in content
    )


def _is_local_command_text(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "<local-command-caveat>",
            "<command-name>",
            "<command-message>",
            "<command-args>",
            "<local-command-stdout>",
            "<local-command-stderr>",
        )
    )


def _stat_signature(stat: os.stat_result) -> tuple[int, int]:
    return (stat.st_mtime_ns, stat.st_size)


def _refresh_observed_status(
    session: AgentSession | None,
    active_within_seconds: int,
    now: datetime,
) -> AgentSession | None:
    if session is None or session.status == SessionStatus.DONE or session.last_seen_at is None:
        return session
    age = (now - session.last_seen_at).total_seconds()
    status = SessionStatus.ACTIVE if age <= active_within_seconds else SessionStatus.IDLE
    if status == session.status:
        return session
    return replace(session, status=status)
