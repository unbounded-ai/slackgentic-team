from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from agent_harness.models import AgentEvent, AgentSession, Provider, UsageSnapshot


class AgentProvider(Protocol):
    provider: Provider

    def discover(self) -> list[AgentSession]:
        raise NotImplementedError

    def iter_events(self, transcript_path: Path) -> Iterator[AgentEvent]:
        raise NotImplementedError

    def iter_events_after(
        self,
        transcript_path: Path,
        line_number: int,
    ) -> Iterator[AgentEvent]:
        raise NotImplementedError

    def last_event_line_number(self, transcript_path: Path) -> int:
        raise NotImplementedError

    def usage_for_day(self, transcript_paths: Iterable[Path], day: str) -> list[UsageSnapshot]:
        raise NotImplementedError
