from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path

from agent_harness.models import Provider, TokenUsage, UsageSnapshot
from agent_harness.providers.claude import ClaudeProvider
from agent_harness.providers.codex import CodexProvider


def day_string(value: str | None) -> str:
    if value in (None, "today"):
        return datetime.now(UTC).date().isoformat()
    date.fromisoformat(value)
    return value


def collect_daily_usage(day: str, home: Path | None = None) -> list[UsageSnapshot]:
    codex = CodexProvider(home=home)
    claude = ClaudeProvider(home=home)
    codex_paths = [session.transcript_path for session in codex.discover()]
    claude_paths = [session.transcript_path for session in claude.discover()]
    return codex.usage_for_day(codex_paths, day) + claude.usage_for_day(claude_paths, day)


def aggregate_usage(snapshots: Iterable[UsageSnapshot]) -> dict[Provider, TokenUsage]:
    totals: dict[Provider, TokenUsage] = {}
    for snapshot in snapshots:
        totals[snapshot.provider] = totals.get(snapshot.provider, TokenUsage()).plus(snapshot.usage)
    return totals


def format_daily_usage(day: str, snapshots: list[UsageSnapshot]) -> str:
    totals = aggregate_usage(snapshots)
    lines = [f"*Agent usage for {day}*", ""]
    for provider in (Provider.CODEX, Provider.CLAUDE):
        usage = totals.get(provider, TokenUsage())
        provider_snapshots = [item for item in snapshots if item.provider == provider]
        latest = max((item.as_of for item in provider_snapshots), default=None)
        remaining = _latest_remaining(provider_snapshots)
        lines.append(f"*{provider.value.title()}*")
        lines.append(f"- total tokens: {usage.total_tokens:,}")
        lines.append(f"- input tokens: {usage.input_tokens:,}")
        lines.append(f"- cached input tokens: {usage.cached_input_tokens:,}")
        lines.append(f"- cache creation tokens: {usage.cache_creation_input_tokens:,}")
        lines.append(f"- output tokens: {usage.output_tokens:,}")
        if usage.reasoning_output_tokens:
            lines.append(f"- reasoning output tokens: {usage.reasoning_output_tokens:,}")
        if remaining:
            lines.append(f"- remaining: {remaining}")
        if latest:
            lines.append(f"- updated: {latest.isoformat()}")
        lines.append("")
    return "\n".join(lines).strip()


def _latest_remaining(snapshots: list[UsageSnapshot]) -> str | None:
    if not snapshots:
        return None
    latest = max(snapshots, key=lambda item: item.as_of)
    return latest.remaining_description
