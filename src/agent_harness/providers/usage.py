from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
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


def collect_weekly_usage(day: str, home: Path | None = None) -> list[UsageSnapshot]:
    anchor = date.fromisoformat(day)
    start = anchor - timedelta(days=anchor.weekday())
    days = [(start + timedelta(days=offset)).isoformat() for offset in range(7)]
    codex = CodexProvider(home=home)
    claude = ClaudeProvider(home=home)
    codex_paths = [session.transcript_path for session in codex.discover()]
    claude_paths = [session.transcript_path for session in claude.discover()]
    snapshots: list[UsageSnapshot] = []
    for usage_day in days:
        snapshots.extend(codex.usage_for_day(codex_paths, usage_day))
        snapshots.extend(claude.usage_for_day(claude_paths, usage_day))
    return snapshots


def aggregate_usage(snapshots: Iterable[UsageSnapshot]) -> dict[Provider, TokenUsage]:
    totals: dict[Provider, TokenUsage] = {}
    for snapshot in snapshots:
        totals[snapshot.provider] = totals.get(snapshot.provider, TokenUsage()).plus(snapshot.usage)
    return totals


def format_daily_usage(
    day: str,
    snapshots: list[UsageSnapshot],
    weekly_snapshots: list[UsageSnapshot] | None = None,
) -> str:
    weekly_snapshots = weekly_snapshots if weekly_snapshots is not None else snapshots
    totals = aggregate_usage(snapshots)
    weekly_totals = aggregate_usage(weekly_snapshots)
    all_week_tokens = sum(usage.total_tokens for usage in weekly_totals.values())
    week_start, week_end = _week_bounds(day)
    lines = [
        "*Agent status*",
        f"`{day}`  Week `{week_start}` to `{week_end}`",
        "",
    ]
    for provider in (Provider.CODEX, Provider.CLAUDE):
        usage = totals.get(provider, TokenUsage())
        weekly_usage = weekly_totals.get(provider, TokenUsage())
        provider_snapshots = [item for item in snapshots if item.provider == provider]
        latest = max((item.as_of for item in provider_snapshots), default=None)
        weekly_provider_percent = _percent(weekly_usage.total_tokens, all_week_tokens)
        today_of_week_percent = _percent(usage.total_tokens, weekly_usage.total_tokens)
        window_percent = _latest_primary_window_percent(provider_snapshots)
        lines.append(f"*{provider.value.title()}*")
        lines.extend(
            _metric_table(
                usage,
                weekly_usage,
                today_of_week_percent,
                weekly_provider_percent,
                window_percent,
            )
        )
        lines.append(
            f"`today` is share of {provider.value.title()} week; "
            "`week share` is share of all weekly agent tokens."
        )
        lines.append(
            f"Tokens: `{usage.total_tokens:,}` today / `{weekly_usage.total_tokens:,}` week"
        )
        session_lines = _session_usage_lines(provider_snapshots, usage.total_tokens)
        if session_lines:
            lines.append("Top sessions today:")
            lines.extend(_code_block(["session       usage     pct      tokens", *session_lines]))
        detail = _token_mix(usage)
        if detail:
            lines.append(f"Mix: {detail}")
        if usage.reasoning_output_tokens:
            lines.append(f"Reasoning output: `{usage.reasoning_output_tokens:,}`")
        if latest:
            lines.append(f"Updated: `{_format_timestamp(latest)}`")
        if not provider_snapshots:
            lines.append("No local transcript usage found today.")
        lines.append("")
    return "\n".join(lines).strip()


def _metric_table(
    usage: TokenUsage,
    weekly_usage: TokenUsage,
    today_of_week_percent: float,
    weekly_provider_percent: float,
    window_percent: float | None,
) -> list[str]:
    rows = ["metric        usage          pct      tokens"]
    if window_percent is not None:
        rows.append(_metric_row("quota window", window_percent, "window"))
    rows.append(
        _metric_row(
            "today",
            today_of_week_percent,
            _short_token_count(usage.total_tokens),
        )
    )
    rows.append(
        _metric_row(
            "week share",
            weekly_provider_percent,
            _short_token_count(weekly_usage.total_tokens),
        )
    )
    return _code_block(rows)


def _metric_row(label: str, percent: float, tokens: str) -> str:
    return f"{label:<12}  {_usage_bar(percent)}  {_format_percent(percent):>6}  {tokens:>8}"


def _code_block(lines: list[str]) -> list[str]:
    return ["```", *lines, "```"]


def _week_bounds(day: str) -> tuple[str, str]:
    anchor = date.fromisoformat(day)
    start = anchor - timedelta(days=anchor.weekday())
    end = start + timedelta(days=6)
    return start.isoformat(), end.isoformat()


def _latest_primary_window_percent(snapshots: list[UsageSnapshot]) -> float | None:
    candidates = [
        snapshot
        for snapshot in snapshots
        if snapshot.primary_limit and snapshot.primary_limit.used_percent is not None
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: item.as_of)
    assert latest.primary_limit is not None
    return latest.primary_limit.used_percent


def _session_usage_lines(snapshots: list[UsageSnapshot], total_tokens: int) -> list[str]:
    ranked = sorted(snapshots, key=lambda item: item.usage.total_tokens, reverse=True)
    lines: list[str] = []
    for snapshot in ranked[:3]:
        if not snapshot.session_id or snapshot.usage.total_tokens <= 0:
            continue
        percent = _percent(snapshot.usage.total_tokens, total_tokens)
        session = snapshot.session_id[:12]
        lines.append(
            f"{session:<12}  {_usage_bar(percent, width=8)}  "
            f"{_format_percent(percent):>6}  {_short_token_count(snapshot.usage.total_tokens):>8}"
        )
    return lines


def _token_mix(usage: TokenUsage) -> str:
    parts = [
        f"in `{usage.input_tokens:,}`",
        f"cache `{usage.cached_input_tokens + usage.cache_creation_input_tokens:,}`",
        f"out `{usage.output_tokens:,}`",
    ]
    return " / ".join(parts)


def _percent(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, (value / total) * 100.0))


def _format_percent(value: float) -> str:
    return f"{value:.1f}%"


def _short_token_count(value: int) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


def _usage_bar(percent: float, width: int = 12) -> str:
    clamped = max(0.0, min(100.0, percent))
    filled = round((clamped / 100.0) * width)
    return "█" * filled + "░" * (width - filled)
