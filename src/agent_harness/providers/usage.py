from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from agent_harness.models import Provider, TokenUsage, UsageSnapshot
from agent_harness.providers.claude import ClaudeProvider
from agent_harness.providers.codex import CodexProvider
from agent_harness.providers.quota import (
    ClaudeQuota,
    ClaudeSignIns,
    QuotaWindow,
    plan_label,
    window_label,
)
from agent_harness.timezones import zone_for


def day_string(value: str | None, timezone: str | None = None) -> str:
    """An ISO day; "today" is the current day in ``timezone`` (UTC when unset)."""
    if value in (None, "today"):
        return datetime.now(zone_for(timezone) or UTC).date().isoformat()
    date.fromisoformat(value)
    return value


def collect_daily_usage(
    day: str,
    home: Path | None = None,
    timezone: str | None = None,
) -> list[UsageSnapshot]:
    codex = CodexProvider(home=home)
    claude = ClaudeProvider(home=home)
    codex_paths = [session.transcript_path for session in codex.discover()]
    claude_paths = [session.transcript_path for session in claude.discover()]
    return codex.usage_for_day(codex_paths, day, timezone) + claude.usage_for_day(
        claude_paths, day, timezone
    )


def collect_weekly_usage(
    day: str,
    home: Path | None = None,
    timezone: str | None = None,
) -> list[UsageSnapshot]:
    anchor = date.fromisoformat(day)
    start = anchor - timedelta(days=anchor.weekday())
    days = [(start + timedelta(days=offset)).isoformat() for offset in range(7)]
    codex = CodexProvider(home=home)
    claude = ClaudeProvider(home=home)
    codex_paths = [session.transcript_path for session in codex.discover()]
    claude_paths = [session.transcript_path for session in claude.discover()]
    snapshots: list[UsageSnapshot] = []
    for usage_day in days:
        snapshots.extend(codex.usage_for_day(codex_paths, usage_day, timezone))
        snapshots.extend(claude.usage_for_day(claude_paths, usage_day, timezone))
    return snapshots


def aggregate_usage(snapshots: Iterable[UsageSnapshot]) -> dict[Provider, TokenUsage]:
    totals: dict[Provider, TokenUsage] = {}
    for snapshot in snapshots:
        totals[snapshot.provider] = totals.get(snapshot.provider, TokenUsage()).plus(snapshot.usage)
    return totals


CLAUDE_APP_SURFACE = "claude-desktop"
CODEX_APP_SURFACE = "Codex Desktop"
TOP_SESSIONS_PER_ACCOUNT = 3
# Two readings of one account's window report the same reset, give or take drift.
CODEX_RESET_MATCH_SECONDS = 300


@dataclass(frozen=True)
class SessionUsage:
    session_id: str
    label: str
    tokens: int
    percent: float


@dataclass(frozen=True)
class AccountStatus:
    """One signed-in account: its quota windows and the tokens its sessions used."""

    key: str
    provider: Provider
    title: str
    surfaces: str
    plan: str | None
    windows: tuple[QuotaWindow, ...]
    quota_note: str | None
    today_tokens: int
    week_tokens: int
    session_count: int
    top_sessions: tuple[SessionUsage, ...]
    quota_as_of: datetime | None = None


def build_status_report(
    day: str,
    snapshots: list[UsageSnapshot],
    weekly_snapshots: list[UsageSnapshot] | None = None,
    *,
    claude_quota: ClaudeQuota | None = None,
    claude_sign_ins: ClaudeSignIns | None = None,
    session_label: Callable[[UsageSnapshot], str | None] | None = None,
    now: datetime | None = None,
) -> list[AccountStatus]:
    """Usage grouped by sign-in rather than by provider.

    The Claude CLI and the Claude app can be signed in to different accounts,
    each with its own quota. Codex CLI and the Codex app share one sign-in.
    """
    now = now or datetime.now(UTC)
    weekly_snapshots = weekly_snapshots if weekly_snapshots is not None else snapshots
    sign_ins = claude_sign_ins or ClaudeSignIns()
    accounts: list[AccountStatus] = []

    def claude_account(snapshot: UsageSnapshot) -> str:
        if sign_ins.app_is_separate and snapshot.surface == CLAUDE_APP_SURFACE:
            return "claude.app"
        return "claude.cli"

    claude_today = [item for item in snapshots if item.provider == Provider.CLAUDE]
    claude_week = [item for item in weekly_snapshots if item.provider == Provider.CLAUDE]
    claude_keys = ["claude.cli"]
    if sign_ins.app_is_separate:
        claude_keys.append("claude.app")
    for key in claude_keys:
        today = [item for item in claude_today if claude_account(item) == key]
        week = [item for item in claude_week if claude_account(item) == key]
        if key == "claude.app" and not today and not week:
            continue
        is_cli = key == "claude.cli"
        surfaces = "app"
        if is_cli:
            surfaces = "CLI" if sign_ins.app_is_separate else "CLI + app"
        windows: tuple[QuotaWindow, ...] = ()
        note = None
        if is_cli and claude_quota is not None:
            windows = _live_windows(claude_quota.windows, now)
            if claude_quota.using_overage:
                note = "Using extra usage"
        elif is_cli:
            note = "Quota unavailable right now"
        else:
            note = "Separate sign-in · its quota shows only in the Claude app"
        accounts.append(
            _account_status(
                key,
                Provider.CLAUDE,
                "Claude",
                surfaces,
                sign_ins.cli_plan if is_cli else None,
                windows,
                note,
                today,
                week,
                session_label,
                quota_as_of=claude_quota.as_of if is_cli and claude_quota else None,
            )
        )

    codex_today = [item for item in snapshots if item.provider == Provider.CODEX]
    codex_week = [item for item in weekly_snapshots if item.provider == Provider.CODEX]

    def codex_surface(snapshot: UsageSnapshot) -> str:
        return "app" if snapshot.surface == CODEX_APP_SURFACE else "cli"

    by_surface = {
        surface: [item for item in codex_week if codex_surface(item) == surface]
        for surface in ("cli", "app")
    }
    if by_surface["app"] and (
        not by_surface["cli"]
        or _same_codex_sign_in(
            _latest_codex_limits(by_surface["cli"]),
            _latest_codex_limits(by_surface["app"]),
            now,
        )
    ):
        groups = [("codex", "CLI + app" if by_surface["cli"] else "app", ("cli", "app"))]
    elif by_surface["app"]:
        groups = [("codex.cli", "CLI", ("cli",)), ("codex.app", "app", ("app",))]
    else:
        groups = [("codex", "CLI", ("cli",))]
    for key, surfaces, members in groups:
        today = [item for item in codex_today if codex_surface(item) in members]
        week = [item for item in codex_week if codex_surface(item) in members]
        latest_limits = _latest_codex_limits(today or week)
        windows: tuple[QuotaWindow, ...] = ()
        if latest_limits is not None:
            windows = _live_windows(_codex_windows(latest_limits), now)
        accounts.append(
            _account_status(
                key,
                Provider.CODEX,
                "Codex",
                surfaces,
                plan_label(latest_limits.plan_type) if latest_limits else None,
                windows,
                None if windows else "No recent quota reading",
                today,
                week,
                session_label,
                quota_as_of=latest_limits.as_of if latest_limits else None,
            )
        )
    return accounts


def format_daily_usage(
    day: str,
    snapshots: list[UsageSnapshot],
    weekly_snapshots: list[UsageSnapshot] | None = None,
    *,
    claude_quota: ClaudeQuota | None = None,
    claude_sign_ins: ClaudeSignIns | None = None,
    session_label: Callable[[UsageSnapshot], str | None] | None = None,
    now: datetime | None = None,
) -> str:
    """Plain-text status, for the CLI and as the Slack notification fallback."""
    accounts = build_status_report(
        day,
        snapshots,
        weekly_snapshots,
        claude_quota=claude_quota,
        claude_sign_ins=claude_sign_ins,
        session_label=session_label,
        now=now,
    )
    lines = [f"Agent status · {day}"]
    for account in accounts:
        lines.append("")
        lines.append(account_heading(account))
        for window in account.windows:
            reset = f" · resets {_format_timestamp(window.resets_at)}" if window.resets_at else ""
            lines.append(
                f"  {window.label:<4} {usage_bar(window.used_percent)} "
                f"{window.used_percent:.0f}%{reset}"
            )
        if account.quota_note:
            lines.append(f"  {account.quota_note}")
        lines.append(f"  {account_tokens_line(account)}")
        for session in account.top_sessions:
            lines.append(
                f"    {session.label} · {short_token_count(session.tokens)} "
                f"({session.percent:.0f}%)"
            )
    return "\n".join(lines)


def account_heading(account: AccountStatus) -> str:
    heading = f"{account.title} {account.surfaces}"
    return f"{heading} · {account.plan}" if account.plan else heading


def account_tokens_line(account: AccountStatus) -> str:
    if not account.today_tokens and not account.week_tokens:
        return "No sessions this week"
    sessions = f"{account.session_count} session{'s' if account.session_count != 1 else ''}"
    return (
        f"Today {short_token_count(account.today_tokens)} tokens · {sessions} · "
        f"week {short_token_count(account.week_tokens)}"
    )


def usage_bar(percent: float, width: int = 10) -> str:
    clamped = max(0.0, min(100.0, percent))
    filled = int(clamped / 100.0 * width + 0.5)
    if clamped > 0 and filled == 0:
        filled = 1
    return "▰" * filled + "▱" * (width - filled)


def _account_status(
    key: str,
    provider: Provider,
    title: str,
    surfaces: str,
    plan: str | None,
    windows: tuple[QuotaWindow, ...],
    note: str | None,
    today: list[UsageSnapshot],
    week: list[UsageSnapshot],
    session_label: Callable[[UsageSnapshot], str | None] | None,
    *,
    quota_as_of: datetime | None,
) -> AccountStatus:
    today_tokens = sum(item.usage.total_tokens for item in today)
    per_session: dict[str, UsageSnapshot] = {}
    tokens: dict[str, int] = {}
    for item in today:
        if not item.session_id or item.usage.total_tokens <= 0:
            continue
        tokens[item.session_id] = tokens.get(item.session_id, 0) + item.usage.total_tokens
        per_session.setdefault(item.session_id, item)
    ranked = sorted(tokens.items(), key=lambda pair: pair[1], reverse=True)
    top = tuple(
        SessionUsage(
            session_id=session_id,
            label=(session_label(per_session[session_id]) if session_label else None)
            or f"session {session_id[:8]}",
            tokens=count,
            percent=_percent(count, today_tokens),
        )
        for session_id, count in ranked[:TOP_SESSIONS_PER_ACCOUNT]
        # A sliver of the day is noise, not a top session.
        if _percent(count, today_tokens) >= 1.0
    )
    return AccountStatus(
        key=key,
        provider=provider,
        title=title,
        surfaces=surfaces,
        plan=plan,
        windows=windows,
        quota_note=note,
        today_tokens=today_tokens,
        week_tokens=sum(item.usage.total_tokens for item in week),
        session_count=len(tokens),
        top_sessions=top,
        quota_as_of=quota_as_of,
    )


def _latest_codex_limits(snapshots: list[UsageSnapshot]) -> UsageSnapshot | None:
    candidates = [
        item
        for item in snapshots
        if item.primary_limit is not None and item.primary_limit.used_percent is not None
    ]
    return max(candidates, key=lambda item: item.as_of, default=None)


def _codex_windows(snapshot: UsageSnapshot) -> tuple[QuotaWindow, ...]:
    return tuple(
        QuotaWindow(
            label=window_label(limit.window_minutes),
            used_percent=limit.used_percent,
            resets_at=limit.resets_at,
        )
        for limit in (snapshot.primary_limit, snapshot.secondary_limit)
        if limit is not None and limit.used_percent is not None
    )


def _same_codex_sign_in(
    first: UsageSnapshot | None,
    second: UsageSnapshot | None,
    now: datetime,
) -> bool:
    """Whether two Codex readings come from one account.

    Transcripts do not name the account, but quota windows belong to it: two
    readings of the same live window share a plan and reset time. Anything
    short of that match keeps the surfaces apart.
    """
    if first is None or second is None or first.plan_type != second.plan_type:
        return False
    first_windows = {window.label: window for window in _live_windows(_codex_windows(first), now)}
    second_windows = {window.label: window for window in _live_windows(_codex_windows(second), now)}
    shared = set(first_windows) & set(second_windows)
    if not shared:
        return False
    for label in shared:
        first_reset = first_windows[label].resets_at
        second_reset = second_windows[label].resets_at
        if first_reset is None or second_reset is None:
            return False
        if abs((first_reset - second_reset).total_seconds()) > CODEX_RESET_MATCH_SECONDS:
            return False
    return True


def _live_windows(windows: tuple[QuotaWindow, ...], now: datetime) -> tuple[QuotaWindow, ...]:
    # A reading from before a window reset says nothing about the new window.
    return tuple(window for window in windows if window.resets_at is None or window.resets_at > now)


def _percent(value: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, (value / total) * 100.0))


def _format_percent(value: float) -> str:
    return f"{value:.1f}%"


def short_token_count(value: int) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
