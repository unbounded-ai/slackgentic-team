"""Subscription quota and sign-in details for the status card.

Codex records its rate limits in every transcript, but Claude transcripts do
not. Claude reports them only to a running client, so a tiny headless request
reads them from its ``rate_limit_event``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

CLAUDE_QUOTA_PROBE_MODEL = "haiku"
CLAUDE_QUOTA_PROBE_TIMEOUT_SECONDS = 60
# Environment a parent Claude session sets for its children. A probe that
# inherited it would report the parent's sign-in instead of the CLI's own.
_INHERITED_CLAUDE_ENV = (
    "ANTHROPIC_BASE_URL",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_OAUTH_TOKEN",
)
_CLAUDE_WINDOW_LABELS = {"five_hour": "5h", "seven_day": "Week"}
_CLAUDE_WINDOW_ORDER = ("five_hour", "seven_day")
_PLAN_WORDS = {"prolite": "Pro Lite", "chatgpt": "ChatGPT"}


@dataclass(frozen=True)
class QuotaWindow:
    label: str
    used_percent: float
    resets_at: datetime | None = None


@dataclass(frozen=True)
class ClaudeQuota:
    windows: tuple[QuotaWindow, ...]
    as_of: datetime
    using_overage: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "as_of": self.as_of.isoformat(),
                "using_overage": self.using_overage,
                "windows": [
                    {
                        "label": window.label,
                        "used_percent": window.used_percent,
                        "resets_at": window.resets_at.isoformat() if window.resets_at else None,
                    }
                    for window in self.windows
                ],
            }
        )

    @classmethod
    def from_json(cls, text: str | None) -> ClaudeQuota | None:
        if not text:
            return None
        try:
            data = json.loads(text)
            windows = tuple(
                QuotaWindow(
                    label=str(item["label"]),
                    used_percent=float(item["used_percent"]),
                    resets_at=_parse_iso(item.get("resets_at")),
                )
                for item in data.get("windows") or ()
            )
            as_of = _parse_iso(data.get("as_of"))
        except (ValueError, TypeError, KeyError, AttributeError):
            return None
        if as_of is None:
            return None
        return cls(windows=windows, as_of=as_of, using_overage=bool(data.get("using_overage")))


@dataclass(frozen=True)
class ClaudeSignIns:
    """Who the Claude CLI and the Claude app are signed in as."""

    cli_name: str | None = None
    cli_plan: str | None = None
    app_is_separate: bool = False


def parse_claude_rate_limit_event(
    output: str,
    now: datetime | None = None,
) -> ClaudeQuota | None:
    """The latest subscription windows from Claude ``stream-json`` output."""
    latest: dict[str, Any] | None = None
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info")
            if isinstance(info, dict):
                latest = info
    if latest is None:
        return None
    windows_data = latest.get("unifiedWindows")
    if not isinstance(windows_data, dict):
        # Older clients report only the window closest to its limit.
        kind = latest.get("rateLimitType")
        windows_data = {kind: latest} if isinstance(kind, str) else {}
    windows: list[QuotaWindow] = []
    for kind in _CLAUDE_WINDOW_ORDER:
        window = windows_data.get(kind)
        if not isinstance(window, dict) or window.get("utilization") is None:
            continue
        try:
            used = float(window["utilization"]) * 100.0
        except (TypeError, ValueError):
            continue
        windows.append(
            QuotaWindow(
                label=_CLAUDE_WINDOW_LABELS[kind],
                used_percent=used,
                resets_at=_parse_epoch(window.get("resetsAt")),
            )
        )
    if not windows:
        return None
    return ClaudeQuota(
        windows=tuple(windows),
        as_of=now or datetime.now(UTC),
        using_overage=bool(latest.get("isUsingOverage")),
    )


def probe_claude_quota(
    claude_binary: str = "claude",
    *,
    runner=None,
    cwd: Path | None = None,
) -> ClaudeQuota | None:
    """Read the CLI sign-in's quota with a one-token request (a few hundred tokens)."""
    env = {key: value for key, value in os.environ.items() if key not in _INHERITED_CLAUDE_ENV}
    try:
        completed = (runner or subprocess.run)(
            [
                claude_binary,
                "-p",
                ".",
                "--model",
                CLAUDE_QUOTA_PROBE_MODEL,
                "--system-prompt",
                "Reply with one character.",
                "--tools",
                "",
                "--setting-sources",
                "",
                "--strict-mcp-config",
                "--no-session-persistence",
                "--max-turns",
                "1",
                "--output-format",
                "stream-json",
                "--verbose",
            ],
            capture_output=True,
            text=True,
            timeout=CLAUDE_QUOTA_PROBE_TIMEOUT_SECONDS,
            check=False,
            env=env,
            cwd=str(cwd) if cwd else None,
        )
    except (OSError, subprocess.SubprocessError):
        LOGGER.debug("Claude quota probe failed", exc_info=True)
        return None
    return parse_claude_rate_limit_event(completed.stdout or "")


def read_claude_sign_ins(home: Path | None = None) -> ClaudeSignIns:
    """Profile fields only; credentials are never read."""
    home = home or Path.home()
    account = _read_json(home / ".claude.json").get("oauthAccount")
    account = account if isinstance(account, dict) else {}
    app_config = _read_json(home / "Library" / "Application Support" / "Claude" / "config.json")
    app_uuid = app_config.get("lastKnownAccountUuid")
    cli_uuid = account.get("accountUuid")
    name = account.get("displayName") or account.get("fullName")
    return ClaudeSignIns(
        cli_name=str(name).split()[0] if name else None,
        cli_plan=claude_plan_label(
            account.get("organizationType"),
            account.get("userRateLimitTier") or account.get("organizationRateLimitTier"),
        ),
        # Only a matching account id proves the app shares the CLI's sign-in.
        app_is_separate=not (app_uuid and cli_uuid and app_uuid == cli_uuid),
    )


def claude_plan_label(organization_type: Any, rate_limit_tier: Any) -> str | None:
    tier = str(rate_limit_tier or "")
    multiplier = re.search(r"max_(\d+x)\b", tier)
    kind = str(organization_type or "")
    if kind.startswith("claude_"):
        kind = kind.removeprefix("claude_")
    if not kind:
        return None
    label = kind.replace("_", " ").title()
    if multiplier and label == "Max":
        label = f"Max {multiplier.group(1)}"
    return label


def plan_label(plan_type: Any) -> str | None:
    """``self_serve_business_prolite`` -> ``Business Pro Lite``."""
    if not plan_type:
        return None
    words = [
        word
        for word in str(plan_type).replace("-", "_").split("_")
        if word not in {"self", "serve"}
    ]
    return " ".join(_PLAN_WORDS.get(word, word.title()) for word in words) or None


def window_label(minutes: int | None) -> str:
    if minutes is None:
        return "Window"
    if minutes == 10080:
        return "Week"
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _parse_epoch(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
