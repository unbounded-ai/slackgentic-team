"""The owner's timezone: where it is stored, how it is inferred, and how times read in it.

Slack date tokens already show each viewer their own local time, but plain-text
times (fallbacks, roster details, schedule confirmations) and agent prompts need
one concrete zone. That zone is the owner's, detected from their Slack profile
when they are first seen and changeable from the settings card.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo, available_timezones

SETTING_USER_TIMEZONE = "slack.human_timezone"
# Where the stored zone came from: "slack" or "system" when inferred, "manual"
# once the owner picks one. Inferred zones may be refreshed; manual ones are kept.
SETTING_USER_TIMEZONE_SOURCE = "slack.human_timezone_source"
TIMEZONE_SOURCE_SLACK = "slack"
TIMEZONE_SOURCE_SYSTEM = "system"
TIMEZONE_SOURCE_MANUAL = "manual"

_ZONEINFO_MARKER = "zoneinfo/"
_TIMEZONE_ALIASES = {"utc": "UTC", "gmt": "UTC", "z": "UTC"}


class SettingStore(Protocol):
    def get_setting(self, key: str) -> str | None: ...

    def set_setting(self, key: str, value: str) -> None: ...


def normalize_timezone(value: object) -> str | None:
    """Return the canonical IANA name for ``value``, or None when it is not one."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    alias = _TIMEZONE_ALIASES.get(text.lower())
    if alias:
        return alias
    return _timezone_names().get(text.lower().replace(" ", "_"))


@cache
def _timezone_names() -> dict[str, str]:
    """Every known IANA name, keyed by its lowercase spelling."""
    return {name.lower(): name for name in available_timezones()}


def zone_for(name: str | None) -> ZoneInfo | None:
    normalized = normalize_timezone(name)
    return ZoneInfo(normalized) if normalized else None


def system_timezone_name() -> str | None:
    """The machine's IANA zone, from ``TZ`` or the ``/etc/localtime`` link."""
    configured = normalize_timezone(os.environ.get("TZ", "").lstrip(":"))
    if configured:
        return configured
    try:
        target = str(Path("/etc/localtime").resolve())
    except OSError:
        return None
    index = target.rfind(_ZONEINFO_MARKER)
    if index < 0:
        return None
    return normalize_timezone(target[index + len(_ZONEINFO_MARKER) :])


def configured_timezone(store: SettingStore) -> str | None:
    return normalize_timezone(store.get_setting(SETTING_USER_TIMEZONE))


def timezone_is_manual(store: SettingStore) -> bool:
    return store.get_setting(SETTING_USER_TIMEZONE_SOURCE) == TIMEZONE_SOURCE_MANUAL


def set_user_timezone(store: SettingStore, timezone: str, source: str) -> None:
    store.set_setting(SETTING_USER_TIMEZONE, timezone)
    store.set_setting(SETTING_USER_TIMEZONE_SOURCE, source)


def remember_inferred_timezone(store: SettingStore, slack_timezone: object) -> str | None:
    """Store an inferred zone unless the owner already picked one.

    Prefers the zone on the owner's Slack profile and falls back to this machine's.
    Returns the zone now in effect.
    """
    current = configured_timezone(store)
    if current and timezone_is_manual(store):
        return current
    inferred = normalize_timezone(slack_timezone)
    source = TIMEZONE_SOURCE_SLACK
    if inferred is None:
        if current:
            return current
        inferred = system_timezone_name()
        source = TIMEZONE_SOURCE_SYSTEM
    if inferred is None:
        return current
    if inferred != current or store.get_setting(SETTING_USER_TIMEZONE_SOURCE) != source:
        set_user_timezone(store, inferred, source)
    return inferred


def format_user_time(value: datetime, timezone: str | None) -> str:
    """``Wed Sep 23, 2:05 PM EDT`` in ``timezone`` (UTC when it is unknown)."""
    zone = zone_for(timezone) or ZoneInfo("UTC")
    local = value.astimezone(zone) if value.tzinfo else value.replace(tzinfo=UTC).astimezone(zone)
    clock = local.strftime("%I:%M %p").lstrip("0")
    return f"{local.strftime('%a %b')} {local.day}, {clock} {local.strftime('%Z')}"


def timezone_label(timezone: str | None, *, now: datetime | None = None) -> str:
    """``America/New_York (EDT, UTC-04:00)`` for display next to a zone name."""
    zone = zone_for(timezone)
    if zone is None:
        return "UTC"
    local = (now or datetime.now(UTC)).astimezone(zone)
    offset = local.strftime("%z")
    offset_text = f"UTC{offset[:3]}:{offset[3:]}" if offset else "UTC"
    abbreviation = local.strftime("%Z")
    if abbreviation and abbreviation != timezone and not abbreviation.startswith(("+", "-")):
        return f"{timezone} ({abbreviation}, {offset_text})"
    return f"{timezone} ({offset_text})"


def timezone_prompt_lines(timezone: str | None, now: datetime | None = None) -> list[str]:
    """Prompt lines that tell an agent which zone the owner reads times in.

    Pass ``now`` to include the owner's current local time; leave it out where the
    prompt must stay identical across launches.
    """
    zone = zone_for(timezone)
    if zone is None:
        return []
    zone_line = f"Owner timezone: {timezone}."
    if now is not None:
        local = now.astimezone(zone)
        zone_line = (
            f"Owner timezone: {timezone_label(timezone, now=now)}. "
            f"Owner's current local time: {local.strftime('%Y-%m-%d %H:%M %Z')}."
        )
    return [
        zone_line,
        "Read times the owner gives without a timezone (for example '9am' or 'tomorrow "
        f"evening') as {timezone}. Whenever you mention a time, date, or timestamp to the "
        f"owner, including UTC times from logs or tools, convert it to {timezone} and "
        "label the zone (for example 'Tue Sep 22, 2:05 PM EDT'). Keep machine fields and "
        "control lines that ask for UTC in UTC.",
    ]
