from __future__ import annotations

import re
from dataclasses import dataclass

from agent_harness.models import Provider
from agent_harness.team import normalize_handle


@dataclass(frozen=True)
class HireCommand:
    count: int
    provider: Provider | None = None


@dataclass(frozen=True)
class FireCommand:
    handle: str


@dataclass(frozen=True)
class RosterCommand:
    pass


TeamCommand = HireCommand | FireCommand | RosterCommand


def parse_team_command(text: str) -> TeamCommand | None:
    cleaned = _strip_bot_mention(_collapse_spaces(text))
    if not cleaned:
        return None
    return _parse_hire(cleaned) or _parse_fire(cleaned) or _parse_roster(cleaned)


def _parse_hire(text: str) -> HireCommand | None:
    match = re.match(
        r"^(?:please\s+)?hire(?:\s+(?P<count>\d+|one|two|three|four|five|ten))?"
        r"(?:\s+(?:new|more))?(?:\s+(?P<provider>codex|claude))?"
        r"(?:\s+agents?)?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    count = _count_value(match.group("count"))
    provider_text = match.group("provider")
    provider = Provider(provider_text.lower()) if provider_text else None
    return HireCommand(count=count, provider=provider)


def _parse_fire(text: str) -> FireCommand | None:
    match = re.match(
        r"^(?:please\s+)?fire\s+(?P<handle>@?[a-zA-Z][a-zA-Z0-9_-]{1,31})\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return FireCommand(handle=normalize_handle(match.group("handle")))


def _parse_roster(text: str) -> RosterCommand | None:
    if re.match(r"^(?:show\s+)?(?:team|roster|agents)\s*$", text, flags=re.IGNORECASE):
        return RosterCommand()
    return None


def _count_value(value: str | None) -> int:
    if value is None:
        return 1
    lookup = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "ten": 10,
    }
    lowered = value.lower()
    if lowered in lookup:
        return lookup[lowered]
    return int(value)


def _strip_bot_mention(text: str) -> str:
    return re.sub(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*", "", text).strip()


def _collapse_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
