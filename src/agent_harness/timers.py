from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from agent_harness.models import parse_timestamp, utc_now

AGENT_TIMER_SIGNAL_PREFIX = "SLACKGENTIC: TIMER "


@dataclass(frozen=True)
class TimerRequest:
    due_at: datetime
    prompt: str


@dataclass(frozen=True)
class TimerParseResult:
    request: TimerRequest | None = None
    error: str | None = None


def parse_agent_timer_signal(
    signal: str,
    *,
    now: datetime | None = None,
) -> TimerParseResult:
    stripped = signal.strip()
    if not stripped.upper().startswith(AGENT_TIMER_SIGNAL_PREFIX):
        return TimerParseResult()
    body = stripped[len(AGENT_TIMER_SIGNAL_PREFIX) :].strip()
    if not body:
        return TimerParseResult(error="missing timer body")
    if body.startswith("{"):
        return _parse_json_timer_body(body, now=now)
    when_text, separator, prompt = body.partition("|")
    if not separator:
        return TimerParseResult(error="expected '<delay-or-utc-time> | <prompt>'")
    prompt = prompt.strip()
    if not prompt:
        return TimerParseResult(error="missing timer prompt")
    due_at = _parse_due_at(when_text.strip(), now=now)
    if due_at is None:
        return TimerParseResult(error=f"could not parse timer due time: {when_text.strip()}")
    return TimerParseResult(request=TimerRequest(due_at=due_at, prompt=prompt))


def is_agent_timer_signal(signal: str) -> bool:
    return signal.strip().upper().startswith(AGENT_TIMER_SIGNAL_PREFIX)


def _parse_json_timer_body(body: str, *, now: datetime | None) -> TimerParseResult:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return TimerParseResult(error=f"invalid timer JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return TimerParseResult(error="timer JSON must be an object")
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return TimerParseResult(error="timer JSON must include a non-empty prompt")
    due_at = _json_due_at(payload, now=now)
    if due_at is None:
        return TimerParseResult(
            error="timer JSON must include due_at, delay_seconds, delay_minutes, or delay"
        )
    return TimerParseResult(request=TimerRequest(due_at=due_at, prompt=prompt.strip()))


def _json_due_at(payload: dict[str, Any], *, now: datetime | None) -> datetime | None:
    due_at = payload.get("due_at")
    if isinstance(due_at, str):
        parsed = parse_timestamp(due_at)
        if parsed is not None:
            return parsed
    delay_seconds = payload.get("delay_seconds")
    if isinstance(delay_seconds, int | float) and delay_seconds >= 0:
        return (now or utc_now()) + timedelta(seconds=float(delay_seconds))
    delay_minutes = payload.get("delay_minutes")
    if isinstance(delay_minutes, int | float) and delay_minutes >= 0:
        return (now or utc_now()) + timedelta(minutes=float(delay_minutes))
    delay = payload.get("delay")
    if isinstance(delay, str):
        return _parse_due_at(delay, now=now)
    return None


def _parse_due_at(text: str, *, now: datetime | None) -> datetime | None:
    if not text:
        return None
    delay = _parse_delay(text)
    if delay is not None:
        return (now or utc_now()) + delay
    return parse_timestamp(text)


def _parse_delay(text: str) -> timedelta | None:
    total_seconds = 0.0
    position = 0
    matches = list(
        re.finditer(
            r"(?P<amount>\d+(?:\.\d+)?)\s*"
            r"(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\b",
            text,
            flags=re.IGNORECASE,
        )
    )
    if not matches:
        return None
    for match in matches:
        gap = text[position : match.start()]
        if gap.strip() and gap.strip().lower() not in {"and", ","}:
            return None
        amount = float(match.group("amount"))
        unit = match.group("unit").lower()
        if unit.startswith(("s", "sec")):
            total_seconds += amount
        elif unit.startswith(("m", "min")):
            total_seconds += amount * 60
        elif unit.startswith(("h", "hr")):
            total_seconds += amount * 3600
        elif unit.startswith("d"):
            total_seconds += amount * 86400
        else:
            return None
        position = match.end()
    tail = text[position:]
    if tail.strip():
        return None
    return timedelta(seconds=total_seconds)
