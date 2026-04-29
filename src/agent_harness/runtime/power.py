from __future__ import annotations

import logging
import platform
import shutil
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)
POWER_SOURCE_AC = "AC Power"
POWER_SOURCE_BATTERY = "Battery Power"


class ActiveSessionAwakeKeeper:
    def __init__(
        self,
        is_active: Callable[[], bool],
        *,
        poll_seconds: float = 15.0,
        system: str | None = None,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
        which: Callable[[str], str | None] = shutil.which,
    ):
        self.is_active = is_active
        self.poll_seconds = poll_seconds
        self.system = (system or platform.system()).lower()
        self.popen_factory = popen_factory
        self.which = which
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sync_once()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="slackgentic-awake-keeper",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._stop_caffeinate()

    def sync_once(self) -> None:
        if self.system != "darwin" or not self.which("caffeinate"):
            return
        try:
            active = self.is_active()
        except Exception:
            LOGGER.debug("failed to inspect active sessions for awake keeper", exc_info=True)
            active = False
        if active:
            self._start_caffeinate()
        else:
            self._stop_caffeinate()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.sync_once()

    def _start_caffeinate(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        try:
            self._process = self.popen_factory(
                ["caffeinate", "-dimsu"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            LOGGER.debug("failed to start caffeinate", exc_info=True)
            self._process = None

    def _stop_caffeinate(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except Exception:
            process.kill()


@dataclass(frozen=True)
class MacPowerStatus:
    system: str
    caffeinate_available: bool
    pmset_available: bool
    settings: dict[str, dict[str, str]]
    wake_on_wireless: str | None
    scheduled_wakes: list[str]


def inspect_macos_power(
    *,
    system: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    run: Callable[..., Any] = subprocess.run,
) -> MacPowerStatus:
    resolved_system = (system or platform.system()).lower()
    caffeinate_available = bool(which("caffeinate")) if resolved_system == "darwin" else False
    pmset_available = bool(which("pmset")) if resolved_system == "darwin" else False
    if resolved_system != "darwin" or not pmset_available:
        return MacPowerStatus(
            system=resolved_system,
            caffeinate_available=caffeinate_available,
            pmset_available=pmset_available,
            settings={},
            wake_on_wireless=None,
            scheduled_wakes=[],
        )

    settings = parse_pmset_custom(_run_text(run, ["pmset", "-g", "custom"]))
    wake_on_wireless = parse_wake_on_wireless(
        _run_text(run, ["system_profiler", "SPAirPortDataType"])
    )
    scheduled_wakes = parse_scheduled_wakes(_run_text(run, ["pmset", "-g", "sched"]))
    return MacPowerStatus(
        system=resolved_system,
        caffeinate_available=caffeinate_available,
        pmset_available=pmset_available,
        settings=settings,
        wake_on_wireless=wake_on_wireless,
        scheduled_wakes=scheduled_wakes,
    )


def format_power_doctor_lines(status: MacPowerStatus) -> list[str]:
    if status.system != "darwin":
        return ["info macOS power checks skipped on this platform"]

    lines = [
        _check_line(
            status.caffeinate_available,
            "active-session keep-awake",
            "`caffeinate` available",
            "`caffeinate` missing",
        )
    ]
    if not status.pmset_available:
        lines.append("missing pmset power diagnostics")
        return lines

    ac_womp = _enabled_setting(status, POWER_SOURCE_AC, "womp")
    battery_womp = _enabled_setting(status, POWER_SOURCE_BATTERY, "womp")
    ac_tcp = _enabled_setting(status, POWER_SOURCE_AC, "tcpkeepalive")
    battery_tcp = _enabled_setting(status, POWER_SOURCE_BATTERY, "tcpkeepalive")

    lines.append(
        _status_line(
            "ok" if ac_womp else "off",
            "Wake for network access on power adapter",
            _raw_setting(status, POWER_SOURCE_AC, "womp"),
        )
    )
    lines.append(
        _status_line(
            "ok" if battery_womp else "off",
            "Wake for network access on battery",
            _raw_setting(status, POWER_SOURCE_BATTERY, "womp"),
        )
    )
    lines.append(
        _status_line(
            "ok" if ac_tcp and battery_tcp else "off",
            "TCP keepalive during sleep",
            f"AC={_raw_setting(status, POWER_SOURCE_AC, 'tcpkeepalive')} "
            f"Battery={_raw_setting(status, POWER_SOURCE_BATTERY, 'tcpkeepalive')}",
        )
    )
    if status.wake_on_wireless:
        lines.append(_status_line("ok", "Wi-Fi wake support", status.wake_on_wireless))
    else:
        lines.append(_status_line("info", "Wi-Fi wake support", "not reported"))
    if status.scheduled_wakes:
        lines.append(
            _status_line("info", "scheduled wake events", str(len(status.scheduled_wakes)))
        )
    lines.append(
        "info network wake is best-effort; active work is kept awake, but Slackgentic "
        "cannot guarantee wake from closed-lid or deep sleep"
    )
    return lines


def parse_pmset_custom(output: str) -> dict[str, dict[str, str]]:
    settings: dict[str, dict[str, str]] = {}
    current_source: str | None = None
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            current_source = line[:-1]
            settings.setdefault(current_source, {})
            continue
        if current_source is None:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        settings[current_source][parts[0]] = parts[-1]
    return settings


def parse_wake_on_wireless(output: str) -> str | None:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("wake on wireless:"):
            return line.split(":", 1)[1].strip() or None
    return None


def parse_scheduled_wakes(output: str) -> list[str]:
    wakes: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("["):
            continue
        if "wake" in line.lower():
            wakes.append(line)
    return wakes


def _run_text(run: Callable[..., Any], command: list[str]) -> str:
    try:
        completed = run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except Exception:
        LOGGER.debug("failed to run power diagnostic command %s", command, exc_info=True)
        return ""
    if getattr(completed, "returncode", 1) != 0:
        return ""
    return str(getattr(completed, "stdout", "") or "")


def _enabled_setting(status: MacPowerStatus, source: str, key: str) -> bool:
    return _raw_setting(status, source, key) == "1"


def _raw_setting(status: MacPowerStatus, source: str, key: str) -> str:
    return status.settings.get(source, {}).get(key, "unknown")


def _check_line(passed: bool, name: str, ok_detail: str, missing_detail: str) -> str:
    return _status_line(
        "ok" if passed else "missing", name, ok_detail if passed else missing_detail
    )


def _status_line(state: str, name: str, detail: str) -> str:
    return f"{state} {name} ({detail})"
