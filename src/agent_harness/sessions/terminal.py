from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent_harness.models import AgentSession, Provider

ProcessLister = Callable[[], list[str]]
CwdResolver = Callable[[int], Path | None]
StartResolver = Callable[[int], datetime | None]
LogWriter = Callable[[Path, str], None]
TtyWriter = Callable[[str, str], None]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TerminalTarget:
    pid: int
    tty: str
    cwd: Path | None
    command: str
    started_at: datetime | None = None


class SessionTerminalNotifier:
    def __init__(
        self,
        process_lister: ProcessLister | None = None,
        cwd_resolver: CwdResolver | None = None,
        start_resolver: StartResolver | None = None,
        log_root: Path | None = None,
        log_writer: LogWriter | None = None,
        tty_writer: TtyWriter | None = None,
        write_tui_notice: bool = False,
        write_user_tui_notice: bool = False,
    ):
        # Process discovery is retained for diagnostics, but notifications are
        # kept short when written into live TUIs. Full content still goes to the
        # sidecar log so long Slack messages do not wreck terminal layout.
        self.process_lister = process_lister or _list_process_rows
        self.cwd_resolver = cwd_resolver or _process_cwd
        self.start_resolver = start_resolver or _process_started_at
        self.log_root = log_root
        self.log_writer = log_writer or _append_log
        self.tty_writer = tty_writer or _write_tty
        self.write_tui_notice = write_tui_notice
        self.write_user_tui_notice = write_user_tui_notice

    def notify_user_message(self, session: AgentSession, text: str) -> int:
        return self._notify(
            session,
            "Slack reply",
            text,
            footer="Session is being resumed in the background.",
            write_tui=self.write_user_tui_notice,
        )

    def notify_agent_response(self, session: AgentSession, text: str) -> int:
        return self._notify(session, "Slackgentic response", text, write_tui=True)

    def _notify(
        self,
        session: AgentSession,
        label: str,
        text: str,
        footer: str | None = None,
        write_tui: bool = True,
    ) -> int:
        body = text.strip()
        if not body:
            return 0
        message = _terminal_message(label, session, body, footer)
        path = self.notification_log_path(session)
        try:
            self.log_writer(path, message)
        except Exception:
            LOGGER.debug("failed to write session terminal notification", exc_info=True)
            return 0
        if self.write_tui_notice and write_tui:
            self._notify_tui(session, label, body, footer)
        return 1

    def _notify_tui(
        self,
        session: AgentSession,
        label: str,
        text: str,
        footer: str | None,
    ) -> int:
        if session.provider == Provider.CLAUDE:
            return 0
        notice = _tui_notice(label, session, text, footer)
        count = 0
        for target in self.targets_for_session(session):
            try:
                self.tty_writer(target.tty, notice)
            except Exception:
                LOGGER.debug("failed to write session TUI notification", exc_info=True)
                continue
            count += 1
        return count

    def notification_log_path(self, session: AgentSession) -> Path:
        root = self.log_root or session.cwd or session.transcript_path.parent
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", session.session_id[:32] or "unknown")
        filename = f"{session.provider.value}-{safe_id}.log"
        return root / ".slackgentic" / "terminal-notifications" / filename

    def targets_for_session(self, session: AgentSession) -> list[TerminalTarget]:
        targets: list[TerminalTarget] = []
        for target in self.provider_processes(session.provider):
            if session.cwd and target.cwd and _same_path(session.cwd, target.cwd):
                targets.append(target)
        return _best_session_targets(session, targets)

    def provider_process_for_pid(self, provider: Provider, pid: int) -> TerminalTarget | None:
        for target in self.provider_processes(provider, require_tty=False):
            if target.pid == pid:
                return target
        return None

    def provider_processes(
        self,
        provider: Provider,
        *,
        require_tty: bool = True,
    ) -> list[TerminalTarget]:
        targets: list[TerminalTarget] = []
        for row in self.process_lister():
            parsed = _parse_process_row(row)
            if parsed is None:
                continue
            pid, tty, command = parsed
            if not _is_provider_process(provider, command):
                continue
            if require_tty and (tty == "??" or not tty):
                continue
            cwd = self.cwd_resolver(pid)
            try:
                started_at = self.start_resolver(pid)
            except Exception:
                LOGGER.debug("failed to resolve session process start time", exc_info=True)
                started_at = None
            targets.append(
                TerminalTarget(
                    pid=pid,
                    tty=tty,
                    cwd=cwd,
                    command=command,
                    started_at=started_at,
                )
            )
        return targets


def _terminal_message(
    label: str,
    session: AgentSession,
    text: str,
    footer: str | None,
) -> str:
    lines = [
        "",
        f"[slackgentic] {label} for {session.provider.value} session {session.session_id[:8]}:",
        text,
    ]
    if footer:
        lines.append(f"[slackgentic] {footer}")
    return "\n".join(lines) + "\n"


def _tui_notice(
    label: str,
    session: AgentSession,
    text: str,
    footer: str | None,
) -> str:
    rendered_lines = _bounded_notice_lines(text, max_lines=24)
    title = f"[slackgentic] {label} ({session.provider.value} {session.session_id[:8]})"
    border = "=" * min(120, max(48, len(title)))
    lines = [border, title, *rendered_lines]
    if footer and label != "Slack reply":
        lines.append(footer)
    lines.append(border)
    body = "\r\n".join(lines)
    return f"\r\n{body}\r\n"


def _bounded_notice_lines(text: str, max_lines: int = 8, width: int = 110) -> list[str]:
    cleaned = _strip_terminal_controls(text)
    physical_lines = cleaned.splitlines() or [cleaned]
    lines: list[str] = []
    truncated = False
    for line in physical_lines:
        if len(lines) >= max_lines:
            truncated = True
            break
        remaining = line
        if not remaining:
            lines.append("")
            continue
        while remaining and len(lines) < max_lines:
            lines.append(remaining[:width])
            remaining = remaining[width:]
        if remaining:
            truncated = True
            break
    if truncated:
        if len(lines) >= max_lines:
            lines[-1] = f"{lines[-1][: max(0, width - 14)]} ... [more]"
        else:
            lines.append("[more in log]")
    return lines


def _strip_terminal_controls(text: str) -> str:
    without_osc = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    no_ansi = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", without_osc)
    no_ansi = re.sub(r"\x1b[@-_][0-?]*[ -/]*[@-~]", "", no_ansi)
    return "".join(ch for ch in no_ansi if ch in "\n\t" or ord(ch) >= 32).strip()


def _parse_process_row(row: str) -> tuple[int, str, str] | None:
    parts = row.strip().split(None, 2)
    if len(parts) != 3:
        return None
    try:
        pid = int(parts[0])
    except ValueError:
        return None
    return pid, parts[1], parts[2]


def _is_provider_process(provider: Provider, command: str) -> bool:
    if provider == Provider.CLAUDE:
        return bool(re.search(r"(^|[/\s])claude(\s|$)", command)) and "--print" not in command
    if provider == Provider.CODEX:
        return bool(re.search(r"(^|[/\s])codex(\s|$)", command)) and " exec " not in command
    return False


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.expanduser().resolve() == right.expanduser().resolve()
    except OSError:
        return left.expanduser() == right.expanduser()


def _list_process_rows() -> list[str]:
    try:
        completed = subprocess.run(
            ["ps", "-axo", "pid=,tty=,command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return completed.stdout.splitlines()


def _process_cwd(pid: int) -> Path | None:
    try:
        completed = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.startswith("n"):
            return Path(line[1:])
    return None


def _process_started_at(pid: int) -> datetime | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    if not text:
        return None
    try:
        return datetime.strptime(text, "%a %b %d %H:%M:%S %Y").astimezone()
    except ValueError:
        return None


def _best_session_targets(
    session: AgentSession,
    targets: list[TerminalTarget],
) -> list[TerminalTarget]:
    if len(targets) <= 1:
        return targets
    dated = [target for target in targets if target.started_at is not None]
    if not dated:
        return targets
    if session.started_at is not None:
        reference_time = session.started_at.astimezone()
        best = min(
            dated,
            key=lambda target: abs(
                (target.started_at.astimezone() - reference_time).total_seconds()
            ),
        )
        return [best]
    reference_time = session.last_seen_at
    if reference_time is None:
        return targets
    reference_time = reference_time.astimezone()
    before_or_equal = [
        target for target in dated if target.started_at.astimezone() <= reference_time
    ]
    if before_or_equal:
        return [max(before_or_equal, key=lambda target: target.started_at.astimezone())]
    best = min(
        dated,
        key=lambda target: abs((target.started_at.astimezone() - reference_time).total_seconds()),
    )
    return [best]


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text.replace("\n", os.linesep))


def _write_tty(tty: str, text: str) -> None:
    path = Path("/dev") / tty
    with path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write(text.replace("\n", os.linesep))


def _short_path(path: Path) -> str:
    try:
        home = Path.home().resolve()
        return f"~/{path.expanduser().resolve().relative_to(home).as_posix()}"
    except (OSError, ValueError):
        return str(path)
