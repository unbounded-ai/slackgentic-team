from __future__ import annotations

import json
import logging
import os
import signal
from dataclasses import dataclass
from pathlib import Path

from agent_harness.models import DEFAULT_PERMISSION_MODE, PermissionMode, Provider
from agent_harness.permissions import (
    CLAUDE_CHANNEL_PERMISSION_MODE_ENV,
    claude_extra_allowed_tools,
    claude_permission_flag,
    codex_sandbox_for,
)
from agent_harness.slack import dangerous_flag

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaunchRequest:
    provider: Provider
    prompt: str
    cwd: Path
    permission_mode: PermissionMode = DEFAULT_PERMISSION_MODE
    model: str | None = None
    worktree: str | None = None
    resume_session_id: str | None = None
    claude_channel: bool = False
    slack_channel_id: str | None = None
    slack_thread_ts: str | None = None
    allowed_tools: tuple[str, ...] = ()
    safe_auto_extra_roots: tuple[Path, ...] = ()
    codex_binary: str = "codex"
    claude_binary: str = "claude"

    @property
    def dangerous(self) -> bool:
        return self.permission_mode == PermissionMode.DANGEROUS


def build_command(request: LaunchRequest) -> tuple[str, list[str]]:
    mode = request.permission_mode
    if request.provider == Provider.CODEX:
        args: list[str] = ["exec"]
        if request.resume_session_id:
            args.append("resume")
            args.extend(["--json", "--skip-git-repo-check"])
        else:
            args.extend(["--json", "--color", "never", "--skip-git-repo-check"])
        args.extend(["-c", _codex_trust_override(request.cwd)])
        extra_roots = _safe_auto_extra_roots(request)
        if mode == PermissionMode.DANGEROUS:
            args.append(dangerous_flag(Provider.CODEX))
        else:
            sandbox = codex_sandbox_for(mode)
            if sandbox is not None:
                if request.resume_session_id:
                    args.extend(["-c", f"sandbox_mode={json.dumps(sandbox)}"])
                    if extra_roots:
                        args.extend(
                            [
                                "-c",
                                _codex_workspace_writable_roots_override(extra_roots),
                            ]
                        )
                else:
                    args.extend(["--sandbox", sandbox])
                    for root in extra_roots:
                        args.extend(["--add-dir", str(root)])
        if request.model:
            args.extend(["--model", request.model])
        if request.resume_session_id:
            args.extend([request.resume_session_id, request.prompt])
        else:
            args.extend(["-C", str(request.cwd), "-"])
        return request.codex_binary, args
    if request.provider == Provider.CLAUDE:
        args = [
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
        ]
        merged_allowed: list[str] = []
        seen: set[str] = set()
        for tool in (*request.allowed_tools, *claude_extra_allowed_tools(mode)):
            if tool not in seen:
                merged_allowed.append(tool)
                seen.add(tool)
        for allowed_tool in merged_allowed:
            args.append(f"--allowedTools={allowed_tool}")
        if request.claude_channel:
            args.append("--dangerously-load-development-channels=server:slackgentic")
        if request.resume_session_id:
            args.extend(["--resume", request.resume_session_id])
        if mode == PermissionMode.DANGEROUS:
            args.append(dangerous_flag(Provider.CLAUDE))
        else:
            permission_flag = claude_permission_flag(mode)
            if permission_flag:
                for root in _safe_auto_extra_roots(request):
                    args.extend(["--add-dir", str(root)])
                args.extend(["--permission-mode", permission_flag])
        if request.model:
            args.extend(["--model", request.model])
        if request.worktree:
            args.extend(["--worktree", request.worktree])
        args.append(request.prompt)
        return request.claude_binary, args
    raise ValueError(f"unsupported provider: {request.provider}")


def _codex_trust_override(cwd: Path) -> str:
    return f"projects.{json.dumps(str(cwd))}.trust_level={json.dumps('trusted')}"


def _safe_auto_extra_roots(request: LaunchRequest) -> tuple[Path, ...]:
    if request.permission_mode != PermissionMode.SAFE_AUTO:
        return ()
    roots: list[Path] = []
    seen: set[str] = set()
    for root in request.safe_auto_extra_roots:
        expanded = root.expanduser()
        key = str(expanded)
        if key in seen:
            continue
        seen.add(key)
        roots.append(expanded)
    return tuple(roots)


def _codex_workspace_writable_roots_override(roots: tuple[Path, ...]) -> str:
    return "sandbox_workspace_write.writable_roots=" + json.dumps([str(root) for root in roots])


class ManagedAgentProcess:
    def __init__(self, request: LaunchRequest, env: dict[str, str] | None = None):
        self.request = request
        self.env = env
        self.child = None

    def start(self) -> None:
        import pexpect

        command, args = build_command(self.request)
        child_env = os.environ.copy()
        if self.env:
            child_env.update(self.env)
        if (
            self.request.provider == Provider.CLAUDE
            and self.request.slack_channel_id
            and self.request.slack_thread_ts
        ):
            from agent_harness.sessions.claude_channel import (
                SLACK_THREAD_CHANNEL_ENV,
                SLACK_THREAD_TS_ENV,
            )

            child_env[SLACK_THREAD_CHANNEL_ENV] = self.request.slack_channel_id
            child_env[SLACK_THREAD_TS_ENV] = self.request.slack_thread_ts
        if self.request.provider == Provider.CLAUDE:
            child_env[CLAUDE_CHANNEL_PERMISSION_MODE_ENV] = self.request.permission_mode.value
        if (
            self.request.provider == Provider.CLAUDE
            and self.request.permission_mode == PermissionMode.DANGEROUS
        ):
            from agent_harness.sessions.claude_channel import DANGEROUS_MODE_ENV

            child_env[DANGEROUS_MODE_ENV] = "1"
        if self._reads_prompt_from_stdin():
            from pexpect.popen_spawn import PopenSpawn

            # Large Slack-derived prompts can exceed PTY line-buffer limits before Codex starts.
            self.child = PopenSpawn(
                [command, *args],
                cwd=str(self.request.cwd),
                env=child_env,
                encoding="utf-8",
                codec_errors="replace",
                timeout=0.1,
            )
            self.child.send(self.request.prompt)
            if not self.request.prompt.endswith("\n"):
                self.child.send("\n")
            self.child.sendeof()
            return
        self.child = pexpect.spawn(
            command,
            args,
            cwd=str(self.request.cwd),
            env=child_env,
            encoding="utf-8",
            codec_errors="replace",
            timeout=0.1,
            echo=False,
        )

    def _reads_prompt_from_stdin(self) -> bool:
        return self.request.provider == Provider.CODEX and not self.request.resume_session_id

    def send(self, message: str) -> None:
        if self.child is None:
            raise RuntimeError("process is not started")
        self.child.sendline(message)

    def interrupt(self) -> None:
        if self.child is None:
            raise RuntimeError("process is not started")
        self.child.send("\x1b")

    def read_available(self, max_reads: int = 20, timeout: float = 0.05) -> str:
        if self.child is None:
            raise RuntimeError("process is not started")
        import pexpect

        chunks: list[str] = []
        for _ in range(max_reads):
            try:
                self.child.expect(".+", timeout=timeout)
            except pexpect.EOF:
                before = getattr(self.child, "before", "")
                if before:
                    chunks.append(before)
                break
            except pexpect.TIMEOUT:
                break
            chunks.append(self.child.after)
        return "".join(chunks)

    def is_alive(self) -> bool:
        if self.child is None:
            return False
        isalive = getattr(self.child, "isalive", None)
        if callable(isalive):
            try:
                return bool(isalive())
            except Exception:
                LOGGER.debug("failed to poll managed agent process", exc_info=True)
                return False
        proc = getattr(self.child, "proc", None)
        poll = getattr(proc, "poll", None)
        if callable(poll):
            return poll() is None
        return False

    def terminate(self) -> None:
        if self.child is None:
            return
        terminate = getattr(self.child, "terminate", None)
        if callable(terminate):
            try:
                terminate(force=False)
            except TypeError:
                terminate()
            return
        proc = getattr(self.child, "proc", None)
        proc_terminate = getattr(proc, "terminate", None)
        if callable(proc_terminate):
            proc_terminate()
            return
        kill = getattr(self.child, "kill", None)
        if callable(kill):
            kill(signal.SIGTERM)
