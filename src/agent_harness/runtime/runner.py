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
    claude_effort: str | None = None
    codex_reasoning_effort: str | None = None

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
        if request.codex_reasoning_effort:
            args.extend(
                [
                    "-c",
                    f"model_reasoning_effort={json.dumps(request.codex_reasoning_effort)}",
                ]
            )
        extra_roots = _safe_auto_extra_roots(request)
        if mode == PermissionMode.DANGEROUS:
            args.extend(_codex_dangerous_overrides())
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
        # --input-format stream-json keeps Claude alive across turns: each user
        # follow-up is a JSON message on stdin instead of a fresh subprocess.
        # The initial prompt is written to stdin by ManagedAgentProcess.start
        # below, so it is not passed as an argv positional here.
        args = [
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--input-format",
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
        if request.claude_effort:
            args.extend(["--effort", request.claude_effort])
        if request.worktree:
            args.extend(["--worktree", request.worktree])
        return request.claude_binary, args
    raise ValueError(f"unsupported provider: {request.provider}")


def _claude_stream_json_user_turn(text: str) -> str:
    # stream-json input mode expects one JSON value per line. The Claude CLI
    # treats each user-shaped value as a new conversation turn.
    payload = {"type": "user", "message": {"role": "user", "content": text}}
    return json.dumps(payload) + "\n"


def _codex_trust_override(cwd: Path) -> str:
    return f"projects.{json.dumps(str(cwd))}.trust_level={json.dumps('trusted')}"


def _codex_dangerous_overrides() -> list[str]:
    return [
        "-c",
        f"sandbox_mode={json.dumps('danger-full-access')}",
        "-c",
        f"approval_policy={json.dumps('never')}",
    ]


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

            # Large Slack-derived prompts can exceed PTY line-buffer limits before
            # the provider starts; pipe-backed stdin avoids that. Claude also uses
            # this path so we can keep its stdin open across follow-up turns.
            self.child = PopenSpawn(
                [command, *args],
                cwd=str(self.request.cwd),
                env=child_env,
                encoding="utf-8",
                codec_errors="replace",
                timeout=0.1,
            )
            if self._uses_stream_json_stdin():
                self.child.send(_claude_stream_json_user_turn(self.request.prompt))
                # No sendeof: Claude --input-format=stream-json stays alive
                # waiting for more user turns. send() below appends them.
                return
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
        if self.request.provider == Provider.CLAUDE:
            return True
        return self.request.provider == Provider.CODEX and not self.request.resume_session_id

    def _uses_stream_json_stdin(self) -> bool:
        return self.request.provider == Provider.CLAUDE

    def send(self, message: str) -> None:
        if self.child is None:
            raise RuntimeError("process is not started")
        if self._uses_stream_json_stdin():
            self.child.send(_claude_stream_json_user_turn(message))
            return
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

    def kill(self) -> None:
        if self.child is None:
            return
        # pexpect's terminate(force=True) walks SIGHUP/SIGCONT/SIGINT/SIGTERM
        # then SIGKILL, which is the strongest shutdown the child exposes.
        terminate = getattr(self.child, "terminate", None)
        if callable(terminate):
            try:
                terminate(force=True)
                return
            except TypeError:
                pass
        proc = getattr(self.child, "proc", None)
        proc_kill = getattr(proc, "kill", None)
        if callable(proc_kill):
            proc_kill()
            return
        kill = getattr(self.child, "kill", None)
        if callable(kill):
            kill(signal.SIGKILL)
