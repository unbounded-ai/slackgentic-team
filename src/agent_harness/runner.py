from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from agent_harness.models import Provider
from agent_harness.slack import dangerous_flag


@dataclass(frozen=True)
class LaunchRequest:
    provider: Provider
    prompt: str
    cwd: Path
    dangerous: bool = False
    model: str | None = None
    worktree: str | None = None
    codex_binary: str = "codex"
    claude_binary: str = "claude"


def build_command(request: LaunchRequest) -> tuple[str, list[str]]:
    if request.provider == Provider.CODEX:
        args: list[str] = [
            "exec",
            "--json",
            "--color",
            "never",
            "--skip-git-repo-check",
            "--ephemeral",
            "-c",
            _codex_trust_override(request.cwd),
        ]
        if request.dangerous:
            args.append(dangerous_flag(Provider.CODEX))
        else:
            args.extend(["--sandbox", "workspace-write"])
        if request.model:
            args.extend(["--model", request.model])
        args.extend(["-C", str(request.cwd), "-"])
        return request.codex_binary, args
    if request.provider == Provider.CLAUDE:
        args = [
            "--print",
            "--output-format",
            "json",
            "--no-session-persistence",
        ]
        if request.dangerous:
            args.append(dangerous_flag(Provider.CLAUDE))
        if request.model:
            args.extend(["--model", request.model])
        if request.worktree:
            args.extend(["--worktree", request.worktree])
        args.append("-")
        return request.claude_binary, args
    raise ValueError(f"unsupported provider: {request.provider}")


def _codex_trust_override(cwd: Path) -> str:
    return f"projects.{json.dumps(str(cwd))}.trust_level={json.dumps('trusted')}"


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
        self.child = pexpect.spawn(
            command,
            args,
            cwd=str(self.request.cwd),
            env=child_env,
            encoding="utf-8",
            timeout=0.1,
        )
        self.child.send(self.request.prompt)
        if not self.request.prompt.endswith("\n"):
            self.child.send("\n")
        self.child.sendeof()

    def send(self, message: str) -> None:
        if self.child is None:
            raise RuntimeError("process is not started")
        self.child.sendline(message)

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
        return bool(self.child.isalive())

    def terminate(self) -> None:
        if self.child is not None:
            self.child.terminate(force=False)
