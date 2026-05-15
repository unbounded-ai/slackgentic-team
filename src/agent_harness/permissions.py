"""Helpers for the three-way agent permission mode.

Slackgentic launches agents in one of three permission modes:

* ``locked`` — every tool call needs human approval (the historical default
  before this module existed).
* ``safe-auto`` — file edits and read-only inspection commands are auto-approved
  but anything genuinely destructive still prompts. This is the default.
* ``dangerous`` — full permission bypass for the underlying CLI.

The mode is persisted on :class:`AgentTask` metadata under
``permission_mode``. Legacy tasks that only carry the older
``dangerous_mode`` boolean continue to resolve to ``dangerous``.
"""

from __future__ import annotations

from agent_harness.models import (
    DANGEROUS_MODE_METADATA_KEY,
    DEFAULT_PERMISSION_MODE,
    PERMISSION_MODE_METADATA_KEY,
    AgentTask,
    PermissionMode,
)

CLAUDE_SAFE_AUTO_PERMISSION_FLAG = "acceptEdits"

CLAUDE_SAFE_AUTO_ALLOWED_TOOLS: tuple[str, ...] = (
    "Read",
    "Grep",
    "Glob",
    "WebFetch",
    "WebSearch",
    "Bash(git status:*)",
    "Bash(git log:*)",
    "Bash(git diff:*)",
    "Bash(git show:*)",
    "Bash(git fetch:*)",
    "Bash(git pull:*)",
    "Bash(git branch:*)",
    "Bash(git remote:*)",
    "Bash(git rev-parse:*)",
    "Bash(git ls-files:*)",
    "Bash(gh pr view:*)",
    "Bash(gh pr list:*)",
    "Bash(gh pr diff:*)",
    "Bash(gh issue view:*)",
    "Bash(gh issue list:*)",
    "Bash(gh api:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(rg:*)",
    "Bash(grep:*)",
    "Bash(find:*)",
    "Bash(pwd)",
    "Bash(which:*)",
)

CODEX_SANDBOX_BY_MODE: dict[PermissionMode, str | None] = {
    PermissionMode.LOCKED: "read-only",
    PermissionMode.SAFE_AUTO: "workspace-write",
    PermissionMode.DANGEROUS: None,
}


def task_permission_mode(task: AgentTask) -> PermissionMode:
    raw = task.metadata.get(PERMISSION_MODE_METADATA_KEY)
    if isinstance(raw, str):
        try:
            return PermissionMode(raw)
        except ValueError:
            pass
    if task.metadata.get(DANGEROUS_MODE_METADATA_KEY):
        return PermissionMode.DANGEROUS
    return DEFAULT_PERMISSION_MODE


def codex_sandbox_for(mode: PermissionMode) -> str | None:
    return CODEX_SANDBOX_BY_MODE.get(mode)


def claude_permission_flag(mode: PermissionMode) -> str | None:
    if mode == PermissionMode.SAFE_AUTO:
        return CLAUDE_SAFE_AUTO_PERMISSION_FLAG
    return None


def claude_extra_allowed_tools(mode: PermissionMode) -> tuple[str, ...]:
    if mode == PermissionMode.SAFE_AUTO:
        return CLAUDE_SAFE_AUTO_ALLOWED_TOOLS
    return ()
