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

import json
import shlex
from collections.abc import Mapping

from agent_harness.models import (
    DANGEROUS_MODE_METADATA_KEY,
    DEFAULT_PERMISSION_MODE,
    PERMISSION_MODE_METADATA_KEY,
    AgentTask,
    PermissionMode,
)

CLAUDE_CHANNEL_PERMISSION_MODE_ENV = "SLACKGENTIC_CLAUDE_PERMISSION_MODE"
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
    "Bash(git blame:*)",
    "Bash(git grep:*)",
    "Bash(git rev-parse:*)",
    "Bash(git ls-files:*)",
    "Bash(git ls-tree:*)",
    "Bash(git cat-file:*)",
    "Bash(git merge-base:*)",
    "Bash(git for-each-ref:*)",
    "Bash(gh pr view:*)",
    "Bash(gh pr list:*)",
    "Bash(gh pr diff:*)",
    "Bash(gh issue view:*)",
    "Bash(gh issue list:*)",
    "Bash(ls:*)",
    "Bash(cat:*)",
    "Bash(head:*)",
    "Bash(tail:*)",
    "Bash(nl:*)",
    "Bash(wc:*)",
    "Bash(sed -n:*)",
    "Bash(rg:*)",
    "Bash(grep:*)",
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


def claude_channel_permission_mode_from_env(
    env: Mapping[str, str] | None = None,
) -> PermissionMode | None:
    env = env or {}
    raw = env.get(CLAUDE_CHANNEL_PERMISSION_MODE_ENV)
    if not isinstance(raw, str):
        return None
    try:
        return PermissionMode(raw)
    except ValueError:
        return None


def claude_safe_auto_permission_request_allowed(params: Mapping[str, object]) -> bool:
    tool_name = str(params.get("tool_name") or "")
    if tool_name in {
        "Read",
        "Grep",
        "Glob",
        "WebFetch",
        "WebSearch",
        "Edit",
        "MultiEdit",
        "Write",
    }:
        return True
    if tool_name != "Bash":
        return False
    command = _permission_request_bash_command(params)
    return bool(command and _safe_auto_bash_command(command))


def _permission_request_bash_command(params: Mapping[str, object]) -> str:
    value = _permission_input_object(params)
    if isinstance(value, dict):
        command = value.get("command")
        return command.strip() if isinstance(command, str) else ""
    return ""


def _permission_input_object(params: Mapping[str, object]) -> object:
    for key in ("input", "arguments", "tool_input", "parameters"):
        if key not in params:
            continue
        value = params[key]
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    preview = params.get("input_preview")
    if isinstance(preview, str) and preview.strip():
        try:
            return json.loads(preview)
        except json.JSONDecodeError:
            return preview
    return None


def _safe_auto_bash_command(command: str) -> bool:
    command = command.strip()
    if not command or any(value in command for value in ("\n", "\r", "$(", "`")):
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    if not parts:
        return False
    segments = _split_shell_segments(parts)
    if not segments:
        return False
    return all(_safe_auto_bash_segment(segment) for segment in segments)


def _split_shell_segments(parts: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for part in parts:
        if part == "&&":
            if not segments[-1]:
                return []
            segments.append([])
            continue
        if part in {";", "||", "|", "&"}:
            return []
        segments[-1].append(part)
    if not segments[-1]:
        return []
    return segments


def _safe_auto_bash_segment(parts: list[str]) -> bool:
    command_parts = _strip_safe_redirections(parts)
    if not command_parts:
        return False
    executable = command_parts[0]
    if executable == "cd":
        return len(command_parts) == 2 and _safe_shell_arg(command_parts[1])
    if executable == "git":
        return _safe_auto_git_command(command_parts)
    if executable == "gh":
        return _safe_auto_gh_command(command_parts)
    if executable in {"ls", "cat", "head", "tail", "nl", "wc", "rg", "grep", "pwd", "which"}:
        return all(_safe_shell_arg(part) for part in command_parts[1:])
    if executable == "sed":
        return _safe_auto_sed_command(command_parts)
    return False


def _strip_safe_redirections(parts: list[str]) -> list[str]:
    return [
        part
        for part in parts
        if part
        not in {
            "2>&1",
            "1>&2",
            ">/dev/null",
            "1>/dev/null",
            "2>/dev/null",
            "&>/dev/null",
        }
    ]


def _safe_auto_git_command(parts: list[str]) -> bool:
    parsed = _git_subcommand(parts)
    if parsed is None:
        return False
    action, remaining = parsed
    if _contains_git_output_redirect_flag(remaining):
        return False
    if action not in {
        "status",
        "log",
        "diff",
        "show",
        "blame",
        "grep",
        "rev-parse",
        "ls-files",
        "ls-tree",
        "cat-file",
        "merge-base",
        "for-each-ref",
    }:
        return False
    return all(_safe_shell_arg(part) for part in remaining)


def _contains_git_output_redirect_flag(parts: list[str]) -> bool:
    return any(part == "--output" or part.startswith("--output=") for part in parts)


def _git_subcommand(parts: list[str]) -> tuple[str, list[str]] | None:
    if len(parts) < 2 or parts[0] != "git":
        return None
    index = 1
    while index < len(parts):
        part = parts[index]
        if part == "-C":
            if index + 1 >= len(parts) or not _safe_shell_arg(parts[index + 1]):
                return None
            index += 2
            continue
        if part.startswith("-C") and len(part) > 2:
            if not _safe_shell_arg(part[2:]):
                return None
            index += 1
            continue
        if part in {"--no-pager", "--paginate", "--no-optional-locks"}:
            index += 1
            continue
        if part in {"-c", "--git-dir", "--work-tree", "--namespace"}:
            if index + 1 >= len(parts) or not _safe_shell_arg(parts[index + 1]):
                return None
            index += 2
            continue
        if part.startswith(("-c", "--git-dir=", "--work-tree=", "--namespace=")):
            index += 1
            continue
        if part.startswith("-"):
            return None
        return part, parts[index + 1 :]
    return None


def _safe_auto_gh_command(parts: list[str]) -> bool:
    if parts == ["gh", "auth", "status"]:
        return True
    if len(parts) < 3:
        return False
    group, action = parts[1], parts[2]
    safe = {
        ("pr", "checks"),
        ("pr", "diff"),
        ("pr", "list"),
        ("pr", "status"),
        ("pr", "view"),
        ("repo", "view"),
        ("run", "list"),
        ("run", "view"),
        ("search", "prs"),
        ("issue", "list"),
        ("issue", "view"),
    }
    if (group, action) not in safe:
        return False
    return all(_safe_shell_arg(part) for part in parts[3:])


def _safe_auto_sed_command(parts: list[str]) -> bool:
    if "-i" in parts or any(part.startswith("-i") and part != "-i" for part in parts):
        return False
    return (
        len(parts) >= 2 and "-n" in parts[1:] and all(_safe_shell_arg(part) for part in parts[1:])
    )


def _safe_shell_arg(value: str) -> bool:
    return not any(
        marker in value for marker in (";", "&&", "||", "|", ">", "<", "&", "$(", "`", "\n", "\r")
    )
