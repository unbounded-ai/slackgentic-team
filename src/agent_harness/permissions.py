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
import logging
from collections.abc import Mapping

from agent_harness.bash_policy import BASH_SAFE_AUTO_ALLOWED_TOOLS, classify_bash_command
from agent_harness.models import (
    DANGEROUS_MODE_METADATA_KEY,
    DEFAULT_PERMISSION_MODE,
    PERMISSION_MODE_METADATA_KEY,
    AgentTask,
    PermissionMode,
)

LOGGER = logging.getLogger(__name__)

CLAUDE_CHANNEL_PERMISSION_MODE_ENV = "SLACKGENTIC_CLAUDE_PERMISSION_MODE"
CLAUDE_SAFE_AUTO_PERMISSION_FLAG = "acceptEdits"

CLAUDE_SAFE_AUTO_ALLOWED_TOOLS: tuple[str, ...] = (
    "Read",
    "Grep",
    "Glob",
    "WebFetch",
    "WebSearch",
    *BASH_SAFE_AUTO_ALLOWED_TOOLS,
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
    if not command:
        return False
    decision = classify_bash_command(command)
    if not decision.safe:
        LOGGER.info("safe-auto denied Claude Bash permission request: %s", decision.reason)
    return decision.safe


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
    return classify_bash_command(command).safe
