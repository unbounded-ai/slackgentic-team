"""Structural Bash policy for Claude safe-auto permissions."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class BashSegment:
    argv: tuple[str, ...]
    redirects: tuple[str, ...] = ()


@dataclass(frozen=True)
class BashCommandDecision:
    command: str
    safe: bool
    reason: str
    segments: tuple[BashSegment, ...] = ()
    operators: tuple[str, ...] = ()
    safe_allowed_tools: tuple[str, ...] = ()
    approval_allowed_tools: tuple[str, ...] = ()


_ALLOWED_OPERATORS = {"&&", "|", ";"}
_REJECTED_OPERATORS = {"||", "&"}
_REDIRECT_OPERATORS = {">", ">>", "<", "<<", "<>", ">|", ">&", "<&", "&>", "<(", ">(", "<<<"}
_SAFE_SIMPLE_EXECUTABLE_TOOLS: dict[str, tuple[str, ...]] = {
    "ls": ("Bash(ls:*)",),
    "cat": ("Bash(cat:*)",),
    "head": ("Bash(head:*)",),
    "tail": ("Bash(tail:*)",),
    "nl": ("Bash(nl:*)",),
    "wc": ("Bash(wc:*)",),
    "rg": ("Bash(rg:*)",),
    "grep": ("Bash(grep:*)",),
    "pwd": ("Bash(pwd)",),
    "which": ("Bash(which:*)",),
}
_GIT_SAFE_SUBCOMMAND_TOOLS: dict[str, tuple[str, ...]] = {
    "status": ("Bash(git status)", "Bash(git status:*)", "Bash(git status *)"),
    "log": ("Bash(git log:*)", "Bash(git log *)"),
    "diff": ("Bash(git diff:*)", "Bash(git diff *)"),
    "show": ("Bash(git show:*)", "Bash(git show *)"),
    "blame": ("Bash(git blame:*)", "Bash(git blame *)"),
    "grep": ("Bash(git grep:*)", "Bash(git grep *)"),
    "rev-parse": ("Bash(git rev-parse:*)", "Bash(git rev-parse *)"),
    "ls-files": ("Bash(git ls-files:*)", "Bash(git ls-files *)"),
    "ls-tree": ("Bash(git ls-tree:*)", "Bash(git ls-tree *)"),
    "cat-file": ("Bash(git cat-file:*)", "Bash(git cat-file *)"),
    "merge-base": ("Bash(git merge-base:*)", "Bash(git merge-base *)"),
    "for-each-ref": ("Bash(git for-each-ref:*)", "Bash(git for-each-ref *)"),
}
_GIT_PR_AUTHORING_SUBCOMMAND_TOOLS: dict[str, tuple[str, ...]] = {
    "add": ("Bash(git add:*)", "Bash(git add *)"),
    "commit": ("Bash(git commit:*)", "Bash(git commit *)"),
}
_GIT_PULL_TOOLS = ("Bash(git pull:*)", "Bash(git pull *)")
_GIT_CONFIG_READ_TOOLS = ("Bash(git config user.name)", "Bash(git config user.email)")
_GH_SAFE_COMMAND_TOOLS: dict[tuple[str, str], tuple[str, ...]] = {
    ("pr", "checks"): ("Bash(gh pr checks:*)", "Bash(gh pr checks *)"),
    ("pr", "create"): ("Bash(gh pr create:*)", "Bash(gh pr create *)"),
    ("pr", "diff"): ("Bash(gh pr diff:*)", "Bash(gh pr diff *)"),
    ("pr", "list"): ("Bash(gh pr list:*)", "Bash(gh pr list *)"),
    ("pr", "status"): ("Bash(gh pr status:*)", "Bash(gh pr status *)"),
    ("pr", "view"): ("Bash(gh pr view:*)", "Bash(gh pr view *)"),
    ("repo", "view"): ("Bash(gh repo view:*)", "Bash(gh repo view *)"),
    ("run", "list"): ("Bash(gh run list:*)", "Bash(gh run list *)"),
    ("run", "view"): ("Bash(gh run view:*)", "Bash(gh run view *)"),
    ("search", "prs"): ("Bash(gh search prs:*)", "Bash(gh search prs *)"),
    ("issue", "list"): ("Bash(gh issue list:*)", "Bash(gh issue list *)"),
    ("issue", "view"): ("Bash(gh issue view:*)", "Bash(gh issue view *)"),
}
_SED_SAFE_TOOLS = ("Bash(sed -n:*)", "Bash(sed -n *)")
BASH_SAFE_AUTO_ALLOWED_TOOLS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *[tool for tools in _GIT_SAFE_SUBCOMMAND_TOOLS.values() for tool in tools],
            *[tool for tools in _GIT_PR_AUTHORING_SUBCOMMAND_TOOLS.values() for tool in tools],
            *_GIT_PULL_TOOLS,
            *_GIT_CONFIG_READ_TOOLS,
            "Bash(gh auth status)",
            *[tool for tools in _GH_SAFE_COMMAND_TOOLS.values() for tool in tools],
            *[tool for tools in _SAFE_SIMPLE_EXECUTABLE_TOOLS.values() for tool in tools],
            *_SED_SAFE_TOOLS,
        )
    )
)
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def classify_bash_command(command: str) -> BashCommandDecision:
    command = command.strip()
    if not command:
        return _decision(command, False, "empty command")
    unparsed_git_commit = _classify_unparsed_git_commit(command)
    if unparsed_git_commit is not None:
        return unparsed_git_commit
    if "\n" in command or "\r" in command:
        return _decision(command, False, "multiline command")
    if "$(" in command or "`" in command:
        return _decision(command, False, "command substitution")

    tokens = _shell_tokens(command)
    if isinstance(tokens, str):
        return _decision(command, False, tokens)

    parsed = _parse_segments(tokens)
    if isinstance(parsed, str):
        return _decision(command, False, parsed)
    segments, operators = parsed
    if not segments:
        return _decision(command, False, "empty command")

    for segment in segments:
        reason = _unsafe_segment_reason(segment.argv)
        if reason:
            return _decision(
                command,
                False,
                reason,
                segments=segments,
                operators=operators,
                approval_allowed_tools=_manual_approval_tools(command, segments, operators),
            )

    return _decision(
        command,
        True,
        "safe read-only command",
        segments=segments,
        operators=operators,
        safe_allowed_tools=_safe_allowed_tools(command, segments, operators),
        approval_allowed_tools=_approval_tools_for_safe_command(command, segments, operators),
    )


def allowed_bash_tools_for_command(command: str) -> tuple[str, ...]:
    return classify_bash_command(command).approval_allowed_tools


def allowed_bash_session_tools_for_command(command: str) -> tuple[str, ...]:
    decision = classify_bash_command(command)
    if not decision.segments:
        return ()
    executable = shlex.join(decision.segments[0].argv[:1])
    if any(value in executable for value in (")", ",", "\n", "\r")):
        return ()
    return (f"Bash({executable}:*)", f"Bash({executable} *)")


def _decision(
    command: str,
    safe: bool,
    reason: str,
    *,
    segments: tuple[BashSegment, ...] = (),
    operators: tuple[str, ...] = (),
    safe_allowed_tools: tuple[str, ...] = (),
    approval_allowed_tools: tuple[str, ...] | None = None,
) -> BashCommandDecision:
    if approval_allowed_tools is None:
        approval_allowed_tools = _manual_approval_tools(command, segments, operators)
    return BashCommandDecision(
        command=command,
        safe=safe,
        reason=reason,
        segments=segments,
        operators=operators,
        safe_allowed_tools=safe_allowed_tools,
        approval_allowed_tools=approval_allowed_tools,
    )


def _shell_tokens(command: str) -> list[str] | str:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return "invalid shell quoting"


def _parse_segments(tokens: list[str]) -> tuple[tuple[BashSegment, ...], tuple[str, ...]] | str:
    segments: list[BashSegment] = []
    operators: list[str] = []
    current: list[str] = []
    redirects: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _ALLOWED_OPERATORS:
            if not current:
                return f"empty segment before {token}"
            segments.append(BashSegment(tuple(current), tuple(redirects)))
            operators.append(token)
            current = []
            redirects = []
            index += 1
            continue
        if token in _REJECTED_OPERATORS:
            return f"unsupported operator {token}"

        redirection = _parse_redirection(tokens, index)
        if isinstance(redirection, str):
            return redirection
        if redirection is not None:
            redirect, next_index = redirection
            redirects.append(redirect)
            index = next_index
            continue
        if token in _REDIRECT_OPERATORS:
            return f"unsupported redirection {token}"
        if _word_has_shell_control(token):
            return "quoted or embedded shell operator in argument"
        if not current and _ASSIGNMENT_RE.match(token):
            return "environment assignment"
        current.append(token)
        index += 1

    if not current:
        return "empty trailing segment"
    segments.append(BashSegment(tuple(current), tuple(redirects)))
    return tuple(segments), tuple(operators)


def _parse_redirection(tokens: list[str], index: int) -> tuple[str, int] | str | None:
    token = tokens[index]
    fd = ""
    op = ""
    target_index = index + 1
    if (
        token.isdigit()
        and target_index < len(tokens)
        and tokens[target_index] in _REDIRECT_OPERATORS
    ):
        fd = token
        op = tokens[target_index]
        target_index += 1
    elif token in _REDIRECT_OPERATORS:
        op = token
    else:
        return None

    if target_index >= len(tokens):
        return f"missing target for redirection {op}"
    target = tokens[target_index]
    if (
        target in _ALLOWED_OPERATORS
        or target in _REJECTED_OPERATORS
        or target in _REDIRECT_OPERATORS
    ):
        return f"invalid target for redirection {op}"
    if _word_has_shell_control(target):
        return "unsafe redirection target"
    redirect = _safe_redirection(fd, op, target)
    if redirect is None:
        return f"unsupported redirection {fd}{op}{target}"
    return redirect, target_index + 1


def _safe_redirection(fd: str, op: str, target: str) -> str | None:
    if op == ">&" and ((fd, target) == ("2", "1") or (fd, target) == ("1", "2")):
        return f"{fd}{op}{target}"
    if op == ">" and target == "/dev/null" and fd in {"", "1", "2"}:
        return f"{fd}{op}{target}"
    if op == "&>" and target == "/dev/null" and not fd:
        return f"{op}{target}"
    return None


def _word_has_shell_control(value: str) -> bool:
    return any(
        marker in value
        for marker in (";", "&&", "||", "|", ">", "<", "&", "$(", "`", "(", ")", "\n", "\r")
    )


def _unsafe_segment_reason(parts: tuple[str, ...]) -> str | None:
    if not parts:
        return "empty segment"
    executable = parts[0]
    if executable == "cd":
        return None if len(parts) == 2 and _safe_shell_arg(parts[1]) else "unsafe cd command"
    if executable == "git":
        return _unsafe_git_reason(parts)
    if executable == "gh":
        return _unsafe_gh_reason(parts)
    if executable in _SAFE_SIMPLE_EXECUTABLE_TOOLS:
        return None if all(_safe_shell_arg(part) for part in parts[1:]) else "unsafe argument"
    if executable == "sed":
        return _unsafe_sed_reason(parts)
    return f"unsupported executable {executable}"


def _unsafe_git_reason(parts: tuple[str, ...]) -> str | None:
    parsed = _git_subcommand(parts)
    if parsed is None:
        return "unsupported git invocation"
    action, remaining = parsed
    if _contains_git_output_redirect_flag(remaining):
        return "git output redirect flag"
    if action == "pull":
        return _unsafe_git_pull_reason(remaining)
    if action == "config":
        return _unsafe_git_config_reason(remaining)
    if action == "add":
        return _unsafe_git_add_reason(remaining)
    if action == "commit":
        return _unsafe_git_commit_reason(remaining)
    if action not in _GIT_SAFE_SUBCOMMAND_TOOLS:
        return f"unsupported git subcommand {action}"
    if not all(_safe_shell_arg(part) for part in remaining):
        return "unsafe git argument"
    return None


def _contains_git_output_redirect_flag(parts: tuple[str, ...] | list[str]) -> bool:
    return any(part == "--output" or part.startswith("--output=") for part in parts)


def _unsafe_git_pull_reason(parts: tuple[str, ...]) -> str | None:
    for part in parts:
        if not _safe_shell_arg(part):
            return "unsafe git pull argument"
    return None


def _unsafe_git_config_reason(parts: tuple[str, ...]) -> str | None:
    if parts in {("user.name",), ("user.email",)}:
        return None
    if len(parts) == 2 and parts[0] == "--get" and parts[1] in {"user.name", "user.email"}:
        return None
    return "unsupported git config lookup"


def _unsafe_git_add_reason(parts: tuple[str, ...]) -> str | None:
    if not parts:
        return "git add without pathspec"
    if not all(_safe_shell_arg(part) for part in parts):
        return "unsafe git add argument"
    return None


def _unsafe_git_commit_reason(parts: tuple[str, ...]) -> str | None:
    if not parts:
        return "git commit without arguments"
    for part in parts:
        if _safe_shell_arg(part) or _safe_git_commit_message_arg(part):
            continue
        return "unsafe git commit argument"
    return None


def _safe_git_commit_message_arg(value: str) -> bool:
    opener = "$(cat <<'EOF'\n"
    terminator = "\nEOF\n)"
    return (
        value.startswith(opener)
        and value.endswith(terminator)
        and value.count(terminator) == 1
        and "\r" not in value
        and "`" not in value
    )


def _classify_unparsed_git_commit(command: str) -> BashCommandDecision | None:
    if "commit" not in command or "$(cat <<'EOF'\n" not in command:
        return None
    tokens = _shell_tokens(command)
    if isinstance(tokens, str):
        return _decision(command, False, tokens)

    parsed = _git_commit_segments_from_tokens(tokens)
    if isinstance(parsed, str):
        return _decision(command, False, parsed)
    if parsed is None:
        return None
    segments, operators = parsed
    git_segment = segments[-1]
    parsed_git = _git_subcommand(git_segment.argv)
    if parsed_git is None:
        return None
    action, remaining = parsed_git
    if action != "commit":
        return None
    reason = _unsafe_git_commit_reason(remaining)
    if reason:
        return _decision(command, False, reason, segments=segments, operators=operators)
    return _decision(
        command,
        True,
        "safe git commit command",
        segments=segments,
        operators=operators,
        safe_allowed_tools=_safe_allowed_tools(command, segments, operators),
        approval_allowed_tools=_approval_tools_for_safe_command(command, segments, operators),
    )


def _git_commit_segments_from_tokens(
    tokens: list[str],
) -> tuple[tuple[BashSegment, ...], tuple[str, ...]] | str | None:
    if not tokens:
        return "empty command"
    if tokens[0] == "git":
        if any(token in _ALLOWED_OPERATORS or token in _REJECTED_OPERATORS for token in tokens[1:]):
            return "unsupported git commit shell structure"
        return (BashSegment(tuple(tokens)),), ()
    if (
        len(tokens) >= 5
        and tokens[0] == "cd"
        and _safe_shell_arg(tokens[1])
        and tokens[2] == "&&"
        and tokens[3] == "git"
    ):
        git_tokens = tokens[3:]
        if any(
            token in _ALLOWED_OPERATORS or token in _REJECTED_OPERATORS for token in git_tokens[1:]
        ):
            return "unsupported git commit shell structure"
        return (
            BashSegment(("cd", tokens[1])),
            BashSegment(tuple(git_tokens)),
        ), ("&&",)
    return None


def _git_subcommand(parts: tuple[str, ...]) -> tuple[str, tuple[str, ...]] | None:
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


def _unsafe_gh_reason(parts: tuple[str, ...]) -> str | None:
    if parts == ("gh", "auth", "status"):
        return None
    if len(parts) < 3:
        return "unsupported gh invocation"
    group, action = parts[1], parts[2]
    if (group, action) not in _GH_SAFE_COMMAND_TOOLS:
        return f"unsupported gh command {group} {action}"
    if not all(_safe_shell_arg(part) for part in parts[3:]):
        return "unsafe gh argument"
    return None


def _unsafe_sed_reason(parts: tuple[str, ...]) -> str | None:
    if any(part in {"-i", "-I"} or part.startswith(("-i", "-I")) for part in parts):
        return "sed edit flag"
    if len(parts) < 2 or "-n" not in parts[1:]:
        return "sed without -n"
    if not all(_safe_shell_arg(part) for part in parts[1:]):
        return "unsafe sed argument"
    return None


def _safe_shell_arg(value: str) -> bool:
    return not _word_has_shell_control(value)


def _safe_allowed_tools(
    command: str,
    segments: tuple[BashSegment, ...],
    operators: tuple[str, ...],
) -> tuple[str, ...]:
    if len(segments) == 1 and not operators:
        return _allowed_tools_for_simple_safe_command(command, segments[0].argv)
    cd_prefixed_tools = _cd_prefixed_safe_tools(command, segments, operators)
    if cd_prefixed_tools:
        return cd_prefixed_tools
    return (f"Bash({command})",) if _exact_bash_permission_allowed(command) else ()


def _approval_tools_for_safe_command(
    command: str,
    segments: tuple[BashSegment, ...],
    operators: tuple[str, ...],
) -> tuple[str, ...]:
    if len(segments) == 1 and not operators:
        return _allowed_tools_for_simple_safe_command(command, segments[0].argv)
    cd_prefixed_tools = _cd_prefixed_safe_tools(command, segments, operators)
    if cd_prefixed_tools:
        return cd_prefixed_tools
    exact = (f"Bash({command})",) if _exact_bash_permission_allowed(command) else ()
    return tuple(dict.fromkeys(exact))


def _cd_prefixed_safe_tools(
    command: str,
    segments: tuple[BashSegment, ...],
    operators: tuple[str, ...],
) -> tuple[str, ...]:
    if len(segments) != 2 or operators != ("&&",):
        return ()
    cd_segment, command_segment = segments
    if cd_segment.argv[:1] != ("cd",) or cd_segment.redirects or command_segment.redirects:
        return ()
    tools: list[str] = []
    if _exact_bash_permission_allowed(command):
        tools.append(f"Bash({command})")
    tools.extend(
        _allowed_tools_for_simple_safe_command(
            shlex.join(command_segment.argv),
            command_segment.argv,
            include_exact=False,
        )
    )
    return tuple(dict.fromkeys(tools))


def _manual_approval_tools(
    command: str,
    segments: tuple[BashSegment, ...],
    operators: tuple[str, ...],
) -> tuple[str, ...]:
    command = command.strip()
    if not command:
        return ()
    if _exact_bash_permission_allowed(command):
        return (f"Bash({command})",)
    if len(segments) == 1 and not operators:
        return _generic_bash_tools(command, list(segments[0].argv))
    if not segments:
        try:
            return _generic_bash_tools(command, shlex.split(command))
        except ValueError:
            return ()
    return ()


def _allowed_tools_for_simple_safe_command(
    command: str,
    parts: tuple[str, ...],
    *,
    include_exact: bool = True,
) -> tuple[str, ...]:
    tools: list[str] = []
    exact_allowed = _exact_bash_permission_allowed(command)
    if parts and parts[0] == "gh":
        prefix_tools = _allowed_gh_bash_tools(parts, include_generic=not exact_allowed)
    elif parts and parts[0] == "git":
        prefix_tools = _allowed_git_bash_tools(parts)
    elif parts and parts[0] in _SAFE_SIMPLE_EXECUTABLE_TOOLS:
        prefix_tools = _SAFE_SIMPLE_EXECUTABLE_TOOLS[parts[0]]
    elif parts and parts[0] == "sed":
        prefix_tools = _SED_SAFE_TOOLS
    else:
        prefix_tools = ()
    if include_exact and exact_allowed:
        tools.append(f"Bash({command})")
    if prefix_tools:
        tools.extend(prefix_tools)
    elif not exact_allowed:
        tools.extend(_generic_bash_tools(command, list(parts)))
    return tuple(dict.fromkeys(tools))


def _exact_bash_permission_allowed(command: str) -> bool:
    return not any(value in command for value in (")", ",", "\n", "\r"))


def _generic_bash_tools(command: str, parts: list[str]) -> tuple[str, ...]:
    git_prefix = _git_action_prefix(tuple(parts))
    if git_prefix is not None and command.startswith(git_prefix):
        return (f"Bash({git_prefix}:*)", f"Bash({git_prefix} *)")
    prefix = _generic_bash_prefix(command, parts)
    if prefix is None:
        return ()
    return (f"Bash({prefix}:*)", f"Bash({prefix} *)")


def _generic_bash_prefix(command: str, parts: list[str]) -> str | None:
    for prefix_len in range(len(parts), 1, -1):
        prefix = shlex.join(parts[:prefix_len])
        if ")" in prefix or "," in prefix or "\n" in prefix or "\r" in prefix:
            continue
        if command.startswith(prefix):
            return prefix
    return None


def _allowed_gh_bash_tools(
    parts: tuple[str, ...],
    *,
    include_generic: bool = True,
) -> tuple[str, ...]:
    if parts == ("gh", "auth", "status"):
        return ("Bash(gh auth status)",)
    if len(parts) < 3:
        return ()
    group, action = parts[1], parts[2]
    patterns = list(_GH_SAFE_COMMAND_TOOLS.get((group, action), ()))
    prefix = shlex.join(parts[:3])
    if include_generic and ")" not in prefix and "," not in prefix:
        patterns.extend((f"Bash({prefix}:*)", f"Bash({prefix} *)"))
    return tuple(dict.fromkeys(patterns))


def _allowed_git_bash_tools(parts: tuple[str, ...]) -> tuple[str, ...]:
    parsed = _git_subcommand(parts)
    if parsed is not None:
        action, remaining = parsed
        if action == "pull" and _unsafe_git_pull_reason(remaining) is None:
            prefix = _git_action_prefix(parts)
            tools = list(_GIT_PULL_TOOLS)
            if prefix and prefix != "git pull":
                tools.extend((f"Bash({prefix}:*)", f"Bash({prefix} *)"))
            return tuple(dict.fromkeys(tools))
        if action == "config" and _unsafe_git_config_reason(remaining) is None:
            key = remaining[-1]
            return (f"Bash(git config {key})",)
        if action in _GIT_PR_AUTHORING_SUBCOMMAND_TOOLS:
            if action == "add":
                reason = _unsafe_git_add_reason(remaining)
            else:
                reason = _unsafe_git_commit_reason(remaining)
            if reason is None:
                prefix = _git_action_prefix(parts)
                tools = list(_GIT_PR_AUTHORING_SUBCOMMAND_TOOLS[action])
                if prefix and prefix != f"git {action}":
                    tools.extend((f"Bash({prefix}:*)", f"Bash({prefix} *)"))
                return tuple(dict.fromkeys(tools))
    git_command = _git_read_only_command(parts)
    if git_command is None:
        return ()
    action, prefix = git_command
    patterns = list(_GIT_SAFE_SUBCOMMAND_TOOLS.get(action, ()))
    if ")" not in prefix and "," not in prefix:
        patterns.extend((f"Bash({prefix}:*)", f"Bash({prefix} *)"))
    return tuple(dict.fromkeys(patterns))


def _git_read_only_command(parts: tuple[str, ...]) -> tuple[str, str] | None:
    prefix = _git_action_prefix(parts)
    if prefix is None:
        return None
    parsed = _git_subcommand(parts)
    if parsed is None:
        return None
    action, _ = parsed
    if action not in _GIT_SAFE_SUBCOMMAND_TOOLS:
        return None
    return action, prefix


def _git_action_prefix(parts: tuple[str, ...]) -> str | None:
    parsed = _git_subcommand(parts)
    if parsed is None:
        return None
    _, remaining = parsed
    prefix = shlex.join(parts[: len(parts) - len(remaining)])
    if ")" in prefix or "," in prefix or "\n" in prefix or "\r" in prefix:
        return None
    return prefix
