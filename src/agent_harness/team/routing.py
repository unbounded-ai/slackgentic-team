from __future__ import annotations

import re
from dataclasses import replace

from agent_harness.models import AgentTaskKind, AssignmentMode, TeamAgent, WorkRequest
from agent_harness.team import normalize_handle

ANYONE_WORDS = ("somebody", "someone", "anyone", "any agent", "whoever")
TASK_VERBS = ("do", "handle", "take", "work on", "start", "pick up", "review")
BOT_MENTION_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*")
AGENT_MENTION_RE = re.compile(r"(?<![\w.-])@([a-zA-Z][a-zA-Z0-9_-]{1,31})\b")
PR_URL_RE = re.compile(r"https://github\.com/[^\s>/]+/[^\s>/]+/pull/\d+[^\s>]*")
DANGEROUS_MODE_TAG_RE = re.compile(r"(?<![\w.-])#dangerous-mode\b", re.IGNORECASE)


def parse_work_request(text: str, known_handles: list[str] | tuple[str, ...]) -> WorkRequest | None:
    stripped_text, dangerous_mode = strip_dangerous_mode_tag(text)
    cleaned = BOT_MENTION_RE.sub("", _collapse_spaces(stripped_text))
    if not cleaned:
        return None
    anyone = _parse_anyone_request(cleaned, known_handles)
    if anyone:
        return _with_dangerous_mode(anyone, dangerous_mode)
    return _with_dangerous_mode(_parse_specific_request(cleaned, known_handles), dangerous_mode)


def strip_dangerous_mode_tag(text: str) -> tuple[str, bool]:
    if not DANGEROUS_MODE_TAG_RE.search(text):
        return text, False
    stripped = DANGEROUS_MODE_TAG_RE.sub("", text)
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n[ \t]+", "\n", stripped)
    return stripped.strip(), True


def parse_lightweight_handles(text: str) -> list[str]:
    handles: list[str] = []
    for match in AGENT_MENTION_RE.finditer(text):
        handle = normalize_handle(match.group(1))
        if handle not in handles:
            handles.append(handle)
    return handles


def canonicalize_agent_mentions(text: str, agents: list[TeamAgent] | tuple[TeamAgent, ...]) -> str:
    if not agents:
        return text

    def replace(match: re.Match[str]) -> str:
        typed = match.group(1)
        canonical = canonical_agent_handle(typed, agents)
        if not canonical or canonical == typed.lower():
            return match.group(0)
        return f"@{canonical}"

    return AGENT_MENTION_RE.sub(replace, text)


def canonical_agent_handle(
    mention: str,
    agents: list[TeamAgent] | tuple[TeamAgent, ...],
) -> str | None:
    normalized = normalize_handle(mention)
    aliases = agent_handle_aliases(agents)
    canonical = aliases.get(normalized)
    if canonical:
        return canonical
    fuzzy_matches = [
        normalize_handle(agent.handle)
        for agent in agents
        if _is_single_edit_typo(normalized, normalize_handle(agent.handle))
    ]
    distinct_matches = sorted(set(fuzzy_matches))
    if len(distinct_matches) == 1:
        return distinct_matches[0]
    return None


def agent_handle_aliases(
    agents: list[TeamAgent] | tuple[TeamAgent, ...],
) -> dict[str, str]:
    exact_handles = {normalize_handle(agent.handle) for agent in agents}
    aliases: dict[str, str] = {handle: handle for handle in exact_handles}
    candidates: dict[str, set[str]] = {}
    for agent in agents:
        handle = normalize_handle(agent.handle)
        for alias in _agent_alias_candidates(agent):
            if alias == handle or alias in exact_handles:
                continue
            candidates.setdefault(alias, set()).add(handle)
    for alias, handles in candidates.items():
        if len(handles) == 1:
            aliases[alias] = next(iter(handles))
    return aliases


def _agent_alias_candidates(agent: TeamAgent) -> set[str]:
    candidates = {normalize_handle(agent.handle)}
    name_parts = re.findall(r"[A-Za-z][A-Za-z0-9_-]*", agent.full_name)
    if name_parts:
        first_name = name_parts[0].lower()
        if len(first_name) >= 2:
            candidates.add(first_name)
    return {candidate for candidate in candidates if _is_valid_handle(candidate)}


def _is_valid_handle(value: str) -> bool:
    try:
        normalize_handle(value)
    except ValueError:
        return False
    return True


def _is_single_edit_typo(typed: str, handle: str) -> bool:
    if typed == handle or min(len(typed), len(handle)) < 3:
        return False
    if abs(len(typed) - len(handle)) > 1:
        return False
    index = 0
    other_index = 0
    edits = 0
    while index < len(typed) and other_index < len(handle):
        if typed[index] == handle[other_index]:
            index += 1
            other_index += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(typed) == len(handle):
            index += 1
            other_index += 1
        elif len(typed) < len(handle):
            other_index += 1
        else:
            index += 1
    if index < len(typed) or other_index < len(handle):
        edits += 1
    return edits == 1


def _parse_anyone_request(
    text: str, known_handles: list[str] | tuple[str, ...]
) -> WorkRequest | None:
    anyone_pattern = "|".join(re.escape(word) for word in ANYONE_WORDS)
    patterns = [
        (
            rf"^(?:please\s+)?(?:{anyone_pattern})\s+can\s+"
            rf"(?:please\s+)?(?P<prompt>.+)$"
        ),
        rf"^(?:please\s+)?(?:{anyone_pattern})\s+(?:please\s+)?(?P<prompt>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            prompt = _clean_prompt(match.group("prompt"))
            if prompt:
                return _work_request(
                    prompt=prompt,
                    assignment_mode=AssignmentMode.ANYONE,
                    verb=_infer_verb(prompt),
                    known_handles=known_handles,
                )
    return None


def _parse_specific_request(
    text: str,
    known_handles: list[str] | tuple[str, ...],
) -> WorkRequest | None:
    handles = sorted((normalize_handle(handle) for handle in known_handles), key=len, reverse=True)
    for handle in handles:
        handle_pattern = rf"@?{re.escape(handle)}"
        verb_pattern = "|".join(re.escape(verb) for verb in TASK_VERBS)
        separator_pattern = r"(?::|,|-|\u2013|\u2014)"
        patterns = [
            (
                rf"^(?:please\s+)?{handle_pattern}\b\s*{separator_pattern}?\s*(?:please\s+)?"
                rf"(?P<verb>{verb_pattern})\s+(?P<prompt>.+)$"
            ),
            rf"^(?:please\s+)?ask\s+{handle_pattern}\b\s+to\s+(?P<prompt>.+)$",
            (
                rf"^(?:can|could)\s+{handle_pattern}\b\s+(?:please\s+)?"
                rf"(?P<verb>{verb_pattern})\s+(?P<prompt>.+)$"
            ),
            (
                rf"^(?:please\s+)?{handle_pattern}\b\s*{separator_pattern}?"
                rf"\s+(?P<prompt>.+)$"
            ),
        ]
        for pattern in patterns:
            match = re.match(pattern, text, flags=re.IGNORECASE)
            if match:
                verb = match.groupdict().get("verb") or ""
                prompt = _clean_prompt(match.group("prompt"))
                if prompt:
                    return _work_request(
                        prompt=_command_prompt(verb, prompt),
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=handle,
                        verb=verb or _infer_verb(prompt),
                        known_handles=handles,
                    )
    return None


def _work_request(
    prompt: str,
    assignment_mode: AssignmentMode,
    verb: str,
    known_handles: list[str] | tuple[str, ...],
    requested_handle: str | None = None,
) -> WorkRequest:
    pr_url = _extract_pr_url(prompt)
    task_kind = _task_kind_for(verb, prompt, pr_url)
    return WorkRequest(
        prompt=prompt,
        assignment_mode=assignment_mode,
        requested_handle=requested_handle,
        task_kind=task_kind,
        author_handle=_extract_author_handle(prompt, known_handles),
        pr_url=pr_url,
    )


def _with_dangerous_mode(
    request: WorkRequest | None,
    dangerous_mode: bool,
) -> WorkRequest | None:
    if request is None or not dangerous_mode:
        return request
    return replace(request, dangerous_mode=True)


def _task_kind_for(verb: str, prompt: str, pr_url: str | None) -> AgentTaskKind:
    normalized = f"{verb} {prompt}".lower()
    if re.search(r"\breview\b", normalized):
        return AgentTaskKind.REVIEW
    return AgentTaskKind.WORK


def _extract_pr_url(prompt: str) -> str | None:
    match = PR_URL_RE.search(prompt)
    return match.group(0) if match else None


def _extract_author_handle(prompt: str, known_handles: list[str] | tuple[str, ...]) -> str | None:
    handles = sorted((normalize_handle(handle) for handle in known_handles), key=len, reverse=True)
    for handle in handles:
        handle_pattern = re.escape(handle)
        patterns = [
            rf"@?{handle_pattern}'?s\s+(?:pr|pull request)",
            rf"(?:by|from|for|authored by)\s+@?{handle_pattern}\b",
        ]
        for pattern in patterns:
            if re.search(pattern, prompt, flags=re.IGNORECASE):
                return handle
    return None


def _clean_prompt(value: str) -> str:
    return value.strip(" \t\n\r.").strip()


def _command_prompt(verb: str, prompt: str) -> str:
    if verb.lower() == "review" and not re.match(r"^review\b", prompt, flags=re.IGNORECASE):
        return f"review {prompt}"
    return prompt


def _infer_verb(prompt: str) -> str:
    first = prompt.split(maxsplit=1)[0].lower() if prompt.split() else "do"
    return first if first in TASK_VERBS else "do"


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
