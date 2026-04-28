from __future__ import annotations

import re

from agent_harness.models import AgentTaskKind, AssignmentMode, WorkRequest
from agent_harness.team import normalize_handle

ANYONE_WORDS = ("somebody", "someone", "anyone", "any agent", "whoever")
TASK_VERBS = ("do", "handle", "take", "work on", "start", "pick up", "review")
BOT_MENTION_RE = re.compile(r"^\s*<@[A-Z0-9]+>\s*[:,]?\s*")
PR_URL_RE = re.compile(r"https://github\.com/[^\s>/]+/[^\s>/]+/pull/\d+[^\s>]*")


def parse_work_request(text: str, known_handles: list[str] | tuple[str, ...]) -> WorkRequest | None:
    cleaned = BOT_MENTION_RE.sub("", _collapse_spaces(text))
    if not cleaned:
        return None
    anyone = _parse_anyone_request(cleaned, known_handles)
    if anyone:
        return anyone
    return _parse_specific_request(cleaned, known_handles)


def parse_lightweight_handles(text: str) -> list[str]:
    handles: list[str] = []
    for match in re.finditer(r"(?<![\w.-])@([a-zA-Z][a-zA-Z0-9_-]{1,31})\b", text):
        handle = normalize_handle(match.group(1))
        if handle not in handles:
            handles.append(handle)
    return handles


def _parse_anyone_request(
    text: str, known_handles: list[str] | tuple[str, ...]
) -> WorkRequest | None:
    anyone_pattern = "|".join(re.escape(word) for word in ANYONE_WORDS)
    verb_pattern = "|".join(re.escape(verb) for verb in TASK_VERBS)
    patterns = [
        (
            rf"^(?:please\s+)?(?:{anyone_pattern})\s+(?:please\s+)?"
            rf"(?P<verb>{verb_pattern})\s+(?P<prompt>.+)$"
        ),
        (
            rf"^(?:please\s+)?(?:{anyone_pattern})\s+can\s+"
            rf"(?P<verb>{verb_pattern})\s+(?P<prompt>.+)$"
        ),
        rf"^(?:please\s+)?(?:{anyone_pattern})\s+(?P<prompt>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            verb = match.groupdict().get("verb") or ""
            prompt = _clean_prompt(match.group("prompt"))
            if prompt:
                return _work_request(
                    prompt=_command_prompt(verb, prompt),
                    assignment_mode=AssignmentMode.ANYONE,
                    verb=verb or _infer_verb(prompt),
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
        patterns = [
            (
                rf"^(?:please\s+)?{handle_pattern}\b\s*[:,]?\s*(?:please\s+)?"
                rf"(?P<verb>{verb_pattern})\s+(?P<prompt>.+)$"
            ),
            rf"^(?:please\s+)?ask\s+{handle_pattern}\b\s+to\s+(?P<prompt>.+)$",
            (
                rf"^(?:can|could)\s+{handle_pattern}\b\s+(?:please\s+)?"
                rf"(?P<verb>{verb_pattern})\s+(?P<prompt>.+)$"
            ),
            rf"^(?:please\s+)?{handle_pattern}\b\s*[:,]?\s+(?P<prompt>.+)$",
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
