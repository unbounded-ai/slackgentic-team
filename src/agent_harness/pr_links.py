from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from agent_harness.models import PR_URL_METADATA_KEY, PR_URLS_METADATA_KEY

GITHUB_PR_URL_RE = re.compile(
    r"https://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+)/"
    r"pull/"
    r"(?P<number>\d+)"
    r"(?=$|[^\w-])"
)


def extract_github_pr_urls(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    urls: list[str] = []
    for match in GITHUB_PR_URL_RE.finditer(text):
        url = match.group(0)
        if url not in urls:
            urls.append(url)
    return tuple(urls)


def pr_urls_from_metadata(metadata: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    single = metadata.get(PR_URL_METADATA_KEY)
    if isinstance(single, str):
        values.append(single)
    many = metadata.get(PR_URLS_METADATA_KEY)
    if isinstance(many, str):
        values.append(many)
    elif isinstance(many, Iterable):
        values.extend(item for item in many if isinstance(item, str))
    return merge_pr_urls((), *values)


def merge_pr_urls(existing: Iterable[str], *texts_or_urls: str | None) -> tuple[str, ...]:
    merged: list[str] = []
    for url in existing:
        if isinstance(url, str) and url not in merged:
            merged.append(url)
    for value in texts_or_urls:
        for url in extract_github_pr_urls(value):
            if url not in merged:
                merged.append(url)
    return tuple(merged)


def metadata_with_pr_urls(
    metadata: Mapping[str, Any],
    *texts_or_urls: str | None,
) -> dict[str, Any]:
    updated = dict(metadata)
    urls = merge_pr_urls(pr_urls_from_metadata(metadata), *texts_or_urls)
    if not urls:
        return updated
    updated[PR_URL_METADATA_KEY] = urls[0]
    updated[PR_URLS_METADATA_KEY] = list(urls)
    return updated


def slack_pr_link(url: str) -> str:
    match = GITHUB_PR_URL_RE.fullmatch(url)
    if not match:
        return url
    label = f"{match.group('owner')}/{match.group('repo')}#{match.group('number')}"
    return f"<{url}|{label}>"


def slack_pr_links(urls: Iterable[str], *, limit: int | None = None) -> str:
    deduped = list(dict.fromkeys(urls))
    shown = deduped if limit is None else deduped[:limit]
    text = ", ".join(slack_pr_link(url) for url in shown)
    if limit is not None and len(deduped) > limit:
        text = f"{text}, +{len(deduped) - limit} more"
    return text
