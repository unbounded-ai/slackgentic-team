from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield line_number, value


def first_jsonl_record(path: Path) -> tuple[int, dict[str, Any]] | None:
    for item in iter_jsonl(path):
        return item
    return None


def last_jsonl_records(path: Path, limit: int = 20) -> list[tuple[int, dict[str, Any]]]:
    items: list[tuple[int, dict[str, Any]]] = []
    for item in iter_jsonl(path):
        items.append(item)
        if len(items) > limit:
            items.pop(0)
    return items
