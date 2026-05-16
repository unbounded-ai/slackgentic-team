from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path, *, after_line: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number <= after_line:
                continue
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


def last_jsonl_line_number(path: Path, *, chunk_size: int = 64 * 1024) -> int:
    chunk_size = max(1, chunk_size)
    newline_count = 0
    last_byte = b""
    tail_chunks: list[bytes] = []
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
            last_newline = chunk.rfind(b"\n")
            if last_newline >= 0:
                tail_chunks = [chunk[last_newline + 1 :]]
            else:
                tail_chunks.append(chunk)
    if last_byte != b"\n" and _is_jsonl_record(b"".join(tail_chunks)):
        return newline_count + 1
    return newline_count


def tail_jsonl_records(
    path: Path,
    limit: int = 20,
    *,
    chunk_size: int = 64 * 1024,
    max_bytes: int = 16 * 1024 * 1024,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    chunk_size = max(1, chunk_size)
    max_bytes = max(chunk_size, max_bytes)
    size = path.stat().st_size
    position = size
    remaining = min(size, max_bytes)
    chunks: list[bytes] = []
    with path.open("rb") as handle:
        while position > 0 and remaining > 0:
            read_size = min(chunk_size, position, remaining)
            position -= read_size
            remaining -= read_size
            handle.seek(position)
            chunks.append(handle.read(read_size))
            lines = b"".join(reversed(chunks)).splitlines()
            if position > 0:
                lines = lines[1:]
            if _count_jsonl_records(lines, limit) >= limit:
                break
    if not chunks:
        return []
    lines = b"".join(reversed(chunks)).splitlines()
    if position > 0:
        lines = lines[1:]
    records: list[dict[str, Any]] = []
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
            if len(records) >= limit:
                break
    records.reverse()
    return records


def _count_jsonl_records(lines: list[bytes], limit: int) -> int:
    count = 0
    for line in reversed(lines):
        if _is_jsonl_record(line):
            count += 1
            if count >= limit:
                return count
    return count


def _is_jsonl_record(line: bytes) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    try:
        value = json.loads(stripped.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict)
