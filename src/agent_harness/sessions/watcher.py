from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from watchfiles import Change, awatch


class TranscriptWatcher:
    def __init__(self, roots: list[Path]):
        self.roots = [root for root in roots if root.exists()]

    async def changes(self) -> AsyncIterator[Path]:
        if not self.roots:
            return
        async for batch in awatch(*self.roots):
            for change, path_text in batch:
                if change not in (Change.added, Change.modified):
                    continue
                path = Path(path_text)
                if path.suffix == ".jsonl":
                    yield path
