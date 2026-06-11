from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathDiscovery:
    paths: list[Path]
    full_scan: bool


class TranscriptPathIndex:
    def __init__(
        self,
        root: Callable[[], Path],
        *,
        pattern: str = "*.jsonl",
        full_scan_interval_seconds: float = 300.0,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.root = root
        self.pattern = pattern
        self.full_scan_interval_seconds = max(0.0, full_scan_interval_seconds)
        self.monotonic = monotonic
        self._paths: set[Path] = set()
        self._last_full_scan_monotonic: float | None = None

    def full_scan_due(self) -> bool:
        if self._last_full_scan_monotonic is None:
            return True
        if self.full_scan_interval_seconds <= 0:
            return True
        return self.monotonic() - self._last_full_scan_monotonic >= self.full_scan_interval_seconds

    def discover(
        self,
        *,
        hot_paths: Iterable[Path] = (),
        scan_roots: Iterable[Path] = (),
    ) -> PathDiscovery:
        root = self.root()
        if not root.exists():
            self._paths.clear()
            self._last_full_scan_monotonic = None
            return PathDiscovery([], full_scan=True)

        if self.full_scan_due():
            paths = set(root.rglob(self.pattern))
            self._paths = paths
            self._last_full_scan_monotonic = self.monotonic()
            return PathDiscovery(sorted(paths), full_scan=True)

        paths: set[Path] = set()
        for path in hot_paths:
            if path.exists():
                self._paths.add(path)
                paths.add(path)
            else:
                self._paths.discard(path)

        for scan_root in scan_roots:
            if not scan_root.exists():
                continue
            if scan_root.is_file():
                if scan_root.match(self.pattern):
                    self._paths.add(scan_root)
                    paths.add(scan_root)
                continue
            for path in scan_root.rglob(self.pattern):
                self._paths.add(path)
                paths.add(path)

        return PathDiscovery(sorted(paths), full_scan=False)
