from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

# A transcript untouched for this long cannot belong to a session that is still
# running, so a full scan skips it. Without a horizon the full scan turns every
# transcript ever written into a live session, and its cost grows with total
# history forever rather than with how much is actually running.
#
# This also bounds what the session mirror polls each cycle, and polling is not
# free: it reads a transcript end to end per session. A week was far too loose --
# on a busy machine "written to this week" described 377 transcripts while only
# five had a running process, costing about 5.3s of CPU per cycle and starving
# the Slack listener so acks missed Slack's three second deadline. A day still
# covers resuming yesterday's work, and anything older reappears as soon as
# something writes to it.
DEFAULT_STALE_AFTER_SECONDS = 24 * 60 * 60


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
        stale_after_seconds: float | None = DEFAULT_STALE_AFTER_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        self.root = root
        self.pattern = pattern
        self.full_scan_interval_seconds = max(0.0, full_scan_interval_seconds)
        self.stale_after_seconds = stale_after_seconds
        self.monotonic = monotonic
        self.wall_clock = wall_clock
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
            paths = {path for path in root.rglob(self.pattern) if self._recent_enough(path)}
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

    def _recent_enough(self, path: Path) -> bool:
        """True when ``path`` was modified recently enough to be a live session.

        Hot paths and explicit scan roots bypass this: the caller already knows
        it wants them. Only the unbounded full scan is filtered.
        """

        if self.stale_after_seconds is None:
            return True
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            return False
        return self.wall_clock() - modified_at <= self.stale_after_seconds
