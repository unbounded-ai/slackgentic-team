import os
import tempfile
import time
import unittest
from pathlib import Path

from agent_harness.providers.path_index import (
    DEFAULT_STALE_AFTER_SECONDS,
    TranscriptPathIndex,
)

NOW = 1_800_000_000.0


def _touch(path: Path, *, age_seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")
    stamp = NOW - age_seconds
    os.utime(path, (stamp, stamp))
    return path


class TranscriptPathIndexStaleFilterTests(unittest.TestCase):
    def _index(self, root: Path, **kwargs) -> TranscriptPathIndex:
        kwargs.setdefault("wall_clock", lambda: NOW)
        return TranscriptPathIndex(lambda: root, **kwargs)

    def test_full_scan_skips_transcripts_older_than_the_horizon(self):
        # Without this bound a full scan turns every transcript ever written into
        # a live session, so its cost grows with total history rather than with
        # how much is actually running.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fresh = _touch(root / "a" / "fresh.jsonl", age_seconds=60)
            recent = _touch(root / "b" / "recent.jsonl", age_seconds=6 * 24 * 3600)
            stale = _touch(root / "c" / "stale.jsonl", age_seconds=30 * 24 * 3600)

            discovery = self._index(root).discover()

            self.assertTrue(discovery.full_scan)
            self.assertIn(fresh, discovery.paths)
            self.assertIn(recent, discovery.paths)
            self.assertNotIn(stale, discovery.paths)

    def test_horizon_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = _touch(root / "old.jsonl", age_seconds=365 * 24 * 3600)

            discovery = self._index(root, stale_after_seconds=None).discover()

            self.assertIn(stale, discovery.paths)

    def test_hot_paths_bypass_the_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = _touch(root / "old.jsonl", age_seconds=90 * 24 * 3600)
            index = self._index(root)
            index.discover()  # first call is the full scan

            discovery = index.discover(hot_paths=[stale])

            self.assertFalse(discovery.full_scan)
            self.assertIn(stale, discovery.paths)

    def test_scan_roots_bypass_the_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = _touch(root / "nested" / "old.jsonl", age_seconds=90 * 24 * 3600)
            index = self._index(root)
            index.discover()

            discovery = index.discover(scan_roots=[root / "nested"])

            self.assertFalse(discovery.full_scan)
            self.assertIn(stale, discovery.paths)

    def test_unreadable_path_is_treated_as_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch(root / "gone.jsonl", age_seconds=60)
            index = self._index(root)
            (root / "gone.jsonl").unlink()

            discovery = index.discover()

            self.assertEqual(discovery.paths, [])

    def test_default_horizon_is_a_week(self):
        self.assertEqual(DEFAULT_STALE_AFTER_SECONDS, 7 * 24 * 60 * 60)

    def test_real_clock_is_used_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "now.jsonl"
            path.write_text("{}\n")
            os.utime(path, (time.time(), time.time()))

            discovery = TranscriptPathIndex(lambda: root).discover()

            self.assertIn(path, discovery.paths)


if __name__ == "__main__":
    unittest.main()
