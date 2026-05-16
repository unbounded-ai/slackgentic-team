import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_harness.providers.codex import CodexProvider


class CodexProviderTests(unittest.TestCase):
    def test_discover_reuses_cached_session_when_transcript_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            session_dir = home / ".codex" / "sessions" / "2026" / "04" / "27"
            session_dir.mkdir(parents=True)
            path = (
                session_dir
                / "rollout-2026-04-27T12-00-00-019dcf88-26b6-7cc3-a23e-3c9e45e12e24.jsonl"
            )
            path.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "timestamp": "2026-04-27T12:00:00.000Z",
                        "payload": {
                            "id": "019dcf88-26b6-7cc3-a23e-3c9e45e12e24",
                            "cwd": str(session_dir),
                            "model": "gpt-test",
                        },
                    }
                )
                + "\n"
            )
            path.touch()
            provider = CodexProvider(home=home, active_within_seconds=3600)

            self.assertEqual(
                provider.discover()[0].session_id, "019dcf88-26b6-7cc3-a23e-3c9e45e12e24"
            )

            with patch.object(provider, "_session_from_path", side_effect=AssertionError):
                self.assertEqual(
                    provider.discover()[0].session_id,
                    "019dcf88-26b6-7cc3-a23e-3c9e45e12e24",
                )


if __name__ == "__main__":
    unittest.main()
