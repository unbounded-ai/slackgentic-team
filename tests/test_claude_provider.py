import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from agent_harness.providers.claude import ClaudeProvider


class ClaudeProviderTests(unittest.TestCase):
    def test_session_start_uses_first_transcript_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            records = [
                {"type": "permission-mode", "sessionId": "session-1"},
                {
                    "type": "file-history-snapshot",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "sessionId": "session-1",
                },
                {
                    "type": "user",
                    "timestamp": "2026-04-27T12:00:01.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                },
                {
                    "type": "assistant",
                    "timestamp": "2026-04-27T13:00:00.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                },
            ]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            now = datetime(2026, 4, 27, 13, 0, tzinfo=UTC).timestamp()
            os.utime(path, (now, now))

            sessions = ClaudeProvider(home=home, active_within_seconds=3600).discover()

            self.assertEqual(len(sessions), 1)
            self.assertEqual(
                sessions[0].started_at,
                datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
            )
            self.assertEqual(sessions[0].cwd, project)


if __name__ == "__main__":
    unittest.main()
