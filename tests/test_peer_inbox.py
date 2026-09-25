import json
import os
import socket
import tempfile
import unittest
from pathlib import Path

from agent_harness.sessions.peer_inbox import (
    find_claude_peer_inbox,
    peer_message_line,
    transcript_shows_prompt,
)


def _register(home: Path, pid: int, session_id: str, socket_path: Path) -> None:
    sessions = home / ".claude" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / f"{pid}.json").write_text(
        json.dumps(
            {
                "pid": pid,
                "sessionId": session_id,
                "entrypoint": "claude-desktop",
                "messagingSocketPath": str(socket_path),
            }
        ),
        encoding="utf-8",
    )


class PeerInboxTests(unittest.TestCase):
    def test_finds_inbox_of_live_process_with_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            socket_path = home / "inbox.sock"
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                server.bind(str(socket_path))
                _register(home, os.getpid(), "s1", socket_path)

                inbox = find_claude_peer_inbox("s1", home)

                self.assertIsNotNone(inbox)
                self.assertEqual(inbox.pid, os.getpid())
                self.assertEqual(inbox.socket_path, socket_path)
                self.assertIsNone(find_claude_peer_inbox("other", home))
            finally:
                server.close()

    def test_ignores_dead_process_and_missing_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            _register(home, 0, "dead", home / "dead.sock")
            _register(home, os.getpid(), "plain", home / "plain.sock")
            (home / "plain.sock").write_text("not a socket", encoding="utf-8")

            self.assertIsNone(find_claude_peer_inbox("dead", home))
            self.assertIsNone(find_claude_peer_inbox("plain", home))
            self.assertIsNone(find_claude_peer_inbox("s1", home / "missing"))

    def test_peer_message_line_is_a_now_priority_user_frame(self):
        line = peer_message_line("ship it")

        self.assertTrue(line.endswith("\n"))
        frame = json.loads(line)
        self.assertEqual(frame["type"], "user")
        self.assertEqual(frame["priority"], "now")
        self.assertEqual(frame["from"], "uds:slackgentic")
        content = frame["message"]["content"]
        self.assertEqual(frame["message"]["role"], "user")
        self.assertTrue(content.startswith('<cross-session-message from="uds:slackgentic"'))
        self.assertIn('from-name="Slackgentic"', content)
        self.assertIn("\nship it\n", content)
        self.assertTrue(content.endswith("</cross-session-message>"))

    def test_transcript_shows_prompt_after_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "claude.jsonl"
            transcript.write_text(
                json.dumps({"type": "user", "message": {"content": "ship it"}}) + "\n",
                encoding="utf-8",
            )
            offset = transcript.stat().st_size
            self.assertFalse(transcript_shows_prompt(transcript, offset, "ship it"))

            with transcript.open("a", encoding="utf-8") as handle:
                handle.write("not json\n")
                handle.write(
                    json.dumps(
                        {
                            "type": "queue-operation",
                            "operation": "enqueue",
                            "content": "<cross-session-message>\nship it\n</cross-session-message>",
                        }
                    )
                    + "\n"
                )
            self.assertTrue(transcript_shows_prompt(transcript, offset, "ship it"))
            self.assertFalse(transcript_shows_prompt(transcript, offset, "something else"))

            offset = transcript.stat().st_size
            with transcript.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "user",
                            "message": {"content": [{"type": "text", "text": "later: ship it"}]},
                        }
                    )
                    + "\n"
                )
            self.assertTrue(transcript_shows_prompt(transcript, offset, "ship it"))
            self.assertFalse(transcript_shows_prompt(Path(tmp) / "missing.jsonl", 0, "ship it"))
            self.assertFalse(transcript_shows_prompt(transcript, 0, "   "))


if __name__ == "__main__":
    unittest.main()
