import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from agent_harness.providers.codex import CodexProvider


class CodexProviderTests(unittest.TestCase):
    def test_response_item_transcript_marks_legacy_messages_as_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-test-019dcf88-26b6-7cc3-a23e-3c9e45e12e24.jsonl"
            records = [
                {
                    "type": "response_item",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                },
                {
                    "type": "event_msg",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "payload": {"type": "user_message", "message": "hello"},
                },
                {
                    "type": "event_msg",
                    "timestamp": "2026-04-27T12:00:01.000Z",
                    "payload": {"type": "agent_message", "message": "done"},
                },
                {
                    "type": "response_item",
                    "timestamp": "2026-04-27T12:00:01.000Z",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    },
                },
            ]
            path.write_text("".join(f"{json.dumps(record)}\n" for record in records))

            events = list(CodexProvider().iter_events(path))

            self.assertFalse(events[0].metadata.get("_slackgentic_duplicate_message", False))
            self.assertTrue(events[1].metadata["_slackgentic_duplicate_message"])
            self.assertTrue(events[2].metadata["_slackgentic_duplicate_message"])
            self.assertFalse(events[3].metadata.get("_slackgentic_duplicate_message", False))

    def test_legacy_only_transcript_keeps_legacy_messages_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-test-019dcf88-26b6-7cc3-a23e-3c9e45e12e24.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "event_msg",
                        "timestamp": "2026-04-27T12:00:01.000Z",
                        "payload": {"type": "agent_message", "message": "done"},
                    }
                )
                + "\n"
            )

            event = next(CodexProvider().iter_events(path))

            self.assertFalse(event.metadata.get("_slackgentic_duplicate_message", False))

    def test_response_item_recovery_rewinds_only_schema_without_legacy_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-test-019dcf88-26b6-7cc3-a23e-3c9e45e12e24.jsonl"
            records = [
                {
                    "type": "response_item",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "before observation"}],
                    },
                },
                {
                    "type": "response_item",
                    "timestamp": "2026-04-27T12:00:02.000Z",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "after observation"}],
                    },
                },
            ]
            path.write_text("".join(f"{json.dumps(record)}\n" for record in records))
            provider = CodexProvider()

            recovered = provider.response_item_recovery_cursor(
                path,
                last_line_number=2,
                observed_after=datetime(2026, 4, 27, 12, 0, 1, tzinfo=UTC),
            )

            self.assertEqual(recovered, 1)

            records.insert(
                2,
                {
                    "type": "event_msg",
                    "timestamp": "2026-04-27T12:00:02.000Z",
                    "payload": {"type": "agent_message", "message": "after observation"},
                },
            )
            path.write_text("".join(f"{json.dumps(record)}\n" for record in records))
            self.assertIsNone(
                provider.response_item_recovery_cursor(
                    path,
                    last_line_number=3,
                    observed_after=datetime(2026, 4, 27, 12, 0, 1, tzinfo=UTC),
                )
            )

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
                            "source": {"subagent": {"thread_spawn": {"depth": 1}}},
                            "thread_source": "subagent",
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
            self.assertEqual(provider.discover()[0].metadata["thread_source"], "subagent")
            self.assertEqual(
                provider.discover()[0].metadata["source"],
                {"subagent": {"thread_spawn": {"depth": 1}}},
            )

            with patch.object(provider, "_session_from_path", side_effect=AssertionError):
                self.assertEqual(
                    provider.discover()[0].session_id,
                    "019dcf88-26b6-7cc3-a23e-3c9e45e12e24",
                )

    def test_discover_finds_new_recent_session_before_full_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            day = datetime.now(UTC)
            session_dir = (
                home
                / ".codex"
                / "sessions"
                / f"{day.year:04d}"
                / f"{day.month:02d}"
                / f"{day.day:02d}"
            )
            session_dir.mkdir(parents=True)
            first = session_dir / "rollout-a-019dcf88-26b6-7cc3-a23e-3c9e45e12e24.jsonl"
            first.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "timestamp": "2026-04-27T12:00:00.000Z",
                        "payload": {
                            "id": "019dcf88-26b6-7cc3-a23e-3c9e45e12e24",
                            "cwd": str(session_dir),
                        },
                    }
                )
                + "\n"
            )
            provider = CodexProvider(
                home=home,
                active_within_seconds=3600,
                full_discovery_interval_seconds=3600,
            )

            self.assertEqual(
                {session.session_id for session in provider.discover()},
                {"019dcf88-26b6-7cc3-a23e-3c9e45e12e24"},
            )

            second = session_dir / "rollout-b-019dcf88-3067-75e3-b9da-c52efeb3bb99.jsonl"
            second.write_text(
                json.dumps(
                    {
                        "type": "session_meta",
                        "timestamp": "2026-04-27T12:01:00.000Z",
                        "payload": {
                            "id": "019dcf88-3067-75e3-b9da-c52efeb3bb99",
                            "cwd": str(session_dir),
                        },
                    }
                )
                + "\n"
            )

            self.assertEqual(
                {session.session_id for session in provider.discover()},
                {
                    "019dcf88-26b6-7cc3-a23e-3c9e45e12e24",
                    "019dcf88-3067-75e3-b9da-c52efeb3bb99",
                },
            )


if __name__ == "__main__":
    unittest.main()
