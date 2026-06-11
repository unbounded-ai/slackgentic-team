import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from agent_harness.models import SessionStatus
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

    def test_discover_keeps_primary_transcript_when_subagent_reuses_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            subagents = project / "session-1" / "subagents"
            subagents.mkdir(parents=True)
            path = project / "session-1.jsonl"
            subagent_path = subagents / "agent-a1.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-27T12:00:00.000Z",
                        "cwd": str(project),
                        "sessionId": "session-1",
                    }
                )
                + "\n"
            )
            subagent_path.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-27T12:00:01.000Z",
                        "cwd": str(project),
                        "sessionId": "session-1",
                    }
                )
                + "\n"
            )
            primary_mtime = datetime(2026, 4, 27, 12, 5, tzinfo=UTC).timestamp()
            subagent_mtime = datetime(2026, 4, 27, 12, 10, tzinfo=UTC).timestamp()
            os.utime(path, (primary_mtime, primary_mtime))
            os.utime(subagent_path, (subagent_mtime, subagent_mtime))

            sessions = ClaudeProvider(home=home, active_within_seconds=3600).discover()

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].session_id, "session-1")
            self.assertEqual(sessions[0].transcript_path, path)

    def test_session_metadata_tracks_latest_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            records = [
                {
                    "type": "user",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "entrypoint": "cli",
                },
                {
                    "type": "assistant",
                    "timestamp": "2026-04-27T12:00:01.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "entrypoint": "sdk-cli",
                },
            ]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

            sessions = ClaudeProvider(home=home, active_within_seconds=3600).discover()

            self.assertEqual(sessions[0].metadata["entrypoint"], "sdk-cli")

    def test_discover_reuses_cached_session_when_transcript_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-27T12:00:00.000Z",
                        "cwd": str(project),
                        "sessionId": "session-1",
                    }
                )
                + "\n"
            )
            now = datetime.now(UTC).timestamp()
            os.utime(path, (now, now))
            provider = ClaudeProvider(home=home, active_within_seconds=3600)

            self.assertEqual(provider.discover()[0].session_id, "session-1")

            with patch.object(provider, "_session_from_path", side_effect=AssertionError):
                self.assertEqual(provider.discover()[0].session_id, "session-1")

    def test_discover_reparses_cached_session_when_transcript_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-27T12:00:00.000Z",
                        "cwd": str(project),
                        "sessionId": "session-1",
                        "entrypoint": "cli",
                    }
                )
                + "\n"
            )
            provider = ClaudeProvider(home=home, active_within_seconds=3600)

            self.assertEqual(provider.discover()[0].metadata["entrypoint"], "cli")

            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "type": "assistant",
                            "timestamp": "2026-04-27T12:00:01.000Z",
                            "cwd": str(project),
                            "sessionId": "session-1",
                            "entrypoint": "sdk-cli",
                        }
                    )
                    + "\n"
                )

            self.assertEqual(provider.discover()[0].metadata["entrypoint"], "sdk-cli")

    def test_discover_finds_new_session_in_changed_project_before_full_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            first = project / "session-1.jsonl"
            first.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-27T12:00:00.000Z",
                        "cwd": str(project),
                        "sessionId": "session-1",
                    }
                )
                + "\n"
            )
            provider = ClaudeProvider(
                home=home,
                active_within_seconds=3600,
                full_discovery_interval_seconds=3600,
            )

            self.assertEqual([session.session_id for session in provider.discover()], ["session-1"])

            second = project / "session-2.jsonl"
            second.write_text(
                json.dumps(
                    {
                        "type": "assistant",
                        "timestamp": "2026-04-27T12:00:01.000Z",
                        "cwd": str(project),
                        "sessionId": "session-2",
                    }
                )
                + "\n"
            )
            now = datetime.now(UTC).timestamp()
            os.utime(project, (now, now))

            self.assertEqual(
                {session.session_id for session in provider.discover()},
                {"session-1", "session-2"},
            )

    def test_exit_command_marks_recent_session_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            records = [
                {
                    "type": "user",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {"role": "user", "content": "hello"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2026-04-27T12:00:01.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {"role": "assistant", "content": "hi"},
                },
                {
                    "type": "user",
                    "timestamp": "2026-04-27T12:00:02.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {
                        "role": "user",
                        "content": (
                            "<command-name>/exit</command-name>\n"
                            "<command-message>exit</command-message>"
                        ),
                    },
                },
                {
                    "type": "user",
                    "timestamp": "2026-04-27T12:00:02.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {
                        "role": "user",
                        "content": "<local-command-stdout>Bye!</local-command-stdout>",
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            now = datetime.now(UTC).timestamp()
            os.utime(path, (now, now))

            sessions = ClaudeProvider(home=home, active_within_seconds=3600).discover()

            self.assertEqual(sessions[0].status, SessionStatus.DONE)

    def test_system_local_command_exit_marks_recent_session_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            records = [
                {
                    "type": "user",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {"role": "user", "content": "hello"},
                },
                {
                    "type": "assistant",
                    "timestamp": "2026-04-27T12:00:01.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {"role": "assistant", "content": "hi"},
                },
                {
                    "type": "system",
                    "subtype": "local_command",
                    "timestamp": "2026-04-27T12:00:02.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "content": (
                        "<command-name>/exit</command-name>\n"
                        "<command-message>exit</command-message>"
                    ),
                },
                {
                    "type": "system",
                    "subtype": "local_command",
                    "timestamp": "2026-04-27T12:00:02.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "content": "<local-command-stdout>Bye!</local-command-stdout>",
                },
            ]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            now = datetime.now(UTC).timestamp()
            os.utime(path, (now, now))

            sessions = ClaudeProvider(home=home, active_within_seconds=3600).discover()

            self.assertEqual(sessions[0].status, SessionStatus.DONE)

    def test_synthetic_no_response_after_exit_stays_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            records = [
                {
                    "type": "user",
                    "timestamp": "2026-04-27T12:00:00.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {
                        "role": "user",
                        "content": "<command-name>/exit</command-name>",
                    },
                },
                {
                    "type": "assistant",
                    "timestamp": "2026-04-27T12:00:01.000Z",
                    "cwd": str(project),
                    "sessionId": "session-1",
                    "message": {
                        "model": "<synthetic>",
                        "role": "assistant",
                        "content": [{"type": "text", "text": "No response requested."}],
                    },
                },
            ]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            now = datetime.now(UTC).timestamp()
            os.utime(path, (now, now))

            sessions = ClaudeProvider(home=home, active_within_seconds=3600).discover()

            self.assertEqual(sessions[0].status, SessionStatus.DONE)


class IsSyntheticClaudeAssistantRecordTests(unittest.TestCase):
    def test_detects_synthetic_model_marker(self):
        from agent_harness.providers.claude import is_synthetic_claude_assistant_record

        record = {
            "type": "assistant",
            "message": {
                "model": "<synthetic>",
                "content": [{"type": "text", "text": "anything at all"}],
            },
        }

        self.assertTrue(is_synthetic_claude_assistant_record(record))

    def test_detects_no_response_text_without_model_marker(self):
        from agent_harness.providers.claude import is_synthetic_claude_assistant_record

        record = {
            "type": "assistant",
            "message": {
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "No response requested."}],
            },
        }

        self.assertTrue(is_synthetic_claude_assistant_record(record))

    def test_genuine_assistant_message_is_not_synthetic(self):
        from agent_harness.providers.claude import is_synthetic_claude_assistant_record

        record = {
            "type": "assistant",
            "message": {
                "model": "claude-opus-4-7",
                "content": [{"type": "text", "text": "Here is the answer."}],
            },
        }

        self.assertFalse(is_synthetic_claude_assistant_record(record))


if __name__ == "__main__":
    unittest.main()
