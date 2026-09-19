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

            sessions = ClaudeProvider(
                home=home, active_within_seconds=3600, stale_after_seconds=None
            ).discover()

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

            sessions = ClaudeProvider(
                home=home, active_within_seconds=3600, stale_after_seconds=None
            ).discover()

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

            sessions = ClaudeProvider(
                home=home, active_within_seconds=3600, stale_after_seconds=None
            ).discover()

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

            sessions = ClaudeProvider(
                home=home, active_within_seconds=3600, stale_after_seconds=None
            ).discover()

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

            sessions = ClaudeProvider(
                home=home, active_within_seconds=3600, stale_after_seconds=None
            ).discover()

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

            sessions = ClaudeProvider(
                home=home, active_within_seconds=3600, stale_after_seconds=None
            ).discover()

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


def _prompt(text, **fields):
    return {"type": "user", "message": {"role": "user", "content": text}, **fields}


class ClaudeUserRecordAuthorshipTests(unittest.TestCase):
    MODERN = "2.1.228"
    LEGACY = "1.0.35"

    def test_only_a_human_origin_is_attributed_to_a_person(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        for source in ("typed", "queued", "suggestion_accepted"):
            record = _prompt("ship it", origin={"kind": "human"}, promptSource=source)
            with self.subTest(source=source):
                self.assertTrue(is_human_claude_user_record(record, provenance_labelled=True))

        for kind in ("task-notification", "coordinator", "peer", "some-future-kind"):
            record = _prompt("ship it", origin={"kind": kind}, promptSource="system")
            with self.subTest(kind=kind):
                self.assertFalse(is_human_claude_user_record(record, provenance_labelled=True))

    def test_headless_launcher_prompt_counts_but_system_prompts_do_not(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        launched = _prompt("review the diff", promptSource="sdk", version=self.MODERN)
        injected = _prompt("scheduled wake-up", promptSource="system", version=self.MODERN)

        self.assertTrue(is_human_claude_user_record(launched, provenance_labelled=True))
        self.assertFalse(is_human_claude_user_record(injected, provenance_labelled=True))

    def test_cli_written_flags_win_over_any_origin(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        for flag in ("isMeta", "isCompactSummary", "isVisibleInTranscriptOnly", "isSidechain"):
            record = _prompt("internal state", origin={"kind": "human"}, **{flag: True})
            with self.subTest(flag=flag):
                self.assertFalse(is_human_claude_user_record(record, provenance_labelled=False))

    def test_unlabelled_record_from_a_labelling_cli_is_never_a_person(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        # Shaped like any record type the CLI might add later: plain text, no
        # flags, nothing to pattern-match on.
        future = _prompt("An internal note nobody has thought of yet.", version=self.MODERN)
        newer = _prompt("An internal note nobody has thought of yet.", version="3.0.0")

        self.assertFalse(is_human_claude_user_record(future, provenance_labelled=False))
        self.assertFalse(is_human_claude_user_record(newer, provenance_labelled=False))

    def test_unlabelled_record_in_a_transcript_seen_labelling_is_not_a_person(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        record = _prompt("An internal note.", version=self.LEGACY)

        self.assertFalse(is_human_claude_user_record(record, provenance_labelled=True))

    def test_legacy_transcript_keeps_typed_prompts_and_drops_cli_plumbing(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        typed = _prompt("ship it", version=self.LEGACY)
        unversioned = _prompt("ship it")
        parts = {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": "ship it"}]},
        }
        self.assertTrue(is_human_claude_user_record(typed, provenance_labelled=False))
        self.assertTrue(is_human_claude_user_record(unversioned, provenance_labelled=False))
        self.assertTrue(is_human_claude_user_record(parts, provenance_labelled=False))

        tool_result = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
            },
        }
        self.assertFalse(is_human_claude_user_record(tool_result, provenance_labelled=False))
        for text in (
            "<command-name>/model</command-name>",
            "<local-command-stdout>done</local-command-stdout>",
            "<bash-input>ls</bash-input>",
            "<bash-stdout>README.md</bash-stdout>",
            "<task-notification><summary>finished</summary></task-notification>",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    is_human_claude_user_record(_prompt(text), provenance_labelled=False)
                )

    def test_malformed_origin_fails_closed(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        for origin in ({}, {"kind": None}, {"kind": ["human"]}, ["human"], 7, ""):
            with self.subTest(origin=origin):
                record = _prompt("ship it", origin=origin)
                self.assertFalse(is_human_claude_user_record(record, provenance_labelled=False))

        self.assertTrue(
            is_human_claude_user_record(
                _prompt("ship it", origin="human"), provenance_labelled=True
            )
        )

    def test_non_user_records_are_never_a_person(self):
        from agent_harness.providers.claude import is_human_claude_user_record

        record = {"type": "assistant", "origin": {"kind": "human"}, "message": {"content": "hi"}}

        self.assertFalse(is_human_claude_user_record(record, provenance_labelled=True))


class ClaudeEventAuthorshipTests(unittest.TestCase):
    def _events(self, records, *, after=0, provider=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session-1.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            provider = provider or ClaudeProvider(home=Path(tmp))
            return list(provider.iter_events_after(path, after))

    def test_compaction_summary_and_notifications_are_not_human_authored(self):
        version = "2.1.277"
        records = [
            _prompt(
                "take over the rollout",
                origin={"kind": "human"},
                promptSource="typed",
                version=version,
            ),
            {"type": "assistant", "version": version, "message": {"content": "On it."}},
            _prompt(
                "This session is being continued from a previous conversation.\n\n"
                "## 6. All user messages\n1. take over the rollout",
                isCompactSummary=True,
                isVisibleInTranscriptOnly=True,
                version=version,
            ),
            _prompt(
                "<task-notification><task-id>t1</task-id><summary>done</summary>"
                "</task-notification>",
                origin={"kind": "task-notification"},
                promptSource="system",
                version=version,
            ),
            _prompt("[Request interrupted by user]", version=version),
            _prompt("<bash-input>git status</bash-input>", version=version),
            _prompt("keep going", origin={"kind": "human"}, promptSource="queued", version=version),
        ]

        events = self._events(records)

        self.assertEqual(
            [event.metadata["message"]["content"] for event in events if event.human_authored],
            ["take over the rollout", "keep going"],
        )

    def test_resumed_transcript_that_opens_with_a_summary_is_not_human_authored(self):
        records = [
            _prompt(
                "This session is being continued from a previous conversation.",
                isCompactSummary=True,
                version="2.1.277",
            ),
            _prompt("carry on", origin={"kind": "human"}, promptSource="typed", version="2.1.277"),
        ]

        events = self._events(records)

        self.assertEqual([event.human_authored for event in events], [False, True])

    def test_older_cli_is_calibrated_by_what_its_transcript_labels(self):
        # A release below the verified floor that nevertheless labels prompts:
        # once that is seen, its unlabelled records are CLI state as well.
        old = "2.0.10"
        records = [
            _prompt("first ask", origin={"kind": "human"}, promptSource="typed", version=old),
            {"type": "assistant", "version": old, "message": {"content": "Done."}},
            _prompt("An internal note with nothing to pattern-match on.", version=old),
        ]

        events = self._events(records)
        self.assertEqual([event.human_authored for event in events], [True, False, False])

        # Reading only the tail, as the mirror does from its cursor, must reach
        # the same verdict even though the labelled record is not re-read.
        tail = self._events(records, after=2)
        self.assertEqual([event.human_authored for event in tail], [False])

    def test_transcript_that_never_labels_keeps_mirroring_typed_prompts(self):
        records = [
            _prompt("first ask", version="1.0.35"),
            {"type": "assistant", "version": "1.0.35", "message": {"content": "Done."}},
            _prompt("<command-name>/clear</command-name>", version="1.0.35"),
            _prompt("second ask", version="1.0.35"),
        ]

        events = self._events(records)

        self.assertEqual([event.human_authored for event in events], [True, False, False, True])

    def test_compaction_summary_after_exit_does_not_revive_the_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            project = home / ".claude" / "projects" / "-tmp-repo"
            project.mkdir(parents=True)
            path = project / "session-1.jsonl"
            base = {"cwd": str(project), "sessionId": "session-1", "version": "2.1.277"}
            records = [
                _prompt("hello", origin={"kind": "human"}, promptSource="typed", **base),
                _prompt("<command-name>/exit</command-name>", **base),
                _prompt("This session is being continued.", isCompactSummary=True, **base),
            ]
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            now = datetime.now(UTC).timestamp()
            os.utime(path, (now, now))
            provider = ClaudeProvider(
                home=home, active_within_seconds=3600, stale_after_seconds=None
            )

            self.assertEqual(provider.discover()[0].status, SessionStatus.DONE)

            records.append(
                _prompt("back again", origin={"kind": "human"}, promptSource="typed", **base)
            )
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            os.utime(path, (now + 5, now + 5))

            self.assertNotEqual(provider.discover()[0].status, SessionStatus.DONE)


if __name__ == "__main__":
    unittest.main()
