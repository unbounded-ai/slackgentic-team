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


def _session_meta(cli_version):
    payload = {"id": "019dcf88-26b6-7cc3-a23e-3c9e45e12e24", "originator": "codex-tui"}
    if cli_version is not None:
        payload["cli_version"] = cli_version
    return {"type": "session_meta", "payload": payload}


def _input_item(text, role="user"):
    return {
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text" if role == "user" else "output_text", "text": text}],
        },
    }


def _user_message_item(text):
    return {
        "type": "event_msg",
        "payload": {
            "type": "item_completed",
            "item": {
                "type": "UserMessage",
                "id": "item-1",
                "content": [{"type": "text", "text": text, "text_elements": []}],
            },
        },
    }


def _user_message_event(text):
    return {"type": "event_msg", "payload": {"type": "user_message", "message": text}}


class CodexEventAuthorshipTests(unittest.TestCase):
    def _events(self, records, *, after=0):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout-test-019dcf88-26b6-7cc3-a23e-3c9e45e12e24.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
            return list(CodexProvider(home=Path(tmp)).iter_events_after(path, after))

    def _human_lines(self, events):
        return [event.line_number for event in events if event.human_authored]

    def test_only_submission_events_are_human_authored(self):
        records = [
            _session_meta("0.152.0"),
            _input_item("# AGENTS.md instructions for /workspace/repos/example-project"),
            _input_item("<environment_context><cwd>/workspace</cwd></environment_context>"),
            _input_item("fix the flaky test"),
            _user_message_item("fix the flaky test"),
            _input_item("On it.", role="assistant"),
            _input_item("<user_shell_command>git status</user_shell_command>"),
            _input_item("<hook_prompt>stop hook feedback</hook_prompt>"),
            _input_item("<recommended_plugins>example</recommended_plugins>"),
            # Shaped like any injection Codex might add later: plain text with
            # nothing to pattern-match on.
            _input_item("An internal note nobody has thought of yet."),
        ]

        events = self._events(records)

        self.assertEqual(self._human_lines(events), [5])

    def test_prompts_replayed_after_compaction_are_not_human_authored(self):
        records = [
            _session_meta("0.154.0"),
            {"type": "compacted", "payload": {"message": "summary of earlier work"}},
            _input_item("fix the flaky test"),
            _input_item("and open a PR when done"),
            _input_item("Continuing.", role="assistant"),
            _input_item("one more thing"),
            _user_message_item("one more thing"),
        ]

        events = self._events(records)

        self.assertEqual(self._human_lines(events), [7])

    def test_user_message_events_are_human_authored_and_not_marked_duplicate(self):
        records = [
            _session_meta("0.152.0"),
            _input_item("private task context"),
            _user_message_event("private task context"),
            _input_item("Done.", role="assistant"),
        ]

        events = self._events(records)

        self.assertEqual(self._human_lines(events), [3])
        self.assertNotIn("_slackgentic_duplicate_message", events[2].metadata)

    def test_one_submission_recorded_twice_is_human_authored_once(self):
        records = [
            _session_meta("0.137.0"),
            _user_message_event("review the current changes"),
            _user_message_event("review the current changes"),
            _input_item("Looks fine.", role="assistant"),
            _user_message_event("review the current changes"),
        ]

        events = self._events(records)

        # The adjacent pair is one submission; the later one is a person
        # sending the same text again.
        self.assertEqual(self._human_lines(events), [2, 5])

    def test_authorship_is_the_same_when_reading_from_a_cursor(self):
        records = [
            _session_meta("0.152.0"),
            _input_item("first ask"),
            _user_message_item("first ask"),
            _input_item("<hook_prompt>stop hook feedback</hook_prompt>"),
            _input_item("second ask"),
            _user_message_item("second ask"),
        ]

        events = self._events(records, after=3)

        self.assertEqual(self._human_lines(events), [6])

    def test_rollouts_without_a_verified_version_keep_their_previous_behaviour(self):
        context = (
            "# AGENTS.md instructions for /workspace/repos/example-project\n"
            "<INSTRUCTIONS>Follow the guide.</INSTRUCTIONS>\n"
            "<environment_context><cwd>/workspace</cwd></environment_context>"
        )
        for cli_version in (None, "0.98.0"):
            with self.subTest(cli_version=cli_version):
                records = [
                    _session_meta(cli_version),
                    _input_item(context),
                    _input_item("ship it"),
                    _user_message_event("ship it"),
                ]

                events = self._events(records)

                # Response items are preferred and the matching legacy event is
                # a duplicate, exactly as before.
                self.assertEqual(self._human_lines(events), [3])
                self.assertTrue(events[3].metadata["_slackgentic_duplicate_message"])

    def test_submission_text_is_read_from_either_event_shape(self):
        from agent_harness.providers.codex import codex_user_submission_text

        self.assertEqual(
            codex_user_submission_text(_user_message_item("ship it")["payload"]), "ship it"
        )
        self.assertEqual(
            codex_user_submission_text(_user_message_event("ship it")["payload"]), "ship it"
        )
        agent_item = {"type": "item_completed", "item": {"type": "AgentMessage", "content": []}}
        self.assertIsNone(codex_user_submission_text(agent_item))
        self.assertIsNone(codex_user_submission_text(_input_item("ship it")["payload"]))
        self.assertIsNone(codex_user_submission_text(None))


if __name__ == "__main__":
    unittest.main()
