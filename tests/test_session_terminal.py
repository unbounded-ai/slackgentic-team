import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_harness.models import AgentSession, Provider
from agent_harness.sessions.terminal import SessionTerminalNotifier


class SessionTerminalNotifierTests(unittest.TestCase):
    def test_writes_notification_to_session_sidecar_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = []
            notifier = SessionTerminalNotifier(
                log_writer=lambda path, text: written.append((path, text)),
                write_tui_notice=False,
            )
            session = AgentSession(
                provider=Provider.CLAUDE,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
            )

            count = notifier.notify_user_message(session, "hello from Slack")

            self.assertEqual(count, 1)
            self.assertEqual(
                written[0][0],
                Path(tmp) / ".slackgentic" / "terminal-notifications" / "claude-abcdef123456.log",
            )
            self.assertIn("hello from Slack", written[0][1])
            self.assertIn("abcdef12", written[0][1])

    def test_targets_for_session_filter_provider_resume_and_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            notifier = SessionTerminalNotifier(
                process_lister=lambda: [
                    "101 ttys002 claude --dangerously-skip-permissions",
                    "102 ttys003 claude --print --resume s1",
                    "103 ?? claude --dangerously-skip-permissions",
                    "104 ttys004 codex",
                    "105 ttys005 claude --dangerously-skip-permissions",
                ],
                cwd_resolver=lambda pid: Path(tmp) if pid == 101 else Path("/other"),
            )
            session = AgentSession(
                provider=Provider.CLAUDE,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
            )

            targets = notifier.targets_for_session(session)

            self.assertEqual([target.tty for target in targets], ["ttys002"])

    def test_provider_process_for_pid_matches_provider_without_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            starts = {101: datetime(2026, 4, 27, 12, 0, tzinfo=UTC)}
            notifier = SessionTerminalNotifier(
                process_lister=lambda: [
                    "101 ttys002 codex --remote ws://127.0.0.1:47684",
                    "102 ttys003 claude --dangerously-skip-permissions",
                ],
                cwd_resolver=lambda pid: Path(tmp) if pid == 101 else Path("/other"),
                start_resolver=lambda pid: starts.get(pid),
            )

            target = notifier.provider_process_for_pid(Provider.CODEX, 101)

            self.assertIsNotNone(target)
            self.assertEqual(target.pid, 101)
            self.assertEqual(target.cwd, Path(tmp))

    def test_claude_agent_response_never_writes_to_tty(self):
        with tempfile.TemporaryDirectory() as tmp:
            tty_writes = []
            notifier = SessionTerminalNotifier(
                process_lister=lambda: ["101 ttys002 claude --dangerously-skip-permissions"],
                cwd_resolver=lambda pid: Path(tmp),
                start_resolver=lambda pid: datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                log_writer=lambda path, text: None,
                tty_writer=lambda tty, text: tty_writes.append((tty, text)),
                write_tui_notice=True,
            )
            session = AgentSession(
                provider=Provider.CLAUDE,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
                started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
            )

            notifier.notify_agent_response(session, "done")

            self.assertEqual(tty_writes, [])

    def test_codex_agent_response_can_write_bounded_notice_to_matching_tty(self):
        with tempfile.TemporaryDirectory() as tmp:
            tty_writes = []
            notifier = SessionTerminalNotifier(
                process_lister=lambda: ["101 ttys002 codex"],
                cwd_resolver=lambda pid: Path(tmp),
                start_resolver=lambda pid: datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                log_writer=lambda path, text: None,
                tty_writer=lambda tty, text: tty_writes.append((tty, text)),
                write_tui_notice=True,
            )
            session = AgentSession(
                provider=Provider.CODEX,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
                started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
            )

            notifier.notify_agent_response(session, "done")

            self.assertEqual(tty_writes[0][0], "ttys002")
            self.assertIn("Slackgentic response", tty_writes[0][1])
            self.assertIn("done", tty_writes[0][1])
            self.assertNotIn("> done", tty_writes[0][1])
            self.assertNotIn("log:", tty_writes[0][1])

    def test_user_message_does_not_write_to_tty_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tty_writes = []
            notifier = SessionTerminalNotifier(
                process_lister=lambda: ["101 ttys002 claude --dangerously-skip-permissions"],
                cwd_resolver=lambda pid: Path(tmp),
                log_writer=lambda path, text: None,
                tty_writer=lambda tty, text: tty_writes.append((tty, text)),
            )
            session = AgentSession(
                provider=Provider.CLAUDE,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
            )

            notifier.notify_user_message(session, "from Slack")

            self.assertEqual(tty_writes, [])

    def test_multiple_matching_processes_choose_start_time_closest_to_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_started = datetime(2026, 4, 27, 13, 46, tzinfo=UTC)
            starts = {
                101: session_started - timedelta(hours=2),
                102: session_started + timedelta(seconds=5),
            }
            notifier = SessionTerminalNotifier(
                process_lister=lambda: [
                    "101 ttys002 claude --dangerously-skip-permissions",
                    "102 ttys003 claude --dangerously-skip-permissions",
                ],
                cwd_resolver=lambda pid: Path(tmp),
                start_resolver=lambda pid: starts[pid],
            )
            session = AgentSession(
                provider=Provider.CLAUDE,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
                started_at=session_started,
            )

            targets = notifier.targets_for_session(session)

            self.assertEqual([target.tty for target in targets], ["ttys003"])

    def test_multiple_matching_processes_with_session_start_uses_closest_start_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_started = datetime(2026, 4, 27, 13, 46, tzinfo=UTC)
            starts = {
                101: session_started - timedelta(hours=2),
                102: session_started - timedelta(seconds=5),
                103: session_started + timedelta(minutes=20),
            }
            notifier = SessionTerminalNotifier(
                process_lister=lambda: [
                    "101 ttys002 claude --dangerously-skip-permissions",
                    "102 ttys003 claude --dangerously-skip-permissions",
                    "103 ttys004 claude --dangerously-skip-permissions",
                ],
                cwd_resolver=lambda pid: Path(tmp),
                start_resolver=lambda pid: starts[pid],
            )
            session = AgentSession(
                provider=Provider.CLAUDE,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
                started_at=session_started,
            )

            targets = notifier.targets_for_session(session)

            self.assertEqual([target.tty for target in targets], ["ttys003"])

    def test_multiple_matching_processes_without_session_start_prefers_newest_before_last_seen(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            last_seen = datetime(2026, 4, 27, 13, 46, tzinfo=UTC)
            starts = {
                101: last_seen - timedelta(hours=2),
                102: last_seen - timedelta(seconds=5),
                103: last_seen + timedelta(minutes=20),
            }
            notifier = SessionTerminalNotifier(
                process_lister=lambda: [
                    "101 ttys002 claude --dangerously-skip-permissions",
                    "102 ttys003 claude --dangerously-skip-permissions",
                    "103 ttys004 claude --dangerously-skip-permissions",
                ],
                cwd_resolver=lambda pid: Path(tmp),
                start_resolver=lambda pid: starts[pid],
            )
            session = AgentSession(
                provider=Provider.CLAUDE,
                session_id="abcdef123456",
                transcript_path=Path(tmp) / "session.jsonl",
                cwd=Path(tmp),
                last_seen_at=last_seen,
            )

            targets = notifier.targets_for_session(session)

            self.assertEqual([target.tty for target in targets], ["ttys003"])


if __name__ == "__main__":
    unittest.main()
