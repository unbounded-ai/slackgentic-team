import json
import signal
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path

from agent_harness.config import AgentCommandConfig
from agent_harness.models import AgentSession, Provider, SessionStatus, SlackThreadRef
from agent_harness.sessions.bridge import (
    ExternalSessionBridge,
    _codex_remote_enabled,
    _slackgentic_channel_enabled,
    build_external_session_prompt,
    is_session_exit_request,
)
from agent_harness.sessions.terminal import TerminalTarget
from agent_harness.storage.store import Store


class FakeGateway:
    def __init__(self):
        self.replies = []

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        self.replies.append((thread, text))


class FakeTerminalNotifier:
    def __init__(self, targets=None):
        self.user_messages = []
        self.agent_responses = []
        self.targets = targets or []

    def notify_user_message(self, session, text):
        self.user_messages.append((session, text))
        return 1

    def notify_agent_response(self, session, text):
        self.agent_responses.append((session, text))
        return 1

    def targets_for_session(self, session):
        return self.targets


class FakeCodexLiveClient:
    def __init__(self, handled=True):
        self.handled = handled
        self.calls = []

    def send_to_thread(self, thread_id, text, cwd=None):
        self.calls.append((thread_id, text, cwd))
        if isinstance(self.handled, Exception):
            raise self.handled
        return self.handled


def _write_claude_slackgentic_mcp_marker(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "attachment": {
                    "type": "deferred_tools_delta",
                    "addedNames": ["mcp__slackgentic__request_approval"],
                }
            }
        )
        + "\n"
    )


class SessionBridgeTests(unittest.TestCase):
    def test_build_external_session_prompt_forwards_raw_text(self):
        prompt = build_external_session_prompt("do the thing", "U1")

        self.assertEqual(prompt, "do the thing")

    def test_is_session_exit_request_matches_exact_exit_command(self):
        self.assertTrue(is_session_exit_request(" /exit "))
        self.assertFalse(is_session_exit_request("/exit now"))
        self.assertFalse(is_session_exit_request("/quit"))

    def test_slackgentic_channel_flag_detection(self):
        self.assertTrue(
            _slackgentic_channel_enabled(
                "claude --dangerously-load-development-channels server:slackgentic"
            )
        )
        self.assertTrue(_slackgentic_channel_enabled("claude --channels=server:slackgentic"))
        self.assertTrue(
            _slackgentic_channel_enabled("claude --channels 'server:slackgentic server:other'")
        )
        self.assertFalse(_slackgentic_channel_enabled("claude"))
        self.assertFalse(_slackgentic_channel_enabled("claude --channels plugin:fakechat"))

    def test_codex_remote_flag_detection(self):
        expected = "ws://127.0.0.1:47684"

        self.assertTrue(_codex_remote_enabled("codex --remote ws://127.0.0.1:47684", expected))
        self.assertTrue(_codex_remote_enabled("codex --remote=ws://localhost:47684", expected))
        self.assertFalse(_codex_remote_enabled("codex", expected))
        self.assertFalse(_codex_remote_enabled("codex --remote ws://127.0.0.1:59026", expected))

    def test_send_to_claude_session_runs_resume_command_and_records_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()
            terminal = FakeTerminalNotifier()

            def runner(args, **kwargs):
                calls.append((args, kwargs))
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(
                        claude_binary="claude-bin",
                        default_cwd=Path(tmp),
                    ),
                    command_runner=runner,
                    terminal_notifier=terminal,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    permission_mode="bypassPermissions",
                )

                self.assertTrue(
                    bridge.send_to_session(
                        session,
                        "continue",
                        SlackThreadRef("C1", "171.000001"),
                        slack_user="U1",
                    )
                )

                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(
                    calls[0][0][:5],
                    ["claude-bin", "--print", "--output-format", "json", "--resume"],
                )
                self.assertIn("s1", calls[0][0])
                self.assertIn("--permission-mode", calls[0][0])
                self.assertEqual(calls[0][1]["input"], "")
                prompt = calls[0][0][-1]
                self.assertIn("continue", prompt)
                self.assertTrue(store.consume_session_bridge_prompt(Provider.CLAUDE, "s1", prompt))
                self.assertEqual(terminal.user_messages[0][1], "continue")
            finally:
                store.close()

    def test_exit_request_terminates_live_session_without_resuming(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            killed = []
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude",
                    )
                ]
            )

            def runner(args, **kwargs):
                calls.append(args)
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    process_killer=lambda pid, sig: killed.append((pid, sig)),
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                )

                handled = bridge.send_to_session(session, " /exit ", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertEqual(killed, [(123, signal.SIGTERM)])
                self.assertEqual(calls, [])
                self.assertEqual(gateway.replies[0][1], "Terminated the external claude session.")
                self.assertFalse(
                    store.consume_session_bridge_prompt(Provider.CLAUDE, "s1", "/exit")
                )
            finally:
                store.close()

    def test_exit_request_uses_claude_channel_before_sigterm(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            killed = []
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude --dangerously-load-development-channels server:slackgentic",
                        started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    )
                ]
            )
            delivered = threading.Event()

            def deliver_channel_message():
                worker_store = Store(Path(tmp) / "state.sqlite")
                try:
                    for _ in range(50):
                        rows = worker_store.pending_claude_channel_messages(123)
                        if rows:
                            worker_store.mark_claude_channel_message_delivered(int(rows[0]["id"]))
                            delivered.set()
                            return
                        delivered.wait(0.01)
                finally:
                    worker_store.close()

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.5,
                    process_killer=lambda pid, sig: killed.append((pid, sig)),
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime.now(UTC),
                )
                _write_claude_slackgentic_mcp_marker(session.transcript_path)
                store.upsert_session(session)
                store.set_setting("external_session_agent.claude.s1", "agent-1")
                thread = threading.Thread(target=deliver_channel_message)
                thread.start()

                handled = bridge.send_to_session(session, "/exit", SlackThreadRef("C1", "171"))

                thread.join(timeout=1)
                self.assertTrue(handled)
                self.assertTrue(delivered.is_set())
                self.assertEqual(killed, [])
                self.assertEqual(
                    gateway.replies[0][1],
                    "Sent `/exit` to the live claude session.",
                )
                self.assertIsNone(store.get_setting("external_session_agent.claude.s1"))
                self.assertIsNotNone(store.get_setting("external_session_ignored.claude.s1"))
                self.assertEqual(
                    store.get_session(Provider.CLAUDE, "s1").status, SessionStatus.DONE
                )
            finally:
                store.close()

    def test_exit_request_uses_codex_remote_then_terminates_matching_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            killed = []
            live = FakeCodexLiveClient()
            started = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="codex --remote ws://localhost:47684",
                        started_at=started,
                    )
                ]
            )

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(codex_binary="codex-bin", default_cwd=Path(tmp)),
                    terminal_notifier=terminal,
                    codex_app_server_url="ws://127.0.0.1:47684",
                    codex_live_client=live,
                    process_killer=lambda pid, sig: killed.append((pid, sig)),
                )
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    started_at=started,
                    last_seen_at=datetime.now(UTC),
                )
                store.upsert_session(session)
                store.set_setting("external_session_agent.codex.codex-s1", "agent-1")

                handled = bridge.send_to_session(session, "/exit", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertEqual(live.calls, [("codex-s1", "/exit", Path(tmp))])
                self.assertEqual(killed, [(123, signal.SIGTERM)])
                self.assertIn("Sent `/exit` to the live codex session.", gateway.replies[0][1])
                self.assertIn("Terminated the matching process", gateway.replies[0][1])
                self.assertIsNone(store.get_setting("external_session_agent.codex.codex-s1"))
                self.assertIsNotNone(store.get_setting("external_session_ignored.codex.codex-s1"))
                self.assertEqual(
                    store.get_session(Provider.CODEX, "codex-s1").status,
                    SessionStatus.DONE,
                )
            finally:
                store.close()

    def test_codex_exit_uses_app_server_even_without_matching_terminal_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            killed = []
            live = FakeCodexLiveClient()
            terminal = FakeTerminalNotifier()

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(codex_binary="codex-bin", default_cwd=Path(tmp)),
                    terminal_notifier=terminal,
                    codex_app_server_url="ws://127.0.0.1:47684",
                    codex_live_client=live,
                    process_killer=lambda pid, sig: killed.append((pid, sig)),
                )
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime.now(UTC),
                )
                store.upsert_session(session)
                store.set_setting("external_session_agent.codex.codex-s1", "agent-1")

                handled = bridge.send_to_session(session, "/exit", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertEqual(live.calls, [("codex-s1", "/exit", Path(tmp))])
                self.assertEqual(killed, [])
                self.assertEqual(
                    gateway.replies[0][1],
                    "Sent `/exit` to the live codex session.",
                )
                self.assertTrue(
                    store.consume_session_bridge_prompt(Provider.CODEX, "codex-s1", "/exit")
                )
                self.assertIsNone(store.get_setting("external_session_agent.codex.codex-s1"))
                self.assertIsNotNone(store.get_setting("external_session_ignored.codex.codex-s1"))
                self.assertEqual(
                    store.get_session(Provider.CODEX, "codex-s1").status,
                    SessionStatus.DONE,
                )
            finally:
                store.close()

    def test_exit_request_does_not_kill_ambiguous_session_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            killed = []
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(pid=123, tty="ttys001", cwd=Path(tmp), command="claude"),
                    TerminalTarget(pid=456, tty="ttys002", cwd=Path(tmp), command="claude"),
                ]
            )

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    terminal_notifier=terminal,
                    process_killer=lambda pid, sig: killed.append((pid, sig)),
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                )

                handled = bridge.send_to_session(session, "/exit", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertEqual(killed, [])
                self.assertIn("I found 2 matching claude processes", gateway.replies[0][1])
            finally:
                store.close()

    def test_exit_request_does_not_kill_stale_session_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            killed = []
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude",
                        started_at=datetime(2026, 4, 27, 15, 0, tzinfo=UTC),
                    )
                ]
            )

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    terminal_notifier=terminal,
                    process_killer=lambda pid, sig: killed.append((pid, sig)),
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="old",
                    transcript_path=Path(tmp) / "old.jsonl",
                    cwd=Path(tmp),
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                )

                handled = bridge.send_to_session(session, "/exit", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertEqual(killed, [])
                self.assertIn("could not find a live claude process", gateway.replies[0][1])
            finally:
                store.close()

    def test_send_to_claude_session_uses_live_channel_when_delivered(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude --dangerously-load-development-channels server:slackgentic",
                        started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    )
                ]
            )
            delivered = threading.Event()

            def runner(args, **kwargs):
                calls.append(args)
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            def deliver_channel_message():
                for _ in range(50):
                    rows = store.pending_claude_channel_messages(123)
                    if rows:
                        store.mark_claude_channel_message_delivered(int(rows[0]["id"]))
                        delivered.set()
                        return
                    delivered.wait(0.01)

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.5,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime.now(UTC),
                )
                _write_claude_slackgentic_mcp_marker(session.transcript_path)
                store.upsert_session(session)
                thread = threading.Thread(target=deliver_channel_message)
                thread.start()

                handled = bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))

                thread.join(timeout=1)
                self.assertTrue(handled)
                self.assertTrue(delivered.is_set())
                self.assertEqual(calls, [])
                self.assertTrue(
                    store.consume_session_bridge_prompt(Provider.CLAUDE, "s1", "continue")
                )
            finally:
                store.close()

    def test_idle_claude_session_with_matching_live_channel_uses_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude --dangerously-load-development-channels server:slackgentic",
                        started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    )
                ]
            )
            delivered = threading.Event()

            def runner(args, **kwargs):
                calls.append(args)
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            def deliver_channel_message():
                for _ in range(50):
                    rows = store.pending_claude_channel_messages(123)
                    if rows:
                        store.mark_claude_channel_message_delivered(int(rows[0]["id"]))
                        delivered.set()
                        return
                    delivered.wait(0.01)

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.5,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.IDLE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime(2026, 4, 27, 12, 10, tzinfo=UTC),
                )
                _write_claude_slackgentic_mcp_marker(session.transcript_path)
                store.upsert_session(session)
                thread = threading.Thread(target=deliver_channel_message)
                thread.start()

                handled = bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))

                thread.join(timeout=1)
                self.assertTrue(handled)
                self.assertTrue(delivered.is_set())
                self.assertEqual(calls, [])
            finally:
                store.close()

    def test_stale_claude_session_does_not_channel_into_new_process_same_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude --dangerously-load-development-channels server:slackgentic",
                        started_at=datetime(2026, 4, 27, 15, 0, tzinfo=UTC),
                    )
                ]
            )

            def runner(args, **kwargs):
                calls.append(args)
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.01,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="old",
                    transcript_path=Path(tmp) / "old.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.IDLE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime(2026, 4, 27, 12, 10, tzinfo=UTC),
                )
                store.upsert_session(session)

                self.assertTrue(
                    bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))
                )

                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(
                    calls[0][:5], ["claude-bin", "--print", "--output-format", "json", "--resume"]
                )
                row = store.conn.execute(
                    "SELECT COUNT(*) AS count FROM claude_channel_messages"
                ).fetchone()
                self.assertEqual(row["count"], 0)
            finally:
                store.close()

    def test_undelivered_claude_channel_message_falls_back_to_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude --dangerously-load-development-channels server:slackgentic",
                        started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    )
                ]
            )

            def runner(args, **kwargs):
                calls.append(args)
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.01,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime.now(UTC),
                )
                _write_claude_slackgentic_mcp_marker(session.transcript_path)
                store.upsert_session(session)

                self.assertTrue(
                    bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))
                )

                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(
                    calls[0][:5], ["claude-bin", "--print", "--output-format", "json", "--resume"]
                )
                row = store.conn.execute(
                    "SELECT cancelled_at FROM claude_channel_messages"
                ).fetchone()
                self.assertIsNotNone(row["cancelled_at"])
            finally:
                store.close()

    def test_claude_session_with_channel_flag_but_missing_mcp_uses_resume_not_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude --dangerously-load-development-channels server:slackgentic",
                        started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    )
                ]
            )

            def runner(args, **kwargs):
                calls.append(args)
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.01,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime.now(UTC),
                )
                store.upsert_session(session)

                self.assertTrue(
                    bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))
                )

                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(
                    calls[0][:5], ["claude-bin", "--print", "--output-format", "json", "--resume"]
                )
                row = store.conn.execute(
                    "SELECT COUNT(*) AS count FROM claude_channel_messages"
                ).fetchone()
                self.assertEqual(row["count"], 0)
            finally:
                store.close()

    def test_claude_session_without_channel_opt_in_uses_resume_not_channel(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()
            terminal = FakeTerminalNotifier(
                [
                    TerminalTarget(
                        pid=123,
                        tty="ttys001",
                        cwd=Path(tmp),
                        command="claude",
                        started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    )
                ]
            )

            def runner(args, **kwargs):
                calls.append(args)
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.01,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                    started_at=datetime(2026, 4, 27, 12, 0, tzinfo=UTC),
                    last_seen_at=datetime.now(UTC),
                )
                store.upsert_session(session)

                self.assertTrue(
                    bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))
                )

                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(
                    calls[0][:5], ["claude-bin", "--print", "--output-format", "json", "--resume"]
                )
                row = store.conn.execute(
                    "SELECT COUNT(*) AS count FROM claude_channel_messages"
                ).fetchone()
                self.assertEqual(row["count"], 0)
            finally:
                store.close()

    def test_send_to_codex_session_prefers_app_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            live = FakeCodexLiveClient()

            def runner(args, **kwargs):
                calls.append(args)
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(codex_binary="codex-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    codex_app_server_url="ws://127.0.0.1:47684",
                    codex_live_client=live,
                )
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                )

                handled = bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertEqual(live.calls, [("codex-s1", "continue", Path(tmp))])
                self.assertEqual(calls, [])
                self.assertTrue(
                    store.consume_session_bridge_prompt(Provider.CODEX, "codex-s1", "continue")
                )
            finally:
                store.close()

    def test_codex_slash_text_uses_regular_app_server_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            live = FakeCodexLiveClient()

            def runner(args, **kwargs):
                calls.append(args)
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(codex_binary="codex-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    codex_app_server_url="ws://127.0.0.1:47684",
                    codex_live_client=live,
                )
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                )

                handled = bridge.send_to_session(
                    session,
                    "/model gpt-5",
                    SlackThreadRef("C1", "171"),
                )

                self.assertTrue(handled)
                self.assertEqual(live.calls, [("codex-s1", "/model gpt-5", Path(tmp))])
                self.assertEqual(calls, [])
                self.assertTrue(
                    store.consume_session_bridge_prompt(
                        Provider.CODEX,
                        "codex-s1",
                        "/model gpt-5",
                    )
                )
            finally:
                store.close()

    def test_codex_slash_text_without_app_server_uses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()

            def runner(args, **kwargs):
                calls.append(args)
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(
                        codex_binary="codex-bin",
                        default_cwd=Path(tmp),
                        codex_app_server_url="off",
                    ),
                    command_runner=runner,
                )
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                )

                handled = bridge.send_to_session(
                    session,
                    "/model gpt-5",
                    SlackThreadRef("C1", "171"),
                )

                self.assertTrue(handled)
                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(calls[0][:4], ["codex-bin", "exec", "resume", "--json"])
                self.assertTrue(
                    store.consume_session_bridge_prompt(
                        Provider.CODEX,
                        "codex-s1",
                        "/model gpt-5",
                    )
                )
            finally:
                store.close()

    def test_codex_app_server_failure_falls_back_to_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()
            live = FakeCodexLiveClient(RuntimeError("no app server"))

            def runner(args, **kwargs):
                calls.append(args)
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(codex_binary="codex-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    codex_app_server_url="ws://127.0.0.1:47684",
                    codex_live_client=live,
                )
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="codex-s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                )

                handled = bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(calls[0][:4], ["codex-bin", "exec", "resume", "--json"])
            finally:
                store.close()

    def test_claude_slash_text_without_channel_uses_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            calls = []
            ran = threading.Event()
            terminal = FakeTerminalNotifier()

            def runner(args, **kwargs):
                calls.append(args)
                ran.set()
                return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

            try:
                store.init_schema()
                gateway = FakeGateway()
                bridge = ExternalSessionBridge(
                    store,
                    gateway,
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                    channel_delivery_timeout=0.01,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )

                handled = bridge.send_to_session(session, "/compact", SlackThreadRef("C1", "171"))

                self.assertTrue(handled)
                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(
                    calls[0][:5],
                    ["claude-bin", "--print", "--output-format", "json", "--resume"],
                )
            finally:
                store.close()

    def test_successful_send_writes_agent_response_to_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            ran = threading.Event()
            terminal = FakeTerminalNotifier()

            def runner(args, **kwargs):
                ran.set()
                return type(
                    "Completed",
                    (),
                    {
                        "returncode": 0,
                        "stdout": '{"type":"result","result":"done"}\n',
                        "stderr": "",
                    },
                )()

            try:
                store.init_schema()
                bridge = ExternalSessionBridge(
                    store,
                    FakeGateway(),
                    AgentCommandConfig(claude_binary="claude-bin", default_cwd=Path(tmp)),
                    command_runner=runner,
                    terminal_notifier=terminal,
                )
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                )

                bridge.send_to_session(session, "continue", SlackThreadRef("C1", "171"))

                self.assertTrue(ran.wait(timeout=1))
                self.assertEqual(terminal.agent_responses[0][1], "done")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
