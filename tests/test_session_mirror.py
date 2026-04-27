import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_harness.models import AgentEvent, AgentSession, Provider, SessionStatus, SlackThreadRef
from agent_harness.session_mirror import SessionMirror, render_session_event
from agent_harness.session_terminal import TerminalTarget
from agent_harness.store import Store


class FakeGateway:
    def __init__(self):
        self.parents = []
        self.replies = []

    def post_session_parent(self, channel_id, text, persona, icon_url=None, blocks=None):
        self.parents.append((channel_id, text, persona))
        ts = f"171.{len(self.parents):06d}"
        return type("Posted", (), {"ts": ts})()

    def post_thread_reply(
        self,
        thread,
        text,
        persona=None,
        username=None,
        icon_url=None,
        icon_emoji=None,
        blocks=None,
    ):
        self.replies.append((thread, text, persona, username, icon_url))
        ts = f"173.{len(self.replies):06d}"
        return type("Posted", (), {"ts": ts})()


class FakeProvider:
    provider = Provider.CODEX

    def __init__(self, session, events):
        self.session = session
        self.events = events

    def discover(self):
        return [self.session]

    def iter_events(self, transcript_path):
        return iter(self.events)

    def usage_for_day(self, transcript_paths, day):
        return []


class FakeTerminalNotifier:
    def __init__(self, targets=None):
        self.targets = targets or []

    def targets_for_session(self, session):
        return self.targets


class SessionMirrorTests(unittest.TestCase):
    def test_initial_sync_marks_existing_events_without_posting(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=1,
                        metadata={"payload": {"type": "agent_message", "message": "old"}},
                    )
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once(backfill_new_sessions=False)

                self.assertEqual(gateway.parents, [])
                self.assertEqual(gateway.replies, [])
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 1)
            finally:
                store.close()

    def test_unthreaded_session_posts_parent_with_first_visible_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="response_item",
                        line_number=1,
                        metadata={"payload": {"type": "function_call", "name": "exec_command"}},
                    ),
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=2,
                        metadata={"payload": {"type": "agent_message", "message": "visible"}},
                    ),
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual(len(gateway.parents), 1)
                self.assertEqual([reply[1] for reply in gateway.replies], ["visible"])
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 2)
            finally:
                store.close()

    def test_later_sync_posts_only_new_user_visible_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                store.set_session_mirror_cursor(Provider.CODEX, "s1", 1)
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="response_item",
                        line_number=2,
                        metadata={"payload": {"type": "function_call", "name": "exec_command"}},
                    ),
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=3,
                        metadata={"payload": {"type": "agent_message", "message": "visible"}},
                    ),
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual([reply[1] for reply in gateway.replies], ["visible"])
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 3)
            finally:
                store.close()

    def test_local_user_messages_are_mirrored_into_existing_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=1,
                        metadata={"payload": {"type": "user_message", "message": "hello"}},
                    )
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual([reply[1] for reply in gateway.replies], ["hello"])
                self.assertTrue(store.is_mirrored_slack_message("C1", "173.000001"))
            finally:
                store.close()

    def test_slack_injected_user_messages_are_not_mirrored_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                store.add_session_bridge_prompt(Provider.CODEX, "s1", "from slack")
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=1,
                        metadata={"payload": {"type": "user_message", "message": "from slack"}},
                    ),
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=2,
                        metadata={"payload": {"type": "agent_message", "message": "done"}},
                    ),
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual([reply[1] for reply in gateway.replies], ["done"])
                self.assertFalse(
                    store.consume_session_bridge_prompt(Provider.CODEX, "s1", "from slack")
                )
            finally:
                store.close()

    def test_local_user_messages_use_configured_human_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.set_setting("slack.human_display_name", "Local User")
                store.set_setting("slack.human_image_url", "https://example.com/avatar.png")
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=1,
                        metadata={"payload": {"type": "user_message", "message": "hello"}},
                    )
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual(gateway.replies[0][1], "hello")
                self.assertIsNone(gateway.replies[0][2])
                self.assertEqual(gateway.replies[0][3], "Local User")
                self.assertEqual(gateway.replies[0][4], "https://example.com/avatar.png")
            finally:
                store.close()

    def test_sync_skips_unthreaded_sessions_last_seen_before_mirror_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                    last_seen_at=datetime.now(UTC) - timedelta(minutes=5),
                )
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, [])],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual(gateway.parents, [])
            finally:
                store.close()

    def test_render_claude_event_hides_tool_traffic(self):
        tool_event = AgentEvent(
            provider=Provider.CLAUDE,
            session_id="s1",
            timestamp=None,
            event_type="assistant",
            metadata={"message": {"content": [{"type": "tool_use", "name": "Read"}]}},
        )
        text_event = AgentEvent(
            provider=Provider.CLAUDE,
            session_id="s1",
            timestamp=None,
            event_type="assistant",
            metadata={"message": {"content": [{"type": "text", "text": "done"}]}},
        )

        self.assertIsNone(render_session_event(tool_event))
        self.assertEqual(render_session_event(text_event), "done")

    def test_codex_parent_warns_when_session_lacks_remote_app_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=1,
                        metadata={"payload": {"type": "agent_message", "message": "visible"}},
                    )
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                    terminal_notifier=FakeTerminalNotifier(
                        [
                            TerminalTarget(
                                pid=123,
                                tty="ttys001",
                                cwd=Path(tmp),
                                command="codex",
                            )
                        ]
                    ),
                    codex_app_server_url="ws://127.0.0.1:47684",
                )

                mirror.sync_once()

                self.assertEqual(len(gateway.parents), 1)
                self.assertIn(
                    "not started against Slackgentic's Codex app-server", gateway.parents[0][1]
                )
                self.assertIn("codex --remote ws://127.0.0.1:47684", gateway.parents[0][1])
            finally:
                store.close()

    def test_codex_parent_skips_warning_when_remote_app_server_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=1,
                        metadata={"payload": {"type": "agent_message", "message": "visible"}},
                    )
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                    terminal_notifier=FakeTerminalNotifier(
                        [
                            TerminalTarget(
                                pid=123,
                                tty="ttys001",
                                cwd=Path(tmp),
                                command="codex --remote ws://localhost:47684",
                            )
                        ]
                    ),
                    codex_app_server_url="ws://127.0.0.1:47684",
                )

                mirror.sync_once()

                self.assertEqual(len(gateway.parents), 1)
                self.assertNotIn("Restart it with", gateway.parents[0][1])
            finally:
                store.close()

    def test_claude_parent_warns_when_session_lacks_channel_launch_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )
                events = [
                    AgentEvent(
                        provider=Provider.CLAUDE,
                        session_id="s1",
                        timestamp=None,
                        event_type="assistant",
                        line_number=1,
                        metadata={"message": {"content": [{"type": "text", "text": "visible"}]}},
                    )
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                    terminal_notifier=FakeTerminalNotifier(
                        [
                            TerminalTarget(
                                pid=123,
                                tty="ttys001",
                                cwd=Path(tmp),
                                command="claude",
                            )
                        ]
                    ),
                )

                mirror.sync_once()

                self.assertEqual(len(gateway.parents), 1)
                self.assertIn(
                    "not started with Slackgentic's Claude channel", gateway.parents[0][1]
                )
                self.assertIn(
                    "claude --dangerously-load-development-channels server:slackgentic",
                    gateway.parents[0][1],
                )
            finally:
                store.close()

    def test_claude_parent_skips_warning_when_channel_launch_flag_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )
                events = [
                    AgentEvent(
                        provider=Provider.CLAUDE,
                        session_id="s1",
                        timestamp=None,
                        event_type="assistant",
                        line_number=1,
                        metadata={"message": {"content": [{"type": "text", "text": "visible"}]}},
                    )
                ]
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                    terminal_notifier=FakeTerminalNotifier(
                        [
                            TerminalTarget(
                                pid=123,
                                tty="ttys001",
                                cwd=Path(tmp),
                                command=(
                                    "claude --dangerously-load-development-channels "
                                    "server:slackgentic"
                                ),
                            )
                        ]
                    ),
                )

                mirror.sync_once()

                self.assertEqual(len(gateway.parents), 1)
                self.assertNotIn("Restart it with", gateway.parents[0][1])
            finally:
                store.close()

    def test_existing_claude_thread_gets_missing_channel_warning_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_slack_thread_for_session(Provider.CLAUDE, "s1", "T1", thread)
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, [])],
                    team_id="T1",
                    channel_id="C1",
                    terminal_notifier=FakeTerminalNotifier(
                        [
                            TerminalTarget(
                                pid=123,
                                tty="ttys001",
                                cwd=Path(tmp),
                                command="claude",
                            )
                        ]
                    ),
                )

                mirror.sync_once()
                mirror.sync_once()

                self.assertEqual(len(gateway.replies), 1)
                self.assertIn(
                    "not started with Slackgentic's Claude channel", gateway.replies[0][1]
                )
            finally:
                store.close()

    def test_existing_codex_thread_gets_missing_remote_warning_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, [])],
                    team_id="T1",
                    channel_id="C1",
                    terminal_notifier=FakeTerminalNotifier(
                        [
                            TerminalTarget(
                                pid=123,
                                tty="ttys001",
                                cwd=Path(tmp),
                                command="codex",
                            )
                        ]
                    ),
                    codex_app_server_url="ws://127.0.0.1:47684",
                )

                mirror.sync_once()
                mirror.sync_once()

                self.assertEqual(len(gateway.replies), 1)
                self.assertIn(
                    "not started against Slackgentic's Codex app-server",
                    gateway.replies[0][1],
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
