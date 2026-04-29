import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_harness.models import AgentEvent, AgentSession, Provider, SessionStatus, SlackThreadRef
from agent_harness.sessions.mirror import SessionMirror, render_session_event
from agent_harness.sessions.terminal import TerminalTarget
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task


class FakeGateway:
    def __init__(self):
        self.parents = []
        self.replies = []
        self.posts = []
        self.updates = []
        self.calls = []

    def post_message(self, channel_id, text, blocks=None, thread_ts=None):
        self.calls.append(("post_message", channel_id, text))
        self.posts.append((channel_id, text, blocks, thread_ts))
        ts = f"170.{len(self.posts):06d}"
        return type("Posted", (), {"ts": ts})()

    def update_message(self, channel_id, ts, text, blocks=None):
        self.calls.append(("update_message", channel_id, ts, text))
        self.updates.append((channel_id, ts, text, blocks))

    def post_session_parent(self, channel_id, text, persona, icon_url=None, blocks=None):
        self.calls.append(("post_session_parent", channel_id, text))
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
        self.calls.append(("post_thread_reply", thread.channel_id, thread.thread_ts, text))
        self.replies.append((thread, text, persona, username, icon_url))
        ts = f"173.{len(self.replies):06d}"
        return type("Posted", (), {"ts": ts})()


class FakeProvider:
    provider = Provider.CODEX

    def __init__(self, session, events):
        self.session = session
        self.events = events

    def discover(self):
        if isinstance(self.session, list):
            return self.session
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


def _add_team(store, codex_count=1, claude_count=1):
    for agent in build_initial_model_team(codex_count=codex_count, claude_count=claude_count):
        store.upsert_team_agent(agent)


class SessionMirrorTests(unittest.TestCase):
    def test_initial_sync_marks_existing_events_without_posting(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store)
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

    def test_ignored_external_session_is_not_remirrored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store)
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
                        metadata={"payload": {"type": "agent_message", "message": "visible"}},
                    )
                ]
                store.set_setting("external_session_ignored.codex.s1", "now")
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(session, events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual(gateway.parents, [])
                self.assertEqual(store.get_setting("external_session_agent.codex.s1"), None)
            finally:
                store.close()

    def test_unthreaded_session_posts_parent_with_first_visible_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store)
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
                self.assertEqual(len(gateway.updates), 1)
                self.assertIn("Task: visible", gateway.updates[0][2])
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 2)
            finally:
                store.close()

    def test_external_session_parent_summary_prefers_user_task_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store)
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
                        metadata={
                            "payload": {
                                "type": "user_message",
                                "message": "make the README punchier",
                            }
                        },
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

                self.assertEqual(len(gateway.updates), 1)
                self.assertIn("Task: make the README punchier", gateway.updates[0][2])
                self.assertEqual(
                    store.get_setting("external_session_summary.codex.s1"),
                    "make the README punchier",
                )
            finally:
                store.close()

    def test_external_session_summary_is_not_overwritten_by_later_assistant_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(session)
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                store.set_setting(
                    "external_session_agent.codex.s1", store.list_team_agents()[0].agent_id
                )
                store.set_setting("external_session_summary.codex.s1", "make the README punchier")
                store.set_session_mirror_cursor(Provider.CODEX, "s1", 1)
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=2,
                        metadata={
                            "payload": {
                                "type": "agent_message",
                                "message": "I inspected several files and found more details.",
                            }
                        },
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

                self.assertEqual(
                    store.get_setting("external_session_summary.codex.s1"),
                    "make the README punchier",
                )
                self.assertEqual(gateway.updates, [])
            finally:
                store.close()

    def test_new_external_session_is_mirrored_before_existing_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=2, claude_count=0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                existing = AgentSession(
                    provider=Provider.CODEX,
                    session_id="existing",
                    transcript_path=Path(tmp) / "existing.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                new = AgentSession(
                    provider=Provider.CODEX,
                    session_id="new",
                    transcript_path=Path(tmp) / "new.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                store.set_setting("external_session_agent.codex.existing", agents[0].agent_id)
                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "existing",
                    "T1",
                    SlackThreadRef("C1", "171.existing", "171.existing"),
                )
                existing_provider = FakeProvider(
                    existing,
                    [
                        AgentEvent(
                            provider=Provider.CODEX,
                            session_id="existing",
                            timestamp=None,
                            event_type="event_msg",
                            line_number=1,
                            metadata={
                                "payload": {
                                    "type": "agent_message",
                                    "message": "existing visible",
                                }
                            },
                        )
                    ],
                )
                new_provider = FakeProvider(
                    new,
                    [
                        AgentEvent(
                            provider=Provider.CODEX,
                            session_id="new",
                            timestamp=None,
                            event_type="event_msg",
                            line_number=1,
                            metadata={
                                "payload": {"type": "agent_message", "message": "new visible"}
                            },
                        )
                    ],
                )
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [existing_provider, new_provider],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual(gateway.calls[0][0], "post_session_parent")
                self.assertEqual(
                    gateway.calls[1], ("post_thread_reply", "C1", "171.000001", "new visible")
                )
                self.assertEqual(
                    gateway.calls[3],
                    ("post_thread_reply", "C1", "171.existing", "existing visible"),
                )
            finally:
                store.close()

    def test_unthreaded_session_without_team_posts_provider_capacity_message(self):
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
                )

                mirror.sync_once()
                mirror.sync_once()

                self.assertEqual(gateway.parents, [])
                self.assertEqual(gateway.replies, [])
                self.assertEqual(len(gateway.posts), 1)
                self.assertIn("No Codex team seat is available", gateway.posts[0][1])
                self.assertIn("Hire 1 Codex agent", str(gateway.posts[0][2]))
            finally:
                store.close()

    def test_external_sessions_do_not_overfill_available_provider_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store, codex_count=1, claude_count=0)
                first = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex-1.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                second = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s2",
                    transcript_path=Path(tmp) / "codex-2.jsonl",
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
                    [FakeProvider([first, second], events)],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                self.assertEqual(len(gateway.parents), 1)
                self.assertEqual(len(gateway.posts), 1)
                self.assertIn("No Codex team seat is available", gateway.posts[0][1])
                self.assertIn("Hire 1 Codex agent", str(gateway.posts[0][2]))
            finally:
                store.close()

    def test_external_session_assignment_refreshes_occupancy_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                refreshed_channels = []
                mirror = SessionMirror(
                    store,
                    FakeGateway(),
                    [FakeProvider(session, [])],
                    team_id="T1",
                    channel_id="C1",
                    on_external_session_occupancy_change=refreshed_channels.append,
                )

                mirror.sync_once(backfill_new_sessions=False)

                self.assertEqual(
                    store.get_setting("external_session_agent.codex.s1"),
                    agents[0].agent_id,
                )
                self.assertEqual(refreshed_channels, ["C1"])
            finally:
                store.close()

    def test_inactive_external_session_cleanup_updates_status_and_refreshes_occupancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                active = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                done = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.DONE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_session(active)
                store.set_setting("external_session_agent.codex.s1", agents[0].agent_id)
                store.set_setting(
                    "external_session_summary.codex.s1",
                    "<local-command-caveat>internal</local-command-caveat>",
                )
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                gateway = FakeGateway()
                refreshed_channels = []
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider(done, [])],
                    team_id="T1",
                    channel_id="C1",
                    on_external_session_occupancy_change=refreshed_channels.append,
                )

                mirror.sync_once()

                self.assertIsNone(store.get_setting("external_session_agent.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_summary.codex.s1"))
                self.assertEqual(store.get_session(Provider.CODEX, "s1").status, SessionStatus.DONE)
                self.assertEqual(
                    [reply[1] for reply in gateway.replies],
                    ["Session ended; freed up this agent."],
                )
                self.assertEqual(refreshed_channels, ["C1"])
            finally:
                store.close()

    def test_external_session_does_not_take_agent_with_managed_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                store.upsert_agent_task(create_agent_task(agents[0], "busy task", "C1"))
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
                )

                mirror.sync_once()

                self.assertEqual(gateway.parents, [])
                self.assertEqual(len(gateway.posts), 1)
                self.assertIn("No Codex team seat is available", gateway.posts[0][1])
            finally:
                store.close()

    def test_managed_task_session_is_not_mirrored_as_external_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                task = create_agent_task(agents[0], "managed task", "C1")
                store.upsert_agent_task(task)
                store.update_agent_task_session(task.task_id, Provider.CODEX, "s1")
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.set_setting("external_session_agent.codex.s1", agents[0].agent_id)
                store.set_setting("external_session_pending.codex.s1", "now")
                store.set_setting("external_session_summary.codex.s1", "stale")
                store.set_setting("session_channel_notice.codex.s1", "now")
                store.set_setting("external_session_capacity_notice_ts.codex", "170.000001")
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                store.set_session_mirror_cursor(Provider.CODEX, "s1", 3)
                events = [
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=4,
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
                )

                mirror.sync_once()

                self.assertEqual(gateway.parents, [])
                self.assertEqual(gateway.replies, [])
                self.assertIsNone(store.get_setting("external_session_agent.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_pending.codex.s1"))
                self.assertIsNone(store.get_setting("external_session_summary.codex.s1"))
                self.assertIsNone(store.get_setting("session_channel_notice.codex.s1"))
                self.assertIsNone(
                    store.get_slack_thread_for_session(Provider.CODEX, "s1", "T1", "C1")
                )
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 0)
                self.assertEqual(len(gateway.updates), 1)
                self.assertIn(
                    "Codex capacity for sessions started outside Slack is available now.",
                    gateway.updates[0][2],
                )
            finally:
                store.close()

    def test_external_session_requires_matching_provider_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store, codex_count=1, claude_count=0)
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
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
                )

                mirror.sync_once()

                self.assertEqual(gateway.parents, [])
                self.assertEqual(gateway.replies, [])
                self.assertEqual(len(gateway.posts), 1)
                self.assertIn("No Claude team seat is available", gateway.posts[0][1])
                self.assertIn("Hire 1 Claude agent", str(gateway.posts[0][2]))
            finally:
                store.close()

    def test_pending_external_session_backfills_after_matching_agent_is_hired(self):
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
                        metadata={"payload": {"type": "agent_message", "message": "first"}},
                    ),
                    AgentEvent(
                        provider=Provider.CODEX,
                        session_id="s1",
                        timestamp=None,
                        event_type="event_msg",
                        line_number=2,
                        metadata={"payload": {"type": "agent_message", "message": "second"}},
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
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 0)

                _add_team(store, codex_count=1, claude_count=0)
                mirror.sync_once()

                self.assertEqual(len(gateway.parents), 1)
                self.assertEqual([reply[1] for reply in gateway.replies], ["first", "second"])
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 2)
            finally:
                store.close()

    def test_later_sync_posts_only_new_user_visible_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store)
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
                _add_team(store)
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
                _add_team(store)
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
                _add_team(store)
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
                _add_team(store)
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

    def test_render_claude_event_hides_slackgentic_channel_user_event(self):
        event = AgentEvent(
            provider=Provider.CLAUDE,
            session_id="s1",
            timestamp=None,
            event_type="user",
            metadata={
                "message": {
                    "content": (
                        '<channel source="slackgentic" session_id="s1" '
                        'slack_channel="C1" slack_thread_ts="171.000001">'
                        "approve it"
                        "</channel>"
                    )
                }
            },
        )

        self.assertIsNone(render_session_event(event))

    def test_render_claude_event_hides_local_command_records(self):
        for text in (
            "<local-command-caveat>internal</local-command-caveat>",
            "<command-name>/exit</command-name>\n<command-message>exit</command-message>",
            "<local-command-stdout>Bye!</local-command-stdout>",
        ):
            event = AgentEvent(
                provider=Provider.CLAUDE,
                session_id="s1",
                timestamp=None,
                event_type="user",
                metadata={"message": {"content": text}},
            )

            self.assertIsNone(render_session_event(event))

    def test_render_claude_event_strips_echoed_slackgentic_channel_block(self):
        event = AgentEvent(
            provider=Provider.CLAUDE,
            session_id="s1",
            timestamp=None,
            event_type="assistant",
            metadata={
                "message": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I saw this:\n"
                                "&lt;channel source=&quot;slackgentic&quot;&gt;"
                                "approve it"
                                "&lt;/channel&gt;\nDone."
                            ),
                        }
                    ]
                }
            },
        )

        rendered = render_session_event(event)
        self.assertEqual(rendered, "I saw this:\n\nDone.")
        assert rendered is not None
        self.assertNotIn("channel source", rendered)

    def test_codex_parent_warns_when_session_lacks_remote_app_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                _add_team(store)
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
                _add_team(store)
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
                _add_team(store)
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
                _add_team(store)
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
                _add_team(store)
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
                _add_team(store)
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

    def test_ended_external_session_frees_assigned_agent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=0)
                for agent in agents:
                    store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CODEX,
                    session_id="s1",
                    transcript_path=Path(tmp) / "codex.jsonl",
                    status=SessionStatus.ACTIVE,
                )
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                store.upsert_slack_thread_for_session(Provider.CODEX, "s1", "T1", thread)
                provider = FakeProvider([session], [])
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [provider],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()
                self.assertEqual(
                    store.get_setting("external_session_agent.codex.s1"),
                    agents[0].agent_id,
                )

                provider.session = []
                mirror.sync_once()

                self.assertIsNone(store.get_setting("external_session_agent.codex.s1"))
                self.assertEqual(gateway.replies[-1][1], "Session ended; freed up this agent.")
            finally:
                store.close()

    def test_closed_live_claude_terminal_frees_assigned_agent_without_exit_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=0, claude_count=1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                session = AgentSession(
                    provider=Provider.CLAUDE,
                    session_id="s1",
                    transcript_path=Path(tmp) / "claude.jsonl",
                    cwd=Path(tmp),
                    status=SessionStatus.ACTIVE,
                )
                notifier = FakeTerminalNotifier(
                    [
                        TerminalTarget(
                            pid=123,
                            tty="ttys001",
                            cwd=Path(tmp),
                            command=(
                                "claude --dangerously-load-development-channels server:slackgentic"
                            ),
                        )
                    ]
                )
                gateway = FakeGateway()
                mirror = SessionMirror(
                    store,
                    gateway,
                    [FakeProvider([session], [])],
                    team_id="T1",
                    channel_id="C1",
                    terminal_notifier=notifier,
                )

                mirror.sync_once(backfill_new_sessions=False)
                self.assertEqual(
                    store.get_setting("external_session_agent.claude.s1"),
                    agents[0].agent_id,
                )
                self.assertEqual(store.get_setting("external_session_live_target.claude.s1"), "123")
                store.upsert_slack_thread_for_session(
                    Provider.CLAUDE,
                    "s1",
                    "T1",
                    SlackThreadRef("C1", "171.000001", "171.000001"),
                )

                notifier.targets = []
                mirror.sync_once()
                self.assertEqual(
                    store.get_setting("external_session_agent.claude.s1"),
                    agents[0].agent_id,
                )

                mirror.sync_once()

                self.assertIsNone(store.get_setting("external_session_agent.claude.s1"))
                self.assertIsNone(store.get_setting("external_session_live_target.claude.s1"))
                self.assertIsNone(store.get_setting("external_session_missing_target.claude.s1"))
                self.assertEqual(
                    store.get_session(Provider.CLAUDE, "s1").status, SessionStatus.DONE
                )
                self.assertEqual(gateway.replies[-1][1], "Session ended; freed up this agent.")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
