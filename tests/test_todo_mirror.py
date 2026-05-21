import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.models import AgentSession, Provider, SessionStatus, SlackThreadRef
from agent_harness.sessions.todo_mirror import (
    TodoItem,
    TodoMirror,
    TodoSnapshot,
    claude_todo_snapshot,
    codex_latest_plan_after,
    render_snapshot,
    todo_message_ts_key,
)
from agent_harness.storage.store import Store


class FakeGateway:
    def __init__(self):
        self.replies = []
        self.updates = []
        self.fail_post = False
        self.fail_update = False

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
        if self.fail_post:
            raise RuntimeError("post failed")
        self.replies.append((thread.channel_id, thread.thread_ts, text))
        ts = f"170.{len(self.replies):06d}"
        return type("Posted", (), {"ts": ts})()

    def update_message(self, channel_id, ts, text, blocks=None):
        if self.fail_update:
            raise RuntimeError("update failed")
        self.updates.append((channel_id, ts, text))


def _write_claude_task(dir_path: Path, task_id: int, **fields) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {"id": str(task_id), **fields}
    (dir_path / f"{task_id}.json").write_text(json.dumps(payload))


def _claude_session(home: Path, session_id: str = "s-claude") -> AgentSession:
    return AgentSession(
        provider=Provider.CLAUDE,
        session_id=session_id,
        transcript_path=home / ".claude" / "projects" / "x" / f"{session_id}.jsonl",
        status=SessionStatus.ACTIVE,
    )


def _codex_session(transcript_path: Path, session_id: str = "s-codex") -> AgentSession:
    return AgentSession(
        provider=Provider.CODEX,
        session_id=session_id,
        transcript_path=transcript_path,
        status=SessionStatus.ACTIVE,
    )


def _write_codex_plan(transcript_path: Path, plan: list[dict]) -> None:
    record = {
        "timestamp": "2026-04-27T15:00:00.000Z",
        "type": "response_item",
        "payload": {
            "type": "function_call",
            "name": "update_plan",
            "arguments": json.dumps({"plan": plan}),
            "call_id": "call_x",
        },
    }
    with transcript_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class ClaudeTodoSnapshotTests(unittest.TestCase):
    def test_returns_none_when_no_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(claude_todo_snapshot(Path(tmp), "missing"))

    def test_sorts_tasks_numerically_by_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            tasks_dir = home / ".claude" / "tasks" / "s1"
            _write_claude_task(tasks_dir, 10, subject="ten", activeForm="Ten-ing", status="pending")
            _write_claude_task(
                tasks_dir, 2, subject="two", activeForm="Two-ing", status="completed"
            )
            _write_claude_task(
                tasks_dir, 1, subject="one", activeForm="One-ing", status="in_progress"
            )
            snapshot = claude_todo_snapshot(home, "s1")
            self.assertIsNotNone(snapshot)
            assert snapshot is not None  # for type checkers
            self.assertEqual([item.title for item in snapshot.items], ["one", "two", "ten"])
            self.assertEqual(
                [item.status for item in snapshot.items],
                ["in_progress", "completed", "pending"],
            )
            self.assertEqual(snapshot.items[0].detail, "One-ing")

    def test_skips_invalid_json_and_dotfiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            tasks_dir = home / ".claude" / "tasks" / "s1"
            tasks_dir.mkdir(parents=True)
            (tasks_dir / "1.json").write_text("{not json")
            (tasks_dir / ".highwatermark").write_text("3")
            (tasks_dir / "2.json").write_text(json.dumps({"id": "2", "subject": "ok"}))
            snapshot = claude_todo_snapshot(home, "s1")
            assert snapshot is not None
            self.assertEqual([item.title for item in snapshot.items], ["ok"])


class CodexLatestPlanTests(unittest.TestCase):
    def test_returns_none_when_path_missing(self):
        snapshot, cursor = codex_latest_plan_after(Path("/no/such/path.jsonl"), 0)
        self.assertIsNone(snapshot)
        self.assertEqual(cursor, 0)

    def test_returns_latest_plan_and_advances_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "codex.jsonl"
            transcript.write_text("")
            _write_codex_plan(
                transcript,
                [
                    {"step": "A", "status": "in_progress"},
                    {"step": "B", "status": "pending"},
                ],
            )
            _write_codex_plan(
                transcript,
                [
                    {"step": "A", "status": "completed"},
                    {"step": "B", "status": "in_progress"},
                ],
            )
            snapshot, cursor = codex_latest_plan_after(transcript, 0)
            assert snapshot is not None
            self.assertEqual([item.title for item in snapshot.items], ["A", "B"])
            self.assertEqual(
                [item.status for item in snapshot.items],
                ["completed", "in_progress"],
            )
            self.assertEqual(cursor, 2)

    def test_returns_none_when_no_new_plan_after_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "codex.jsonl"
            transcript.write_text("")
            _write_codex_plan(transcript, [{"step": "A", "status": "pending"}])
            snapshot, cursor = codex_latest_plan_after(transcript, 1)
            self.assertIsNone(snapshot)
            self.assertEqual(cursor, 1)


class RenderSnapshotTests(unittest.TestCase):
    def test_in_progress_prefers_active_form(self):
        snapshot = TodoSnapshot(
            items=(
                TodoItem(title="Find release path", status="completed"),
                TodoItem(
                    title="Fix fallback",
                    status="in_progress",
                    detail="Fixing the fallback",
                ),
                TodoItem(title="Dedup follow-ups", status="pending"),
            )
        )
        text = render_snapshot(snapshot)
        self.assertIn("Find release path", text)
        self.assertIn("Fixing the fallback", text)
        self.assertIn("Dedup follow-ups", text)
        self.assertTrue(text.startswith("*Tasks*"))

    def test_unknown_status_falls_back_to_pending_glyph(self):
        snapshot = TodoSnapshot(items=(TodoItem(title="x", status="weird"),))
        text = render_snapshot(snapshot)
        self.assertIn("◻", text)


class TodoMirrorIntegrationTests(unittest.TestCase):
    def _store(self, tmp: Path) -> Store:
        store = Store(tmp / "state.sqlite")
        store.init_schema()
        return store

    def test_claude_first_run_posts_then_subsequent_updates_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = self._store(home)
            try:
                gateway = FakeGateway()
                mirror = TodoMirror(store, gateway, home=home)
                session = _claude_session(home)
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                tasks_dir = home / ".claude" / "tasks" / session.session_id
                _write_claude_task(
                    tasks_dir, 1, subject="First", activeForm="First-ing", status="in_progress"
                )

                mirror.sync_session(session, thread)
                self.assertEqual(len(gateway.replies), 1)
                self.assertEqual(gateway.updates, [])
                first_ts = store.get_setting(todo_message_ts_key(session))
                self.assertEqual(first_ts, "170.000001")

                _write_claude_task(
                    tasks_dir, 1, subject="First", activeForm="First-ing", status="completed"
                )
                _write_claude_task(
                    tasks_dir, 2, subject="Second", activeForm="Second-ing", status="in_progress"
                )

                mirror.sync_session(session, thread)
                self.assertEqual(len(gateway.replies), 1)
                self.assertEqual(len(gateway.updates), 1)
                self.assertEqual(gateway.updates[0][1], "170.000001")
                self.assertIn("Second-ing", gateway.updates[0][2])
            finally:
                store.close()

    def test_no_slack_call_when_snapshot_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = self._store(home)
            try:
                gateway = FakeGateway()
                mirror = TodoMirror(store, gateway, home=home)
                session = _claude_session(home)
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                _write_claude_task(
                    home / ".claude" / "tasks" / session.session_id,
                    1,
                    subject="Same",
                    status="in_progress",
                    activeForm="Same-ing",
                )
                mirror.sync_session(session, thread)
                mirror.sync_session(session, thread)
                self.assertEqual(len(gateway.replies), 1)
                self.assertEqual(gateway.updates, [])
            finally:
                store.close()

    def test_codex_uses_transcript_function_call_and_caches_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = self._store(home)
            try:
                gateway = FakeGateway()
                mirror = TodoMirror(store, gateway, home=home)
                transcript = home / "codex.jsonl"
                transcript.write_text("")
                _write_codex_plan(
                    transcript,
                    [
                        {"step": "Alpha", "status": "in_progress"},
                        {"step": "Beta", "status": "pending"},
                    ],
                )
                session = _codex_session(transcript)
                thread = SlackThreadRef("C1", "171.000001", "171.000001")

                mirror.sync_session(session, thread)
                self.assertEqual(len(gateway.replies), 1)
                first_text = gateway.replies[0][2]
                self.assertIn("Alpha", first_text)
                self.assertIn("Beta", first_text)

                # No new plan -> no update, even though we re-scan.
                mirror.sync_session(session, thread)
                self.assertEqual(gateway.updates, [])

                _write_codex_plan(
                    transcript,
                    [
                        {"step": "Alpha", "status": "completed"},
                        {"step": "Beta", "status": "in_progress"},
                    ],
                )
                mirror.sync_session(session, thread)
                self.assertEqual(len(gateway.updates), 1)
                self.assertIn("Alpha", gateway.updates[0][2])
                self.assertIn("Beta", gateway.updates[0][2])
            finally:
                store.close()

    def test_post_failure_does_not_persist_ts(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            store = self._store(home)
            try:
                gateway = FakeGateway()
                gateway.fail_post = True
                mirror = TodoMirror(store, gateway, home=home)
                session = _claude_session(home)
                thread = SlackThreadRef("C1", "171.000001", "171.000001")
                _write_claude_task(
                    home / ".claude" / "tasks" / session.session_id,
                    1,
                    subject="x",
                    status="pending",
                )
                mirror.sync_session(session, thread)
                self.assertIsNone(store.get_setting(todo_message_ts_key(session)))

                # Recovery: a successful post on the next call should now persist.
                gateway.fail_post = False
                mirror.sync_session(session, thread)
                self.assertEqual(store.get_setting(todo_message_ts_key(session)), "170.000001")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
