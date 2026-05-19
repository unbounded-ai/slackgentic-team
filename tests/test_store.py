import tempfile
import threading
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from agent_harness.models import (
    AgentSession,
    AgentTaskKind,
    AgentTaskStatus,
    AssignmentMode,
    ControlMode,
    DeferredWorkStatus,
    PendingWorkRequestStatus,
    Provider,
    ScheduledWorkKind,
    SessionDependency,
    SessionStatus,
    SlackThreadRef,
    WorkDependency,
    WorkDependencyKind,
    WorkRequest,
    deferred_work_dependency_id,
    utc_now,
)
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task


class StoreTests(unittest.TestCase):
    def test_dependencies_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.add_dependency(
                    SessionDependency(
                        blocked_session_id="blocked",
                        blocking_thread=SlackThreadRef(
                            channel_id="C1",
                            thread_ts="171.000001",
                            message_ts="171.000001",
                            permalink="https://example.slack.com/archives/C1/p0000000171000001",
                        ),
                        created_by_slack_user="U1",
                        reason="wait for this",
                    )
                )
                deps = store.dependencies_for("blocked")
                self.assertEqual(len(deps), 1)
                self.assertEqual(deps[0].blocking_thread.channel_id, "C1")
                self.assertEqual(deps[0].created_by_slack_user, "U1")
            finally:
                store.close()

    def test_team_agents_and_tasks_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(codex_count=1, claude_count=1)
                for agent in agents:
                    store.upsert_team_agent(agent)

                listed = store.list_team_agents()
                self.assertEqual(len(listed), 2)
                self.assertEqual(listed[0].provider_preference, Provider.CODEX)

                task = create_agent_task(listed[0], "do the thing", "C1")
                store.upsert_agent_task(task)
                self.assertEqual(store.active_task_for_agent(listed[0].agent_id), task)
                self.assertEqual(len(store.idle_team_agents()), 1)

                fired = store.fire_team_agent(listed[1].handle)
                self.assertIsNotNone(fired)
                assert fired is not None
                self.assertEqual(fired.status.value, "fired")
                active_task = store.active_task_for_agent(listed[0].agent_id)
                self.assertIsNotNone(active_task)
                assert active_task is not None
                self.assertEqual(active_task.status, AgentTaskStatus.QUEUED)
            finally:
                store.close()

    def test_deferred_work_round_trip_and_ready_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                blocking_task = create_agent_task(agent, "blocking work", "C1")
                store.upsert_agent_task(blocking_task)
                store.update_agent_task_thread(blocking_task.task_id, "171.000001", "171.parent")
                request = WorkRequest(
                    prompt="follow-up after the blocker",
                    assignment_mode=AssignmentMode.ANYONE,
                    task_kind=AgentTaskKind.WORK,
                )
                thread_ref = SlackThreadRef(
                    channel_id="C1",
                    thread_ts="171.000002",
                    message_ts="171.000002",
                )
                deps = (
                    WorkDependency(
                        kind=WorkDependencyKind.THREAD,
                        channel_id="C1",
                        thread_ts="171.000001",
                        permalink="https://example.slack.com/archives/C1/p0000000171000001",
                        task_id=blocking_task.task_id,
                    ),
                )
                deferred = store.create_deferred_work_request(
                    thread_ref,
                    request,
                    depends_on=deps,
                )
                self.assertEqual(deferred.status, DeferredWorkStatus.WAITING_DEPS)
                self.assertEqual(len(store.list_waiting_deferred_work()), 1)
                satisfied, missing = store.evaluate_deferred_dependencies(deferred)
                self.assertFalse(satisfied)
                self.assertEqual(len(missing), 1)
                store.update_agent_task_status(blocking_task.task_id, AgentTaskStatus.DONE)
                satisfied, _ = store.evaluate_deferred_dependencies(deferred)
                self.assertTrue(satisfied)
                fire_at = utc_now() + timedelta(seconds=30)
                ready = store.mark_deferred_work_ready(
                    deferred.deferred_id,
                    fire_at=fire_at,
                    deps_satisfied_at=utc_now(),
                )
                assert ready is not None
                self.assertEqual(ready.status, DeferredWorkStatus.READY)
                self.assertEqual(
                    store.list_due_deferred_work(now=fire_at + timedelta(seconds=1)), [ready]
                )
                claimed = store.claim_deferred_work(ready.deferred_id)
                assert claimed is not None
                self.assertEqual(claimed.status, DeferredWorkStatus.CLAIMED)
                self.assertIsNone(store.claim_deferred_work(ready.deferred_id))
                store.complete_deferred_work(claimed.deferred_id, last_task_id="task_xyz")
                stored = store.get_deferred_work(claimed.deferred_id)
                assert stored is not None
                self.assertEqual(stored.status, DeferredWorkStatus.DONE)
                self.assertEqual(stored.last_task_id, "task_xyz")
            finally:
                store.close()

    def test_deferred_dependency_waits_for_launched_task_to_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                launched_task = create_agent_task(agent, "first deferred task", "C1")
                store.upsert_agent_task(launched_task)
                first = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.first", "171.first"),
                    WorkRequest(
                        prompt="first deferred task",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.blocker",
                        ),
                    ),
                )
                store.complete_deferred_work(first.deferred_id, last_task_id=launched_task.task_id)
                second = store.create_deferred_work_request(
                    SlackThreadRef("C1", "171.second", "171.second"),
                    WorkRequest(
                        prompt="second deferred task", assignment_mode=AssignmentMode.ANYONE
                    ),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.AGENT_BUSY,
                            handle=agent.handle,
                            task_id=deferred_work_dependency_id(first.deferred_id),
                        ),
                    ),
                )

                satisfied, missing = store.evaluate_deferred_dependencies(second)

                self.assertFalse(satisfied)
                self.assertEqual(missing[0].task_id, deferred_work_dependency_id(first.deferred_id))

                store.update_agent_task_status(launched_task.task_id, AgentTaskStatus.DONE)
                satisfied, missing = store.evaluate_deferred_dependencies(second)

                self.assertTrue(satisfied)
                self.assertEqual(missing, [])
            finally:
                store.close()

    def test_deferred_work_cancellation_by_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                blocking = create_agent_task(agent, "blocker", "C1")
                store.upsert_agent_task(blocking)
                store.update_agent_task_thread(blocking.task_id, "171.000001", "171.parent")
                deferred = store.create_deferred_work_request(
                    SlackThreadRef(channel_id="C1", thread_ts="171.000002"),
                    WorkRequest(
                        prompt="follow up",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                    depends_on=(
                        WorkDependency(
                            kind=WorkDependencyKind.THREAD,
                            channel_id="C1",
                            thread_ts="171.000001",
                            task_id=blocking.task_id,
                        ),
                    ),
                )
                cancelled = store.cancel_deferred_work_for_thread("C1", "171.000002")
                self.assertEqual(cancelled, 1)
                stored = store.get_deferred_work(deferred.deferred_id)
                assert stored is not None
                self.assertEqual(stored.status, DeferredWorkStatus.CANCELLED)
            finally:
                store.close()

    def test_managed_thread_task_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.sqlite"
            store = Store(path)
            try:
                store.init_schema()
                agent = build_initial_model_team(codex_count=1, claude_count=0)[0]
                store.upsert_team_agent(agent)
                task = create_agent_task(agent, "keep ownership", "C1")
                store.upsert_agent_task(task)
                store.update_agent_task_thread(task.task_id, "171.000001", "171.parent")
                store.update_agent_task_status(task.task_id, AgentTaskStatus.ACTIVE)
                store.update_agent_task_session(task.task_id, Provider.CODEX, "codex-thread-1")
                current = store.get_agent_task(task.task_id)
                assert current is not None
                store.upsert_managed_thread_task(current, SlackThreadRef("C1", "171.000001"))
            finally:
                store.close()

            store = Store(path)
            try:
                store.init_schema()
                restored = store.get_managed_thread_task(
                    "C1",
                    "171.000001",
                    agent.agent_id,
                )

                self.assertIsNotNone(restored)
                assert restored is not None
                self.assertEqual(restored.task_id, task.task_id)
                self.assertEqual(restored.session_id, "codex-thread-1")

                store.update_agent_task_status(task.task_id, AgentTaskStatus.DONE)

                self.assertIsNone(store.get_managed_thread_task("C1", "171.000001", agent.agent_id))
            finally:
                store.close()

    def test_slack_thread_and_mirror_cursor_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                thread = SlackThreadRef("C1", "171.000001", "171.000001")

                store.upsert_slack_thread_for_session(
                    Provider.CODEX,
                    "s1",
                    "T1",
                    thread,
                )
                stored = store.get_slack_thread_for_session(
                    Provider.CODEX,
                    "s1",
                    "T1",
                    "C1",
                )

                self.assertEqual(stored, thread)
                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 0)

                store.set_session_mirror_cursor(Provider.CODEX, "s1", 42)

                self.assertEqual(store.get_session_mirror_cursor(Provider.CODEX, "s1"), 42)
            finally:
                store.close()

    def test_session_bridge_prompt_consumes_prefixed_transcript_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                store.add_session_bridge_prompt(Provider.CLAUDE, "s1", "reply exactly OK")

                consumed = store.consume_session_bridge_prompt(
                    Provider.CLAUDE,
                    "s1",
                    "reply exactly OK\nextra inherited stdin",
                )

                self.assertTrue(consumed)
                self.assertFalse(
                    store.consume_session_bridge_prompt(Provider.CLAUDE, "s1", "reply exactly OK")
                )
            finally:
                store.close()

    def test_claude_channel_message_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                message_id = store.enqueue_claude_channel_message(
                    123,
                    "s1",
                    "hello from Slack",
                    {"slack_channel": "C1"},
                )

                rows = store.pending_claude_channel_messages(123)

                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["id"], message_id)
                self.assertEqual(rows[0]["content"], "hello from Slack")

                store.mark_claude_channel_message_delivered(message_id)

                self.assertTrue(store.is_claude_channel_message_delivered(message_id))
                self.assertEqual(store.pending_claude_channel_messages(123), [])
            finally:
                store.close()

    def test_scheduled_work_round_trip_claim_and_reschedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                due_at = utc_now() - timedelta(seconds=1)
                scheduled = store.create_scheduled_work_request(
                    SlackThreadRef("C1", "171.thread", "171.message"),
                    WorkRequest(
                        prompt="check CI",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                    schedule_kind=ScheduledWorkKind.RECURRING,
                    next_run_at=due_at,
                    recurrence={
                        "frequency": "daily",
                        "time": "17:00",
                        "timezone": "America/New_York",
                    },
                    timezone="America/New_York",
                    requested_by_slack_user="U1",
                )

                due = store.list_due_scheduled_work()
                self.assertEqual([item.schedule_id for item in due], [scheduled.schedule_id])

                claimed = store.claim_scheduled_work(scheduled.schedule_id)
                self.assertIsNotNone(claimed)
                self.assertEqual(store.list_due_scheduled_work(), [])

                next_run_at = utc_now() + timedelta(days=1)
                store.complete_scheduled_work(
                    scheduled.schedule_id,
                    last_task_id="task_123",
                    next_run_at=next_run_at,
                )
                row = store.conn.execute(
                    "SELECT status, next_run_at, last_task_id FROM scheduled_work_requests "
                    "WHERE schedule_id = ?",
                    (scheduled.schedule_id,),
                ).fetchone()

                self.assertEqual(row["status"], "pending")
                self.assertEqual(row["next_run_at"], next_run_at.isoformat())
                self.assertEqual(row["last_task_id"], "task_123")

                listed = store.list_scheduled_work()
                self.assertEqual([item.schedule_id for item in listed], [scheduled.schedule_id])

                changed_run_at = utc_now() + timedelta(days=2)
                changed = store.reschedule_scheduled_work(
                    scheduled.schedule_id,
                    next_run_at=changed_run_at,
                    recurrence={
                        "frequency": "daily",
                        "time": "18:00",
                        "timezone": "America/Chicago",
                    },
                    timezone="America/Chicago",
                )
                self.assertIsNotNone(changed)
                assert changed is not None
                self.assertEqual(changed.next_run_at, changed_run_at)
                self.assertEqual(changed.recurrence["time"], "18:00")

                cancelled = store.cancel_scheduled_work(scheduled.schedule_id)
                self.assertIsNotNone(cancelled)
                assert cancelled is not None
                self.assertEqual(cancelled.status.value, "cancelled")
                self.assertEqual(store.list_scheduled_work(), [])
            finally:
                store.close()

    def test_cancelled_claude_channel_message_is_not_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                message_id = store.enqueue_claude_channel_message(123, "s1", "hello", {})

                store.cancel_claude_channel_message(message_id)

                self.assertFalse(store.is_claude_channel_message_delivered(message_id))
                self.assertEqual(store.pending_claude_channel_messages(123), [])
            finally:
                store.close()

    def test_setting_operations_are_thread_safe(self):
        # The Store connection is shared across threads (mirror loop, Slack handlers,
        # task runtime). Without locking, concurrent get/set/delete against the same
        # sqlite3.Connection raise InterfaceError: bad parameter or other API misuse.
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                errors: list[BaseException] = []
                start = threading.Barrier(8)

                def worker(index: int) -> None:
                    start.wait()
                    try:
                        for round_index in range(50):
                            key = f"setting.{index}.{round_index}"
                            store.set_setting(key, str(round_index))
                            store.get_setting(key)
                            store.delete_setting(key)
                    except BaseException as exc:  # pragma: no cover - exercised on failure
                        errors.append(exc)

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                self.assertEqual(errors, [])
            finally:
                store.close()

    def test_runtime_store_operations_are_thread_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                base_agent = build_initial_model_team(1, 0)[0]
                errors: list[BaseException] = []
                start = threading.Barrier(6)

                def worker(index: int) -> None:
                    start.wait()
                    try:
                        agent = replace(
                            base_agent,
                            agent_id=f"agent_{index}",
                            handle=f"agent{index}",
                            sort_order=index,
                        )
                        store.upsert_team_agent(agent)
                        for round_index in range(12):
                            suffix = f"{index}-{round_index}"
                            ts_suffix = f"{index:03d}{round_index:03d}"
                            session = AgentSession(
                                provider=Provider.CODEX,
                                session_id=f"session-{suffix}",
                                transcript_path=Path(tmp) / f"session-{suffix}.jsonl",
                                cwd=Path(tmp),
                                status=SessionStatus.ACTIVE,
                                control_mode=ControlMode.MANAGED,
                            )
                            store.upsert_session(session)

                            task = create_agent_task(agent, f"task {suffix}", "C1")
                            task = replace(task, thread_ts=f"171.{ts_suffix}")
                            store.upsert_agent_task(task)
                            store.update_agent_task_status(task.task_id, AgentTaskStatus.ACTIVE)
                            store.update_agent_task_session(
                                task.task_id,
                                Provider.CODEX,
                                session.session_id,
                            )
                            thread = SlackThreadRef("C1", task.thread_ts or "171.0")
                            store.upsert_managed_thread_task(task, thread)
                            store.get_agent_task(task.task_id)
                            store.get_managed_thread_task("C1", thread.thread_ts, agent.agent_id)

                            pending = store.create_pending_work_request(
                                SlackThreadRef("C1", thread.thread_ts, f"172.{ts_suffix}"),
                                WorkRequest(f"follow-up {suffix}", AssignmentMode.ANYONE),
                            )
                            store.list_pending_work_requests("C1", limit=10)
                            store.update_pending_work_request_status(
                                pending.pending_id,
                                PendingWorkRequestStatus.ASSIGNED,
                            )

                            token = f"token-{suffix}"
                            store.create_slack_agent_request(
                                token,
                                "Codex",
                                "item/permissions/requestApproval",
                                {"command": "git status"},
                                thread,
                                message_ts=f"173.{ts_suffix}",
                            )
                            store.get_slack_agent_request_response(token)
                            store.resolve_slack_agent_request(token, {"behavior": "allow"})
                            resolved, response = store.get_slack_agent_request_response(token)
                            self.assertTrue(resolved)
                            self.assertEqual(response, {"behavior": "allow"})
                    except BaseException as exc:  # pragma: no cover - exercised on failure
                        errors.append(exc)

                threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

                self.assertEqual(errors, [])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
