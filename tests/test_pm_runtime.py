from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from agent_harness.models import (
    AgentTaskStatus,
    AssignmentMode,
    DeferredWorkStatus,
    PmInitiativeStatus,
    SlackThreadRef,
    WorkRequest,
    utc_now,
)
from agent_harness.pm import (
    AGENT_PM_PLAN_SIGNAL_PREFIX,
    PM_INITIATIVE_ID_METADATA_KEY,
    PM_RESOLUTION_ATTEMPTS_METADATA_KEY,
    PM_RESOLUTION_METADATA_KEY,
    PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY,
)
from agent_harness.slack.app import (
    PMInitiativeRunner,
    SlackTeamController,
)
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team, create_agent_task
from tests.test_slack_app import FakeGateway, FakeRuntime


def _build_plan_signal(initiative_id: str, *, handles: list[str]) -> str:
    payload = {
        "title": "Ship feature X",
        "summary": "Roll out feature X.",
        "subtasks": [
            {
                "id": "investigate",
                "title": "Investigate",
                "task": "Look at the current state and write a one-paragraph summary.",
                "target": "somebody",
                "depends_on": [],
            },
            {
                "id": "implement",
                "title": "Implement",
                "task": "Implement feature X based on the investigation summary.",
                "target": "somebody",
                "depends_on": ["investigate"],
            },
        ],
    }
    return f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}"


class PmDispatchTests(unittest.TestCase):
    def test_pm_message_creates_initiative_and_starts_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm: ship a new logging stack",
                            "ts": "171.000001",
                        }
                    }
                )

                initiatives = store.list_pm_initiatives(
                    statuses=(PmInitiativeStatus.PLANNING, PmInitiativeStatus.ACTIVE)
                )
                self.assertEqual(len(initiatives), 1)
                self.assertEqual(initiatives[0].status, PmInitiativeStatus.PLANNING)
                self.assertEqual(initiatives[0].summary, "ship a new logging stack")
                self.assertEqual(initiatives[0].requested_by_slack_user, "U1")
                self.assertEqual(len(runtime.started), 1)
                started_task, started_agent, _thread = runtime.started[0]
                self.assertTrue(started_task.metadata.get(PM_RESOLUTION_METADATA_KEY))
                self.assertEqual(
                    started_task.metadata.get(PM_INITIATIVE_ID_METADATA_KEY),
                    initiatives[0].initiative_id,
                )
                self.assertEqual(started_agent.agent_id, agent.agent_id)
                self.assertIn(
                    "ship a new logging stack",
                    gateway.thread_replies[-1]["text"],
                )
            finally:
                store.close()

    def test_pm_message_does_not_match_non_pm_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "please open the PR",
                            "ts": "171.000002",
                        }
                    }
                )

                self.assertEqual(store.list_pm_initiatives(), [])
            finally:
                store.close()


class PmRuntimeTests(unittest.TestCase):
    def test_pm_plan_signal_creates_initiative_subtasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                    requested_by_slack_user="U1",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                signal = _build_plan_signal(
                    initiative.initiative_id,
                    handles=[agent.handle],
                )
                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    signal,
                )

                self.assertTrue(handled)
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.ACTIVE)
                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual([s.local_id for s in subtasks], ["investigate", "implement"])
                root_deferred = store.get_deferred_work(subtasks[0].deferred_id)
                assert root_deferred is not None
                # The root subtask kicks off immediately because the PM resolver
                # was marked DONE before the eager fire.
                self.assertIn(
                    root_deferred.status,
                    {DeferredWorkStatus.DONE, DeferredWorkStatus.READY},
                )
                child_deferred = store.get_deferred_work(subtasks[1].deferred_id)
                assert child_deferred is not None
                self.assertEqual(child_deferred.status, DeferredWorkStatus.WAITING_DEPS)
                # The resolver task itself is now done.
                resolver_task = store.get_agent_task(pm_task.task_id)
                assert resolver_task is not None
                self.assertEqual(resolver_task.status, AgentTaskStatus.DONE)
                # The root subtask was dispatched to the (now idle) agent.
                self.assertTrue(any(t.task_id != pm_task.task_id for t, *_ in runtime.started))
            finally:
                store.close()

    def test_invalid_pm_plan_retries_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="Ship feature X",
                    summary="Roll out feature X.",
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "Ship feature X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{{not json",
                )

                self.assertTrue(handled)
                self.assertEqual(len(runtime.started), 1)
                retry_task = runtime.started[0][0]
                self.assertIn("previous PM_PLAN control line was invalid", retry_task.prompt)
                self.assertEqual(retry_task.metadata[PM_RESOLUTION_ATTEMPTS_METADATA_KEY], 1)
            finally:
                store.close()

    def test_invalid_pm_plan_at_retry_cap_cancels_initiative(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="S")
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 3,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                handled = controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{{not json",
                )

                self.assertTrue(handled)
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.CANCELLED)
                resolver_task = store.get_agent_task(pm_task.task_id)
                assert resolver_task is not None
                self.assertEqual(resolver_task.status, AgentTaskStatus.CANCELLED)
                self.assertEqual(runtime.started, [])
            finally:
                store.close()

    def test_pm_watchdog_marks_initiative_done_when_subtasks_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref, title="T", summary="S", requested_by_slack_user="U1"
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed_initiative = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed_initiative is not None
                subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed_initiative,
                    local_id="s1",
                    title="Solo",
                    request=WorkRequest(prompt="solo task", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                store.update_deferred_work_status(subtask.deferred_id, DeferredWorkStatus.DONE)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                )

                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                surfaced = runner.sync_once()

                self.assertGreaterEqual(surfaced, 1)
                final = store.get_pm_initiative(initiative.initiative_id)
                assert final is not None
                self.assertEqual(final.status, PmInitiativeStatus.DONE)
                # A recap message was posted in the initiative thread.
                self.assertTrue(any("complete" in r["text"] for r in gateway.thread_replies))
            finally:
                store.close()

    def test_pm_watchdog_surfaces_stalled_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref,
                    title="T",
                    summary="S",
                    requested_by_slack_user="U1",
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed_initiative = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed_initiative is not None
                subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed_initiative,
                    local_id="s1",
                    title="Block on approval",
                    request=WorkRequest(
                        prompt="touch CI",
                        assignment_mode=AssignmentMode.SPECIFIC,
                        requested_handle=agent.handle,
                    ),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                # Simulate the deferred row firing into a task in an ACTIVE
                # state by hand.
                subtask_task = replace(
                    create_agent_task(agent, "touch CI", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.subtask",
                    parent_message_ts="171.subtask-parent",
                )
                store.upsert_agent_task(subtask_task)
                store.update_deferred_work_last_task(
                    subtask.deferred_id, last_task_id=subtask_task.task_id
                )
                # An old, unresolved Slack approval request blocks the subtask.
                store.create_slack_agent_request(
                    token="tok-1",
                    provider_label="Codex",
                    method="codex/approve",
                    params={},
                    thread=SlackThreadRef("C1", "171.subtask", "171.approve"),
                    message_ts="171.approve",
                )
                old_created = (utc_now() - timedelta(minutes=10)).isoformat()
                store.conn.execute(
                    "UPDATE slack_agent_requests SET created_at = ? WHERE token = ?",
                    (old_created, "tok-1"),
                )
                store.conn.commit()
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=runtime,
                )

                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()

                surfaced_texts = [reply["text"] for reply in gateway.thread_replies]
                self.assertTrue(
                    any("waiting on `codex/approve`" in text for text in surfaced_texts),
                    surfaced_texts,
                )
                # Second tick should NOT post the same blocker twice.
                first_count = len(gateway.thread_replies)
                runner.sync_once()
                self.assertEqual(len(gateway.thread_replies), first_count)
            finally:
                store.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
