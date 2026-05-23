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
    PM_EXTENSION_KNOWN_IDS_METADATA_KEY,
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
    def test_pm_plan_signal_parks_for_approval_without_firing(self):
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
                self.assertEqual(refreshed.status, PmInitiativeStatus.AWAITING_APPROVAL)
                self.assertIsNotNone(refreshed.pending_plan_json)
                # No subtasks should be persisted, and nothing should fire.
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                self.assertEqual(
                    [t for t, *_ in runtime.started if t.task_id != pm_task.task_id], []
                )
                # Plan ack mentions the approval gate.
                plan_message = gateway.thread_replies[-1]
                self.assertIn("Start executing", plan_message.get("text", ""))
            finally:
                store.close()

    def test_pm_initiative_start_button_dispatches_root_subtasks(self):
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
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)
                parked = store.get_pm_initiative(initiative.initiative_id)
                assert parked is not None
                self.assertEqual(parked.status, PmInitiativeStatus.AWAITING_APPROVAL)

                action_value = json.dumps(
                    {
                        "v": 1,
                        "action": "pm_initiative.start",
                        "initiative_id": initiative.initiative_id,
                    }
                )
                controller.handle_block_action(
                    {
                        "actions": [{"action_id": "pm_initiative.start", "value": action_value}],
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.plan"},
                        "user": {"id": "U2"},
                    }
                )

                approved = store.get_pm_initiative(initiative.initiative_id)
                assert approved is not None
                self.assertEqual(approved.status, PmInitiativeStatus.ACTIVE)
                self.assertIsNone(approved.pending_plan_json)
                subtasks = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual([s.local_id for s in subtasks], ["investigate", "implement"])
                root_deferred = store.get_deferred_work(subtasks[0].deferred_id)
                assert root_deferred is not None
                self.assertIn(
                    root_deferred.status,
                    {DeferredWorkStatus.DONE, DeferredWorkStatus.READY},
                )
                child_deferred = store.get_deferred_work(subtasks[1].deferred_id)
                assert child_deferred is not None
                self.assertEqual(child_deferred.status, DeferredWorkStatus.WAITING_DEPS)
                # The root subtask was dispatched to the (now idle) agent.
                self.assertTrue(any(t.task_id != pm_task.task_id for t, *_ in runtime.started))
            finally:
                store.close()

    def test_pm_initiative_cancel_button_cancels_without_firing(self):
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
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)

                action_value = json.dumps(
                    {
                        "v": 1,
                        "action": "pm_initiative.cancel",
                        "initiative_id": initiative.initiative_id,
                    }
                )
                controller.handle_block_action(
                    {
                        "actions": [{"action_id": "pm_initiative.cancel", "value": action_value}],
                        "channel": {"id": "C1"},
                        "message": {"ts": "171.plan"},
                        "user": {"id": "U2"},
                    }
                )

                cancelled = store.get_pm_initiative(initiative.initiative_id)
                assert cancelled is not None
                self.assertEqual(cancelled.status, PmInitiativeStatus.CANCELLED)
                self.assertEqual(store.list_pm_subtasks(initiative.initiative_id), [])
                self.assertEqual(
                    [t for t, *_ in runtime.started if t.task_id != pm_task.task_id], []
                )
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


class _FakeSlackApiError(Exception):
    """Stand-in for slack_sdk.errors.SlackApiError used by the dead-thread tests."""

    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.response = {"error": error_code}


class PmWatchdogAdversarialTests(unittest.TestCase):
    """Adversarial coverage: lost threads, fired PMs, watchdog crashes."""

    def _setup_active_initiative(self, store: Store, *, agent_handle: str):
        team = build_initial_model_team(1, 0)
        agent = team[0]
        # Force the handle so the test reads cleanly.
        agent = replace(agent, handle=agent_handle)
        store.upsert_team_agent(agent)
        thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
        initiative = store.create_pm_initiative(
            thread_ref, title="T", summary="S", requested_by_slack_user="U1"
        )
        store.update_pm_initiative_status(initiative.initiative_id, PmInitiativeStatus.ACTIVE)
        refreshed = store.get_pm_initiative(initiative.initiative_id)
        assert refreshed is not None
        return refreshed, agent

    def test_watchdog_cancels_initiative_when_blocker_thread_is_unreachable(self):
        """When the watchdog tries to surface a blocker against a deleted /
        archived thread, it should mark the initiative CANCELLED so subsequent
        ticks do not loop forever against the same dead thread."""

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")

            class DeadThreadGateway(FakeGateway):
                def post_thread_reply(self, *args, **kwargs):
                    raise _FakeSlackApiError("channel_not_found")

            gateway = DeadThreadGateway()
            try:
                store.init_schema()
                initiative, agent = self._setup_active_initiative(store, agent_handle="alice")
                subtask = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="s1",
                    title="Stalled",
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
                # Drive the subtask into an ACTIVE task with a stale updated_at
                # so the watchdog tries to surface a `stalled_task` blocker.
                stalled_task = replace(
                    create_agent_task(agent, "touch CI", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.sub",
                    parent_message_ts="171.sub-parent",
                    updated_at=utc_now() - timedelta(minutes=45),
                )
                store.upsert_agent_task(stalled_task)
                store.update_deferred_work_last_task(
                    subtask.deferred_id, last_task_id=stalled_task.task_id
                )
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()
                final = store.get_pm_initiative(initiative.initiative_id)
                assert final is not None
                self.assertEqual(final.status, PmInitiativeStatus.CANCELLED)
            finally:
                store.close()

    def test_watchdog_heartbeat_not_updated_on_exception(self):
        """``mark_pm_initiative_watchdog_run`` must skip the heartbeat update
        when ``_watch_one_pm_initiative`` raises, so a monitor that watches the
        timestamp can still notice the failure."""

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                initiative, _ = self._setup_active_initiative(store, agent_handle="alice")
                controller = SlackTeamController(store, gateway, default_channel_id="C1")

                def boom(_initiative):
                    raise RuntimeError("simulated watchdog failure")

                controller._watch_one_pm_initiative = boom  # type: ignore[assignment]
                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertIsNone(refreshed.watchdog_last_run_at)
            finally:
                store.close()

    def test_pm_agent_fired_during_active_keeps_initiative_visible(self):
        """If the user fires the PM mid-initiative, the watchdog should not
        crash. The initiative stays ACTIVE until subtasks reach terminal state,
        and the watchdog still posts a recap."""

        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                initiative, agent = self._setup_active_initiative(store, agent_handle="alice")
                subtask = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="s1",
                    title="Solo",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                store.update_deferred_work_status(subtask.deferred_id, DeferredWorkStatus.DONE)
                # Fire the PM mid-run.
                store.fire_team_agent(agent.agent_id)
                controller = SlackTeamController(store, gateway, default_channel_id="C1")
                runner = PMInitiativeRunner(store, controller, poll_seconds=0.01)
                runner.sync_once()
                final = store.get_pm_initiative(initiative.initiative_id)
                assert final is not None
                # All terminal, so the watchdog should close the initiative as DONE.
                self.assertEqual(final.status, PmInitiativeStatus.DONE)
            finally:
                store.close()


class PmRetryFailureReasonTests(unittest.TestCase):
    def test_retry_start_task_failure_surfaces_validator_error(self):
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
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)

                class FailingRuntime(FakeRuntime):
                    def start_task(self, task, agent, thread):
                        self.started.append((task, agent, thread))
                        return False

                runtime = FailingRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )
                # Trigger the retry path with an unparseable PM_PLAN.
                controller.handle_runtime_agent_control(
                    pm_task,
                    agent,
                    thread_ref,
                    f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{{not json",
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                self.assertEqual(refreshed.status, PmInitiativeStatus.CANCELLED)
                cancel_message = gateway.thread_replies[-1]["text"]
                # The user-visible message must include the underlying
                # validator error so they know why the retry failed.
                self.assertIn("last validation error", cancel_message)
                self.assertIn("invalid PM_PLAN JSON", cancel_message)
            finally:
                store.close()


class PmStatusCommandTests(unittest.TestCase):
    def test_pm_status_command_renders_plan_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                agent = build_initial_model_team(1, 0)[0]
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(
                    thread_ref, title="Ship feature X", summary="S"
                )
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="investigate",
                    title="Investigate",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store,
                    gateway,
                    default_channel_id="C1",
                    runtime=FakeRuntime(),
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm status",
                            "ts": "171.000010",
                            "thread_ts": "171.thread",
                        }
                    }
                )
                self.assertTrue(gateway.thread_replies, "expected a thread reply")
                latest = gateway.thread_replies[-1]["text"]
                self.assertIn("investigate", latest)
                self.assertIn("Ship feature X", latest)
            finally:
                store.close()


class PmReplanRuntimeTests(unittest.TestCase):
    def test_pm_replan_cancels_pending_and_restarts_resolver(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                from agent_harness.models import TeamAgentKind

                team = build_initial_model_team(1, 0)
                agent = replace(team[0], kind=TeamAgentKind.PM)
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="ship X")
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                # Two subtasks: one DONE, one still pending.
                done_subtask = store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="done1",
                    title="Done step",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                store.update_deferred_work_status(done_subtask.deferred_id, DeferredWorkStatus.DONE)
                pending = store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="pending1",
                    title="Pending step",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[done_subtask],
                    after_delay_seconds=0,
                    sort_order=1,
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm replan: take a different approach",
                            "ts": "171.000020",
                            "thread_ts": "171.thread",
                        }
                    }
                )
                pending_after = store.get_deferred_work(pending.deferred_id)
                assert pending_after is not None
                self.assertEqual(pending_after.status, DeferredWorkStatus.CANCELLED)
                # Done one stays DONE.
                done_after = store.get_deferred_work(done_subtask.deferred_id)
                assert done_after is not None
                self.assertEqual(done_after.status, DeferredWorkStatus.DONE)
                # Initiative is back in PLANNING.
                planning = store.get_pm_initiative(initiative.initiative_id)
                assert planning is not None
                self.assertEqual(planning.status, PmInitiativeStatus.PLANNING)
                # Resolver was restarted with the new context.
                self.assertEqual(len(runtime.started), 1)
                started_task, _, _ = runtime.started[0]
                self.assertIn("REPLAN", started_task.prompt)
                self.assertIn("take a different approach", started_task.prompt)
            finally:
                store.close()


class PmExtensionRuntimeTests(unittest.TestCase):
    def test_pm_extension_command_seeds_known_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            gateway = FakeGateway()
            try:
                store.init_schema()
                from agent_harness.models import TeamAgentKind

                team = build_initial_model_team(1, 0)
                agent = replace(team[0], kind=TeamAgentKind.PM)
                store.upsert_team_agent(agent)
                thread_ref = SlackThreadRef("C1", "171.thread", "171.parent")
                initiative = store.create_pm_initiative(thread_ref, title="T", summary="ship X")
                store.update_pm_initiative_status(
                    initiative.initiative_id, PmInitiativeStatus.ACTIVE
                )
                refreshed = store.get_pm_initiative(initiative.initiative_id)
                assert refreshed is not None
                store.add_pm_subtask_dispatch(
                    initiative=refreshed,
                    local_id="existing1",
                    title="Existing",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                pm_task = replace(
                    create_agent_task(agent, "resolve PM plan", "C1"),
                    status=AgentTaskStatus.ACTIVE,
                    thread_ts="171.thread",
                    parent_message_ts="171.parent",
                    metadata={
                        PM_RESOLUTION_METADATA_KEY: True,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)
                runtime = FakeRuntime()
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=runtime
                )
                controller.handle_event(
                    {
                        "event": {
                            "type": "message",
                            "channel": "C1",
                            "user": "U1",
                            "text": "pm extend: also write integration tests",
                            "ts": "171.000030",
                            "thread_ts": "171.thread",
                        }
                    }
                )
                # Resolver restarted, prompt mentions EXTENSION mode and existing id.
                self.assertEqual(len(runtime.started), 1)
                started_task, _, _ = runtime.started[0]
                self.assertIn("EXTENSION", started_task.prompt)
                self.assertIn("existing1", started_task.prompt)
                # Metadata carries the known ids for the next plan-parse round.
                self.assertEqual(
                    list(started_task.metadata.get(PM_EXTENSION_KNOWN_IDS_METADATA_KEY) or []),
                    ["existing1"],
                )
                # Existing subtask stays untouched.
                existing_after = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual([s.local_id for s in existing_after], ["existing1"])
            finally:
                store.close()


class PmPlanAckEstimateTests(unittest.TestCase):
    def test_estimate_surfaces_in_plan_approval_ack(self):
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
                        PM_RESOLUTION_ATTEMPTS_METADATA_KEY: 0,
                        PM_INITIATIVE_ID_METADATA_KEY: initiative.initiative_id,
                        PM_RESOLUTION_ORIGINAL_TEXT_METADATA_KEY: "ship X",
                    },
                )
                store.upsert_agent_task(pm_task)
                controller = SlackTeamController(
                    store, gateway, default_channel_id="C1", runtime=FakeRuntime()
                )
                signal = _build_plan_signal(initiative.initiative_id, handles=[agent.handle])
                controller.handle_runtime_agent_control(pm_task, agent, thread_ref, signal)
                plan_message = gateway.thread_replies[-1]["text"]
                self.assertIn("Estimate:", plan_message)
                self.assertIn("subtasks", plan_message)
                self.assertIn("critical path", plan_message)
            finally:
                store.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
