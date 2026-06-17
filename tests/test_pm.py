from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.models import (
    AgentTaskKind,
    AssignmentMode,
    DeferredWorkStatus,
    PermissionMode,
    PmInitiativeStatus,
    SlackThreadRef,
    WorkDependencyKind,
    WorkRequest,
)
from agent_harness.pm import (
    AGENT_PM_PLAN_SIGNAL_PREFIX,
    MAX_PM_SUBTASKS,
    ParsedPmPlan,
    ParsedPmSubtask,
    build_pm_resolution_prompt,
    critical_path_depth,
    estimate_pm_plan,
    expand_codesign_plan,
    extract_pm_request_body,
    filter_pm_agents,
    filter_worker_agents,
    find_dependency_cycle,
    is_agent_pm_plan_signal,
    looks_like_pm_request,
    looks_like_pm_status_request,
    message_targets_pm_agent,
    parse_agent_pm_plan_signal,
    parse_pm_extension_request,
    parse_pm_replan_request,
    render_pm_plan_dag,
    topological_sort,
)
from agent_harness.storage.store import Store


class PmDetectorTests(unittest.TestCase):
    def test_detects_pm_colon_form(self):
        self.assertTrue(looks_like_pm_request("pm: ship the new logging stack"))

    def test_detects_pm_plan_form(self):
        self.assertTrue(looks_like_pm_request("pm plan rolling out feature flag X"))

    def test_detects_pm_break_down_form(self):
        self.assertTrue(looks_like_pm_request("pm break down the migration to fastapi"))

    def test_ignores_pm_with_no_body(self):
        self.assertFalse(looks_like_pm_request("pm:"))
        self.assertFalse(looks_like_pm_request("pm"))
        self.assertFalse(looks_like_pm_request("pm plan"))

    def test_ignores_unrelated_messages(self):
        self.assertFalse(looks_like_pm_request("@riley please open the PR"))
        self.assertFalse(looks_like_pm_request("schedule a check tomorrow"))
        self.assertFalse(looks_like_pm_request(""))

    def test_handles_user_mention_prefix(self):
        self.assertTrue(looks_like_pm_request("<@U123> pm: take ownership of the runbook update"))

    def test_extract_body_strips_verb(self):
        self.assertEqual(
            extract_pm_request_body("pm plan the database upgrade"),
            "the database upgrade",
        )
        self.assertEqual(
            extract_pm_request_body("pm: ship the new logging stack"),
            "ship the new logging stack",
        )
        self.assertEqual(
            extract_pm_request_body("pm break down the migration"),
            "the migration",
        )


class TopologicalSortTests(unittest.TestCase):
    def test_orders_by_dependencies(self):
        nodes = [("s2", ("s1",)), ("s1", ()), ("s3", ("s2",))]
        self.assertEqual(topological_sort(nodes), ["s1", "s2", "s3"])

    def test_preserves_declaration_order_for_roots(self):
        nodes = [("b", ()), ("a", ()), ("c", ("a", "b"))]
        self.assertEqual(topological_sort(nodes), ["b", "a", "c"])

    def test_detects_cycle(self):
        nodes = [("s1", ("s2",)), ("s2", ("s1",))]
        self.assertIsNone(topological_sort(nodes))

    def test_detects_self_loop(self):
        nodes = [("s1", ("s1",))]
        self.assertIsNone(topological_sort(nodes))

    def test_unknown_dependency_returns_none(self):
        nodes = [("s1", ("missing",))]
        self.assertIsNone(topological_sort(nodes))


class PmPlanSignalTests(unittest.TestCase):
    def _build_signal(self, payload: dict) -> str:
        return f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}"

    def test_is_agent_pm_plan_signal_recognizes_prefix(self):
        self.assertTrue(is_agent_pm_plan_signal("SLACKGENTIC: PM_PLAN {}"))
        self.assertFalse(is_agent_pm_plan_signal("SLACKGENTIC: SCHEDULE {}"))

    def test_parse_valid_plan(self):
        payload = {
            "title": "Ship feature X",
            "summary": "Roll out feature X end to end.",
            "subtasks": [
                {
                    "id": "investigate",
                    "title": "Investigate current state",
                    "task": "Look at the existing config and report what needs to change.",
                    "target": "somebody",
                    "task_kind": "work",
                    "dangerous_mode": False,
                    "depends_on": [],
                },
                {
                    "id": "implement",
                    "title": "Implement the change",
                    "task": "Implement feature X based on the investigation report.",
                    "target": "riley",
                    "task_kind": "work",
                    "depends_on": ["investigate"],
                },
                {
                    "id": "review",
                    "title": "Review the implementation",
                    "task": "Review the change. Make sure tests cover the new path.",
                    "task_kind": "review",
                    "depends_on": ["implement"],
                },
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload), known_handles=["riley", "nell"]
        )
        self.assertIsNone(result.error)
        plan = result.plan
        assert plan is not None
        self.assertEqual(plan.title, "Ship feature X")
        self.assertEqual(
            [s.local_id for s in plan.subtasks], ["investigate", "implement", "review"]
        )
        implement = next(s for s in plan.subtasks if s.local_id == "implement")
        self.assertEqual(implement.request.assignment_mode, AssignmentMode.SPECIFIC)
        self.assertEqual(implement.request.requested_handle, "riley")
        review = next(s for s in plan.subtasks if s.local_id == "review")
        self.assertEqual(review.request.task_kind, AgentTaskKind.REVIEW)
        self.assertEqual(review.depends_on, ("implement",))

    def test_dangerous_mode_sets_permission_mode(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {
                    "id": "s1",
                    "title": "X",
                    "task": "do X",
                    "dangerous_mode": True,
                }
            ],
        }
        result = parse_agent_pm_plan_signal(self._build_signal(payload), known_handles=[])
        assert result.plan is not None
        self.assertEqual(result.plan.subtasks[0].request.permission_mode, PermissionMode.DANGEROUS)

    def test_rejects_duplicate_ids(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {"id": "s1", "title": "A", "task": "do A"},
                {"id": "s1", "title": "B", "task": "do B"},
            ],
        }
        result = parse_agent_pm_plan_signal(self._build_signal(payload), known_handles=[])
        self.assertIn("unique", result.error or "")

    def test_rejects_dependency_on_unknown_subtask(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {"id": "s1", "title": "A", "task": "do A", "depends_on": ["missing"]},
            ],
        }
        result = parse_agent_pm_plan_signal(self._build_signal(payload), known_handles=[])
        self.assertIn("not in this plan", result.error or "")

    def test_rejects_cycle(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {"id": "a", "title": "A", "task": "do A", "depends_on": ["b"]},
                {"id": "b", "title": "B", "task": "do B", "depends_on": ["a"]},
            ],
        }
        result = parse_agent_pm_plan_signal(self._build_signal(payload), known_handles=[])
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertTrue(result.error.startswith("plan contains a dependency cycle"))
        self.assertIn("'a'", result.error)
        self.assertIn("'b'", result.error)
        self.assertIn("->", result.error)

    def test_rejects_unknown_target_handle(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {"id": "s1", "title": "A", "task": "do A", "target": "ghost"},
            ],
        }
        result = parse_agent_pm_plan_signal(self._build_signal(payload), known_handles=["riley"])
        self.assertIn("target", result.error or "")

    def test_enforces_max_subtasks(self):
        subtasks = [
            {"id": f"s{i}", "title": "X", "task": "do X"} for i in range(MAX_PM_SUBTASKS + 1)
        ]
        payload = {"title": "T", "summary": "S", "subtasks": subtasks}
        result = parse_agent_pm_plan_signal(self._build_signal(payload), known_handles=[])
        self.assertIn("at most", result.error or "")

    def test_requires_non_empty_subtasks(self):
        payload = {"title": "T", "summary": "S", "subtasks": []}
        result = parse_agent_pm_plan_signal(self._build_signal(payload), known_handles=[])
        self.assertIn("non-empty", result.error or "")

    def test_rejects_invalid_json(self):
        result = parse_agent_pm_plan_signal(
            f"{AGENT_PM_PLAN_SIGNAL_PREFIX}not json", known_handles=[]
        )
        self.assertIn("invalid PM_PLAN JSON", result.error or "")

    def test_resolution_prompt_includes_examples_and_error(self):
        prompt = build_pm_resolution_prompt(
            "Ship feature X",
            ["riley", "nell"],
            initiative_id="pm_test",
            validation_error="bad",
        )
        self.assertIn("Initiative id: pm_test", prompt)
        self.assertIn("SLACKGENTIC: PM_PLAN", prompt)
        self.assertIn("@riley", prompt)
        self.assertIn("exactly 2 available handles", prompt)
        self.assertIn("previous PM_PLAN control line was invalid: bad", prompt)

    def test_render_pm_plan_dag_shows_roots_and_edges(self):
        plan = ParsedPmPlan(
            title="T",
            summary="S",
            subtasks=(
                ParsedPmSubtask(
                    local_id="investigate",
                    title="Investigate",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    depends_on=(),
                    after_delay_seconds=0,
                ),
                ParsedPmSubtask(
                    local_id="design",
                    title="Design",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    depends_on=("investigate",),
                    after_delay_seconds=0,
                ),
                ParsedPmSubtask(
                    local_id="implement",
                    title="Implement",
                    request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
                    depends_on=("design",),
                    after_delay_seconds=0,
                ),
            ),
        )

        chart = render_pm_plan_dag(plan)

        self.assertEqual(
            chart,
            "\n".join(
                (
                    "investigate - Investigate",
                    "`-- design - Design",
                    "    `-- implement - Implement",
                )
            ),
        )


class PmStoreTests(unittest.TestCase):
    def test_subtask_dispatch_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                thread = SlackThreadRef(channel_id="C1", thread_ts="1.1", message_ts="1.1")
                initiative = store.create_pm_initiative(
                    thread, title="Ship X", summary="Roll out X.", requested_by_slack_user="U1"
                )
                self.assertEqual(initiative.status, PmInitiativeStatus.PLANNING)
                root_request = WorkRequest(
                    prompt="Investigate", assignment_mode=AssignmentMode.ANYONE
                )
                root = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="root",
                    title="Investigate",
                    request=root_request,
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                root_deferred = store.get_deferred_work(root.deferred_id)
                assert root_deferred is not None
                self.assertEqual(root_deferred.status, DeferredWorkStatus.READY)
                self.assertIsNotNone(root_deferred.fire_at)
                child_request = WorkRequest(
                    prompt="Implement", assignment_mode=AssignmentMode.ANYONE
                )
                child = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="child",
                    title="Implement",
                    request=child_request,
                    plan_depends_on=("root",),
                    existing_subtasks=[root],
                    after_delay_seconds=0,
                    sort_order=1,
                )
                child_deferred = store.get_deferred_work(child.deferred_id)
                assert child_deferred is not None
                self.assertEqual(child_deferred.status, DeferredWorkStatus.WAITING_DEPS)
                self.assertEqual(len(child_deferred.depends_on), 1)
                dep = child_deferred.depends_on[0]
                self.assertEqual(dep.kind, WorkDependencyKind.SUBTASK)
                self.assertEqual(dep.task_id, root.deferred_id)
                self.assertEqual(dep.initiative_id, initiative.initiative_id)
                self.assertEqual(dep.local_id, "root")
                listed = store.list_pm_subtasks(initiative.initiative_id)
                self.assertEqual([s.local_id for s in listed], ["root", "child"])
            finally:
                store.close()

    def test_subtask_dispatch_rejects_unknown_dep(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                thread = SlackThreadRef(channel_id="C1", thread_ts="1.1", message_ts="1.1")
                initiative = store.create_pm_initiative(thread, title="T", summary="S")
                with self.assertRaises(ValueError):
                    store.add_pm_subtask_dispatch(
                        initiative=initiative,
                        local_id="child",
                        title="X",
                        request=WorkRequest(prompt="X", assignment_mode=AssignmentMode.ANYONE),
                        plan_depends_on=("missing",),
                        existing_subtasks=[],
                        after_delay_seconds=0,
                        sort_order=0,
                    )
            finally:
                store.close()

    def test_subtask_dependency_satisfaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                thread = SlackThreadRef(channel_id="C1", thread_ts="1.1", message_ts="1.1")
                initiative = store.create_pm_initiative(thread, title="T", summary="S")
                root = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="root",
                    title="Root",
                    request=WorkRequest(prompt="Root", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=(),
                    existing_subtasks=[],
                    after_delay_seconds=0,
                    sort_order=0,
                )
                child = store.add_pm_subtask_dispatch(
                    initiative=initiative,
                    local_id="child",
                    title="Child",
                    request=WorkRequest(prompt="Child", assignment_mode=AssignmentMode.ANYONE),
                    plan_depends_on=("root",),
                    existing_subtasks=[root],
                    after_delay_seconds=0,
                    sort_order=1,
                )
                child_deferred = store.get_deferred_work(child.deferred_id)
                assert child_deferred is not None
                satisfied, _ = store.evaluate_deferred_dependencies(child_deferred)
                self.assertFalse(satisfied)
                # Mark the root subtask deferred row done. The store's deferred
                # satisfaction check needs the deferred to be DONE and the
                # last_task_id to be terminal (or absent).
                store.update_deferred_work_status(root.deferred_id, DeferredWorkStatus.DONE)
                child_deferred = store.get_deferred_work(child.deferred_id)
                assert child_deferred is not None
                satisfied, _ = store.evaluate_deferred_dependencies(child_deferred)
                self.assertTrue(satisfied)
            finally:
                store.close()

    def test_cancel_initiative_for_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                thread = SlackThreadRef(channel_id="C1", thread_ts="1.1", message_ts="1.1")
                initiative = store.create_pm_initiative(thread, title="T", summary="S")
                cancelled = store.cancel_pm_initiative_for_thread("C1", "1.1")
                self.assertEqual(cancelled, 1)
                reloaded = store.get_pm_initiative(initiative.initiative_id)
                assert reloaded is not None
                self.assertEqual(reloaded.status, PmInitiativeStatus.CANCELLED)
            finally:
                store.close()


class PmTagTargetingTests(unittest.TestCase):
    def test_returns_handle_when_message_starts_with_at_handle(self):
        self.assertEqual(message_targets_pm_agent("@alice plan it", ["alice"]), "alice")

    def test_returns_handle_for_bare_handle_prefix(self):
        self.assertEqual(message_targets_pm_agent("alice: status?", ["alice"]), "alice")

    def test_strips_leading_bot_mention(self):
        self.assertEqual(
            message_targets_pm_agent("<@U999> @alice what is the status?", ["alice"]),
            "alice",
        )

    def test_returns_none_for_unknown_handle(self):
        self.assertIsNone(message_targets_pm_agent("@charlie hi", ["alice", "bob"]))

    def test_returns_none_when_body_is_empty(self):
        self.assertIsNone(message_targets_pm_agent("@alice", ["alice"]))
        self.assertIsNone(message_targets_pm_agent("alice:", ["alice"]))

    def test_returns_none_when_no_pm_handles(self):
        self.assertIsNone(message_targets_pm_agent("@alice plan it", []))

    def test_case_insensitive_match(self):
        self.assertEqual(message_targets_pm_agent("@Alice plan it", ["alice"]), "alice")


class PmCoDesignExpansionTests(unittest.TestCase):
    def _build_signal(self, payload: dict) -> str:
        return f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}"

    def test_co_designers_validated_min_two(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {
                    "id": "s1",
                    "title": "Design",
                    "task": "Design it.",
                    "co_designers": ["alice"],
                }
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload), known_handles=["alice", "bob"]
        )
        self.assertIn("exactly 2 distinct", result.error or "")

    def test_co_designers_rejected_when_more_than_two(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {
                    "id": "s1",
                    "title": "Design",
                    "task": "Design it.",
                    "co_designers": ["alice", "bob", "cara"],
                }
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload), known_handles=["alice", "bob", "cara"]
        )
        self.assertIn("exactly 2 distinct", result.error or "")

    def test_co_designers_validated_unknown_handle(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {
                    "id": "s1",
                    "title": "Design",
                    "task": "Design it.",
                    "co_designers": ["alice", "ghost"],
                }
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload), known_handles=["alice", "bob"]
        )
        self.assertIn("available handles", result.error or "")

    def test_co_designers_rejected_with_anyone_target(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {
                    "id": "s1",
                    "title": "Design",
                    "task": "Design it.",
                    "co_designers": ["alice", "somebody"],
                }
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload), known_handles=["alice", "bob"]
        )
        self.assertIn("somebody", result.error or "")

    def test_co_designers_rejected_when_same_model_family(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {
                    "id": "s1",
                    "title": "Design",
                    "task": "Design it.",
                    "co_designers": ["alice", "bob"],
                }
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload),
            known_handles=["alice", "bob"],
            handle_models={"alice": "claude", "bob": "claude"},
        )
        self.assertIn("different model/provider families", result.error or "")

    def test_co_designers_parsed_into_subtask(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {
                    "id": "s1",
                    "title": "Design",
                    "task": "Design it.",
                    "co_designers": ["alice", "bob"],
                }
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload),
            known_handles=["alice", "bob"],
            handle_models={"alice": "claude", "bob": "codex"},
        )
        self.assertIsNone(result.error)
        plan = result.plan
        assert plan is not None
        subtask = plan.subtasks[0]
        self.assertTrue(subtask.is_co_design)
        self.assertEqual(subtask.co_designers, ("alice", "bob"))

    def test_expand_fans_drafts_plus_synthesis(self):
        co_subtask = ParsedPmSubtask(
            local_id="design",
            title="Design schema",
            request=WorkRequest(
                prompt="Design the new schema.",
                assignment_mode=AssignmentMode.ANYONE,
            ),
            depends_on=("investigate",),
            after_delay_seconds=0,
            co_designers=("alice", "bob"),
        )
        downstream = ParsedPmSubtask(
            local_id="implement",
            title="Implement",
            request=WorkRequest(prompt="Implement.", assignment_mode=AssignmentMode.ANYONE),
            depends_on=("design",),
            after_delay_seconds=0,
        )
        root = ParsedPmSubtask(
            local_id="investigate",
            title="Investigate",
            request=WorkRequest(prompt="Look at the code.", assignment_mode=AssignmentMode.ANYONE),
            depends_on=(),
            after_delay_seconds=0,
        )
        plan = ParsedPmPlan(title="T", summary="S", subtasks=(root, co_subtask, downstream))
        expanded = expand_codesign_plan(plan)
        ids = [s.local_id for s in expanded.subtasks]
        self.assertIn("design--alice", ids)
        self.assertIn("design--bob", ids)
        self.assertIn("design", ids)  # synthesis keeps original id
        alice_draft = next(s for s in expanded.subtasks if s.local_id == "design--alice")
        self.assertEqual(alice_draft.request.assignment_mode, AssignmentMode.SPECIFIC)
        self.assertEqual(alice_draft.request.requested_handle, "alice")
        # Drafts inherit upstream deps of the original co-design subtask.
        self.assertEqual(alice_draft.depends_on, ("investigate",))
        synthesis = next(s for s in expanded.subtasks if s.local_id == "design")
        self.assertEqual(set(synthesis.depends_on), {"design--alice", "design--bob"})
        # Downstream tasks waiting on the original id still resolve via synthesis.
        impl = next(s for s in expanded.subtasks if s.local_id == "implement")
        self.assertEqual(impl.depends_on, ("design",))

    def test_expand_is_noop_without_co_design(self):
        plain = ParsedPmSubtask(
            local_id="s1",
            title="Do it",
            request=WorkRequest(prompt="Do it.", assignment_mode=AssignmentMode.ANYONE),
            depends_on=(),
            after_delay_seconds=0,
        )
        plan = ParsedPmPlan(title="T", summary="S", subtasks=(plain,))
        self.assertIs(expand_codesign_plan(plan), plan)


class PmAgentFilterTests(unittest.TestCase):
    def _mk_agent(self, handle, kind):
        from agent_harness.models import (
            Provider,
            TeamAgent,
            TeamAgentKind,
            TeamAgentStatus,
            utc_now,
        )

        return TeamAgent(
            agent_id=f"a-{handle}",
            handle=handle,
            full_name=handle.title(),
            initials=handle[:2].upper(),
            color_hex="#abcdef",
            avatar_slug=handle,
            icon_emoji=":robot_face:",
            role="program manager" if kind == TeamAgentKind.PM else "engineer",
            personality="x",
            voice="x",
            unique_strength="x",
            reaction_names=("white_check_mark",),
            sort_order=0,
            provider_preference=Provider.CLAUDE,
            status=TeamAgentStatus.ACTIVE,
            kind=kind,
            hired_at=utc_now(),
        )

    def test_filter_pm_and_worker_partition_agents(self):
        from agent_harness.models import TeamAgentKind

        pm = self._mk_agent("alice", TeamAgentKind.PM)
        eng = self._mk_agent("bob", TeamAgentKind.ENGINEER)
        agents = [pm, eng]
        self.assertEqual([a.handle for a in filter_pm_agents(agents)], ["alice"])
        self.assertEqual([a.handle for a in filter_worker_agents(agents)], ["bob"])


class FindDependencyCycleTests(unittest.TestCase):
    def test_returns_empty_for_acyclic_graph(self):
        self.assertEqual(find_dependency_cycle([("a", ())]), [])
        self.assertEqual(find_dependency_cycle([("a", ()), ("b", ("a",)), ("c", ("b",))]), [])

    def test_finds_two_node_cycle(self):
        cycle = find_dependency_cycle([("a", ("b",)), ("b", ("a",))])
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual({"a", "b"}, set(cycle))
        self.assertEqual(len(cycle), 3)

    def test_finds_three_node_cycle(self):
        cycle = find_dependency_cycle([("a", ("b",)), ("b", ("c",)), ("c", ("a",))])
        self.assertEqual(cycle[0], cycle[-1])
        self.assertEqual({"a", "b", "c"}, set(cycle))
        self.assertEqual(len(cycle), 4)


class CriticalPathDepthTests(unittest.TestCase):
    def _plan(self, *subtasks: ParsedPmSubtask) -> ParsedPmPlan:
        return ParsedPmPlan(title="T", summary="S", subtasks=tuple(subtasks))

    def _subtask(self, local_id: str, deps: tuple[str, ...] = ()) -> ParsedPmSubtask:
        return ParsedPmSubtask(
            local_id=local_id,
            title=local_id,
            request=WorkRequest(prompt="x", assignment_mode=AssignmentMode.ANYONE),
            depends_on=deps,
            after_delay_seconds=0,
        )

    def test_depth_of_single_root_is_one(self):
        plan = self._plan(self._subtask("a"))
        self.assertEqual(critical_path_depth(plan), 1)

    def test_depth_of_linear_chain(self):
        plan = self._plan(
            self._subtask("a"),
            self._subtask("b", ("a",)),
            self._subtask("c", ("b",)),
        )
        self.assertEqual(critical_path_depth(plan), 3)

    def test_depth_uses_longest_branch(self):
        plan = self._plan(
            self._subtask("a"),
            self._subtask("b", ("a",)),
            self._subtask("c", ("a",)),
            self._subtask("d", ("b", "c")),
        )
        self.assertEqual(critical_path_depth(plan), 3)


class EstimatePmPlanTests(unittest.TestCase):
    def test_counts_subtasks_and_dangerous(self):
        plan = ParsedPmPlan(
            title="T",
            summary="S",
            subtasks=(
                ParsedPmSubtask(
                    local_id="a",
                    title="A",
                    request=WorkRequest(
                        prompt="x",
                        assignment_mode=AssignmentMode.ANYONE,
                        permission_mode=PermissionMode.DANGEROUS,
                    ),
                    depends_on=(),
                    after_delay_seconds=0,
                ),
                ParsedPmSubtask(
                    local_id="b",
                    title="B",
                    request=WorkRequest(
                        prompt="x",
                        assignment_mode=AssignmentMode.ANYONE,
                    ),
                    depends_on=("a",),
                    after_delay_seconds=0,
                ),
            ),
        )
        estimate = estimate_pm_plan(plan)
        self.assertEqual(estimate.subtask_count, 2)
        self.assertEqual(estimate.critical_path_depth, 2)
        self.assertEqual(estimate.dangerous_count, 1)
        self.assertEqual(estimate.co_design_count, 0)
        self.assertGreater(estimate.max_wall_clock_seconds, estimate.min_wall_clock_seconds)


class PmStatusCommandDetectorTests(unittest.TestCase):
    def test_matches_pm_status(self):
        for command in ("pm status", "PM status", "pm plan", "pm dag", "pm state"):
            self.assertTrue(looks_like_pm_status_request(command), command)

    def test_matches_with_bot_mention(self):
        self.assertTrue(looks_like_pm_status_request("<@UBOT> pm status"))

    def test_matches_slash_command(self):
        self.assertTrue(looks_like_pm_status_request("/status"))

    def test_does_not_match_arbitrary_text(self):
        self.assertFalse(looks_like_pm_status_request("what is the status of the deploy?"))
        self.assertFalse(looks_like_pm_status_request("plan the migration"))


class PmReplanCommandDetectorTests(unittest.TestCase):
    def test_matches_pm_replan_with_body(self):
        body = parse_pm_replan_request("pm replan: also handle the cache invalidation")
        self.assertEqual(body, "also handle the cache invalidation")

    def test_matches_bare_replan(self):
        body = parse_pm_replan_request("pm replan")
        self.assertEqual(body, "")

    def test_no_match_for_unrelated_text(self):
        self.assertIsNone(parse_pm_replan_request("please update the spec"))

    def test_handles_bot_mention_prefix(self):
        body = parse_pm_replan_request("<@UBOT> pm replan: try claude this time")
        self.assertEqual(body, "try claude this time")


class PmExtensionCommandDetectorTests(unittest.TestCase):
    def test_matches_pm_extend(self):
        self.assertEqual(
            parse_pm_extension_request("pm extend: also lint the new module"),
            "also lint the new module",
        )

    def test_requires_body(self):
        self.assertIsNone(parse_pm_extension_request("pm extend"))
        self.assertIsNone(parse_pm_extension_request("pm extend:"))

    def test_no_match_for_unrelated(self):
        self.assertIsNone(parse_pm_extension_request("status please"))


class PmPlanParserAllowsExternalIdsTests(unittest.TestCase):
    def _build_signal(self, payload):
        return f"{AGENT_PM_PLAN_SIGNAL_PREFIX}{json.dumps(payload)}"

    def test_extension_allows_existing_id_as_dep(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {"id": "new1", "title": "new", "task": "x", "depends_on": ["existing_s1"]},
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload),
            known_handles=[],
            allowed_external_dep_ids=("existing_s1",),
        )
        self.assertIsNone(result.error)
        assert result.plan is not None
        self.assertEqual(result.plan.subtasks[0].depends_on, ("existing_s1",))

    def test_extension_rejects_id_collision_with_existing(self):
        payload = {
            "title": "T",
            "summary": "S",
            "subtasks": [
                {"id": "existing_s1", "title": "dup", "task": "x"},
            ],
        }
        result = parse_agent_pm_plan_signal(
            self._build_signal(payload),
            known_handles=[],
            allowed_external_dep_ids=("existing_s1",),
        )
        self.assertIsNotNone(result.error)
        assert result.error is not None
        self.assertIn("collides", result.error)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
