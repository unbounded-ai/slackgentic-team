import json
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from agent_harness.models import (
    AgentSession,
    AgentTask,
    AgentTaskKind,
    AgentTaskStatus,
    Provider,
    SessionStatus,
)
from agent_harness.runtime.tasks import _clean_terminal_output, _process_output_chunks
from agent_harness.sessions.mirror import EXTERNAL_SESSION_AGENT_PREFIX, SessionMirror
from agent_harness.slack import AgentRosterStatus, build_team_roster_blocks, encode_action_value
from agent_harness.slack.app import SlackTeamController, _agent_authored_review_request
from agent_harness.storage.store import Store
from agent_harness.team import build_initial_model_team
from agent_harness.team.routing import canonical_agent_handle, canonicalize_agent_mentions

ACTIVE_TASK_STATUSES = (AgentTaskStatus.QUEUED, AgentTaskStatus.ACTIVE)
ALL_TASK_STATUSES = (
    AgentTaskStatus.QUEUED,
    AgentTaskStatus.ACTIVE,
    AgentTaskStatus.DONE,
    AgentTaskStatus.CANCELLED,
)
THREADS = ("171.000001", "171.000002", "171.000003")
PROPERTY_MAX_EXAMPLES = 2500


class FakeGateway:
    def __init__(self):
        self.posts = []
        self.updates = []
        self.thread_replies = []

    def post_message(self, channel_id, text, blocks=None, thread_ts=None):
        ts = f"1712345678.{len(self.posts):06d}"
        self.posts.append(
            {
                "channel_id": channel_id,
                "text": text,
                "blocks": blocks,
                "thread_ts": thread_ts,
                "ts": ts,
            }
        )
        return type("Posted", (), {"ts": ts})()

    def update_message(self, channel_id, ts, text, blocks=None):
        self.updates.append({"channel_id": channel_id, "ts": ts, "text": text, "blocks": blocks})

    def post_thread_reply(self, thread, text, persona=None, icon_url=None, blocks=None):
        ts = f"1712345679.{len(self.thread_replies):06d}"
        self.thread_replies.append(
            {
                "thread": thread,
                "text": text,
                "persona": persona,
                "icon_url": icon_url,
                "blocks": blocks,
                "ts": ts,
            }
        )
        return type("Posted", (), {"ts": ts, "thread_ts": thread.thread_ts})()

    def post_session_parent(self, channel_id, text, persona, icon_url=None, blocks=None):
        posted = self.post_message(channel_id, text, blocks=blocks)
        return posted

    def permalink(self, channel_id, message_ts):
        return f"https://example.slack.com/archives/{channel_id}/p{message_ts.replace('.', '')}"

    def pin_message(self, channel_id, message_ts):
        return None


class FakeProvider:
    def __init__(self, provider, sessions):
        self.provider = provider
        self.sessions = sessions

    def discover(self):
        return self.sessions

    def iter_events(self, transcript_path):
        return iter(())

    def usage_for_day(self, transcript_paths, day):
        return []


@st.composite
def task_scope_cases(draw):
    target_agent_index = draw(st.integers(min_value=0, max_value=2))
    target_thread = draw(st.sampled_from(THREADS))
    target_status = draw(st.sampled_from(ACTIVE_TASK_STATUSES))
    extra_specs = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=2),
                st.one_of(st.sampled_from(THREADS), st.none()),
                st.sampled_from(ALL_TASK_STATUSES),
            ),
            min_size=0,
            max_size=18,
        )
    )
    return [(target_agent_index, target_thread, target_status), *extra_specs]


@st.composite
def codex_stream_cases(draw):
    alphabet = st.characters(
        whitelist_categories=("Ll", "Lu", "Nd", "Zs", "Po"),
        whitelist_characters="\n_-/@[]()",
    )
    messages = draw(
        st.lists(st.text(alphabet=alphabet, min_size=1, max_size=120), min_size=1, max_size=8)
    )
    cleaned_messages = [_clean_terminal_output(message) for message in messages]
    assume(all(cleaned_messages))
    shapes = draw(
        st.lists(
            st.integers(min_value=0, max_value=2), min_size=len(messages), max_size=len(messages)
        )
    )
    lines = []
    for shape, message in zip(shapes, messages, strict=True):
        if shape == 0:
            event = {"type": "item.completed", "item": {"type": "agent_message", "text": message}}
        elif shape == 1:
            event = {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": message},
            }
        else:
            event = {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": message}],
                },
            }
        lines.append(json.dumps(event))
    stream = "\n".join(lines) + "\n"
    chunk_size = draw(st.integers(min_value=1, max_value=25))
    chunks = [stream[index : index + chunk_size] for index in range(0, len(stream), chunk_size)]
    return cleaned_messages, chunks


@st.composite
def roster_status_cases(draw):
    count = draw(st.integers(min_value=1, max_value=8))
    kinds = draw(
        st.lists(
            st.sampled_from(("available", "plain_occupied", "task", "external")),
            min_size=count,
            max_size=count,
        )
    )
    return count, kinds


@st.composite
def external_capacity_cases(draw):
    codex_count = draw(st.integers(min_value=0, max_value=3))
    claude_count = draw(st.integers(min_value=0, max_value=3))
    providers = draw(
        st.lists(st.sampled_from((Provider.CODEX, Provider.CLAUDE)), min_size=0, max_size=8)
    )
    return codex_count, claude_count, providers


@st.composite
def handle_alias_cases(draw):
    first = draw(st.sampled_from(("Mina", "Cruz", "Laila", "Mei", "Rian", "Taylor")))
    suffix = draw(st.sampled_from(("a", "x", "1", "aa")))
    handle = f"{first.lower()}{suffix}"
    typo_index = draw(st.integers(min_value=0, max_value=len(handle) - 1))
    typo = handle[:typo_index] + handle[typo_index + 1 :]
    assume(len(typo) >= 3)
    return first, handle, typo


class PropertyTests(unittest.TestCase):
    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=None)
    @given(task_scope_cases())
    def test_task_free_up_only_finishes_same_agent_thread_scope(self, specs):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                agents = build_initial_model_team(2, 1)
                for agent in agents:
                    store.upsert_team_agent(agent)
                created_at = datetime(2026, 1, 1, tzinfo=UTC)
                for index, (agent_index, thread_ts, status) in enumerate(specs):
                    task = AgentTask(
                        task_id=f"task_{index}",
                        agent_id=agents[agent_index].agent_id,
                        prompt=f"task {index}",
                        channel_id="C1",
                        kind=AgentTaskKind.WORK,
                        status=status,
                        created_at=created_at + timedelta(seconds=index),
                        updated_at=created_at + timedelta(seconds=index),
                        thread_ts=thread_ts,
                        parent_message_ts=thread_ts,
                    )
                    store.upsert_agent_task(task)
                controller = SlackTeamController(store, FakeGateway(), default_channel_id="C1")

                controller.handle_block_action(
                    {
                        "type": "block_actions",
                        "channel": {"id": "C1"},
                        "message": {"ts": specs[0][1]},
                        "actions": [{"value": encode_action_value("task.done", task_id="task_0")}],
                    }
                )

                target_agent_id = agents[specs[0][0]].agent_id
                target_thread = specs[0][1]
                for index, (agent_index, thread_ts, original_status) in enumerate(specs):
                    stored = store.get_agent_task(f"task_{index}")
                    should_finish = (
                        agents[agent_index].agent_id == target_agent_id
                        and thread_ts == target_thread
                        and original_status in ACTIVE_TASK_STATUSES
                    )
                    expected = AgentTaskStatus.DONE if should_finish else original_status
                    self.assertEqual(stored.status, expected)
            finally:
                store.close()

    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=None)
    @given(codex_stream_cases())
    def test_codex_stream_parser_preserves_visible_messages_across_json_shapes_and_chunks(
        self, case
    ):
        expected, chunks = case
        output = []
        buffer = ""
        for chunk in chunks:
            rendered, buffer = _process_output_chunks(Provider.CODEX, chunk, buffer)
            output.extend(rendered)
        rendered, buffer = _process_output_chunks(Provider.CODEX, "", buffer, final=True)
        output.extend(rendered)

        self.assertEqual(output, expected)
        self.assertEqual(buffer, "")

    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=None)
    @given(roster_status_cases())
    def test_roster_buttons_match_occupancy_kind(self, case):
        count, kinds = case
        agents = build_initial_model_team(count, 0)
        statuses = {}
        for index, (agent, kind) in enumerate(zip(agents, kinds, strict=True)):
            if kind == "available":
                statuses[agent.agent_id] = AgentRosterStatus("Available")
            elif kind == "plain_occupied":
                statuses[agent.agent_id] = AgentRosterStatus("Occupied", "manual")
            elif kind == "task":
                statuses[agent.agent_id] = AgentRosterStatus(
                    "Occupied",
                    "Slack task",
                    thread_url=f"https://example.slack.com/archives/C1/p17100000{index}",
                    task_id=f"task_{index}",
                )
            else:
                statuses[agent.agent_id] = AgentRosterStatus(
                    "Occupied",
                    "codex session outside Slack",
                    thread_url=f"https://example.slack.com/archives/C1/p17100000{index}",
                    session_provider=Provider.CODEX,
                    session_id=f"s{index}",
                )

        blocks = build_team_roster_blocks(agents, statuses)
        action_blocks = [
            block
            for block in blocks
            if str(block.get("block_id", "")).startswith("team.agent.actions.")
        ]
        self.assertEqual(len(action_blocks), count)
        for kind, block in zip(kinds, action_blocks, strict=True):
            labels = [element["text"]["text"] for element in block["elements"]]
            if kind in {"task", "external"}:
                self.assertEqual(labels, ["Free up", "Open thread", "Fire"])
                open_thread = block["elements"][1]
                self.assertEqual(open_thread["action_id"], "thread.open")
                self.assertIn("/archives/C1/p", open_thread["url"])
            else:
                self.assertEqual(labels, ["Fire"])

    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=None)
    @given(external_capacity_cases())
    def test_external_session_assignment_never_overfills_matching_provider_capacity(self, case):
        codex_count, claude_count, providers = case
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "state.sqlite")
            try:
                store.init_schema()
                if codex_count + claude_count:
                    for agent in build_initial_model_team(codex_count, claude_count):
                        store.upsert_team_agent(agent)
                sessions_by_provider = {Provider.CODEX: [], Provider.CLAUDE: []}
                for index, provider in enumerate(providers):
                    sessions_by_provider[provider].append(
                        AgentSession(
                            provider=provider,
                            session_id=f"s{index}",
                            transcript_path=Path(tmp) / f"{provider.value}-{index}.jsonl",
                            status=SessionStatus.ACTIVE,
                        )
                    )
                mirror = SessionMirror(
                    store,
                    FakeGateway(),
                    [
                        FakeProvider(Provider.CODEX, sessions_by_provider[Provider.CODEX]),
                        FakeProvider(Provider.CLAUDE, sessions_by_provider[Provider.CLAUDE]),
                    ],
                    team_id="T1",
                    channel_id="C1",
                )

                mirror.sync_once()

                active_agents = {agent.agent_id: agent for agent in store.list_team_agents()}
                assignments = store.list_settings(EXTERNAL_SESSION_AGENT_PREFIX)
                assigned_agent_ids = list(assignments.values())
                self.assertEqual(len(assigned_agent_ids), len(set(assigned_agent_ids)))
                assigned_by_provider = {Provider.CODEX: 0, Provider.CLAUDE: 0}
                capacity_by_provider = {
                    Provider.CODEX: codex_count,
                    Provider.CLAUDE: claude_count,
                }
                for key, agent_id in assignments.items():
                    provider = Provider(
                        key.removeprefix(EXTERNAL_SESSION_AGENT_PREFIX).split(".", 1)[0]
                    )
                    self.assertEqual(active_agents[agent_id].provider_preference, provider)
                    assigned_by_provider[provider] += 1
                for provider in (Provider.CODEX, Provider.CLAUDE):
                    self.assertLessEqual(
                        assigned_by_provider[provider], capacity_by_provider[provider]
                    )
                    self.assertLessEqual(
                        assigned_by_provider[provider],
                        providers.count(provider),
                    )
            finally:
                store.close()

    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=None)
    @given(handle_alias_cases())
    def test_agent_handle_aliases_route_unique_display_aliases_and_single_typos(self, case):
        first, handle, typo = case
        agent = replace(
            build_initial_model_team(1, 0)[0],
            full_name=f"{first} Example",
            handle=handle,
        )

        self.assertEqual(
            canonicalize_agent_mentions(f"@{first.lower()} take this", [agent]),
            f"@{handle} take this",
        )
        self.assertEqual(canonical_agent_handle(typo, [agent]), handle)

    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=None)
    @given(st.sampled_from(("Mina", "Cruz", "Laila", "Mei", "Rian", "Taylor")))
    def test_agent_handle_aliases_do_not_guess_ambiguous_display_aliases(self, first):
        base = build_initial_model_team(2, 0)
        first_lower = first.lower()
        agents = [
            replace(
                base[0],
                full_name=f"{first} Example",
                handle=f"{first_lower}a",
            ),
            replace(
                base[1],
                full_name=f"{first} Other",
                handle=f"{first_lower}b",
            ),
        ]

        self.assertEqual(
            canonicalize_agent_mentions(f"@{first_lower} take this", agents),
            f"@{first_lower} take this",
        )

    @settings(max_examples=PROPERTY_MAX_EXAMPLES, deadline=None)
    @given(st.text(alphabet="abc ABC \t", max_size=120))
    def test_agent_authored_review_requests_can_start_after_newline(self, prefix):
        agents = build_initial_model_team(1, 1)

        request = _agent_authored_review_request(
            f"{prefix}\n\nsomebody review this and then hand it back",
            agents,
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.prompt, "review this and then hand it back")


if __name__ == "__main__":
    unittest.main()
