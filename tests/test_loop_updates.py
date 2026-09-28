import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from agent_harness.cli import main
from agent_harness.loop_guard import (
    DENY,
    LOOP_GUARD_RUN_ENV,
    GuardContext,
    evaluate_tool_call,
)
from agent_harness.loop_updates import (
    LOOP_UPDATE_MISSION_MAX_CHARS,
    LOOP_UPDATE_NOTE_MAX_CHARS,
    describe_agent_loop_update_request,
    diff_loop_missions,
    enqueue_agent_loop_update_request,
    preview_agent_loop_update,
    render_mission_diff_text,
    resolve_loop_reference,
    split_mission_sentences,
)
from agent_harness.loops import AGENT_LOOP_REQUEST_MAX_PENDING
from agent_harness.models import LoopStatus, LoopUpdateRequestStatus
from agent_harness.sessions.claude_channel import (
    CHANNEL_INSTRUCTIONS,
    CODEX_MCP_INSTRUCTIONS,
    SLACKGENTIC_MCP_PERMISSION_ALLOW,
    ClaudeChannelServer,
)
from agent_harness.slack import (
    LOOP_UPDATE_CHUNK_TEXT_MAX,
    LOOP_UPDATE_DIFF_COLORS,
    LOOP_UPDATE_SLACK_VISIBLE_ATTACHMENTS,
    build_loop_update_continuation_messages,
    build_loop_update_message,
    decode_action_value,
    encode_action_value,
)
from agent_harness.slack.app import LoopRunner
from agent_harness.storage.store import Store
from tests import test_loop_app

OLD_MISSION = (
    "Every 30 minutes, check CI on main in example-org/example-repo. "
    "Report failing workflows with the failing step. "
    "Only post when something fails."
)
NEW_MISSION = (
    "Every 30 minutes, check CI on main in example-org/example-repo. "
    "Report failing workflows with the failing step and a likely cause. "
    "Investigate each new failure before reporting it. "
    "Only post when something fails."
)


class MissionDiffTests(unittest.TestCase):
    def test_splits_sentences_on_terminal_punctuation_and_lines(self):
        self.assertEqual(
            split_mission_sentences("First one. Second (two)!\nThird e.g. four. 5 five"),
            ["First one.", "Second (two)!", "Third e.g. four.", "5 five"],
        )

    def test_identical_missions_have_no_changes(self):
        diff = diff_loop_missions(OLD_MISSION, OLD_MISSION)

        self.assertFalse(diff.changed)
        self.assertEqual([chunk.kind for chunk in diff.chunks], ["same"])

    def test_edited_sentence_is_a_word_level_change_and_new_sentence_is_added(self):
        diff = diff_loop_missions(OLD_MISSION, NEW_MISSION)

        self.assertEqual(
            [chunk.kind for chunk in diff.chunks], ["same", "changed", "added", "same"]
        )
        changed = diff.chunks[1]
        self.assertEqual(
            [(segment.op, segment.text) for segment in changed.segments if segment.op != "equal"],
            [("insert", " and a likely cause")],
        )
        self.assertEqual(
            diff.chunks[2].sentences, ("Investigate each new failure before reporting it.",)
        )
        self.assertEqual(diff.words_added, 4 + 7)
        self.assertEqual(diff.words_removed, 0)
        self.assertEqual(diff.sentences_changed, 2)
        self.assertEqual((diff.sentences_before, diff.sentences_after), (3, 4))

    def test_rewritten_phrase_reads_as_one_deletion_and_one_insertion(self):
        diff = diff_loop_missions(
            "Report the failing step and the log excerpt today.",
            "Report the broken job and the runner name today.",
        )

        segments = [(segment.op, segment.text) for segment in diff.chunks[0].segments]
        self.assertEqual(
            segments,
            [
                ("equal", "Report the "),
                ("delete", "failing step"),
                ("insert", "broken job"),
                ("equal", " and the "),
                ("delete", "log excerpt"),
                ("insert", "runner name"),
                ("equal", " today."),
            ],
        )

    def test_unrelated_rewrites_group_removals_before_additions(self):
        diff = diff_loop_missions(
            "Keep this. Alpha beta gamma. Delta epsilon zeta. Keep that.",
            "Keep this. Something else entirely. Another new idea here. Keep that.",
        )

        self.assertEqual(
            [chunk.kind for chunk in diff.chunks], ["same", "removed", "added", "same"]
        )
        self.assertEqual(len(diff.chunks[1].sentences), 2)
        self.assertEqual(len(diff.chunks[2].sentences), 2)

    def test_text_rendering_uses_word_diff_markers(self):
        rendered = render_mission_diff_text(diff_loop_missions(OLD_MISSION, NEW_MISSION))

        self.assertIn("+11 words, -0 words; 2 sentences changed", rendered)
        self.assertIn("{+ and a likely cause+}", rendered)
        self.assertIn("+ Investigate each new failure before reporting it.", rendered)
        self.assertIn("... 1 unchanged sentence", rendered)


class LoopUpdateMessageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp_dir.name) / "state.sqlite")
        self.store.init_schema()

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def _loop_and_request(self, old: str, new: str):
        from datetime import UTC, datetime

        from agent_harness.models import LoopUpdateQueueRequest

        loop = _loop_fixture(old)
        now = datetime.now(UTC)
        request = LoopUpdateQueueRequest(
            request_id="loopupd_example",
            loop_id=loop.loop_id,
            mission=new,
            base_mission=old,
            status=LoopUpdateRequestStatus.POSTED,
            created_at=now,
            updated_at=now,
            note="Add an <@UOTHER> investigation step",
        )
        return loop, request

    def test_pending_message_has_colored_diff_and_owner_buttons_last(self):
        loop, request = self._loop_and_request(OLD_MISSION, NEW_MISSION)

        text, blocks, attachments = build_loop_update_message(
            loop, request, diff_loop_missions(OLD_MISSION, NEW_MISSION)
        )

        self.assertIn("<@UOWNER>", text)
        self.assertEqual(blocks[0]["type"], "header")
        self.assertIn("Proposed mission change", blocks[0]["text"]["text"])
        note = next(block for block in blocks if block["type"] == "context")
        self.assertEqual(note["elements"][0]["type"], "plain_text")
        colors = [item.get("color") for item in attachments[:-1]]
        self.assertEqual(
            colors,
            [
                LOOP_UPDATE_DIFF_COLORS["same"],
                LOOP_UPDATE_DIFF_COLORS["changed"],
                LOOP_UPDATE_DIFF_COLORS["added"],
                LOOP_UPDATE_DIFF_COLORS["same"],
            ],
        )
        edited = attachments[1]["blocks"][1]["elements"][0]["elements"]
        self.assertIn(
            {"type": "text", "text": " and a likely cause", "style": {"bold": True}}, edited
        )
        actions = attachments[-1]["blocks"][0]
        self.assertEqual(actions["type"], "actions")
        values = [decode_action_value(button["value"]) for button in actions["elements"]]
        self.assertEqual(
            [(value["action"], value["request_id"]) for value in values],
            [("loop.update.apply", "loopupd_example"), ("loop.update.cancel", "loopupd_example")],
        )

    def test_resolved_message_drops_the_buttons(self):
        loop, request = self._loop_and_request(OLD_MISSION, NEW_MISSION)
        diff = diff_loop_missions(OLD_MISSION, NEW_MISSION)

        for state in ("applied", "cancelled", "stale", "failed"):
            _text, blocks, attachments = build_loop_update_message(
                loop, request, diff, state=state, actor="UOWNER"
            )
            self.assertFalse(
                any(
                    block["type"] == "actions"
                    for item in attachments
                    for block in item.get("blocks", [])
                ),
                state,
            )
            self.assertEqual(blocks[0]["type"], "header")

    def test_no_space_is_added_before_replacement_punctuation(self):
        old, new = "Read otel_logs with SeverityNumber set.", "Read otel_logs, SeverityNumber set."
        loop, request = self._loop_and_request(old, new)

        _text, _blocks, card = build_loop_update_message(
            loop, request, diff_loop_missions(old, new)
        )

        rendered = "".join(
            element["text"] for element in card[0]["blocks"][1]["elements"][0]["elements"]
        )
        self.assertIn("otel_logs with, SeverityNumber", rendered)

    def test_long_runs_split_across_blocks_without_losing_text(self):
        old = " ".join(f"Old sentence number {index} about topic {index}." for index in range(80))
        new = " ".join(
            f"Completely different statement {index} with words {'x' * 200}." for index in range(80)
        )
        loop, request = self._loop_and_request(old, new)
        diff = diff_loop_missions(old, new)

        _text, _blocks, card = build_loop_update_message(loop, request, diff)
        pages = [card] + [page for _t, _b, page in build_loop_update_continuation_messages(diff)]

        texts = [
            "".join(element["text"] for element in block["elements"][0]["elements"])
            for page in pages
            for item in page
            for block in item.get("blocks", [])
            if block["type"] == "rich_text"
        ]
        for text in texts:
            self.assertLessEqual(len(text), LOOP_UPDATE_CHUNK_TEXT_MAX)
            self.assertFalse(text.endswith("…"))
        self.assertIn("Completely different statement 79 with", "\n".join(texts))
        for page in pages:
            self.assertLessEqual(len(page), LOOP_UPDATE_SLACK_VISIBLE_ATTACHMENTS)

    def test_many_edits_page_into_the_thread_and_keep_buttons_visible(self):
        old = " ".join(f"Keep {index}. Change the value {index} here." for index in range(40))
        new = " ".join(f"Keep {index}. Change the amount {index} here." for index in range(40))
        loop, request = self._loop_and_request(old, new)
        diff = diff_loop_missions(old, new)

        _text, _blocks, card = build_loop_update_message(loop, request, diff)
        continuations = build_loop_update_continuation_messages(diff)

        self.assertEqual(len(card), LOOP_UPDATE_SLACK_VISIBLE_ATTACHMENTS)
        self.assertEqual(card[-1]["blocks"][0]["type"], "actions")
        self.assertIn("continues in this message's thread", json.dumps(card[-2]))
        self.assertTrue(continuations)
        for text, _blocks, page in continuations:
            self.assertIn("Diff continued", text)
            self.assertLessEqual(len(page), LOOP_UPDATE_SLACK_VISIBLE_ATTACHMENTS)
        shown = len(card) - 2 + sum(len(page) for _t, _b, page in continuations)
        self.assertEqual(shown, len(diff.chunks))

    def test_struck_and_bold_words_do_not_run_together(self):
        old, new = "Run every query now.", "Source it; every query now."
        loop, request = self._loop_and_request(old, new)

        _text, _blocks, card = build_loop_update_message(
            loop, request, diff_loop_missions(old, new)
        )

        elements = card[0]["blocks"][1]["elements"][0]["elements"]
        texts = [(element["text"], element.get("style")) for element in elements]
        self.assertEqual(
            texts[:3], [("Run", {"strike": True}), (" ", None), ("Source it;", {"bold": True})]
        )


def _loop_fixture(mission: str):
    from datetime import UTC, datetime

    from agent_harness.models import Loop, LoopOverlapPolicy, PermissionMode, Provider

    now = datetime.now(UTC)
    return Loop(
        loop_id="loop_example",
        agent_id="agent_example",
        owner_slack_user_id="UOWNER",
        title="CI <Watch> & Friends",
        mission=mission,
        provider=Provider.CLAUDE,
        permission_mode=PermissionMode.READ_ONLY,
        recurrence={"frequency": "interval", "interval_seconds": 1800},
        status=LoopStatus.ACTIVE,
        overlap_policy=next(iter(LoopOverlapPolicy)),
        anchor_channel_id="CMAIN",
        anchor_thread_ts="100.000001",
        created_at=now,
        updated_at=now,
        channel_id="CLOOP",
        channel_name="loop-ci-watch",
    )


class LoopUpdateFlowTests(unittest.TestCase):
    setUp = test_loop_app.LoopCreationFlowTests.setUp
    tearDown = test_loop_app.LoopCreationFlowTests.tearDown
    _request_loop = test_loop_app.LoopCreationFlowTests._request_loop
    _resolve_loop = test_loop_app.LoopCreationFlowTests._resolve_loop
    _action_payload = test_loop_app.LoopCreationFlowTests._action_payload
    _activate_loop = test_loop_app.LoopCreationFlowTests._activate_loop

    def _propose(self, mission: str = NEW_MISSION, **kwargs):
        loop = self.store.list_loops()[0]
        result = enqueue_agent_loop_update_request(
            self.store, loop.loop_id, mission, environ={}, **kwargs
        )
        assert result.request is not None, result.error
        return result.request

    def _update_action(self, action: str, request, *, user: str = "UOWNER"):
        loop = self.store.get_loop(request.loop_id)
        posted = self.store.get_loop_update_request(request.request_id)
        return {
            "actions": [
                {
                    "value": encode_action_value(
                        action, loop_id=request.loop_id, request_id=request.request_id
                    )
                }
            ],
            "channel": {"id": loop.channel_id},
            "message": {"ts": posted.message_ts},
            "user": {"id": user},
        }

    def _posted_update(self):
        return [post for post in self.gateway.posts if post.get("attachments")]

    def test_runner_posts_the_proposal_in_the_loop_channel_with_full_text_in_thread(self):
        loop = self._activate_loop()
        request = self._propose(note="add investigation")

        LoopRunner(self.store, self.controller).sync_once()

        posted = self.store.get_loop_update_request(request.request_id)
        self.assertEqual(posted.status, LoopUpdateRequestStatus.POSTED)
        preview = self._posted_update()[-1]
        self.assertEqual(preview["channel_id"], loop.channel_id)
        self.assertIsNone(preview["thread_ts"])
        self.assertEqual(posted.message_ts, preview["ts"])
        full = self.gateway.posts[-1]
        self.assertEqual(full["thread_ts"], preview["ts"])
        self.assertLessEqual(len(preview["attachments"]), 10)
        self.assertEqual(full["blocks"][1]["elements"][0]["elements"][0]["text"], NEW_MISSION)
        self.assertEqual(self.store.get_loop(loop.loop_id).mission, loop.mission)

    def test_large_diff_pages_into_the_thread_before_the_full_mission(self):
        loop = self._activate_loop()
        old = " ".join(f"Keep {index}. Change the value {index} here." for index in range(40))
        new = " ".join(f"Keep {index}. Change the amount {index} here." for index in range(40))
        self.controller._update_loop_identity_values(loop, mission=old)
        self._propose(new)

        self.controller.process_queued_loop_update_requests()

        card = self._posted_update()[0]
        thread = [post for post in self.gateway.posts if post["thread_ts"] == card["ts"]]
        self.assertEqual(len(card["attachments"]), 10)
        self.assertTrue(all("Diff continued" in post["text"] for post in thread[:-1]))
        self.assertTrue(all(len(post["attachments"]) <= 10 for post in thread[:-1]))
        self.assertIn("Full proposed mission", thread[-1]["text"])

    def test_owner_apply_stores_the_mission_verbatim_and_marks_the_message(self):
        loop = self._activate_loop()
        request = self._propose()
        self.controller.process_queued_loop_update_requests()

        self.controller.handle_block_action(self._update_action("loop.update.apply", request))

        self.assertEqual(self.store.get_loop(loop.loop_id).mission, NEW_MISSION)
        applied = self.store.get_loop_update_request(request.request_id)
        self.assertEqual(applied.status, LoopUpdateRequestStatus.APPLIED)
        self.assertEqual(applied.resolved_by, "UOWNER")
        update = [item for item in self.gateway.updates if item.get("attachments")][-1]
        self.assertEqual(update["ts"], applied.message_ts)
        self.assertIn("applied", update["blocks"][0]["text"]["text"])
        journal = [entry.content for entry in self.store.list_loop_journal(loop.loop_id)]
        self.assertTrue(any("agent-proposed change the owner applied" in item for item in journal))
        self.assertFalse(any(NEW_MISSION in item for item in journal))

    def test_second_tap_does_not_apply_twice(self):
        self._activate_loop()
        request = self._propose()
        self.controller.process_queued_loop_update_requests()
        self.controller.handle_block_action(self._update_action("loop.update.apply", request))

        self.controller.handle_block_action(self._update_action("loop.update.apply", request))

        self.assertIn("already applied", self.gateway.ephemerals[-1][2])

    def test_non_owner_cannot_apply(self):
        loop = self._activate_loop()
        request = self._propose()
        self.controller.process_queued_loop_update_requests()

        self.controller.handle_block_action(
            self._update_action("loop.update.apply", request, user="UOTHER")
        )

        self.assertEqual(self.store.get_loop(loop.loop_id).mission, loop.mission)
        self.assertEqual(
            self.store.get_loop_update_request(request.request_id).status,
            LoopUpdateRequestStatus.POSTED,
        )
        self.assertIn("Only the loop owner", self.gateway.ephemerals[-1][2])

    def test_cancel_leaves_the_mission_alone(self):
        loop = self._activate_loop()
        request = self._propose()
        self.controller.process_queued_loop_update_requests()

        self.controller.handle_block_action(self._update_action("loop.update.cancel", request))

        self.assertEqual(self.store.get_loop(loop.loop_id).mission, loop.mission)
        self.assertEqual(
            self.store.get_loop_update_request(request.request_id).status,
            LoopUpdateRequestStatus.CANCELLED,
        )

    def test_owner_edit_after_proposal_makes_apply_stale(self):
        loop = self._activate_loop()
        request = self._propose()
        self.controller.process_queued_loop_update_requests()
        self.controller._update_loop_identity_values(loop, mission="Owner rewrote it.")

        self.controller.handle_block_action(self._update_action("loop.update.apply", request))

        self.assertEqual(self.store.get_loop(loop.loop_id).mission, "Owner rewrote it.")
        stale = self.store.get_loop_update_request(request.request_id)
        self.assertEqual(stale.status, LoopUpdateRequestStatus.STALE)
        self.assertIn("changed after", stale.error)

    def test_edit_before_posting_is_not_posted(self):
        loop = self._activate_loop()
        request = self._propose()
        self.controller._update_loop_identity_values(loop, mission="Owner rewrote it.")

        self.controller.process_queued_loop_update_requests()

        self.assertEqual(
            self.store.get_loop_update_request(request.request_id).status,
            LoopUpdateRequestStatus.STALE,
        )
        self.assertEqual(self._posted_update(), [])

    def test_applying_one_proposal_retires_the_others(self):
        self._activate_loop()
        first = self._propose()
        second = self._propose(NEW_MISSION + " Keep reports short.")
        self.controller.process_queued_loop_update_requests()

        self.controller.handle_block_action(self._update_action("loop.update.apply", first))

        self.assertEqual(
            self.store.get_loop_update_request(second.request_id).status,
            LoopUpdateRequestStatus.STALE,
        )


class LoopUpdateRequestTests(unittest.TestCase):
    setUp = test_loop_app.LoopCreationFlowTests.setUp
    tearDown = test_loop_app.LoopCreationFlowTests.tearDown
    _request_loop = test_loop_app.LoopCreationFlowTests._request_loop
    _resolve_loop = test_loop_app.LoopCreationFlowTests._resolve_loop
    _action_payload = test_loop_app.LoopCreationFlowTests._action_payload
    _activate_loop = test_loop_app.LoopCreationFlowTests._activate_loop

    def test_loop_reference_accepts_id_channel_and_title(self):
        loop = self._activate_loop()

        for reference in (
            loop.loop_id,
            f"#{loop.channel_name}",
            loop.channel_id,
            "cloud billing watch",
        ):
            found = resolve_loop_reference(self.store, reference)
            self.assertEqual(getattr(found, "loop_id", found), loop.loop_id, reference)
        self.assertIn("no loop matches", resolve_loop_reference(self.store, "#nope"))

    def test_loop_awaiting_approval_cannot_be_edited(self):
        loop = self._resolve_loop()

        result = resolve_loop_reference(self.store, loop.loop_id)

        self.assertIsInstance(result, str)
        self.assertIn("awaiting_approval", result)

    def test_rejects_unchanged_oversized_and_long_notes(self):
        loop = self._activate_loop()

        self.assertIn(
            "unchanged",
            preview_agent_loop_update(self.store, loop.loop_id, f"  {loop.mission}\n").error,
        )
        self.assertIn(
            "under",
            preview_agent_loop_update(
                self.store, loop.loop_id, "x" * (LOOP_UPDATE_MISSION_MAX_CHARS + 1)
            ).error,
        )
        self.assertIn(
            "note",
            enqueue_agent_loop_update_request(
                self.store,
                loop.loop_id,
                NEW_MISSION,
                note="n" * (LOOP_UPDATE_NOTE_MAX_CHARS + 1),
                environ={},
            ).error,
        )

    def test_refuses_inside_a_loop_run(self):
        loop = self._activate_loop()

        result = enqueue_agent_loop_update_request(
            self.store, loop.loop_id, NEW_MISSION, environ={LOOP_GUARD_RUN_ENV: "looprun_1"}
        )

        self.assertIsNone(result.request)
        self.assertIn("loop runs cannot change loops", result.error)

    def test_caps_waiting_requests_and_records_the_base_mission(self):
        loop = self._activate_loop()
        for index in range(AGENT_LOOP_REQUEST_MAX_PENDING):
            request = enqueue_agent_loop_update_request(
                self.store, loop.loop_id, f"{NEW_MISSION} {index}", environ={}
            ).request
            self.assertEqual(request.base_mission, loop.mission)

        result = enqueue_agent_loop_update_request(
            self.store, loop.loop_id, NEW_MISSION, environ={}
        )

        self.assertIn("already waiting", result.error)
        self.assertIn("queued", describe_agent_loop_update_request(request))


class LoopUpdateToolAndCliTests(unittest.TestCase):
    setUp = test_loop_app.LoopCreationFlowTests.setUp
    tearDown = test_loop_app.LoopCreationFlowTests.tearDown
    _request_loop = test_loop_app.LoopCreationFlowTests._request_loop
    _resolve_loop = test_loop_app.LoopCreationFlowTests._resolve_loop
    _action_payload = test_loop_app.LoopCreationFlowTests._action_payload
    _activate_loop = test_loop_app.LoopCreationFlowTests._activate_loop

    def _loop_with_old_mission(self):
        loop = self._activate_loop()
        self.controller._update_loop_identity_values(loop, mission=OLD_MISSION)
        return self.store.get_loop(loop.loop_id)

    def test_update_loop_tool_is_listed_allowlisted_and_documented(self):
        server = ClaudeChannelServer(self.store, target_pid=123, loop_request_wait_seconds=0)
        output = io.StringIO()

        with redirect_stdout(output):
            server._handle_message({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})

        tools = {tool["name"]: tool for tool in json.loads(output.getvalue())["result"]["tools"]}
        self.assertEqual(tools["update_loop"]["inputSchema"]["required"], ["loop", "mission"])
        self.assertIn("mcp__slackgentic__update_loop", SLACKGENTIC_MCP_PERMISSION_ALLOW)
        self.assertIn("update_loop", CHANNEL_INSTRUCTIONS)
        self.assertIn("update_loop", CODEX_MCP_INSTRUCTIONS)

    def test_dry_run_returns_the_diff_without_queueing(self):
        loop = self._loop_with_old_mission()
        server = ClaudeChannelServer(self.store, target_pid=123, loop_request_wait_seconds=0)

        result = server._handle_tool_call(
            {
                "name": "update_loop",
                "arguments": {"loop": loop.loop_id, "mission": NEW_MISSION, "dry_run": True},
            }
        )

        self.assertNotIn("isError", result)
        self.assertIn("{+", result["content"][0]["text"])
        self.assertEqual(self.store.count_pending_loop_update_requests(), 0)

    def test_tool_queues_the_proposal_and_refuses_in_loop_runs(self):
        loop = self._loop_with_old_mission()
        server = ClaudeChannelServer(self.store, target_pid=123, loop_request_wait_seconds=0)
        call = {"name": "update_loop", "arguments": {"loop": loop.loop_id, "mission": NEW_MISSION}}

        with patch.dict(os.environ, {LOOP_GUARD_RUN_ENV: "looprun_1"}):
            refused = server._handle_tool_call(call)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOOP_GUARD_RUN_ENV, None)
            queued = server._handle_tool_call(call)

        self.assertTrue(refused["isError"])
        self.assertNotIn("isError", queued)
        self.assertIn("queued", queued["content"][0]["text"])
        self.assertEqual(self.store.count_pending_loop_update_requests(), 1)

    def test_loop_guard_denies_the_update_tool(self):
        context = GuardContext(scratch_dir=Path(tempfile.gettempdir()))

        decision = evaluate_tool_call("mcp__slackgentic__update_loop", {}, context=context)

        self.assertEqual(decision.decision, DENY)
        self.assertNotEqual(
            evaluate_tool_call("mcp__slackgentic__read_thread", {}, context=context).decision,
            DENY,
        )

    def test_cli_dry_run_update_and_status(self):
        loop = self._loop_with_old_mission()
        db = Path(self.temp_dir.name) / "state.sqlite"
        mission_file = Path(self.temp_dir.name) / "mission.txt"
        mission_file.write_text(NEW_MISSION + "\n", encoding="utf-8")
        base = [
            "loop",
            "update",
            loop.loop_id,
            "--mission-file",
            str(mission_file),
            "--db",
            str(db),
        ]

        dry = io.StringIO()
        with redirect_stdout(dry), patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOOP_GUARD_RUN_ENV, None)
            self.assertEqual(main([*base, "--dry-run"]), 0)
        self.assertIn("{+ and a likely cause+}", dry.getvalue())
        self.assertEqual(self.store.count_pending_loop_update_requests(), 0)

        queued = io.StringIO()
        with redirect_stdout(queued), patch.dict(os.environ, {}, clear=False):
            os.environ.pop(LOOP_GUARD_RUN_ENV, None)
            self.assertEqual(main([*base, "--no-wait", "--json"]), 0)
        payload = json.loads(queued.getvalue())
        self.assertEqual(payload["status"], "pending")
        self.assertEqual(payload["mission"], NEW_MISSION)

        status = io.StringIO()
        with redirect_stdout(status):
            code = main(["loop", "request-status", payload["request_id"], "--db", str(db)])
        self.assertEqual(code, 0)
        self.assertIn("not picked it up", status.getvalue())

        errors = io.StringIO()
        with redirect_stderr(errors):
            code = main(
                ["loop", "update", "#missing", "--mission-file", str(mission_file), "--db", str(db)]
            )
        self.assertEqual(code, 2)
        self.assertIn("no loop matches", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
