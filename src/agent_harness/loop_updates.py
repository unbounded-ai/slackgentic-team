"""Mission changes for existing loops, proposed by local agents.

A local agent (through the MCP tool or the CLI) proposes a complete replacement
mission for one loop. The daemon posts the proposal in the loop's channel as a
diff with Apply and Cancel buttons, and only the loop owner's Apply stores the
new mission, verbatim. The proposal records the mission it was written against,
so an Apply after any other edit is refused instead of overwriting that edit.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Literal

from agent_harness.loop_guard import LOOP_GUARD_RUN_ENV
from agent_harness.loops import AGENT_LOOP_REQUEST_MAX_PENDING, AGENT_LOOP_REQUEST_WAIT_SECONDS
from agent_harness.models import (
    Loop,
    LoopStatus,
    LoopUpdateQueueRequest,
    LoopUpdateRequestStatus,
)

LOOP_UPDATE_MISSION_MAX_CHARS = 12_000
LOOP_UPDATE_NOTE_MAX_CHARS = 300
LOOP_UPDATE_REQUEST_ID_PREFIX = "loopupd_"
LOOP_UPDATE_EDITABLE_STATUSES = frozenset({LoopStatus.ACTIVE, LoopStatus.PAUSED})
LOOP_UPDATE_STALE_ERROR = (
    "the loop's mission changed after this change was proposed; read the current mission "
    "and propose the change again"
)
# Sentences pair up as one edited sentence when at least this share of their words match.
SENTENCE_PAIRING_RATIO = 0.5

_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
_TOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]")
_WORD_RE = re.compile(r"\w+")

DiffOp = Literal["equal", "delete", "insert"]
ChunkKind = Literal["same", "removed", "added", "changed"]


@dataclass(frozen=True)
class MissionDiffSegment:
    op: DiffOp
    text: str


@dataclass(frozen=True)
class MissionDiffChunk:
    """One run of the diff: unchanged, removed, or added sentences, or one edited sentence."""

    kind: ChunkKind
    sentences: tuple[str, ...] = ()
    segments: tuple[MissionDiffSegment, ...] = ()


@dataclass(frozen=True)
class MissionDiff:
    chunks: tuple[MissionDiffChunk, ...]
    words_added: int
    words_removed: int
    sentences_changed: int
    sentences_before: int
    sentences_after: int

    @property
    def changed(self) -> bool:
        return any(chunk.kind != "same" for chunk in self.chunks)

    def summary(self) -> str:
        return (
            f"+{self.words_added} words, -{self.words_removed} words; "
            f"{self.sentences_changed} sentences changed"
        )


@dataclass(frozen=True)
class AgentLoopUpdateResult:
    loop: Loop | None = None
    request: LoopUpdateQueueRequest | None = None
    diff: MissionDiff | None = None
    error: str | None = None


def split_mission_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(
                part.strip() for part in _SENTENCE_BREAK_RE.split(paragraph) if part.strip()
            )
    return sentences


def diff_loop_missions(old: str, new: str) -> MissionDiff:
    """Diff two missions sentence by sentence, with word-level edits inside changed sentences."""

    before = split_mission_sentences(old)
    after = split_mission_sentences(new)
    raw: list[MissionDiffChunk] = []
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            raw.append(MissionDiffChunk("same", tuple(before[i1:i2])))
        elif tag == "delete":
            raw.extend(MissionDiffChunk("removed", (item,)) for item in before[i1:i2])
        elif tag == "insert":
            raw.extend(MissionDiffChunk("added", (item,)) for item in after[j1:j2])
        else:
            raw.extend(_pair_replaced_sentences(before[i1:i2], after[j1:j2]))
    chunks = _merge_sentence_chunks(raw)
    words_added = words_removed = sentences_changed = 0
    for chunk in chunks:
        if chunk.kind == "added":
            words_added += sum(_word_count(item) for item in chunk.sentences)
            sentences_changed += len(chunk.sentences)
        elif chunk.kind == "removed":
            words_removed += sum(_word_count(item) for item in chunk.sentences)
            sentences_changed += len(chunk.sentences)
        elif chunk.kind == "changed":
            words_added += sum(_word_count(s.text) for s in chunk.segments if s.op == "insert")
            words_removed += sum(_word_count(s.text) for s in chunk.segments if s.op == "delete")
            sentences_changed += 1
    return MissionDiff(
        chunks=tuple(chunks),
        words_added=words_added,
        words_removed=words_removed,
        sentences_changed=sentences_changed,
        sentences_before=len(before),
        sentences_after=len(after),
    )


def render_mission_diff_text(diff: MissionDiff) -> str:
    """Plain-text diff for terminals and tool results, in git word-diff notation."""

    lines = [diff.summary()]
    for chunk in diff.chunks:
        if chunk.kind == "same":
            count = len(chunk.sentences)
            lines.append(f"  ... {count} unchanged sentence{'s' if count != 1 else ''}")
        elif chunk.kind == "removed":
            lines.extend(f"- {item}" for item in chunk.sentences)
        elif chunk.kind == "added":
            lines.extend(f"+ {item}" for item in chunk.sentences)
        else:
            rendered = "".join(
                segment.text
                if segment.op == "equal"
                else f"[-{segment.text}-]"
                if segment.op == "delete"
                else f"{{+{segment.text}+}}"
                for segment in chunk.segments
            )
            lines.append(f"~ {rendered}")
    return "\n".join(lines)


def resolve_loop_reference(store, reference: str) -> Loop | str:
    """Find one live loop by id, channel id, channel name, or exact title."""

    wanted = (reference or "").strip()
    if not wanted:
        return "name the loop by its id, channel, or title"
    candidates = store.list_loops(
        statuses=tuple(status for status in LoopStatus if status != LoopStatus.CANCELLED)
    )
    channel_name = wanted.removeprefix("#").lower()
    matches = [
        loop
        for loop in candidates
        if wanted in {loop.loop_id, loop.channel_id}
        or (loop.channel_name or "").lower() == channel_name
        or loop.title.lower() == wanted.lower()
    ]
    if not matches:
        return f"no loop matches {wanted!r}; list loops with `slackgentic loop list`"
    if len(matches) > 1:
        ids = ", ".join(loop.loop_id for loop in matches)
        return f"{wanted!r} matches several loops ({ids}); use the loop id"
    loop = matches[0]
    if loop.status not in LOOP_UPDATE_EDITABLE_STATUSES or not loop.channel_id:
        return f"{loop.title} is {loop.status.value}; only active or paused loops can be edited"
    return loop


def preview_agent_loop_update(store, reference: str, mission: str) -> AgentLoopUpdateResult:
    """Validate a proposed mission and diff it against the loop's current one."""

    found = resolve_loop_reference(store, reference)
    if isinstance(found, str):
        return AgentLoopUpdateResult(error=found)
    proposed = normalize_proposed_mission(mission)
    if not proposed:
        return AgentLoopUpdateResult(loop=found, error="give the complete new mission")
    if len(proposed) > LOOP_UPDATE_MISSION_MAX_CHARS:
        return AgentLoopUpdateResult(
            loop=found,
            error=f"keep the mission under {LOOP_UPDATE_MISSION_MAX_CHARS} characters",
        )
    if proposed == found.mission.strip():
        return AgentLoopUpdateResult(loop=found, error="the proposed mission is unchanged")
    return AgentLoopUpdateResult(loop=found, diff=diff_loop_missions(found.mission, proposed))


def enqueue_agent_loop_update_request(
    store,
    reference: str,
    mission: str,
    *,
    note: str | None = None,
    source: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> AgentLoopUpdateResult:
    """Queue a proposed mission for the daemon to post for the owner's approval."""

    if (environ if environ is not None else os.environ).get(LOOP_GUARD_RUN_ENV):
        return AgentLoopUpdateResult(
            error="loop runs cannot change loops; report the suggestion to the owner instead"
        )
    preview = preview_agent_loop_update(store, reference, mission)
    if preview.error is not None or preview.loop is None:
        return preview
    cleaned_note = " ".join((note or "").split())
    if len(cleaned_note) > LOOP_UPDATE_NOTE_MAX_CHARS:
        return AgentLoopUpdateResult(
            loop=preview.loop,
            error=f"keep the note under {LOOP_UPDATE_NOTE_MAX_CHARS} characters",
        )
    if store.count_pending_loop_update_requests() >= AGENT_LOOP_REQUEST_MAX_PENDING:
        return AgentLoopUpdateResult(
            loop=preview.loop,
            error=(
                f"{AGENT_LOOP_REQUEST_MAX_PENDING} loop changes are already waiting for the "
                "Slackgentic service; check that it is running with `slackgentic service status`"
            ),
        )
    request = store.enqueue_loop_update_request(
        loop_id=preview.loop.loop_id,
        mission=normalize_proposed_mission(mission),
        base_mission=preview.loop.mission,
        note=cleaned_note or None,
        source=source,
    )
    return AgentLoopUpdateResult(loop=preview.loop, request=request, diff=preview.diff)


def wait_for_agent_loop_update_request(
    store,
    request_id: str,
    *,
    timeout_seconds: float = AGENT_LOOP_REQUEST_WAIT_SECONDS,
    poll_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> LoopUpdateQueueRequest | None:
    """Wait until the daemon posts, refuses, or fails a queued mission change."""

    deadline = monotonic() + max(0.0, timeout_seconds)
    while True:
        current = store.get_loop_update_request(request_id)
        if current is None or current.status not in {
            LoopUpdateRequestStatus.PENDING,
            LoopUpdateRequestStatus.CLAIMED,
        }:
            return current
        if monotonic() >= deadline:
            return current
        sleep(poll_seconds)


def describe_agent_loop_update_request(request: LoopUpdateQueueRequest | None) -> str:
    if request is None:
        return "The loop change could not be found."
    status = request.status
    if status in {LoopUpdateRequestStatus.PENDING, LoopUpdateRequestStatus.CLAIMED}:
        return (
            f"The loop change {request.request_id} is queued, but the Slackgentic service has "
            "not picked it up yet. It will post as soon as the service is running; check with "
            "`slackgentic service status`."
        )
    if status == LoopUpdateRequestStatus.POSTED:
        return (
            "Slackgentic posted the proposed mission as a diff in the loop's channel. Nothing "
            "changes until the owner taps Apply there; Cancel discards it."
        )
    if status == LoopUpdateRequestStatus.APPLIED:
        return "The owner applied the change; future runs use the new mission."
    if status == LoopUpdateRequestStatus.CANCELLED:
        return "The owner cancelled the change; the mission is unchanged."
    if status == LoopUpdateRequestStatus.STALE:
        return f"The change was not applied: {request.error or LOOP_UPDATE_STALE_ERROR}."
    return f"Slackgentic could not post the loop change: {request.error or 'unknown error'}"


def normalize_proposed_mission(text: str) -> str:
    return (text or "").replace("\r\n", "\n").strip()


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _sentence_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None, _WORD_RE.findall(left.lower()), _WORD_RE.findall(right.lower()), autojunk=False
    ).ratio()


def _pair_replaced_sentences(before: list[str], after: list[str]) -> list[MissionDiffChunk]:
    """Align rewritten sentences so edits read as changed sentences, not delete-and-add.

    The alignment keeps sentence order and maximizes total similarity over pairs
    that clear the pairing ratio. Sentences left between two pairs become one
    removed block followed by one added block, the way a line diff reads.
    """

    pairs = _align_sentences(before, after)
    chunks: list[MissionDiffChunk] = []
    previous_i = previous_j = 0
    for i, j in [*pairs, (len(before), len(after))]:
        if previous_i < i:
            chunks.append(MissionDiffChunk("removed", tuple(before[previous_i:i])))
        if previous_j < j:
            chunks.append(MissionDiffChunk("added", tuple(after[previous_j:j])))
        if i < len(before) and j < len(after):
            chunks.append(_changed_sentence(before[i], after[j]))
        previous_i, previous_j = i + 1, j + 1
    return chunks


def _align_sentences(before: list[str], after: list[str]) -> list[tuple[int, int]]:
    similarity = [[_sentence_similarity(left, right) for right in after] for left in before]
    rows, columns = len(before), len(after)
    best = [[0.0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows - 1, -1, -1):
        for j in range(columns - 1, -1, -1):
            score = max(best[i + 1][j], best[i][j + 1])
            if similarity[i][j] >= SENTENCE_PAIRING_RATIO:
                score = max(score, similarity[i][j] + best[i + 1][j + 1])
            best[i][j] = score
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < rows and j < columns:
        paired = similarity[i][j] + best[i + 1][j + 1]
        if similarity[i][j] >= SENTENCE_PAIRING_RATIO and best[i][j] == paired:
            pairs.append((i, j))
            i += 1
            j += 1
        elif best[i + 1][j] >= best[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def _changed_sentence(before: str, after: str) -> MissionDiffChunk:
    left = _TOKEN_RE.findall(before)
    right = _TOKEN_RE.findall(after)
    raw: list[tuple[DiffOp, str]] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if tag == "equal":
            raw.append(("equal", "".join(left[i1:i2])))
            continue
        if tag in {"delete", "replace"}:
            raw.append(("delete", "".join(left[i1:i2])))
        if tag in {"insert", "replace"}:
            raw.append(("insert", "".join(right[j1:j2])))
    return MissionDiffChunk("changed", segments=_readable_segments(raw))


def _readable_segments(raw: list[tuple[DiffOp, str]]) -> tuple[MissionDiffSegment, ...]:
    """Group edits into delete-then-insert runs, absorbing the spaces between them.

    Without this, rewriting three words shows as three struck and three bold
    fragments separated by single spaces, which reads worse than one phrase each.
    """

    groups: list[list[str] | str] = []
    for op, text in raw:
        if op == "equal":
            groups.append(text)
            continue
        if not groups or isinstance(groups[-1], str):
            groups.append(["", ""])
        change = groups[-1]
        assert isinstance(change, list)
        change[0 if op == "delete" else 1] += text
    merged: list[list[str] | str] = []
    index = 0
    while index < len(groups):
        item = groups[index]
        if (
            isinstance(item, str)
            and merged
            and isinstance(merged[-1], list)
            and index + 1 < len(groups)
            and isinstance(groups[index + 1], list)
            and _is_glue(item)
        ):
            following = groups[index + 1]
            assert isinstance(following, list)
            previous = merged[-1]
            previous[0] += item + following[0]
            previous[1] += item + following[1]
            index += 2
            continue
        merged.append(item)
        index += 1
    segments: list[MissionDiffSegment] = []
    for item in merged:
        if isinstance(item, str):
            if item:
                segments.append(MissionDiffSegment("equal", item))
            continue
        deleted, inserted = item
        if deleted:
            segments.append(MissionDiffSegment("delete", deleted))
        if inserted:
            segments.append(MissionDiffSegment("insert", inserted))
    return tuple(segments)


def _is_glue(text: str) -> bool:
    stripped = text.strip()
    return not stripped or (len(stripped) <= 1 and not stripped.isalnum())


def _merge_sentence_chunks(chunks: list[MissionDiffChunk]) -> list[MissionDiffChunk]:
    merged: list[MissionDiffChunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.kind != "changed"
            and merged[-1].kind == chunk.kind
            and chunk.kind in {"same", "removed", "added"}
        ):
            merged[-1] = MissionDiffChunk(chunk.kind, merged[-1].sentences + chunk.sentences)
        else:
            merged.append(chunk)
    return merged
