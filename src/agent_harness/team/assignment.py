from __future__ import annotations

from dataclasses import dataclass, replace

from agent_harness.models import (
    ASSIGNMENT_PROMPT_METADATA_KEY,
    DANGEROUS_MODE_METADATA_KEY,
    AgentTask,
    AssignmentMode,
    TeamAgent,
    WorkRequest,
)
from agent_harness.storage.store import Store
from agent_harness.team import create_agent_task, pick_idle_agent
from agent_harness.team.routing import canonicalize_agent_mentions, parse_work_request


@dataclass(frozen=True)
class AssignmentResult:
    request: WorkRequest
    agent: TeamAgent
    task: AgentTask


def assign_channel_work_request(
    store: Store,
    text: str,
    channel_id: str,
    requested_by_slack_user: str | None = None,
) -> AssignmentResult | None:
    active_agents = store.list_team_agents()
    canonical_text = canonicalize_agent_mentions(text, active_agents)
    request = parse_work_request(canonical_text, [agent.handle for agent in active_agents])
    if request is None:
        return None

    author_agent = store.get_team_agent(request.author_handle) if request.author_handle else None
    return assign_work_request(
        store,
        request,
        channel_id,
        requested_by_slack_user=requested_by_slack_user,
        author_agent=author_agent,
    )


def assign_work_request(
    store: Store,
    request: WorkRequest,
    channel_id: str,
    requested_by_slack_user: str | None = None,
    author_agent: TeamAgent | None = None,
    extra_metadata: dict[str, object] | None = None,
    force_agent: TeamAgent | None = None,
    exclude_agent_ids: set[str] | frozenset[str] | None = None,
) -> AssignmentResult | None:
    idle_agents = store.idle_team_agents()
    if force_agent is None and exclude_agent_ids:
        idle_agents = [agent for agent in idle_agents if agent.agent_id not in exclude_agent_ids]
    if force_agent is not None:
        agent = force_agent
    else:
        candidates = idle_agents
        if (
            request.assignment_mode == AssignmentMode.ANYONE
            and author_agent
            and author_agent.provider_preference is not None
        ):
            cross_model = [
                agent
                for agent in idle_agents
                if agent.provider_preference is not None
                and agent.provider_preference != author_agent.provider_preference
            ]
            if cross_model:
                candidates = cross_model
        agent = pick_idle_agent(candidates, request, author_agent)
    if agent is None:
        return None

    task = create_agent_task(
        agent,
        request.prompt,
        channel_id,
        requested_by_slack_user=requested_by_slack_user,
        kind=request.task_kind,
    )
    metadata = dict(task.metadata)
    if request.author_handle:
        metadata["author_handle"] = request.author_handle
    if request.pr_url:
        metadata["pr_url"] = request.pr_url
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata.setdefault(ASSIGNMENT_PROMPT_METADATA_KEY, request.prompt)
    if request.dangerous_mode:
        metadata[DANGEROUS_MODE_METADATA_KEY] = True
    if metadata:
        task = replace(task, metadata=metadata)

    store.upsert_agent_task(task)
    return AssignmentResult(request=request, agent=agent, task=task)
