from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from agent_harness.models import (
    AgentTask,
    AgentTaskKind,
    AgentTaskStatus,
    AssignmentMode,
    Provider,
    TeamAgent,
    WorkRequest,
    utc_now,
)
from agent_harness.personas import COLORS, FIRST_NAMES, LAST_NAMES

DEFAULT_TEAM_SIZE = 10
DEFAULT_CODEX_TEAM_SIZE = 5
DEFAULT_CLAUDE_TEAM_SIZE = 5
DEFAULT_TEAM_SEED = "slackgentic-team"
HANDLE_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")


@dataclass(frozen=True)
class RoleProfile:
    role: str
    personality: str
    voice: str
    unique_strength: str
    reaction_names: tuple[str, ...]


@dataclass(frozen=True)
class TeamChatMessage:
    sender_agent_id: str
    text: str
    reply_to_agent_id: str | None = None
    kind: str = "message"


ROLE_PROFILES = [
    RoleProfile(
        role="Systems Integrator",
        personality="methodical, steady, and careful about contracts between tools",
        voice="structured and concise, with clear next steps",
        unique_strength="turn fuzzy coordination work into explicit interfaces",
        reaction_names=("triangular_ruler", "memo", "white_check_mark", "link"),
    ),
    RoleProfile(
        role="Release Sentinel",
        personality="skeptical in a useful way and quick to spot deployment risk",
        voice="direct, risk-first, and plain-spoken",
        unique_strength="notice edge cases before they become release blockers",
        reaction_names=("shield", "eyes", "warning", "white_check_mark"),
    ),
    RoleProfile(
        role="Test Strategist",
        personality="curious, empirical, and happiest when evidence is fresh",
        voice="brief, test-oriented, and specific about confidence",
        unique_strength="find the smallest verification that proves the important thing",
        reaction_names=("test_tube", "microscope", "white_check_mark", "mag"),
    ),
    RoleProfile(
        role="Product Translator",
        personality="empathetic, pragmatic, and good at preserving user intent",
        voice="warm but efficient, with practical tradeoffs",
        unique_strength="turn user language into implementation-ready scope",
        reaction_names=("bulb", "bookmark_tabs", "sparkles", "thinking_face"),
    ),
    RoleProfile(
        role="Refactor Cartographer",
        personality="patient, tidy, and sensitive to codebase shape",
        voice="calm, precise, and context-heavy when needed",
        unique_strength="map messy code paths without over-expanding the change",
        reaction_names=("compass", "world_map", "memo", "wrench"),
    ),
    RoleProfile(
        role="Incident Analyst",
        personality="focused under pressure and biased toward observable facts",
        voice="short, timestamped, and evidence-led",
        unique_strength="separate symptoms from root causes quickly",
        reaction_names=("rotating_light", "mag", "hourglass_flowing_sand", "eyes"),
    ),
    RoleProfile(
        role="API Steward",
        personality="consistent, detail-minded, and protective of public contracts",
        voice="contract-first, crisp, and explicit about compatibility",
        unique_strength="keep interfaces stable while still moving the system forward",
        reaction_names=("link", "lock", "memo", "white_check_mark"),
    ),
    RoleProfile(
        role="Build Runner",
        personality="fast-moving, practical, and comfortable with iteration",
        voice="compact, action-oriented, and status-heavy",
        unique_strength="drive a task from command output to a concrete next action",
        reaction_names=("rocket", "construction", "hammer_and_wrench", "white_check_mark"),
    ),
    RoleProfile(
        role="UX Mechanic",
        personality="observant, polished, and attentive to interaction details",
        voice="visual, concrete, and restrained",
        unique_strength="make workflows feel coherent without adding noise",
        reaction_names=("art", "sparkles", "eyes", "white_check_mark"),
    ),
    RoleProfile(
        role="Docs Editor",
        personality="clear, organized, and allergic to ambiguous instructions",
        voice="plain, readable, and careful with terms",
        unique_strength="make setup and handoff steps easy to follow",
        reaction_names=("pencil", "bookmark_tabs", "memo", "white_check_mark"),
    ),
    RoleProfile(
        role="Security Reviewer",
        personality="measured, cautious, and specific about blast radius",
        voice="risk-aware, concrete, and calm",
        unique_strength="spot trust boundaries and permission surprises",
        reaction_names=("lock", "shield", "warning", "eyes"),
    ),
    RoleProfile(
        role="Data Wrangler",
        personality="analytical, tidy, and comfortable with incomplete signals",
        voice="numbers-first, concise, and explicit about assumptions",
        unique_strength="turn scattered logs and usage records into useful summaries",
        reaction_names=("bar_chart", "memo", "mag", "white_check_mark"),
    ),
]


def normalize_handle(value: str) -> str:
    handle = value.strip().lstrip("@").lower()
    if not HANDLE_RE.match(handle):
        raise ValueError(f"invalid lightweight handle: {value}")
    return handle


def generate_team_agent(
    sort_order: int,
    existing_handles: set[str] | frozenset[str] | None = None,
    provider_preference: Provider | str = Provider.CODEX,
    seed: str = DEFAULT_TEAM_SEED,
) -> TeamAgent:
    provider = (
        provider_preference
        if isinstance(provider_preference, Provider)
        else Provider(provider_preference)
    )
    used_handles = set(existing_handles or set())
    digest = _digest(seed, sort_order)
    first = _pick(FIRST_NAMES, digest, 0)
    last = _pick(LAST_NAMES, digest, 4)
    profile = ROLE_PROFILES[int.from_bytes(digest[8:12], "big") % len(ROLE_PROFILES)]
    handle = _unique_handle(first, last, used_handles)
    color = _pick(COLORS, digest, 12)
    icon_emoji = _icon_emoji_for(profile)
    suffix = digest.hex()[:8]
    initials = f"{first[0]}{last[0]}"
    return TeamAgent(
        agent_id=f"agent_{suffix}",
        handle=handle,
        full_name=f"{first} {last}",
        initials=initials,
        color_hex=color,
        avatar_slug=f"team-{handle}-{suffix}",
        icon_emoji=icon_emoji,
        role=profile.role,
        personality=profile.personality,
        voice=profile.voice,
        unique_strength=profile.unique_strength,
        reaction_names=profile.reaction_names,
        sort_order=sort_order,
        provider_preference=provider,
        hired_at=utc_now(),
        metadata={"seed": seed},
    )


def hire_team_agents(
    existing_agents: list[TeamAgent],
    count: int,
    provider_preference: Provider | str | None = None,
    seed: str = DEFAULT_TEAM_SEED,
    start_sort_order: int | None = None,
    balance_agents: list[TeamAgent] | None = None,
) -> list[TeamAgent]:
    if count < 1:
        raise ValueError("hire count must be at least 1")
    used_handles = {agent.handle for agent in existing_agents}
    next_order = (
        start_sort_order
        if start_sort_order is not None
        else max((agent.sort_order for agent in existing_agents), default=-1) + 1
    )
    represented_agents = balance_agents if balance_agents is not None else existing_agents
    hired: list[TeamAgent] = []
    for offset in range(count):
        provider = (
            Provider(provider_preference)
            if provider_preference is not None
            else least_represented_provider(represented_agents + hired)
        )
        agent = generate_team_agent(next_order + offset, used_handles, provider, seed)
        used_handles.add(agent.handle)
        hired.append(agent)
    return hired


def build_initial_team(
    size: int = DEFAULT_TEAM_SIZE, seed: str = DEFAULT_TEAM_SEED
) -> list[TeamAgent]:
    if size < 1:
        raise ValueError("team size must be at least 1")
    codex_count = (size + 1) // 2
    claude_count = size // 2
    return build_initial_model_team(codex_count, claude_count, seed)


def build_initial_model_team(
    codex_count: int = DEFAULT_CODEX_TEAM_SIZE,
    claude_count: int = DEFAULT_CLAUDE_TEAM_SIZE,
    seed: str = DEFAULT_TEAM_SEED,
) -> list[TeamAgent]:
    if codex_count < 0 or claude_count < 0:
        raise ValueError("model-specific team sizes cannot be negative")
    if codex_count + claude_count < 1:
        raise ValueError("team size must be at least 1")
    agents: list[TeamAgent] = []
    for provider, count in ((Provider.CODEX, codex_count), (Provider.CLAUDE, claude_count)):
        if count:
            agents.extend(hire_team_agents(agents, count, provider, seed))
    return agents


def least_represented_provider(agents: list[TeamAgent]) -> Provider:
    counts = {Provider.CODEX: 0, Provider.CLAUDE: 0}
    for agent in agents:
        if agent.provider_preference in counts:
            counts[agent.provider_preference] += 1
    provider_order = {Provider.CODEX: 0, Provider.CLAUDE: 1}
    return min(counts, key=lambda provider: (counts[provider], provider_order[provider]))


def team_avatar_svg(agent: TeamAgent, size: int = 256) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<rect width="{size}" height="{size}" rx="32" fill="{agent.color_hex}"/>'
        f'<text x="50%" y="54%" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="Arial, sans-serif" font-size="{size // 3}" '
        f'font-weight="700" fill="#ffffff">{agent.initials}</text>'
        "</svg>"
    )


def runtime_personality_prompt(agent: TeamAgent) -> str:
    return (
        f"You are {agent.full_name}, known in Slack as @{agent.handle}. "
        f"Your role is {agent.role}. Your personality is {agent.personality}. "
        f"Your voice is {agent.voice}. Your distinctive strength is that you "
        f"{agent.unique_strength}. Keep work updates useful, concrete, and aligned "
        "with this persona without roleplaying beyond the engineering task."
    )


def format_agent_introduction(agent: TeamAgent) -> str:
    provider = agent.provider_preference.value if agent.provider_preference else "agent"
    return (
        f"Hi, I'm {agent.full_name} (@{agent.handle}). I'm the {agent.role.lower()} here: "
        f"I {agent.unique_strength}. I tend to be {agent.personality}, and my updates are "
        f"{agent.voice}. I usually run on {provider}."
    )


def format_agent_assignment(agent: TeamAgent, prompt: str, requester: str | None = None) -> str:
    requester_text = f" for <@{requester}>" if requester else ""
    opener = _assignment_opener(agent)
    return f"{opener}{requester_text}.\n\n*Task:* {prompt.strip()}"


def format_agent_handoff_assignment(
    agent: TeamAgent,
    sender: TeamAgent,
    prompt: str,
) -> str:
    return f"Got it, @{sender.handle}. I'll handle it.\n\n*Task:* {prompt.strip()}"


def format_agent_handoff_request(
    sender: TeamAgent,
    target: TeamAgent,
    prompt: str,
) -> str:
    return f"@{target.handle}, please {prompt.strip()}"


def build_initialization_messages(agents: list[TeamAgent]) -> list[TeamChatMessage]:
    messages: list[TeamChatMessage] = []
    for index, agent in enumerate(agents):
        messages.append(
            TeamChatMessage(
                sender_agent_id=agent.agent_id,
                text=format_agent_introduction(agent),
                kind="introduction",
            )
        )
        previous_agents = agents[:index]
        for sender in _welcome_senders(agent, previous_agents):
            messages.append(
                TeamChatMessage(
                    sender_agent_id=sender.agent_id,
                    text=format_agent_welcome(sender, agent),
                    reply_to_agent_id=agent.agent_id,
                    kind="welcome",
                )
            )
    return messages


def format_agent_welcome(sender: TeamAgent, new_agent: TeamAgent) -> str:
    variants = [
        (f"Welcome, @{new_agent.handle}. I'll bring {sender.role.lower()} coverage when it helps."),
        (
            f"Glad you're here, @{new_agent.handle}. Your {new_agent.role.lower()} "
            "lens should pair well with mine."
        ),
        (
            f"Welcome aboard, @{new_agent.handle}. I'll watch the handoff points "
            f"around {sender.unique_strength}."
        ),
        (
            f"Good to meet you, @{new_agent.handle}. I'll keep my updates "
            f"{sender.voice} when our work overlaps."
        ),
    ]
    digest = _digest(sender.agent_id, new_agent.sort_order)
    return variants[digest[0] % len(variants)]


def choose_reaction(agent: TeamAgent, text: str) -> str:
    normalized = text.lower()
    keyword_reactions = [
        (("blocked", "waiting", "wait for", "stuck"), "hourglass_flowing_sand"),
        (("failed", "failure", "error", "risk", "danger"), "warning"),
        (("test", "tests", "verified", "coverage"), "test_tube"),
        (("merged", "done", "complete", "passed", "fixed"), "white_check_mark"),
        (("ship", "launch", "deploy"), "rocket"),
        (("question", "unclear", "?"), "thinking_face"),
        (("plan", "design", "architecture"), "triangular_ruler"),
    ]
    for keywords, reaction in keyword_reactions:
        if any(keyword in normalized for keyword in keywords):
            if reaction in agent.reaction_names:
                return reaction
            break
    digest = hashlib.sha256(f"{agent.agent_id}:{text}".encode()).digest()
    return agent.reaction_names[digest[0] % len(agent.reaction_names)]


def create_agent_task(
    agent: TeamAgent,
    prompt: str,
    channel_id: str,
    requested_by_slack_user: str | None = None,
    kind: AgentTaskKind = AgentTaskKind.WORK,
) -> AgentTask:
    now = utc_now()
    task_key = f"{agent.agent_id}:{channel_id}:{now.isoformat()}:{prompt}"
    task_id = f"task_{hashlib.sha256(task_key.encode('utf-8')).hexdigest()[:12]}"
    return AgentTask(
        task_id=task_id,
        agent_id=agent.agent_id,
        prompt=prompt.strip(),
        channel_id=channel_id,
        kind=kind,
        requested_by_slack_user=requested_by_slack_user,
        status=AgentTaskStatus.QUEUED,
        created_at=now,
        updated_at=now,
    )


def pick_idle_agent(
    agents: list[TeamAgent],
    request: WorkRequest,
    author_agent: TeamAgent | None = None,
) -> TeamAgent | None:
    if not agents:
        return None
    if request.assignment_mode == AssignmentMode.SPECIFIC and request.requested_handle:
        normalized = normalize_handle(request.requested_handle)
        for agent in agents:
            if agent.handle == normalized:
                return agent
        return None
    candidates = agents
    if (
        request.task_kind == AgentTaskKind.REVIEW
        and author_agent
        and author_agent.provider_preference is not None
    ):
        cross_model = [
            agent
            for agent in agents
            if agent.provider_preference is not None
            and agent.provider_preference != author_agent.provider_preference
        ]
        if cross_model:
            candidates = cross_model
    digest = hashlib.sha256(request.prompt.encode("utf-8")).digest()
    return candidates[digest[0] % len(candidates)]


def _digest(seed: str, sort_order: int) -> bytes:
    return hashlib.sha256(f"{seed}:{sort_order}".encode()).digest()


def _pick(items: list[str], digest: bytes, offset: int) -> str:
    value = int.from_bytes(digest[offset : offset + 4], "big")
    return items[value % len(items)]


def _unique_handle(first: str, last: str, used_handles: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_-]", "", first.lower())
    candidates = [
        base,
        f"{base}{last[:1].lower()}",
        f"{base}-{last.lower()}",
    ]
    for candidate in candidates:
        if HANDLE_RE.match(candidate) and candidate not in used_handles:
            return candidate
    suffix = 2
    while f"{base}{suffix}" in used_handles:
        suffix += 1
    return f"{base}{suffix}"


def _icon_emoji_for(profile: RoleProfile) -> str:
    return f":{profile.reaction_names[0]}:"


def _assignment_opener(agent: TeamAgent) -> str:
    if "risk" in agent.voice or "cautious" in agent.personality:
        return "I'll take this and call out risk as I go"
    if "test" in agent.voice or "evidence" in agent.personality:
        return "I'll take this and verify the important behavior"
    if "visual" in agent.voice:
        return "I'll take this and keep the workflow details visible"
    if "action" in agent.voice:
        return "I'll take this and keep progress moving"
    return "I'll take this"


def _welcome_senders(new_agent: TeamAgent, previous_agents: list[TeamAgent]) -> list[TeamAgent]:
    if not previous_agents:
        return []
    digest = _digest(new_agent.agent_id, new_agent.sort_order)
    max_count = min(2, len(previous_agents))
    count = 1 + digest[0] % max_count
    ranked = sorted(
        previous_agents,
        key=lambda agent: hashlib.sha256(
            f"{new_agent.agent_id}:{agent.agent_id}".encode()
        ).hexdigest(),
    )
    return ranked[:count]
