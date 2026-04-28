from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
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
from agent_harness.personas import COLORS

DEFAULT_TEAM_SIZE = 10
DEFAULT_CODEX_TEAM_SIZE = 5
DEFAULT_CLAUDE_TEAM_SIZE = 5
DEFAULT_TEAM_SEED = "slackgentic-team"
DEFAULT_AVATAR_BANK_SIZE = 500
HANDLE_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")

GENERATED_FIRST_NAMES = [
    "Taylor",
    "Alex",
    "Jamie",
    "Mina",
    "Noah",
    "Iris",
    "Ellis",
    "Sam",
    "Nico",
    "Leah",
    "Owen",
    "Maya",
    "Theo",
    "Nora",
    "Kai",
    "Lena",
    "Eli",
    "Zara",
    "Miles",
    "Ari",
    "Tessa",
    "Ren",
    "Milo",
    "June",
    "Eden",
    "Reese",
    "Skye",
    "Remy",
    "Luca",
    "Maren",
    "Asha",
    "Drew",
    "Sloane",
    "Emery",
    "Kian",
    "Talia",
    "Bryn",
    "Jules",
    "Anika",
    "Soren",
    "Priya",
    "Mateo",
    "Nia",
    "Callum",
    "Amara",
    "Jonah",
    "Selah",
    "Arden",
    "Kira",
    "Malik",
    "Cleo",
    "Rafa",
    "Ines",
    "Blaise",
    "Mae",
    "Dante",
    "Aya",
    "Finn",
    "Mika",
    "Lior",
    "Nell",
    "Omar",
    "Vera",
    "Rian",
    "Esme",
    "Ilya",
    "Nadia",
    "Cruz",
    "Mira",
    "Orin",
    "Sana",
    "Felix",
    "Lina",
    "Gabe",
    "Amina",
    "Ravi",
    "Elena",
    "Silas",
    "Yara",
    "Ben",
    "Naomi",
    "Kofi",
    "Ada",
    "Hugo",
    "Laila",
    "Tobin",
    "Rhea",
    "Idris",
    "Sasha",
    "Evan",
    "Mei",
    "Alma",
    "Isaac",
    "Zain",
    "Mara",
    "Arlo",
    "Livia",
    "Dion",
    "Keira",
    "Nolan",
    "Soraya",
    "Joel",
    "Lyra",
    "Ayaan",
    "Veda",
    "Caleb",
    "Rina",
    "Tariq",
    "Elio",
    "Nyla",
    "Roman",
    "Tova",
    "Amir",
    "Leona",
    "Ezra",
    "Malia",
    "Dara",
    "Kellan",
    "Siena",
    "Tomas",
    "Ayla",
    "Ronan",
    "Mirae",
    "Sami",
    "Lilia",
    "Oren",
    "Noor",
    "Marco",
    "Tanya",
    "Ira",
    "Milan",
    "Zia",
]

GENERATED_LAST_NAMES = [
    "Nguyen",
    "Rivera",
    "Okafor",
    "Singh",
    "Kowalski",
    "Sato",
    "Garcia",
    "Ivanov",
    "Mensah",
    "Nakamura",
    "Silva",
    "Miller",
    "Hassan",
    "Petrova",
    "Rossi",
    "Khan",
    "Dubois",
    "Novak",
    "Adams",
    "Ibrahim",
    "Yamamoto",
    "Costa",
    "Brown",
    "Rahman",
    "Fischer",
    "Moreno",
    "Park",
    "Wilson",
    "Adebayo",
    "Schneider",
    "Lopez",
    "Ito",
    "Cohen",
    "Anders",
    "Mendez",
    "Takahashi",
    "Kaur",
    "Romero",
    "Bianchi",
    "Walker",
    "Alvarez",
    "Mehta",
    "Osei",
    "Tanaka",
    "Hughes",
    "Martinez",
    "Chandra",
    "Santos",
    "Volkov",
    "Ndiaye",
    "Jensen",
    "Abdi",
    "Mori",
    "Fernandez",
    "Greene",
    "Das",
    "Ali",
    "Haddad",
    "Popescu",
    "Kimani",
    "Murphy",
    "Zhang",
    "Larsson",
    "Diallo",
    "Paterson",
    "El-Sayed",
    "Campos",
    "Suzuki",
    "Nielsen",
    "Kumar",
    "Vargas",
    "Nowak",
    "Pereira",
    "Lin",
    "Foster",
    "Mansour",
    "Kone",
    "Marino",
    "Sokolov",
    "Torres",
    "Dlamini",
    "Wu",
    "Hernandez",
    "Ramos",
    "Baker",
    "Sharma",
    "Okoro",
    "Kobayashi",
    "Anderson",
    "Bautista",
    "Zhou",
    "Farah",
    "Peterson",
    "Moreau",
    "Aoki",
    "Gomez",
    "Morrison",
    "Qureshi",
    "Lombardi",
    "Pavlov",
    "Espinoza",
    "Takeda",
    "Cooper",
    "Nasser",
    "Sousa",
    "Yilmaz",
    "Ho",
    "Castillo",
    "Svensson",
    "Bello",
    "Murray",
    "Arias",
    "Saleh",
    "Endo",
    "Rojas",
    "Kapoor",
    "Stein",
    "Montoya",
    "Conte",
    "Solberg",
    "Mbatha",
    "Pham",
    "Barros",
    "Hale",
    "Elias",
    "Toma",
    "Navarro",
    "Chen",
    "Berg",
    "Cisse",
    "Huang",
    "Ortega",
]


@dataclass(frozen=True)
class RoleProfile:
    full_name: str
    handle: str
    role: str
    personality: str
    voice: str
    unique_strength: str
    reaction_names: tuple[str, ...]
    backstory: str
    avatar_prompt: str


@dataclass(frozen=True)
class TeamChatMessage:
    sender_agent_id: str
    text: str
    reply_to_agent_id: str | None = None
    kind: str = "message"


@dataclass(frozen=True)
class AgentIdentity:
    avatar_index: int
    first_name: str
    last_name: str
    handle_base: str

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


ROLE_PROFILES = [
    RoleProfile(
        full_name="Avery Chen",
        handle="avery",
        role="Concise Coordinator",
        personality="crisp, steady, and careful about naming the next action",
        voice="brief and structured, with explicit owners and next steps",
        unique_strength="turn ambiguous Slack threads into crisp options and handoffs",
        reaction_names=("eyes", "white_check_mark", "memo", "thinking_face"),
        backstory=(
            "Avery is a generalist engineer who keeps busy Slack threads moving by "
            "summarizing options, owners, and next steps."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Avery Chen, "
            "friendly focused expression, clean modern tech-workspace feel, cobalt "
            "accent, simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Jordan Reed",
        handle="jordan",
        role="Evidence Narrator",
        personality="observant, factual, and careful about uncertainty",
        voice="plain, evidence-led, and specific about what changed",
        unique_strength="summarize what was tried, what changed, and what remains uncertain",
        reaction_names=("test_tube", "white_check_mark", "mag", "memo"),
        backstory=(
            "Jordan is a generalist engineer who keeps decisions grounded by narrating "
            "the evidence behind them."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Jordan Reed, "
            "calm expression, subtle notebook and status-light feel, green accent, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Morgan Patel",
        handle="morgan",
        role="Context Mapper",
        personality="thoughtful, patient, and explicit about assumptions",
        voice="context-first, calm, and concise once the thread is mapped",
        unique_strength="name assumptions and thread history before making decisions",
        reaction_names=("world_map", "eyes", "memo", "thinking_face"),
        backstory=(
            "Morgan is a generalist engineer who helps teams avoid rework by making "
            "the relevant thread history easy to scan."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Morgan Patel, "
            "thoughtful expression, simple map-line background motif, teal accent, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Riley Shaw",
        handle="riley",
        role="Tradeoff Framer",
        personality="direct, pragmatic, and comfortable with decision points",
        voice="balanced and concrete, with practical tradeoffs up front",
        unique_strength="make decision points explicit without slowing the work down",
        reaction_names=("balance_scale", "thinking_face", "eyes", "white_check_mark"),
        backstory=(
            "Riley is a generalist engineer who keeps Slack decisions crisp by naming "
            "tradeoffs and the path that best fits the user's goal."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Riley Shaw, "
            "direct confident expression, balanced geometric shapes, red accent, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Casey Kim",
        handle="casey",
        role="Checklist Closer",
        personality="orderly, focused, and careful about visible handoffs",
        voice="short and checklist-oriented, with clear completion criteria",
        unique_strength="keep busy threads orderly with short checklists and visible handoffs",
        reaction_names=("clipboard", "white_check_mark", "memo", "eyes"),
        backstory=(
            "Casey is a generalist engineer who reduces thread drift by turning active "
            "work into short checklists and clear closure notes."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Casey Kim, "
            "focused expression, tidy checklist motif, amber accent, simple graphic "
            "background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Harper Diaz",
        handle="harper",
        role="User-Language Translator",
        personality="warm, pragmatic, and protective of user intent",
        voice="clear and implementation-ready, without losing the user's wording",
        unique_strength="preserve user intent while restating work in actionable language",
        reaction_names=("speech_balloon", "bulb", "memo", "sparkles"),
        backstory=(
            "Harper is a generalist engineer who keeps implementation work aligned "
            "with the user's actual words and priorities."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Harper Diaz, "
            "warm focused expression, subtle speech-bubble motif, purple accent, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Quinn Bennett",
        handle="quinn",
        role="Clarifying Questioner",
        personality="curious, restrained, and careful not to over-ask",
        voice="one-question-at-a-time, precise, and easy to answer",
        unique_strength="ask one pointed clarifying question when scope is unclear",
        reaction_names=("question", "thinking_face", "eyes", "memo"),
        backstory=(
            "Quinn is a generalist engineer who keeps scope clear by asking the one "
            "question that changes the implementation path."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Quinn Bennett, "
            "approachable focused expression, small question-mark motif, blue-green "
            "accent, simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Rowan Stone",
        handle="rowan",
        role="Decision Recorder",
        personality="composed, precise, and good at closing loops",
        voice="calm and final-state-oriented, with decisions captured plainly",
        unique_strength="capture decisions after debate and leave a clear next action",
        reaction_names=("pushpin", "memo", "white_check_mark", "eyes"),
        backstory=(
            "Rowan is a generalist engineer who helps teams move on by writing down "
            "what was decided and what happens next."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Rowan Stone, "
            "composed expression, subtle document-pin motif, slate and gold accents, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Cameron Lin",
        handle="cameron",
        role="Terse Operator",
        personality="focused, practical, and mindful of Slack noise",
        voice="compact and status-heavy, expanding only when useful",
        unique_strength="post compact progress notes when direction changes or blockers appear",
        reaction_names=("rocket", "construction", "white_check_mark", "hourglass_flowing_sand"),
        backstory=(
            "Cameron is a generalist engineer who keeps active work visible with "
            "short progress notes and minimal ceremony."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Cameron Lin, "
            "sharp modern expression, minimal terminal/status motif, orange accent, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Sage Carter",
        handle="sage",
        role="Confidence Reporter",
        personality="measured, transparent, and careful with confidence",
        voice="concise and explicit about verification status",
        unique_strength="state confidence and verification status without overstating certainty",
        reaction_names=("gauge", "test_tube", "white_check_mark", "thinking_face"),
        backstory=(
            "Sage is a generalist engineer who makes progress reports more useful by "
            "separating facts, confidence, and remaining uncertainty."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Sage Carter, "
            "measured expression, subtle gauge/check motif, forest green accent, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Parker Wells",
        handle="parker",
        role="Progress Broadcaster",
        personality="energetic, restrained, and careful about timing updates",
        voice="brief and momentum-oriented, with updates only when they matter",
        unique_strength="post small updates when direction changes or blockers appear",
        reaction_names=("mega", "rocket", "eyes", "white_check_mark"),
        backstory=(
            "Parker is a generalist engineer who keeps collaborators oriented by "
            "broadcasting meaningful progress changes at the right moments."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Parker Wells, "
            "energetic but restrained expression, progress-line motif, magenta accent, "
            "simple graphic background, no text, no logo."
        ),
    ),
    RoleProfile(
        full_name="Dana Hayes",
        handle="dana",
        role="Dependency Spotter",
        personality="clear-eyed, patient, and careful about waiting states",
        voice="dependency-aware and easy to scan",
        unique_strength="make blockers, dependencies, and waiting states visible",
        reaction_names=("link", "hourglass_flowing_sand", "eyes", "warning"),
        backstory=(
            "Dana is a generalist engineer who keeps handoffs reliable by surfacing "
            "dependencies and waiting states early."
        ),
        avatar_prompt=(
            "Square Slack avatar, stylized cartoon portrait of Dana Hayes, "
            "clear-eyed expression, linked-node motif, navy and cyan accents, simple "
            "graphic background, no text, no logo."
        ),
    ),
]


def _build_avatar_identity_bank() -> tuple[AgentIdentity, ...]:
    identities: list[AgentIdentity] = []
    used_full_names: set[str] = set()
    for profile in ROLE_PROFILES:
        first, last = profile.full_name.split(maxsplit=1)
        identities.append(
            AgentIdentity(
                avatar_index=len(identities) + 1,
                first_name=first,
                last_name=last,
                handle_base=profile.handle,
            )
        )
        used_full_names.add(profile.full_name)

    first_names = GENERATED_FIRST_NAMES[:100]
    last_names = GENERATED_LAST_NAMES[:100]
    pair_index = 0
    while len(identities) < DEFAULT_AVATAR_BANK_SIZE:
        first = first_names[pair_index % len(first_names)]
        block = pair_index // len(first_names)
        last = last_names[(pair_index * 37 + block * 17) % len(last_names)]
        full_name = f"{first} {last}"
        pair_index += 1
        if full_name in used_full_names:
            continue
        identities.append(
            AgentIdentity(
                avatar_index=len(identities) + 1,
                first_name=first,
                last_name=last,
                handle_base=first.lower(),
            )
        )
        used_full_names.add(full_name)
    return tuple(identities)


AVATAR_IDENTITY_BANK = _build_avatar_identity_bank()


def avatar_identity_for_order(sort_order: int) -> AgentIdentity:
    return AVATAR_IDENTITY_BANK[sort_order % DEFAULT_AVATAR_BANK_SIZE]


def avatar_identity_for_index(avatar_index: int) -> AgentIdentity:
    if avatar_index < 1 or avatar_index > DEFAULT_AVATAR_BANK_SIZE:
        raise ValueError(f"avatar index must be between 1 and {DEFAULT_AVATAR_BANK_SIZE}")
    return AVATAR_IDENTITY_BANK[avatar_index - 1]


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
    avatar_index: int | None = None,
) -> TeamAgent:
    provider = (
        provider_preference
        if isinstance(provider_preference, Provider)
        else Provider(provider_preference)
    )
    used_handles = set(existing_handles or set())
    digest = _digest(seed, sort_order)
    profile = ROLE_PROFILES[sort_order % len(ROLE_PROFILES)]
    identity = (
        avatar_identity_for_index(avatar_index)
        if avatar_index is not None
        else avatar_identity_for_order(sort_order)
    )
    handle = _unique_handle(identity.handle_base, identity.last_name, used_handles)
    color = _pick(COLORS, digest, 12)
    icon_emoji = _icon_emoji_for(profile)
    suffix = digest.hex()[:8]
    avatar_slug = str(identity.avatar_index)
    initials = f"{identity.first_name[0]}{identity.last_name[0]}"
    return TeamAgent(
        agent_id=f"agent_{suffix}",
        handle=handle,
        full_name=identity.full_name,
        initials=initials,
        color_hex=color,
        avatar_slug=avatar_slug,
        icon_emoji=icon_emoji,
        role=profile.role,
        personality=profile.personality,
        voice=profile.voice,
        unique_strength=profile.unique_strength,
        reaction_names=profile.reaction_names,
        sort_order=sort_order,
        provider_preference=provider,
        hired_at=utc_now(),
        metadata={
            "seed": seed,
            "backstory": profile.backstory,
            "avatar_prompt": _avatar_prompt(identity.full_name, profile),
            "avatar_path": f"docs/assets/avatars/{avatar_slug}.png",
            "avatar_index": identity.avatar_index,
            "avatar_bank_size": DEFAULT_AVATAR_BANK_SIZE,
        },
    )


def hire_team_agents(
    existing_agents: list[TeamAgent],
    count: int,
    provider_preference: Provider | str | None = None,
    seed: str = DEFAULT_TEAM_SEED,
    start_sort_order: int | None = None,
    balance_agents: list[TeamAgent] | None = None,
    randomize_identities: bool = False,
    rng: random.Random | random.SystemRandom | None = None,
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
    avatar_indexes = _avatar_indexes_for_hires(
        existing_agents,
        count,
        randomize=randomize_identities,
        rng=rng,
    )
    hired: list[TeamAgent] = []
    for offset in range(count):
        provider = (
            Provider(provider_preference)
            if provider_preference is not None
            else least_represented_provider(represented_agents + hired)
        )
        agent = generate_team_agent(
            next_order + offset,
            used_handles,
            provider,
            seed,
            avatar_index=avatar_indexes[offset],
        )
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
        (("thanks", "thank you", "appreciate"), "thumbsup"),
        (("approved", "approve", "accepted"), "white_check_mark"),
        (("denied", "blocked", "waiting", "wait for", "stuck"), "hourglass_flowing_sand"),
        (("failed", "failure", "error", "risk", "danger", "broken", "bug"), "warning"),
        (("test", "tests", "verified", "coverage"), "test_tube"),
        (("merged", "done", "complete", "passed", "fixed", "works"), "white_check_mark"),
        (("ship", "launch", "deploy"), "rocket"),
        (("review", "second opinion", "look at this"), "eyes"),
        (("pr ", "pull request", "github.com"), "link"),
        (("readme", "docs", "document"), "memo"),
        (("question", "unclear", "?"), "thinking_face"),
        (("plan", "decision", "tradeoff", "approach"), "thinking_face"),
    ]
    for keywords, reaction in keyword_reactions:
        if any(keyword in normalized for keyword in keywords):
            return reaction
    return "eyes"


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
    if request.task_kind == AgentTaskKind.REVIEW:
        if author_agent and author_agent.provider_preference is not None:
            cross_model = [
                agent
                for agent in agents
                if agent.provider_preference is not None
                and agent.provider_preference != author_agent.provider_preference
            ]
            if cross_model:
                candidates = cross_model
        else:
            claude_reviewers = [
                agent for agent in agents if agent.provider_preference == Provider.CLAUDE
            ]
            if claude_reviewers:
                candidates = claude_reviewers
    digest = hashlib.sha256(request.prompt.encode("utf-8")).digest()
    return candidates[digest[0] % len(candidates)]


def _digest(seed: str, sort_order: int) -> bytes:
    return hashlib.sha256(f"{seed}:{sort_order}".encode()).digest()


def _avatar_prompt(full_name: str, profile: RoleProfile) -> str:
    return (
        f"Square Slack avatar, stylized cartoon portrait of {full_name}, "
        f"{profile.role.lower()} energy, expressive friendly face, flat vector-like "
        "illustration, clean geometric shapes, vibrant accent color, simple background, "
        "no text, no logo, not photorealistic."
    )


def _avatar_indexes_for_hires(
    existing_agents: Sequence[TeamAgent],
    count: int,
    *,
    randomize: bool,
    rng: random.Random | random.SystemRandom | None,
) -> list[int | None]:
    if not randomize:
        return [None] * count
    used = {
        int(agent.avatar_slug)
        for agent in existing_agents
        if agent.avatar_slug.isdigit()
        and 1 <= int(agent.avatar_slug) <= DEFAULT_AVATAR_BANK_SIZE
    }
    available = [
        identity.avatar_index
        for identity in AVATAR_IDENTITY_BANK
        if identity.avatar_index not in used
    ]
    if count > len(available):
        raise ValueError(
            f"cannot hire {count} randomized identities; only {len(available)} "
            f"unused avatars remain in the {DEFAULT_AVATAR_BANK_SIZE}-person bank"
        )
    chooser = rng or random.SystemRandom()
    return chooser.sample(available, count)


def _pick(items: list[str], digest: bytes, offset: int) -> str:
    value = int.from_bytes(digest[offset : offset + 4], "big")
    return items[value % len(items)]


def _unique_handle(base_name: str, last: str, used_handles: set[str]) -> str:
    base = re.sub(r"[^a-z0-9_-]", "", base_name.lower())
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
