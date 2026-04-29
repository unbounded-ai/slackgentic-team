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

DEFAULT_TEAM_SIZE = 2
DEFAULT_CODEX_TEAM_SIZE = 1
DEFAULT_CLAUDE_TEAM_SIZE = 1
DEFAULT_TEAM_SEED = "slackgentic-team"
DEFAULT_AVATAR_BANK_SIZE = 500
MAX_TEAM_AGENTS = 500
AGENT_LIMIT_MESSAGE = "You definitely do not need that many agents."
HANDLE_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
AGENT_CONTEXT_PLACEHOLDER = "outside-work context"

COLORS = [
    "#2457a6",
    "#1c7c54",
    "#8f4d12",
    "#7a3b8f",
    "#b53d4d",
    "#0f766e",
    "#71501d",
    "#315f72",
    "#7c5caa",
    "#a4472f",
]

OUTSIDE_WORK_INTERESTS = [
    "bakes sourdough with handwritten fermentation notes",
    "plays pickup futsal before sunrise",
    "collects vintage transit maps",
    "restores mechanical keyboards",
    "hosts tiny dinner parties around regional noodle dishes",
    "keeps a notebook of obscure jazz organ records",
    "does long-distance open-water swimming",
    "paints small gouache studies of storefront signs",
    "builds modular-synth patches on weekends",
    "learns card magic from old library books",
    "repairs espresso machines for friends",
    "runs a neighborhood film-club spreadsheet",
    "throws pottery bowls with uneven glazes",
    "studies abandoned railway stations",
    "plays table tennis with a defensive chopping style",
    "makes hot sauce from balcony-grown peppers",
    "collects matchbooks from closed restaurants",
    "practices bebop lines on a tenor sax",
    "designs tiny crossword puzzles",
    "keeps scorecards from minor-league baseball games",
    "learns folk dances from archival videos",
    "roasts coffee in small batches",
    "maps city staircases during weekend walks",
    "builds elaborate playlists for road trips",
    "restores mid-century desk lamps",
    "cooks one regional dumpling recipe at a time",
    "plays curling in a late-night league",
    "binds notebooks by hand",
    "takes black-and-white photos of old signage",
    "studies chess endgames on paper boards",
    "makes zines about local history",
    "practices calligraphy with fountain pens",
    "tracks every ramen shop visited in a ledger",
    "plays drums in a garage surf-rock band",
    "forages for edible plants with a field guide",
    "collects library stamps in used books",
    "builds miniature room dioramas",
    "keeps a personal museum of obsolete connectors",
    "runs a monthly soup swap",
    "learns lockpicking as a puzzle hobby",
    "plays ultimate frisbee on windy fields",
    "records ambient sounds from train platforms",
    "makes cyanotype prints",
    "studies map projections for fun",
    "cooks elaborate breakfasts on cast iron",
    "plays bocce in a neighborhood league",
    "repairs fountain pens",
    "collects regional potato-chip flavors",
    "learns harmonica standards",
    "keeps a spreadsheet of hiking trail sandwiches",
]

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
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("eyes", "white_check_mark", "memo", "thinking_face"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Jordan Reed",
        handle="jordan",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("test_tube", "white_check_mark", "mag", "memo"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Morgan Patel",
        handle="morgan",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("world_map", "eyes", "memo", "thinking_face"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Riley Shaw",
        handle="riley",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("balance_scale", "thinking_face", "eyes", "white_check_mark"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Casey Kim",
        handle="casey",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("clipboard", "white_check_mark", "memo", "eyes"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Harper Diaz",
        handle="harper",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("speech_balloon", "bulb", "memo", "sparkles"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Quinn Bennett",
        handle="quinn",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("question", "thinking_face", "eyes", "memo"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Rowan Stone",
        handle="rowan",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("pushpin", "memo", "white_check_mark", "eyes"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Cameron Lin",
        handle="cameron",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("rocket", "construction", "white_check_mark", "hourglass_flowing_sand"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Sage Carter",
        handle="sage",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("gauge", "test_tube", "white_check_mark", "thinking_face"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Parker Wells",
        handle="parker",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("mega", "rocket", "eyes", "white_check_mark"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
    ),
    RoleProfile(
        full_name="Dana Hayes",
        handle="dana",
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=("link", "hourglass_flowing_sand", "eyes", "warning"),
        backstory=AGENT_CONTEXT_PLACEHOLDER,
        avatar_prompt=AGENT_CONTEXT_PLACEHOLDER,
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
    outside_interests: Sequence[str] | None = None,
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
    interests = tuple(outside_interests or _outside_interests_for_digest(digest))
    personal_context = _personal_context_from_interests(interests)
    return TeamAgent(
        agent_id=f"agent_{suffix}",
        handle=handle,
        full_name=identity.full_name,
        initials=initials,
        color_hex=color,
        avatar_slug=avatar_slug,
        icon_emoji=icon_emoji,
        role=AGENT_CONTEXT_PLACEHOLDER,
        personality=AGENT_CONTEXT_PLACEHOLDER,
        voice=AGENT_CONTEXT_PLACEHOLDER,
        unique_strength=AGENT_CONTEXT_PLACEHOLDER,
        reaction_names=profile.reaction_names,
        sort_order=sort_order,
        provider_preference=provider,
        hired_at=utc_now(),
        metadata={
            "seed": seed,
            "backstory": personal_context,
            "outside_interests": list(interests),
            "personal_context": personal_context,
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
    avatar_agents: list[TeamAgent] | None = None,
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
    represented_avatars = avatar_agents if avatar_agents is not None else existing_agents
    chooser = rng or (random.SystemRandom() if randomize_identities else None)
    avatar_indexes = _avatar_indexes_for_hires(
        represented_avatars,
        count,
        randomize=randomize_identities,
        rng=chooser,
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
            outside_interests=(
                _random_outside_interests(chooser) if randomize_identities and chooser else None
            ),
        )
        used_handles.add(agent.handle)
        hired.append(agent)
    return hired


def build_initial_team(
    size: int = DEFAULT_TEAM_SIZE, seed: str = DEFAULT_TEAM_SEED
) -> list[TeamAgent]:
    if size < 1:
        raise ValueError("team size must be at least 1")
    if size > MAX_TEAM_AGENTS:
        raise ValueError(AGENT_LIMIT_MESSAGE)
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
    if codex_count + claude_count > MAX_TEAM_AGENTS:
        raise ValueError(AGENT_LIMIT_MESSAGE)
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


def agent_identity_label(agent: TeamAgent) -> str:
    provider = agent.provider_preference.value if agent.provider_preference else "unmapped"
    return f"{agent.full_name} [{provider}] @{agent.handle}"


def agent_personal_context(agent: TeamAgent) -> str:
    configured = agent.metadata.get("personal_context")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    interests_value = agent.metadata.get("outside_interests")
    if isinstance(interests_value, list):
        interests = tuple(str(item).strip() for item in interests_value if str(item).strip())
        if len(interests) >= 3:
            return _personal_context_from_interests(interests[:3])
    digest = hashlib.sha256(f"{agent.agent_id}:{agent.sort_order}".encode()).digest()
    return _personal_context_from_interests(_outside_interests_for_digest(digest))


def runtime_personality_prompt(agent: TeamAgent) -> str:
    return (
        f"You are {agent.full_name}, known in Slack as @{agent.handle}. "
        "Personal context is explicitly included so it can lightly carry into "
        f"your personality when you chat: {agent_personal_context(agent)}. "
        "Do not roleplay beyond the engineering task."
    )


def format_agent_introduction(agent: TeamAgent) -> str:
    return f"Hi, I'm @{agent.handle}. Outside work: {agent_personal_context(agent)}."


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
        f"Welcome, @{new_agent.handle}.",
        f"Glad you're here, @{new_agent.handle}.",
        f"Welcome aboard, @{new_agent.handle}.",
        f"Good to meet you, @{new_agent.handle}.",
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
        "expressive friendly face, flat vector-like illustration, clean geometric "
        "shapes, vibrant accent color, simple background, "
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
        if agent.avatar_slug.isdigit() and 1 <= int(agent.avatar_slug) <= DEFAULT_AVATAR_BANK_SIZE
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


def _outside_interests_for_digest(digest: bytes) -> tuple[str, str, str]:
    ranked = sorted(
        OUTSIDE_WORK_INTERESTS,
        key=lambda item: hashlib.sha256(digest + item.encode("utf-8")).digest(),
    )
    return ranked[0], ranked[1], ranked[2]


def _random_outside_interests(
    rng: random.Random | random.SystemRandom,
) -> tuple[str, str, str]:
    first, second, third = rng.sample(OUTSIDE_WORK_INTERESTS, 3)
    return first, second, third


def _personal_context_from_interests(interests: Sequence[str]) -> str:
    clean = [interest.strip() for interest in interests if interest.strip()]
    if len(clean) >= 3:
        return f"{clean[0]}; {clean[1]}; and {clean[2]}"
    return "; ".join(clean) or "keeps a few low-key hobbies outside work"


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
