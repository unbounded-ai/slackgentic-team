from __future__ import annotations

import hashlib

from agent_harness.models import Persona, Provider

FIRST_NAMES = [
    "Avery",
    "Blair",
    "Cameron",
    "Casey",
    "Dana",
    "Drew",
    "Elliot",
    "Emery",
    "Finley",
    "Harper",
    "Jamie",
    "Jordan",
    "Kai",
    "Kendall",
    "Lane",
    "Logan",
    "Morgan",
    "Parker",
    "Quinn",
    "Reese",
    "Riley",
    "Rowan",
    "Sage",
    "Taylor",
]

LAST_NAMES = [
    "Adler",
    "Bennett",
    "Brooks",
    "Carter",
    "Chen",
    "Diaz",
    "Ellis",
    "Foster",
    "Hayes",
    "Ivers",
    "Kapoor",
    "Kim",
    "Lane",
    "Lin",
    "Mason",
    "Nolan",
    "Patel",
    "Reed",
    "Shaw",
    "Stone",
    "Tan",
    "Voss",
    "Wells",
    "Young",
]

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

EMOJI = [
    ":large_blue_circle:",
    ":large_green_circle:",
    ":large_orange_circle:",
    ":large_purple_circle:",
    ":red_circle:",
    ":teal_circle:",
]


def _digest(provider: Provider, session_id: str) -> bytes:
    key = f"{provider.value}:{session_id}".encode()
    return hashlib.sha256(key).digest()


def _pick(items: list[str], digest: bytes, offset: int) -> str:
    value = int.from_bytes(digest[offset : offset + 4], "big")
    return items[value % len(items)]


def generate_persona(provider: Provider | str, session_id: str) -> Persona:
    provider_value = provider if isinstance(provider, Provider) else Provider(provider)
    digest = _digest(provider_value, session_id)
    first = _pick(FIRST_NAMES, digest, 0)
    last = _pick(LAST_NAMES, digest, 4)
    color = _pick(COLORS, digest, 8)
    emoji = _pick(EMOJI, digest, 12)
    initials = f"{first[0]}{last[0]}"
    suffix = digest.hex()[:6]
    username = f"{first} {last}"
    avatar_slug = f"{provider_value.value}-{first.lower()}-{last.lower()}-{suffix}"
    return Persona(
        provider=provider_value,
        session_id=session_id,
        full_name=f"{first} {last}",
        username=username,
        initials=initials,
        color_hex=color,
        avatar_slug=avatar_slug,
        icon_emoji=emoji,
    )


def avatar_svg(persona: Persona, size: int = 256) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<rect width="{size}" height="{size}" rx="32" fill="{persona.color_hex}"/>'
        f'<text x="50%" y="54%" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="Arial, sans-serif" font-size="{size // 3}" '
        f'font-weight="700" fill="#ffffff">{persona.initials}</text>'
        "</svg>"
    )
