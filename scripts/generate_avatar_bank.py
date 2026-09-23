"""Generate the bundled agent avatar bank.

Each avatar is a monoline portrait: ink outlines over flat fills on a soft
background blob. Traits (skin, hair, clothes, glasses, ...) are derived from
the avatar index and name, so regenerating is deterministic.

Writes ``<out>/<n>.png`` (256px), ``<out>/36/<n>.png`` (card icons, drawn with
heavier lines so they stay crisp small), and ``<out>/manifest.json``. SVGs are
rasterized with ``scripts/svg2png.swift`` and palettized with ImageMagick, so
this runs on macOS with Xcode command line tools and ``magick``. Run ``scripts/generate_provider_badges.sh`` afterwards to
refresh the provider-badged copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_harness.team import (  # noqa: E402
    AGENT_CARD_ICON_SIZE,
    AVATAR_IDENTITY_BANK,
    DEFAULT_AVATAR_BANK_SIZE,
    ROLE_PROFILES,
    _avatar_prompt,
)

SIZE = 256
CX, CY, RX, RY = 128.0, 112.0, 40.0, 49.0
INK = "#1F1D22"
WHITE = "#FFFFFF"

SKINS = ["#F6D8C4", "#EEC3A3", "#E2AE88", "#CF9670", "#B27852", "#915E3F", "#6F4630", "#F2CDB0"]
HAIRS = ["#211C1C", "#3A2A22", "#5B3E2B", "#86492D", "#C99A5B", "#8E8E93", "#DDD4C3", "#A9582C"]
HAIR_WEIGHTS = [5, 5, 4, 2, 2, 2, 1, 1]
HAIR_STYLES = [
    "crop",
    "sidepart",
    "buzz",
    "curly",
    "afro",
    "bob",
    "long",
    "bun",
    "bald",
    "ponytail",
]
HAIR_STYLE_WEIGHTS = [4, 4, 2, 3, 2, 3, 4, 2, 1, 2]
TOPS = ["crew", "vneck", "blazer", "turtleneck", "hoodie", "collar"]
TOP_COLORS = [
    "#2F3E5C",
    "#2E6F6A",
    "#C0654A",
    "#D9A441",
    "#3F6B4B",
    "#6B4A7A",
    "#4A4D55",
    "#4F74A8",
    "#A8553A",
    "#7A7F3F",
    "#E07A67",
    "#8C6D5A",
]
# (page, blob) background pairs.
BACKGROUNDS = [
    ("#E3ECE4", "#C4D8C9"),
    ("#F3EADB", "#E3CFB2"),
    ("#E1EAF4", "#C2D3EA"),
    ("#EAE4F3", "#D0C4E8"),
    ("#F6E5E1", "#EAC5BD"),
    ("#DDF0EB", "#BBDFD5"),
    ("#F6EFD3", "#E8D9A2"),
    ("#E5E8EC", "#C8D0D8"),
]


@dataclass(frozen=True)
class Traits:
    skin: str
    hair: str
    hair_style: str
    top: str
    top_color: str
    background: tuple[str, str]
    glasses: str | None
    beard: str | None
    headscarf: bool
    earrings: bool
    mouth: str
    eyes: str
    blush: bool
    brow_tilt: float
    side: int
    blob_phase: float


def traits_for(index: int, full_name: str) -> Traits:
    digest = hashlib.sha256(f"slackgentic-avatar:{index}:{full_name}".encode()).digest()
    rng = random.Random(digest)
    headscarf = rng.random() < 0.06
    hair_style = rng.choices(HAIR_STYLES, HAIR_STYLE_WEIGHTS)[0]
    beard = None
    beardless = headscarf or hair_style in {"long", "bob", "bun", "ponytail"}
    if not beardless and rng.random() < 0.3:
        beard = rng.choice(["full", "full", "stubble", "mustache"])
    top_color = WHITE if rng.random() < 0.12 else rng.choice(TOP_COLORS)
    return Traits(
        skin=rng.choice(SKINS),
        hair=rng.choices(HAIRS, HAIR_WEIGHTS)[0],
        hair_style=hair_style,
        top=rng.choice(TOPS),
        top_color=top_color,
        background=rng.choice(BACKGROUNDS),
        glasses=rng.choice(["round", "rect"]) if rng.random() < 0.22 else None,
        beard=beard,
        headscarf=headscarf,
        earrings=rng.random() < 0.2,
        mouth=rng.choice(["smile", "smile", "grin", "soft"]),
        eyes="happy" if rng.random() < 0.12 else "dot",
        blush=rng.random() < 0.45,
        brow_tilt=rng.uniform(-1.5, 1.5),
        side=rng.choice([-1, 1]),
        blob_phase=rng.uniform(0, 2 * math.pi),
    )


# ---------------------------------------------------------------- colors


def _rgb(color: str) -> tuple[int, int, int]:
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _hex(values) -> str:
    return "#" + "".join(f"{max(0, min(255, round(v))):02X}" for v in values)


def mix(a: str, b: str, t: float) -> str:
    ca, cb = _rgb(a), _rgb(b)
    return _hex(ca[i] + (cb[i] - ca[i]) * t for i in range(3))


def shade(color: str, factor: float) -> str:
    return _hex(v * factor for v in _rgb(color))


# ---------------------------------------------------------------- geometry


def num(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def pts(*points: tuple[float, float]) -> str:
    return " ".join(f"{num(x)} {num(y)}" for x, y in points)


def circle_d(cx: float, cy: float, r: float) -> str:
    return (
        f"M{pts((cx - r, cy))} a{num(r)} {num(r)} 0 1 0 {num(2 * r)} 0 "
        f"a{num(r)} {num(r)} 0 1 0 {num(-2 * r)} 0 Z"
    )


def polygon_d(points: list[tuple[float, float]]) -> str:
    return "M" + " L".join(pts(p) for p in points) + " Z"


def head_d() -> str:
    points = []
    for i in range(72):
        t = i / 72 * 2 * math.pi
        narrowing = 1 - 0.13 * max(0.0, math.sin(t)) ** 1.5
        points.append((CX + RX * math.cos(t) * narrowing, CY + RY * math.sin(t)))
    return polygon_d(points)


def blob_d(phase: float) -> str:
    points = []
    for i in range(48):
        a = i / 48 * 2 * math.pi
        r = 96 + 6 * math.sin(3 * a + phase) + 4 * math.sin(5 * a + 2 * phase)
        points.append((128 + r * math.cos(a), 130 + r * math.sin(a)))
    return polygon_d(points)


def cap_d(vol: float = 0.0, fringe_y: float | None = None, part: float = 0.0) -> str:
    fy = CY - RY * 0.58 if fringe_y is None else fringe_y
    r, h = RX + 4 + vol, RY + 6 + vol
    return (
        f"M{pts((CX - r, CY - 3))} A{num(r)} {num(h)} 0 0 1 {pts((CX + r, CY - 3))} "
        f"L{pts((CX + RX - 3, CY - 1))} "
        f"C{pts((CX + RX - 6, CY - 30), (CX + RX * 0.5 + part, fy), (CX + part, fy))} "
        f"C{pts((CX - RX * 0.5 + part, fy), (CX - RX + 6, CY - 30), (CX - RX + 3, CY - 1))} Z"
    )


def bangs_d(vol: float = 3.0) -> str:
    edge = CY - 22
    r, h = RX + 4 + vol, RY + 6 + vol
    return (
        f"M{pts((CX - r, CY - 3))} A{num(r)} {num(h)} 0 0 1 {pts((CX + r, CY - 3))} "
        f"L{pts((CX + RX - 4, edge + 6))} Q{pts((CX, edge + 2), (CX - RX + 4, edge + 6))} Z"
    )


def side_panel_d(side: int, bottom: float) -> str:
    x_out, x_in = CX + side * (RX + 9), CX + side * (RX - 7)
    return (
        f"M{pts((x_out, CY - 24))} L{pts((x_in, CY - 26))} "
        f"C{pts((x_in + side, CY + 10), (x_in - side * 2, bottom - 20), (x_in + side * 2, bottom))} "
        f"L{pts((x_out + side * 3, bottom + 4))} Z"
    )


def swoop_d(side: int) -> str:
    return (
        f"M{pts((CX - side * 18, CY - RY - 5))} "
        f"C{pts((CX + side * 12, CY - RY - 6), (CX + side * 36, CY - 40), (CX + side * (RX + 3), CY - 14))} "
        f"L{pts((CX + side * (RX - 6), CY - 20))} "
        f"C{pts((CX + side * 22, CY - 30), (CX + side * 2, CY - 32), (CX - side * 16, CY - 26))} Z"
    )


def long_back_d(bottom: float = 214) -> str:
    r = RX + 15
    return (
        f"M{pts((CX - r, CY - 6))} A{num(r)} {num(RY + 12)} 0 0 1 {pts((CX + r, CY - 6))} "
        f"C{pts((CX + r + 4, CY + 40), (CX + r + 6, bottom - 30), (CX + r - 2, bottom))} "
        f"L{pts((CX - r + 2, bottom))} "
        f"C{pts((CX - r - 6, bottom - 30), (CX - r - 4, CY + 40), (CX - r, CY - 6))} Z"
    )


def bob_back_d() -> str:
    r, b = RX + 13, CY + 44
    return (
        f"M{pts((CX - r, CY - 6))} A{num(r)} {num(RY + 11)} 0 0 1 {pts((CX + r, CY - 6))} "
        f"L{pts((CX + r + 2, b - 10))} Q{pts((CX + r + 2, b), (CX + r - 10, b))} "
        f"L{pts((CX - r + 10, b))} Q{pts((CX - r - 2, b), (CX - r - 2, b - 10))} Z"
    )


def hair_layers(t: Traits) -> tuple[list[str], list[str]]:
    """Back and front hair shapes. Each list is outlined as one merged shape."""
    s = t.side
    back: list[str] = []
    front: list[str] = []
    style = t.hair_style
    if style == "crop":
        front.append(cap_d(0))
    elif style == "sidepart":
        front += [cap_d(2, fringe_y=CY - 30, part=s * 10), swoop_d(s)]
    elif style == "buzz":
        front.append(cap_d(-2.5, fringe_y=CY - 30))
    elif style == "curly":
        back.append(cap_d(5))
        for i in range(12):
            a = math.pi * (0.95 + i / 11 * 1.1)
            back.append(circle_d(CX + (RX + 7) * math.cos(a), CY - 6 + (RY + 6) * math.sin(a), 12))
        front.append(cap_d(4, fringe_y=CY - 26))
        for i in range(6):
            front.append(circle_d(CX - 25 + i * 10, CY - 26 - (i % 2) * 2, 7.5))
    elif style == "afro":
        back.append(circle_d(CX, CY - 14, 62))
        for i in range(16):
            a = math.pi * (0.85 + i / 15 * 1.3)
            back.append(circle_d(CX + 60 * math.cos(a), CY - 14 + 58 * math.sin(a), 14))
        front.append(cap_d(6, fringe_y=CY - 28))
    elif style == "bob":
        back.append(bob_back_d())
        front += [bangs_d(3), side_panel_d(-1, CY + 40), side_panel_d(1, CY + 40)]
    elif style == "long":
        back.append(long_back_d())
        front += [cap_d(3, fringe_y=CY - 32), side_panel_d(-1, CY + 58), side_panel_d(1, CY + 58)]
    elif style == "bun":
        back.append(circle_d(CX + s * 6, CY - RY - 14, 18))
        front.append(cap_d(1, fringe_y=CY - 30))
    elif style == "ponytail":
        back.append(
            f"M{pts((CX + s * 30, CY - 36))} "
            f"C{pts((CX + s * 76, CY - 30), (CX + s * 70, CY + 40), (CX + s * 58, CY + 70))} "
            f"C{pts((CX + s * 52, CY + 30), (CX + s * 50, CY - 4), (CX + s * 26, CY - 18))} Z"
        )
        front.append(cap_d(1, fringe_y=CY - 30, part=-s * 8))
    elif style == "bald":
        for side in (-1, 1):
            front.append(
                f"M{pts((CX + side * (RX + 3), CY - 24))} "
                f"Q{pts((CX + side * (RX + 5), CY - 6), (CX + side * (RX - 2), CY + 2))} "
                f"L{pts((CX + side * (RX - 6), CY - 2))} "
                f"Q{pts((CX + side * (RX - 3), CY - 14), (CX + side * (RX - 4), CY - 28))} Z"
            )
    return back, front


def neck_d() -> str:
    return polygon_d([(CX - 14, CY + 26), (CX + 14, CY + 26), (CX + 17, 214), (CX - 17, 214)])


def body_d(top: str) -> str:
    left, right = (CX - 21, 191), (CX + 21, 191)
    if top == "vneck":
        neckline = f"L{pts((CX, 222), right)}"
    elif top in {"turtleneck", "blazer", "collar"}:
        neckline = f"L{pts(right)}"
    else:
        neckline = f"Q{pts((CX, 211), right)}"
    return (
        f"M8 256 C{pts((10, 224), (30, 202), (74, 195))} L{pts(left)} {neckline} "
        f"L182 195 C{pts((226, 202), (246, 224), (248, 256))} Z"
    )


def neckline_d(top: str) -> str:
    if top == "vneck":
        return f"M{pts((CX - 21, 191))} L{pts((CX, 222), (CX + 21, 191))}"
    return f"M{pts((CX - 21, 191))} Q{pts((CX, 211), (CX + 21, 191))}"


def beard_d() -> str:
    return (
        f"M{pts((CX - RX + 1, CY + 2))} "
        f"C{pts((CX - RX + 2, CY + 44), (CX - 22, CY + RY + 12), (CX, CY + RY + 12))} "
        f"C{pts((CX + 22, CY + RY + 12), (CX + RX - 2, CY + 44), (CX + RX - 1, CY + 2))} "
        f"L{pts((CX + RX - 7, CY + 5))} "
        f"C{pts((CX + RX - 9, CY + 26), (CX + 15, CY + 27), (CX, CY + 26))} "
        f"C{pts((CX - 15, CY + 27), (CX - RX + 9, CY + 26), (CX - RX + 7, CY + 5))} Z"
    )


def mustache_d() -> str:
    y = CY + 30
    return (
        f"M{pts((CX - 13, y + 2))} Q{pts((CX, y - 8), (CX + 13, y + 2))} "
        f"Q{pts((CX, y - 2), (CX - 13, y + 2))} Z"
    )


def headscarf_back_d() -> str:
    r = RX + 13
    return (
        f"M{pts((CX - r, CY))} A{num(r)} {num(RY + 13)} 0 0 1 {pts((CX + r, CY))} "
        f"C{pts((CX + r + 4, CY + 48), (CX + 66, 200), (CX + 86, 230))} L{pts((CX - 86, 230))} "
        f"C{pts((CX - 66, 200), (CX - r - 4, CY + 48), (CX - r, CY))} Z"
    )


def headscarf_front_d() -> str:
    r, h, fy = RX + 8, RY + 9, CY - RY * 0.5
    return (
        f"M{pts((CX - r, CY + 30))} L{pts((CX - r, CY - 3))} "
        f"A{num(r)} {num(h)} 0 0 1 {pts((CX + r, CY - 3))} "
        f"L{pts((CX + r, CY + 30), (CX + RX - 4, CY + 30))} "
        f"C{pts((CX + RX - 2, CY - 10), (CX + RX * 0.6, fy), (CX, fy))} "
        f"C{pts((CX - RX * 0.6, fy), (CX - RX + 2, CY - 10), (CX - RX + 4, CY + 30))} Z"
    )


# ---------------------------------------------------------------- drawing


class Svg:
    def __init__(self, line: float) -> None:
        self.line = line
        self.parts: list[str] = []

    def path(
        self,
        d: str,
        fill: str = "none",
        *,
        stroke: str | None = None,
        width: float | None = None,
        opacity: float | None = None,
    ) -> None:
        attrs = f'd="{d}" fill="{fill}"'
        if stroke:
            attrs += (
                f' stroke="{stroke}" stroke-width="{num(width or self.line)}"'
                ' stroke-linecap="round" stroke-linejoin="round"'
            )
        if opacity is not None:
            attrs += f' opacity="{opacity}"'
        self.parts.append(f"<path {attrs}/>")

    def outlined(self, d: str, fill: str) -> None:
        self.path(d, fill, stroke=INK)

    def merged(self, shapes: list[str], fill: str) -> None:
        """Fill overlapping shapes with a single outline around their union."""
        for d in shapes:
            self.path(d, INK, stroke=INK, width=self.line * 2)
        for d in shapes:
            self.path(d, fill)

    def ellipse(self, cx: float, cy: float, rx: float, ry: float, fill: str, *, outline=False):
        stroke = f' stroke="{INK}" stroke-width="{num(self.line)}"' if outline else ""
        self.parts.append(
            f'<ellipse cx="{num(cx)}" cy="{num(cy)}" rx="{num(rx)}" ry="{num(ry)}" '
            f'fill="{fill}"{stroke}/>'
        )

    def render(self, background: str) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
            f'viewBox="0 0 {SIZE} {SIZE}"><rect width="{SIZE}" height="{SIZE}" '
            f'fill="{background}"/>' + "".join(self.parts) + "</svg>"
        )


def draw_clothes(svg: Svg, t: Traits, skin: str) -> None:
    top = t.top
    fill = t.top_color
    trim = INK if fill == WHITE else shade(fill, 0.7)
    svg.outlined(body_d(top), fill)
    if top in {"crew", "vneck"}:
        svg.path(neckline_d(top), stroke=INK)
    elif top == "hoodie":
        svg.path(neckline_d("crew"), stroke=INK, width=13)
        svg.path(neckline_d("crew"), stroke=mix(fill, WHITE, 0.2), width=13 - 2 * svg.line)
        for side in (-1, 1):
            svg.path(f"M{pts((CX + side * 10, 210))} L{pts((CX + side * 12, 238))}", stroke=INK)
            svg.ellipse(CX + side * 12, 240, 2.6, 2.6, INK)
    elif top == "turtleneck":
        roll = mix(fill, WHITE, 0.12)
        svg.outlined(
            f"M{pts((CX - 23, 197))} L{pts((CX - 21, 174))} Q{pts((CX, 168), (CX + 21, 174))} "
            f"L{pts((CX + 23, 197))} Q{pts((CX, 203), (CX - 23, 197))} Z",
            roll,
        )
        for y in (182, 190):
            svg.path(
                f"M{pts((CX - 15, y))} Q{pts((CX, y + 3), (CX + 15, y))}", stroke=trim, width=2
            )
    else:
        shirt = WHITE if fill != WHITE else "#DCE6F2"
        if top == "blazer":
            svg.outlined(
                polygon_d([(CX - 22, 191), (CX + 22, 191), (CX + 5, 256), (CX - 5, 256)]), shirt
            )
            for side in (-1, 1):
                svg.path(
                    f"M{pts((CX + side * 22, 191))} "
                    f"L{pts((CX + side * 30, 214), (CX + side * 22, 220), (CX + side * 5, 256))}",
                    stroke=INK,
                )
        svg.path(polygon_d([(CX - 16, 186), (CX + 16, 186), (CX, 211)]), skin)
        for side in (-1, 1):
            svg.outlined(
                polygon_d(
                    [
                        (CX + side * 19, 186),
                        (CX + side * 1, 212),
                        (CX + side * 17, 210),
                        (CX + side * 25, 194),
                    ]
                ),
                shirt,
            )


def draw_face(svg: Svg, t: Traits, detail: float) -> None:
    ey, gap = CY + 6, 16
    light_hair = t.hair in {"#DDD4C3", "#C99A5B", "#8E8E93"}
    brow = "#3A2A22" if t.headscarf else shade(t.hair, 0.6 if light_hair else 0.8)
    for side in (-1, 1):
        x = CX + side * gap
        tilt = t.brow_tilt * side
        svg.path(
            f"M{pts((x - 7, ey - 11 + tilt))} Q{pts((x, ey - 15), (x + 7, ey - 11 - tilt))}",
            stroke=brow,
            width=3 * detail,
        )
        if t.eyes == "happy":
            svg.path(
                f"M{pts((x - 4.5, ey + 1))} Q{pts((x, ey - 5), (x + 4.5, ey + 1))}",
                stroke=INK,
                width=2.7 * detail,
            )
        else:
            svg.ellipse(x, ey, 3.4 * detail, 4.3 * detail, INK)
    if t.blush:
        for side in (-1, 1):
            svg.parts.append(
                f'<ellipse cx="{num(CX + side * 23)}" cy="{num(CY + 21)}" rx="7" ry="4.2" '
                'fill="#F08A7A" opacity="0.35"/>'
            )
    svg.path(
        f"M{pts((CX + 1, CY + 12))} Q{pts((CX + 5, CY + 21), (CX - 1, CY + 23))}",
        stroke=shade(t.skin, 0.72),
        width=2.4 * detail,
    )
    y = CY + 32
    if t.mouth == "grin":
        svg.path(
            f"M{pts((CX - 10, y))} Q{pts((CX, y + 14), (CX + 10, y))} Z",
            "#5B2A2E",
            stroke=INK,
            width=2 * detail,
        )
        svg.path(
            f"M{pts((CX - 8, y + 1))} L{pts((CX + 8, y + 1))} Q{pts((CX, y + 5), (CX - 8, y + 1))} Z",
            WHITE,
        )
    else:
        width, depth = (6, 3) if t.mouth == "soft" else (9, 6)
        svg.path(
            f"M{pts((CX - width, y + 1))} Q{pts((CX, y + 1 + depth), (CX + width, y + 1))}",
            stroke=INK,
            width=2.8 * detail,
        )


def draw_glasses(svg: Svg, t: Traits, detail: float) -> None:
    if not t.glasses:
        return
    ey, width = CY + 6, 2.8 * detail
    for side in (-1, 1):
        x = CX + side * 16
        if t.glasses == "round":
            shape = circle_d(x, ey, 10)
        else:
            shape = (
                f"M{pts((x - 6, ey - 8))} L{pts((x + 6, ey - 8))} Q{pts((x + 11, ey - 8), (x + 11, ey - 3))} "
                f"L{pts((x + 11, ey + 3))} Q{pts((x + 11, ey + 8), (x + 6, ey + 8))} "
                f"L{pts((x - 6, ey + 8))} Q{pts((x - 11, ey + 8), (x - 11, ey + 3))} "
                f"L{pts((x - 11, ey - 3))} Q{pts((x - 11, ey - 8), (x - 6, ey - 8))} Z"
            )
        svg.parts.append(
            f'<path d="{shape}" fill="{WHITE}" fill-opacity="0.18" stroke="{INK}" '
            f'stroke-width="{num(width)}"/>'
        )
        svg.path(
            f"M{pts((CX + side * 26, ey - 2))} L{pts((CX + side * (RX - 1), ey - 4))}",
            stroke=INK,
            width=width,
        )
    svg.path(
        f"M{pts((CX - 6, ey - 1))} Q{pts((CX, ey - 4), (CX + 6, ey - 1))}", stroke=INK, width=width
    )


def render_avatar_svg(t: Traits, *, line: float = 3.2, detail: float = 1.0) -> str:
    svg = Svg(line)
    page, blob = t.background
    svg.path(blob_d(t.blob_phase), blob)
    skin = mix(t.skin, WHITE, 0.2)
    back, front = hair_layers(t)
    if back and not t.headscarf:
        svg.merged(back, t.hair)
    svg.outlined(neck_d(), skin)
    svg.path(
        polygon_d([(CX - 14, CY + 40), (CX + 14, CY + 40), (CX + 14, CY + 50), (CX - 14, CY + 54)]),
        shade(skin, 0.9),
    )
    draw_clothes(svg, t, skin)
    if t.headscarf:
        svg.outlined(
            headscarf_back_d(), mix(t.top_color if t.top_color != WHITE else "#6B4A7A", WHITE, 0.25)
        )
    else:
        for side in (-1, 1):
            svg.ellipse(CX + side * (RX - 1), CY + 6, 7, 11, skin, outline=True)
    svg.outlined(head_d(), skin)
    if t.beard == "full":
        svg.merged([beard_d(), mustache_d()], t.hair)
    elif t.beard == "stubble":
        svg.path(beard_d(), t.hair, opacity=0.28)
    elif t.beard == "mustache":
        svg.outlined(mustache_d(), t.hair)
    draw_face(svg, t, detail)
    if t.headscarf:
        scarf = t.top_color if t.top_color != WHITE else "#6B4A7A"
        svg.outlined(headscarf_front_d(), mix(scarf, WHITE, 0.4))
    elif front:
        svg.merged(front, t.hair)
        if t.earrings and t.hair_style not in {"bob", "long"}:
            for side in (-1, 1):
                svg.ellipse(CX + side * (RX - 0.5), CY + 18, 2.8, 2.8, "#D9B45A", outline=True)
    draw_glasses(svg, t, detail)
    return svg.render(page)


# ---------------------------------------------------------------- output


def rasterize(jobs: list[tuple[Path, Path]], size: int, work: Path) -> None:
    renderer = work / "svg2png"
    if not renderer.exists():
        subprocess.run(
            ["swiftc", "-O", str(REPO_ROOT / "scripts" / "svg2png.swift"), "-o", str(renderer)],
            check=True,
        )
    for start in range(0, len(jobs), 200):
        args = [str(renderer), str(size)]
        for svg_path, png_path in jobs[start : start + 200]:
            args += [str(svg_path), str(png_path)]
        subprocess.run(args, check=True)
    # Flat art survives a 128-color palette losslessly to the eye, at a third of the size.
    subprocess.run(
        [
            "magick",
            "mogrify",
            "-strip",
            "-alpha",
            "off",
            "+dither",
            "-colors",
            "128",
            "-define",
            "png:color-type=3",
            *(str(png) for _, png in jobs),
        ],
        check=True,
    )


def generate(out_dir: Path, count: int) -> None:
    card_dir = out_dir / str(AGENT_CARD_ICON_SIZE)
    card_dir.mkdir(parents=True, exist_ok=True)
    for path in [*out_dir.glob("*.png"), *card_dir.glob("*.png")]:
        path.unlink()
    manifest = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        full_jobs: list[tuple[Path, Path]] = []
        card_jobs: list[tuple[Path, Path]] = []
        for identity in AVATAR_IDENTITY_BANK[:count]:
            role = ROLE_PROFILES[(identity.avatar_index - 1) % len(ROLE_PROFILES)]
            traits = traits_for(identity.avatar_index, identity.full_name)
            name = f"{identity.avatar_index}.png"
            full_svg = work / f"{identity.avatar_index}.svg"
            card_svg = work / f"{identity.avatar_index}-card.svg"
            full_svg.write_text(render_avatar_svg(traits))
            # Card icons are tiny: heavier lines and features keep them legible.
            card_svg.write_text(render_avatar_svg(traits, line=7, detail=1.6))
            full_jobs.append((full_svg, out_dir / name))
            card_jobs.append((card_svg, card_dir / name))
            manifest.append(
                {
                    "avatar": name,
                    "avatar_index": identity.avatar_index,
                    "full_name": identity.full_name,
                    "handle_base": identity.handle_base,
                    "role": role.role,
                    "prompt": _avatar_prompt(identity.full_name, role),
                }
            )
        rasterize(full_jobs, SIZE, work)
        rasterize(card_jobs, AGENT_CARD_ICON_SIZE, work)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "docs" / "assets" / "avatars")
    parser.add_argument("--count", type=int, default=DEFAULT_AVATAR_BANK_SIZE)
    args = parser.parse_args()
    if args.count < 1 or args.count > DEFAULT_AVATAR_BANK_SIZE:
        raise ValueError(f"count must be between 1 and {DEFAULT_AVATAR_BANK_SIZE}")
    generate(args.out_dir, args.count)
    print(f"generated {args.count} avatars in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
