from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_harness.team import (  # noqa: E402
    AVATAR_IDENTITY_BANK,
    DEFAULT_AVATAR_BANK_SIZE,
    ROLE_PROFILES,
    _avatar_prompt,
)

SIZE = 256
SCALE = 2
SKIN_COLORS = [
    (255, 224, 189, 255),
    (246, 205, 166, 255),
    (232, 184, 134, 255),
    (198, 134, 86, 255),
    (141, 85, 54, 255),
    (92, 58, 43, 255),
]
HAIR_COLORS = [
    (43, 33, 27, 255),
    (74, 49, 35, 255),
    (117, 73, 45, 255),
    (189, 128, 66, 255),
    (225, 190, 118, 255),
    (45, 47, 57, 255),
    (116, 117, 125, 255),
]
BG_COLORS = [
    (185, 222, 255, 255),
    (203, 238, 221, 255),
    (255, 227, 184, 255),
    (235, 218, 255, 255),
    (255, 211, 221, 255),
    (205, 235, 242, 255),
    (226, 236, 203, 255),
    (245, 224, 207, 255),
]
SHIRT_COLORS = [
    (36, 87, 166, 255),
    (42, 132, 112, 255),
    (175, 73, 73, 255),
    (96, 78, 166, 255),
    (209, 128, 50, 255),
    (46, 64, 83, 255),
    (176, 82, 144, 255),
    (69, 134, 180, 255),
]
LINE = (36, 35, 42, 255)
WHITE = (255, 255, 255, 255)


class Canvas:
    def __init__(self, size: int = SIZE, scale: int = SCALE):
        self.size = size
        self.scale = scale
        self.width = size * scale
        self.height = size * scale
        self.pixels = [(0, 0, 0, 0)] * (self.width * self.height)

    def fill(self, color: tuple[int, int, int, int]) -> None:
        self.pixels = [color] * (self.width * self.height)

    def ellipse(
        self,
        cx: float,
        cy: float,
        rx: float,
        ry: float,
        color: tuple[int, int, int, int],
    ) -> None:
        s = self.scale
        x0 = max(0, int((cx - rx) * s))
        x1 = min(self.width - 1, int((cx + rx) * s))
        y0 = max(0, int((cy - ry) * s))
        y1 = min(self.height - 1, int((cy + ry) * s))
        rsx = rx * s
        rsy = ry * s
        scx = cx * s
        scy = cy * s
        for y in range(y0, y1 + 1):
            dy = (y + 0.5 - scy) / rsy
            for x in range(x0, x1 + 1):
                dx = (x + 0.5 - scx) / rsx
                if dx * dx + dy * dy <= 1.0:
                    self._over(x, y, color)

    def rect(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int, int],
    ) -> None:
        s = self.scale
        ix0 = max(0, int(x0 * s))
        ix1 = min(self.width, int(x1 * s))
        iy0 = max(0, int(y0 * s))
        iy1 = min(self.height, int(y1 * s))
        for y in range(iy0, iy1):
            for x in range(ix0, ix1):
                self._over(x, y, color)

    def polygon(self, points: list[tuple[float, float]], color: tuple[int, int, int, int]) -> None:
        scaled = [(x * self.scale, y * self.scale) for x, y in points]
        min_y = max(0, int(min(y for _, y in scaled)))
        max_y = min(self.height - 1, int(max(y for _, y in scaled)))
        for y in range(min_y, max_y + 1):
            intersections: list[float] = []
            for index, (x1, y1) in enumerate(scaled):
                x2, y2 = scaled[(index + 1) % len(scaled)]
                if y1 == y2:
                    continue
                if min(y1, y2) <= y + 0.5 < max(y1, y2):
                    intersections.append(x1 + (y + 0.5 - y1) * (x2 - x1) / (y2 - y1))
            intersections.sort()
            for left, right in zip(intersections[0::2], intersections[1::2], strict=False):
                for x in range(max(0, int(left)), min(self.width, int(right) + 1)):
                    self._over(x, y, color)

    def _over(self, x: int, y: int, src: tuple[int, int, int, int]) -> None:
        sr, sg, sb, sa = src
        if sa == 255:
            self.pixels[y * self.width + x] = src
            return
        dr, dg, db, da = self.pixels[y * self.width + x]
        alpha = sa / 255.0
        out_alpha = alpha + da / 255.0 * (1 - alpha)
        if out_alpha == 0:
            self.pixels[y * self.width + x] = (0, 0, 0, 0)
            return
        self.pixels[y * self.width + x] = (
            int((sr * alpha + dr * da / 255.0 * (1 - alpha)) / out_alpha),
            int((sg * alpha + dg * da / 255.0 * (1 - alpha)) / out_alpha),
            int((sb * alpha + db * da / 255.0 * (1 - alpha)) / out_alpha),
            int(out_alpha * 255),
        )

    def downsample(self) -> list[tuple[int, int, int, int]]:
        output: list[tuple[int, int, int, int]] = []
        s = self.scale
        area = s * s
        for y in range(self.size):
            for x in range(self.size):
                total = [0, 0, 0, 0]
                for yy in range(y * s, y * s + s):
                    for xx in range(x * s, x * s + s):
                        pixel = self.pixels[yy * self.width + xx]
                        for i, value in enumerate(pixel):
                            total[i] += value
                output.append(tuple(round(value / area) for value in total))  # type: ignore[arg-type]
        return output


def render_avatar(index: int, full_name: str, role_index: int) -> list[tuple[int, int, int, int]]:
    digest = hashlib.sha256(f"slackgentic-avatar:{index}:{full_name}".encode()).digest()
    canvas = Canvas()
    bg = pick(BG_COLORS, digest, 0)
    skin = pick(SKIN_COLORS, digest, 1)
    hair = pick(HAIR_COLORS, digest, 2)
    shirt = pick(SHIRT_COLORS, digest, 3)
    accent = pick(SHIRT_COLORS + BG_COLORS, digest, 4)
    canvas.fill(bg)

    canvas.ellipse(34 + digest[5] % 18, 43 + digest[6] % 16, 38, 22, with_alpha(accent, 70))
    canvas.ellipse(218 - digest[7] % 24, 58 + digest[8] % 20, 30, 30, with_alpha(WHITE, 74))
    canvas.polygon([(0, 221), (72, 184), (158, 256), (0, 256)], with_alpha(accent, 52))
    canvas.polygon([(256, 210), (183, 180), (95, 256), (256, 256)], with_alpha(WHITE, 58))

    canvas.ellipse(128, 238, 76, 42, shirt)
    canvas.rect(111, 154, 145, 204, skin)
    canvas.ellipse(128, 205, 28, 12, darken(skin, 0.9))

    hair_style = digest[9] % 8
    draw_back_hair(canvas, hair_style, hair)
    canvas.ellipse(83, 118, 13, 20, skin)
    canvas.ellipse(173, 118, 13, 20, skin)
    canvas.ellipse(128, 115, 49, 61, skin)
    draw_front_hair(canvas, hair_style, hair)

    eye_y = 117 + digest[10] % 5
    eye_gap = 19 + digest[11] % 4
    canvas.ellipse(128 - eye_gap, eye_y, 5, 6, LINE)
    canvas.ellipse(128 + eye_gap, eye_y, 5, 6, LINE)
    canvas.ellipse(127 - eye_gap, eye_y - 1, 1.6, 1.6, WHITE)
    canvas.ellipse(127 + eye_gap, eye_y - 1, 1.6, 1.6, WHITE)
    canvas.rect(103, eye_y - 14, 119, eye_y - 11, with_alpha(LINE, 170))
    canvas.rect(137, eye_y - 14, 153, eye_y - 11, with_alpha(LINE, 170))

    if digest[12] % 5 == 0:
        draw_glasses(canvas, eye_y, eye_gap)

    canvas.ellipse(128, 135, 4, 10, darken(skin, 0.82))
    mouth_mode = digest[13] % 4
    if mouth_mode == 0:
        canvas.ellipse(128, 153, 16, 6, (116, 48, 62, 255))
        canvas.rect(111, 146, 145, 153, skin)
    elif mouth_mode == 1:
        canvas.ellipse(128, 154, 13, 4, (116, 48, 62, 255))
    elif mouth_mode == 2:
        canvas.rect(116, 152, 140, 156, (116, 48, 62, 255))
    else:
        canvas.ellipse(128, 154, 12, 7, (116, 48, 62, 255))
        canvas.ellipse(128, 151, 14, 5, skin)

    if digest[14] % 4 == 0:
        for offset in (-15, -8, 10, 17):
            canvas.ellipse(128 + offset, 136 + digest[15] % 7, 1.6, 1.2, with_alpha(LINE, 70))

    if digest[16] % 7 == 0:
        canvas.ellipse(128, 161, 19, 8, with_alpha(hair, 175))

    draw_role_badge(canvas, role_index, accent)
    return canvas.downsample()


def draw_back_hair(canvas: Canvas, style: int, hair: tuple[int, int, int, int]) -> None:
    if style in {0, 1, 5}:
        canvas.ellipse(128, 92, 53, 42, hair)
    if style in {2, 4, 6}:
        canvas.ellipse(128, 104, 61, 58, hair)
        canvas.rect(69, 100, 95, 172, hair)
        canvas.rect(161, 100, 187, 172, hair)
    if style == 3:
        for x in range(82, 178, 17):
            canvas.ellipse(x, 86 + (x % 3) * 4, 18, 18, hair)
        for x in range(75, 185, 18):
            canvas.ellipse(x, 111 + (x % 2) * 5, 17, 20, hair)
    if style == 7:
        canvas.ellipse(128, 70, 24, 24, hair)
        canvas.rect(119, 85, 137, 103, hair)
        canvas.ellipse(128, 100, 52, 33, hair)


def draw_front_hair(canvas: Canvas, style: int, hair: tuple[int, int, int, int]) -> None:
    if style == 0:
        canvas.ellipse(108, 84, 39, 24, hair)
        canvas.polygon([(112, 76), (166, 86), (155, 112), (124, 104)], hair)
    elif style == 1:
        canvas.rect(86, 81, 169, 104, hair)
        canvas.ellipse(128, 84, 44, 22, hair)
    elif style == 2:
        canvas.ellipse(106, 87, 34, 26, hair)
        canvas.ellipse(151, 86, 34, 25, hair)
        canvas.rect(78, 95, 92, 151, hair)
        canvas.rect(165, 95, 179, 151, hair)
    elif style == 3:
        for x in range(93, 165, 16):
            canvas.ellipse(x, 92, 13, 13, hair)
    elif style == 4:
        canvas.polygon([(81, 89), (178, 83), (169, 108), (94, 104)], hair)
        canvas.rect(80, 95, 92, 139, hair)
    elif style == 5:
        canvas.polygon([(83, 89), (159, 78), (176, 95), (122, 107), (91, 104)], hair)
    elif style == 6:
        canvas.ellipse(128, 87, 48, 28, hair)
        canvas.rect(74, 100, 89, 160, hair)
        canvas.rect(167, 100, 182, 160, hair)
    else:
        canvas.ellipse(128, 88, 44, 24, hair)


def draw_glasses(canvas: Canvas, eye_y: int, eye_gap: int) -> None:
    face_fill = (255, 255, 255, 48)
    for cx in (128 - eye_gap, 128 + eye_gap):
        canvas.ellipse(cx, eye_y, 12, 10, LINE)
        canvas.ellipse(cx, eye_y, 9, 7, face_fill)
    canvas.rect(128 - eye_gap + 10, eye_y - 1, 128 + eye_gap - 10, eye_y + 2, LINE)


def draw_role_badge(canvas: Canvas, role_index: int, accent: tuple[int, int, int, int]) -> None:
    x = 190
    y = 190
    canvas.ellipse(x, y, 19, 19, with_alpha(WHITE, 215))
    canvas.ellipse(x, y, 14, 14, accent)
    motif = role_index % 4
    if motif == 0:
        canvas.ellipse(x - 5, y - 3, 3, 3, WHITE)
        canvas.ellipse(x + 5, y + 3, 3, 3, WHITE)
        canvas.rect(x - 5, y - 1, x + 5, y + 1, WHITE)
    elif motif == 1:
        canvas.rect(x - 8, y - 7, x + 8, y - 3, WHITE)
        canvas.rect(x - 8, y - 1, x + 6, y + 3, WHITE)
        canvas.rect(x - 8, y + 5, x + 3, y + 8, WHITE)
    elif motif == 2:
        canvas.ellipse(x, y, 8, 8, WHITE)
        canvas.ellipse(x, y, 4, 4, accent)
    else:
        canvas.polygon([(x, y - 9), (x + 7, y + 7), (x - 8, y + 1)], WHITE)


def pick(
    values: list[tuple[int, int, int, int]], digest: bytes, offset: int
) -> tuple[int, int, int, int]:
    return values[digest[offset] % len(values)]


def with_alpha(color: tuple[int, int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (color[0], color[1], color[2], alpha)


def darken(color: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    return (
        math.floor(color[0] * factor),
        math.floor(color[1] * factor),
        math.floor(color[2] * factor),
        color[3],
    )


def write_png(path: Path, width: int, height: int, pixels: list[tuple[int, int, int, int]]) -> None:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels[y * width + x])
    compressed = zlib.compress(bytes(raw), level=9)
    data = bytearray(b"\x89PNG\r\n\x1a\n")
    data.extend(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
    data.extend(chunk(b"IDAT", compressed))
    data.extend(chunk(b"IEND", b""))
    path.write_bytes(data)


def chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def generate(out_dir: Path, count: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.glob("*.png"):
        path.unlink()
    manifest = []
    for identity in AVATAR_IDENTITY_BANK[:count]:
        role_index = (identity.avatar_index - 1) % len(ROLE_PROFILES)
        role = ROLE_PROFILES[role_index]
        pixels = render_avatar(identity.avatar_index, identity.full_name, role_index)
        write_png(out_dir / f"{identity.avatar_index}.png", SIZE, SIZE, pixels)
        manifest.append(
            {
                "avatar": f"{identity.avatar_index}.png",
                "avatar_index": identity.avatar_index,
                "full_name": identity.full_name,
                "handle_base": identity.handle_base,
                "role": role.role,
                "prompt": _avatar_prompt(identity.full_name, role),
            }
        )
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
