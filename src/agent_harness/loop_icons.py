from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

LOOP_BADGE_SIZE = 256
LOOP_BADGE_SCALE = 2
LOOP_BADGE_SHAPES = frozenset({"circle", "rounded", "hex"})

_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_FONT = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "$": ("00100", "01111", "10100", "01110", "00101", "11110", "00100"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "*": ("00000", "10101", "01110", "11111", "01110", "10101", "00000"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
}


@dataclass(frozen=True)
class LoopBadgeSpec:
    background: str
    glyph: str
    glyph_color: str
    shape: str


def parse_loop_badge_spec(value: object) -> LoopBadgeSpec | str:
    if not isinstance(value, dict):
        return "icon.badge must be an object"
    background = value.get("background")
    glyph = value.get("glyph")
    glyph_color = value.get("glyph_color")
    shape = value.get("shape")
    if not isinstance(background, str) or not _HEX_COLOR_RE.fullmatch(background):
        return "icon.badge.background must be #RRGGBB"
    if not isinstance(glyph_color, str) or not _HEX_COLOR_RE.fullmatch(glyph_color):
        return "icon.badge.glyph_color must be #RRGGBB"
    if not isinstance(glyph, str) or not glyph.strip() or len(glyph.strip()) > 2:
        return "icon.badge.glyph must contain 1-2 characters"
    if shape not in LOOP_BADGE_SHAPES:
        return "icon.badge.shape must be circle, rounded, or hex"
    return LoopBadgeSpec(
        background=background.upper(),
        glyph=glyph.strip(),
        glyph_color=glyph_color.upper(),
        shape=str(shape),
    )


def render_loop_badge_png(spec: LoopBadgeSpec) -> bytes:
    size = LOOP_BADGE_SIZE * LOOP_BADGE_SCALE
    background = _rgba(spec.background)
    foreground = _rgba(spec.glyph_color)
    pixels = [(0, 0, 0, 0)] * (size * size)
    _draw_shape(pixels, size, spec.shape, background)
    _draw_glyph(pixels, size, spec.glyph, foreground)
    downsampled = _downsample(pixels, size, LOOP_BADGE_SCALE)
    return _encode_png(LOOP_BADGE_SIZE, LOOP_BADGE_SIZE, downsampled)


def write_loop_badge(path: Path, spec: LoopBadgeSpec) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_loop_badge_png(spec))
    return path


def _draw_shape(pixels, size: int, shape: str, color) -> None:
    scale = LOOP_BADGE_SCALE
    margin = 18 * scale
    center = size / 2
    radius = center - margin
    for y in range(size):
        dy = y + 0.5 - center
        for x in range(size):
            dx = x + 0.5 - center
            inside = False
            if shape == "circle":
                inside = dx * dx + dy * dy <= radius * radius
            elif shape == "hex":
                inside = abs(dx) <= radius and abs(dy) <= radius * 0.866
                inside = inside and abs(dx) + abs(dy) / 1.732 <= radius
            else:
                corner = 38 * scale
                edge = radius - corner
                qx = max(abs(dx) - edge, 0)
                qy = max(abs(dy) - edge, 0)
                inside = qx * qx + qy * qy <= corner * corner
            if inside:
                pixels[y * size + x] = color


def _draw_glyph(pixels, size: int, glyph: str, color) -> None:
    glyphs = [_FONT.get(char.upper(), _FONT["?"]) for char in glyph]
    cell = (27 if len(glyphs) == 1 else 19) * LOOP_BADGE_SCALE
    gap = cell
    width = len(glyphs) * 5 * cell + (len(glyphs) - 1) * gap
    height = 7 * cell
    start_x = (size - width) // 2
    start_y = (size - height) // 2
    for glyph_index, rows in enumerate(glyphs):
        offset_x = start_x + glyph_index * (5 * cell + gap)
        for row_index, row in enumerate(rows):
            for column_index, enabled in enumerate(row):
                if enabled != "1":
                    continue
                _fill_rect(
                    pixels,
                    size,
                    offset_x + column_index * cell,
                    start_y + row_index * cell,
                    cell,
                    cell,
                    color,
                )


def _fill_rect(pixels, size: int, x: int, y: int, width: int, height: int, color) -> None:
    for yy in range(max(0, y), min(size, y + height)):
        for xx in range(max(0, x), min(size, x + width)):
            pixels[yy * size + xx] = color


def _downsample(pixels, size: int, scale: int):
    target_size = size // scale
    output = []
    area = scale * scale
    for y in range(target_size):
        for x in range(target_size):
            totals = [0, 0, 0, 0]
            for yy in range(y * scale, (y + 1) * scale):
                for xx in range(x * scale, (x + 1) * scale):
                    for index, value in enumerate(pixels[yy * size + xx]):
                        totals[index] += value
            output.append(tuple(round(value / area) for value in totals))
    return output


def _rgba(value: str) -> tuple[int, int, int, int]:
    return (
        int(value[1:3], 16),
        int(value[3:5], 16),
        int(value[5:7], 16),
        255,
    )


def _encode_png(width: int, height: int, pixels) -> bytes:
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for x in range(width):
            raw.extend(pixels[y * width + x])
    data = bytearray(b"\x89PNG\r\n\x1a\n")
    data.extend(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
    data.extend(_png_chunk(b"IDAT", zlib.compress(bytes(raw), level=9)))
    data.extend(_png_chunk(b"IEND", b""))
    return bytes(data)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )
