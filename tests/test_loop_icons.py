import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from agent_harness.loop_icons import (
    LOOP_BADGE_SIZE,
    LoopBadgeSpec,
    parse_loop_badge_spec,
    render_loop_badge_png,
    write_loop_badge,
)


class LoopIconTests(unittest.TestCase):
    def test_badge_parser_normalizes_colors_and_validates_fields(self):
        parsed = parse_loop_badge_spec(
            {
                "background": "#0b6e4f",
                "glyph": "$",
                "glyph_color": "#f4fff9",
                "shape": "circle",
            }
        )

        self.assertEqual(parsed, LoopBadgeSpec("#0B6E4F", "$", "#F4FFF9", "circle"))
        self.assertIsInstance(
            parse_loop_badge_spec(
                {
                    "background": "green",
                    "glyph": "ABC",
                    "glyph_color": "#FFFFFF",
                    "shape": "triangle",
                }
            ),
            str,
        )

    def test_renderer_is_deterministic_and_produces_valid_rgba_png(self):
        spec = LoopBadgeSpec("#0B6E4F", "$", "#F4FFF9", "circle")

        first = render_loop_badge_png(spec)
        second = render_loop_badge_png(spec)

        self.assertEqual(first, second)
        width, height, raw = _decode_png(first)
        self.assertEqual((width, height), (LOOP_BADGE_SIZE, LOOP_BADGE_SIZE))
        self.assertEqual(len(raw), height * (1 + width * 4))
        self.assertTrue(all(raw[row * (1 + width * 4)] == 0 for row in range(height)))

    def test_shapes_and_glyphs_change_output(self):
        circle = render_loop_badge_png(LoopBadgeSpec("#112233", "A", "#FFFFFF", "circle"))
        rounded = render_loop_badge_png(LoopBadgeSpec("#112233", "A", "#FFFFFF", "rounded"))
        hex_badge = render_loop_badge_png(LoopBadgeSpec("#112233", "CI", "#FFFFFF", "hex"))

        self.assertNotEqual(circle, rounded)
        self.assertNotEqual(circle, hex_badge)
        self.assertNotEqual(rounded, hex_badge)

    def test_write_badge_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "loop" / "icon.png"

            returned = write_loop_badge(
                path,
                LoopBadgeSpec("#2457A6", "CI", "#FFFFFF", "rounded"),
            )

            self.assertEqual(returned, path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


def _decode_png(value: bytes) -> tuple[int, int, bytes]:
    if value[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("missing PNG signature")
    offset = 8
    width = height = 0
    compressed = bytearray()
    while offset < len(value):
        length = struct.unpack(">I", value[offset : offset + 4])[0]
        kind = value[offset + 4 : offset + 8]
        payload = value[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"IHDR":
            width, height = struct.unpack(">II", payload[:8])
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
    return width, height, zlib.decompress(bytes(compressed))


if __name__ == "__main__":
    unittest.main()
