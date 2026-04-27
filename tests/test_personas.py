import unittest

from agent_harness.models import Provider
from agent_harness.personas import avatar_svg, generate_persona


class PersonaTests(unittest.TestCase):
    def test_persona_is_deterministic(self):
        left = generate_persona(Provider.CODEX, "session-123")
        right = generate_persona("codex", "session-123")
        self.assertEqual(left, right)

    def test_persona_differs_by_provider(self):
        codex = generate_persona("codex", "same")
        claude = generate_persona("claude", "same")
        self.assertNotEqual(codex.avatar_slug, claude.avatar_slug)

    def test_avatar_svg_contains_initials_and_color(self):
        persona = generate_persona("claude", "abc")
        svg = avatar_svg(persona)
        self.assertIn(persona.initials, svg)
        self.assertIn(persona.color_hex, svg)


if __name__ == "__main__":
    unittest.main()
