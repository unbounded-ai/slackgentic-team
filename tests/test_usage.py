import unittest
from datetime import UTC, datetime

from agent_harness.models import Provider
from agent_harness.providers.claude import claude_usage_from_record
from agent_harness.providers.codex import parse_token_count


class UsageTests(unittest.TestCase):
    def test_parse_codex_token_count(self):
        record = {
            "timestamp": "2026-04-26T17:00:00.000Z",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 5,
                        "output_tokens": 3,
                        "reasoning_output_tokens": 2,
                        "total_tokens": 13,
                    },
                    "model_context_window": 100,
                },
                "rate_limits": {
                    "primary": {
                        "used_percent": 25.0,
                        "window_minutes": 300,
                        "resets_at": 1777240000,
                    },
                    "plan_type": "team",
                },
            },
        }
        snapshot = parse_token_count(record, "sid")
        self.assertEqual(snapshot.provider, Provider.CODEX)
        self.assertEqual(snapshot.usage.total_tokens, 13)
        self.assertEqual(snapshot.context_window, 100)
        self.assertEqual(snapshot.remaining_description, "75.0% primary window remaining")
        self.assertEqual(
            snapshot.as_of,
            datetime(2026, 4, 26, 17, 0, tzinfo=UTC),
        )

    def test_parse_claude_usage(self):
        record = {
            "message": {
                "usage": {
                    "input_tokens": 1,
                    "cache_creation_input_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "output_tokens": 4,
                }
            }
        }
        usage = claude_usage_from_record(record)
        self.assertIsNotNone(usage)
        assert usage is not None
        self.assertEqual(usage.total_tokens, 10)


if __name__ == "__main__":
    unittest.main()
