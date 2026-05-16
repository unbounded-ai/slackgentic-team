import json
import unittest
from pathlib import Path

from agent_harness.bash_policy import BASH_SAFE_AUTO_ALLOWED_TOOLS, classify_bash_command

FIXTURE = Path(__file__).parent / "fixtures" / "safe_auto_bash_corpus.txt"


class BashPolicyTests(unittest.TestCase):
    def test_safe_auto_bash_corpus(self):
        for line_number, line in enumerate(FIXTURE.read_text().splitlines(), start=1):
            if not line.strip() or line.startswith("#"):
                continue
            case = json.loads(line)
            with self.subTest(line=line_number, command=case["command"]):
                decision = classify_bash_command(case["command"])

                self.assertEqual(decision.safe, case["decision"] == "allow", decision.reason)
                self.assertEqual(
                    decision.safe_allowed_tools,
                    tuple(case["safe_allowed_tools"]),
                )

    def test_safe_auto_rejects_quoted_operator_arguments(self):
        decision = classify_bash_command('grep "a;b" file')

        self.assertFalse(decision.safe)
        self.assertEqual(decision.reason, "quoted or embedded shell operator in argument")

    def test_startup_allowlist_is_derived_from_policy_tables(self):
        self.assertIn("Bash(git blame:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(gh issue list:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(git pull --ff-only:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(gh pr create:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertNotIn("Bash(git fetch:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertNotIn("Bash(git pull:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)

    def test_safe_auto_preserves_manual_approval_tool_for_multiline_commit(self):
        command = """git -C /workspace/repos/sample-app commit -m "$(cat <<'EOF'
[sample-app] Add parity adapter and rollout bridge

Parity for Codex.
EOF
)" """

        decision = classify_bash_command(command)

        self.assertFalse(decision.safe)
        self.assertIn(
            "Bash(git -C /workspace/repos/sample-app commit:*)",
            decision.approval_allowed_tools,
        )
