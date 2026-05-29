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

    def test_safe_auto_rejects_quoted_command_separator_arguments(self):
        decision = classify_bash_command('grep "a;b" file')

        self.assertFalse(decision.safe)
        self.assertEqual(decision.reason, "quoted or embedded shell operator in argument")

    def test_safe_auto_allows_quoted_grep_alternation(self):
        decision = classify_bash_command('git branch -a | grep -E "design|operator"')

        self.assertTrue(decision.safe, decision.reason)

    def test_startup_allowlist_is_derived_from_policy_tables(self):
        self.assertIn("Bash(git blame:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(gh issue list:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(git add:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(git commit:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(git pull:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(git config user.name)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(git config user.email)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertIn("Bash(gh pr create:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertNotIn("Bash(git fetch:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertNotIn("Bash(git branch:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)
        self.assertNotIn("Bash(git remote:*)", BASH_SAFE_AUTO_ALLOWED_TOOLS)

    def test_safe_auto_allows_read_only_git_branch_and_remote_commands(self):
        cases = (
            "git -C /workspace/repos/example-project branch -a",
            "git -C /workspace/repos/example-project branch --show-current",
            'git -C /workspace/repos/example-project branch -a | grep -E "design|operator"',
            "cd /workspace/repos/example-project && git remote -v 2>/dev/null",
            "git -C /workspace/repos/example-project remote get-url origin",
        )
        for command in cases:
            with self.subTest(command=command):
                decision = classify_bash_command(command)

                self.assertTrue(decision.safe, decision.reason)
                self.assertEqual(decision.safe_allowed_tools, (f"Bash({command})",))

    def test_safe_auto_rejects_mutating_git_branch_and_remote_commands(self):
        cases = (
            "git branch feature/new-work",
            "git branch -D old-work",
            "git remote add origin https://example.com/repo.git",
            "git remote set-url origin https://example.com/repo.git",
        )
        for command in cases:
            with self.subTest(command=command):
                decision = classify_bash_command(command)

                self.assertFalse(decision.safe)

    def test_safe_auto_allows_multiline_commit_by_prefix(self):
        command = """git -C /workspace/repos/sample-app commit -m "$(cat <<'EOF'
[sample-app] Add parity adapter and rollout bridge

Parity for Codex.
EOF
)" """

        decision = classify_bash_command(command)

        self.assertTrue(decision.safe, decision.reason)
        self.assertIn(
            "Bash(git -C /workspace/repos/sample-app commit:*)",
            decision.safe_allowed_tools,
        )

    def test_safe_auto_allows_exact_chmod_executable_bit(self):
        command = "chmod +x /workspace/repos/example-project/scripts/smoke-test.sh"

        decision = classify_bash_command(command)

        self.assertTrue(decision.safe, decision.reason)
        self.assertEqual(decision.safe_allowed_tools, (f"Bash({command})",))

    def test_safe_auto_rejects_broad_chmod(self):
        cases = (
            "chmod -R +x /workspace/repos/example-project",
            "chmod 777 /workspace/repos/example-project/script.sh",
        )
        for command in cases:
            with self.subTest(command=command):
                decision = classify_bash_command(command)

                self.assertFalse(decision.safe)
