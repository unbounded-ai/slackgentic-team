import os
import unittest

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from agent_harness.bash_policy import (
    _GH_SAFE_COMMAND_TOOLS,
    _GIT_SAFE_SUBCOMMAND_TOOLS,
    _SAFE_SIMPLE_EXECUTABLE_TOOLS,
    classify_bash_command,
)

MAX_EXAMPLES = int(os.environ.get("SLACKGENTIC_BASH_POLICY_MAX_EXAMPLES", "300"))
PROPERTY_SETTINGS = settings(
    max_examples=MAX_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.differing_executors],
)

SAFE_LITERAL_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._=-"
SAFE_LITERAL = st.text(SAFE_LITERAL_ALPHABET, min_size=1, max_size=24)
SAFE_GIT_WORD = SAFE_LITERAL.filter(
    lambda value: not value.startswith("-") and not value.startswith("--output")
)
SAFE_PATH_PART = st.text(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._=-",
    min_size=1,
    max_size=12,
).filter(lambda value: not value.startswith("-"))
SAFE_PATH = st.lists(SAFE_PATH_PART, min_size=1, max_size=4).map(
    lambda parts: "/workspace/" + "/".join(parts)
)
SAFE_FILE = st.one_of(
    st.sampled_from(("README.md", "pyproject.toml", "src/agent_harness/bash_policy.py")),
    st.lists(SAFE_PATH_PART, min_size=1, max_size=3).map(lambda parts: "/".join(parts)),
)

SIMPLE_COMMAND_TEMPLATES = (
    "ls {file}",
    "cat {file}",
    "head -{count} {file}",
    "tail -{count} {file}",
    "nl {file}",
    "wc -l {file}",
    "rg {word} {file}",
    "grep -n {word} {file}",
    "which {word}",
    "pwd",
)
SED_COMMAND_TEMPLATES = (
    "sed -n {line}p {file}",
    "sed -n {start},{end}p {file}",
)

NEVER_SAFE_INNER = (
    "rm -rf /tmp/x",
    "sudo ls",
    "curl https://example.com",
    "bash script.sh",
    "python -c print",
    "git commit -m x",
    "git push",
    "git reset --hard",
    "git checkout main",
    "git fetch",
    "git pull",
    "git pull --rebase",
    "git clone https://example.com/x/y",
    "git tag v1",
    "git stash",
    "git rebase main",
    "git apply patch",
    "git config user.email x",
    "gh pr merge 1",
    "gh pr close 1",
    "gh issue create",
    "gh repo clone x/y",
    "gh auth login",
    "sed -i s/a/b/ f",
    "sed -I '' s/a/b/ f",
    "sed s/a/b/ f",
)

NEVER_SAFE_WRAPPERS = (
    "{}",
    "cd /tmp && {}",
    "ls /tmp && {}",
    "cat README.md ; {}",
    "ls /tmp | {}",
    "{} | head -10",
    "{} 2>&1",
)

SAFE_SIMPLE_COMMANDS = (
    "git status",
    "git -C /workspace/repos/sample-app log --oneline -30",
    "git --no-pager show --stat",
    "git -c core.pager=cat log --oneline -5",
    "git -C /path-with-spaces log --oneline -5",
    "git blame src/agent_harness/bash_policy.py",
    "git rev-parse --show-toplevel",
    "gh pr view 35 --json url,state,mergeable",
    "gh pr create --title update --body summary",
    "git pull --ff-only",
    "git -C /workspace/repos/sample-app pull --ff-only",
    "gh issue list",
    "gh issue view 42",
    "gh repo view --json name,owner",
    "ls /tmp",
    "cat README.md",
    "head -30 README.md",
    "tail -20 README.md",
    "wc -l README.md",
    "rg safe-auto src tests",
    "grep -n safe README.md",
    "sed -n 1,80p src/agent_harness/bash_policy.py",
    "pwd",
    "which git",
)

PIPE_SOURCES = (
    "git status 2>&1",
    "git -C /workspace/repos/sample-app log --oneline -30",
    "ls /tmp",
    "cat README.md",
)

PIPE_FILTERS = (
    "head -30",
    "tail -20",
    "wc -l",
    "grep -i example 2>&1",
)

UNSAFE_TEMPLATES = (
    "{} || rm -rf /tmp/nope",
    "{} &",
    "FOO=bar {}",
    "cat <<EOF",
    "cat <<'EOF'",
    "cat <<-EOF",
    "cat <<<here",
    "cat <({})",
    "cat >({})",
    "echo $({})",
    "echo `{}`",
    "( {} )",
    "{} >>/tmp/out",
    "{} >/tmp/out",
    "{} </tmp/in",
)

OPERATOR_ARGS = (";", "&&", "||", "|", ">", "<", "&")
ALL_SAFE_EXECUTABLE_COMMANDS = tuple(
    f"{executable} {{operator}} README.md" for executable in sorted(_SAFE_SIMPLE_EXECUTABLE_TOOLS)
)
QUOTED_OPERATOR_COMMAND_TEMPLATES = (
    *ALL_SAFE_EXECUTABLE_COMMANDS,
    "git status {operator}",
    "gh pr view 1 {operator}",
    "sed -n {operator} README.md",
)
SHELLISH_TEXT = st.text(
    alphabet=st.characters(
        min_codepoint=32,
        max_codepoint=126,
        blacklist_characters=("\x7f",),
    ),
    min_size=0,
    max_size=120,
)


def _safe_command_strategy():
    simple = st.one_of(st.sampled_from(SAFE_SIMPLE_COMMANDS), _generated_safe_simple_command())
    sequence = st.tuples(
        _exact_grantable_safe_command(),
        st.sampled_from(("&&", ";")),
        _exact_grantable_safe_command(),
    ).map(lambda parts: f"{parts[0]} {parts[1]} {parts[2]}")
    pipeline = st.tuples(
        st.sampled_from(PIPE_SOURCES),
        st.sampled_from(PIPE_FILTERS),
    ).map(lambda parts: f"{parts[0]} | {parts[1]}")
    cd_prefix = st.sampled_from(
        (
            "cd /workspace/repos/sample-app && git log --oneline -30",
            "cd /workspace/repos/sample-app && rg safe-auto src tests",
        )
    )
    return st.one_of(simple, sequence, pipeline, cd_prefix)


def _generated_safe_simple_command():
    return st.one_of(
        _generated_git_command(),
        _generated_gh_command(),
        _generated_basic_command(),
        _generated_sed_command(),
    )


def _generated_git_command():
    subcommand = st.sampled_from(sorted(_GIT_SAFE_SUBCOMMAND_TOOLS))
    prefix = st.one_of(
        st.just("git"),
        SAFE_PATH.map(lambda path: f"git -C {path}"),
        st.just("git --no-pager"),
        st.just("git -c core.pager=cat"),
    )
    return st.tuples(prefix, subcommand, SAFE_FILE, SAFE_GIT_WORD, st.integers(1, 99)).map(
        lambda parts: _git_command_for(parts[0], parts[1], parts[2], parts[3], parts[4])
    )


def _git_command_for(prefix: str, subcommand: str, file: str, word: str, count: int) -> str:
    args = {
        "status": "",
        "log": f"--oneline -{count}",
        "diff": "--stat",
        "show": "--stat",
        "blame": file,
        "grep": word,
        "rev-parse": "--show-toplevel",
        "ls-files": file,
        "ls-tree": "HEAD",
        "cat-file": "-p HEAD",
        "merge-base": "HEAD main",
        "for-each-ref": "refs/heads",
    }[subcommand]
    return f"{prefix} {subcommand} {args}".strip()


def _generated_gh_command():
    command = st.sampled_from(sorted(_GH_SAFE_COMMAND_TOOLS))
    extra = st.one_of(st.just(""), st.integers(1, 1000).map(str), st.just("--json name"))
    return st.tuples(command, extra).map(
        lambda parts: f"gh {parts[0][0]} {parts[0][1]} {parts[1]}".strip()
    )


def _generated_basic_command():
    return st.tuples(
        st.sampled_from(SIMPLE_COMMAND_TEMPLATES),
        SAFE_FILE,
        SAFE_LITERAL,
        st.integers(1, 120),
    ).map(lambda parts: parts[0].format(file=parts[1], word=parts[2], count=parts[3]))


def _generated_sed_command():
    return st.tuples(
        st.sampled_from(SED_COMMAND_TEMPLATES),
        SAFE_FILE,
        st.integers(1, 20),
        st.integers(21, 80),
    ).map(
        lambda parts: parts[0].format(
            file=parts[1],
            line=parts[2],
            start=parts[2],
            end=parts[3],
        )
    )


def _exact_grantable_safe_command():
    return _generated_safe_simple_command().filter(
        lambda command: not any(value in command for value in (")", ",", "\n", "\r"))
    )


def _broad_grantable_safe_command():
    return _exact_grantable_safe_command().filter(
        lambda command: any(
            tool.endswith(":*)") for tool in classify_bash_command(command).safe_allowed_tools
        )
    )


class BashPolicyPropertyTests(unittest.TestCase):
    @PROPERTY_SETTINGS
    @given(_safe_command_strategy())
    def test_safe_structural_commands_are_allowed_with_auto_grants(self, command):
        decision = classify_bash_command(command)

        self.assertTrue(decision.safe, decision.reason)
        self.assertTrue(decision.safe_allowed_tools)

    @PROPERTY_SETTINGS
    @given(
        st.sampled_from(UNSAFE_TEMPLATES),
        st.sampled_from(SAFE_SIMPLE_COMMANDS),
    )
    def test_unsafe_shell_structures_are_denied(self, template, command):
        decision = classify_bash_command(template.format(command))

        self.assertFalse(decision.safe)
        self.assertEqual(decision.safe_allowed_tools, ())

    @PROPERTY_SETTINGS
    @given(
        st.sampled_from(NEVER_SAFE_INNER),
        st.sampled_from(NEVER_SAFE_WRAPPERS),
    )
    def test_mutating_commands_are_never_safe_under_wrappers(self, inner, wrapper):
        decision = classify_bash_command(wrapper.format(inner))

        self.assertFalse(decision.safe)
        self.assertEqual(decision.safe_allowed_tools, ())

    @PROPERTY_SETTINGS
    @given(
        st.sampled_from(QUOTED_OPERATOR_COMMAND_TEMPLATES),
        st.sampled_from(OPERATOR_ARGS),
    )
    def test_quoted_operator_arguments_are_denied(self, template, operator):
        decision = classify_bash_command(template.format(operator=f'"{operator}"'))

        self.assertFalse(decision.safe)
        self.assertEqual(decision.safe_allowed_tools, ())

    @PROPERTY_SETTINGS
    @given(SAFE_PATH, _broad_grantable_safe_command())
    def test_cd_safe_compound_returns_exact_and_inner_broad_grants(self, path, inner):
        command = f"cd {path} && {inner}"
        decision = classify_bash_command(command)

        self.assertTrue(decision.safe, decision.reason)
        self.assertIn(f"Bash({command})", decision.safe_allowed_tools)
        self.assertTrue(any(tool.endswith(":*)") for tool in decision.safe_allowed_tools))

    @PROPERTY_SETTINGS
    @given(_safe_command_strategy())
    def test_safe_grants_are_well_formed(self, command):
        decision = classify_bash_command(command)

        self.assertTrue(decision.safe, decision.reason)
        self.assertTrue(decision.safe_allowed_tools)
        for tool in decision.safe_allowed_tools:
            self.assertTrue(tool.startswith("Bash("), tool)
            self.assertTrue(tool.endswith(")"), tool)
            inner = tool[5:-1]
            self.assertNotIn("\n", inner)
            self.assertNotIn("\r", inner)
        if not any(value in command for value in (")", ",", "\n", "\r")):
            self.assertIn(f"Bash({command})", decision.safe_allowed_tools)

    @PROPERTY_SETTINGS
    @given(SHELLISH_TEXT)
    def test_arbitrary_shellish_input_never_crashes(self, command):
        decision = classify_bash_command(command)

        self.assertIsInstance(decision.safe, bool)
        if decision.safe:
            self.assertTrue(decision.safe_allowed_tools)
            self.assertTrue(_safe_shell_control_is_structural(decision))


def _safe_shell_control_is_structural(decision) -> bool:
    command = decision.command
    if "$(" in command or "`" in command or "(" in command or ")" in command:
        return False
    for marker in ("||", ">>", "<<", "<<<", "<(", ">("):
        if marker in command:
            return False
    redirects = tuple(redirect for segment in decision.segments for redirect in segment.redirects)
    if ";" in command and ";" not in decision.operators:
        return False
    if "|" in command and "|" not in decision.operators:
        return False
    if "<" in command:
        return False
    if ">" in command and not redirects:
        return False
    return not (
        "&" in command
        and "&&" not in command
        and not any("&" in redirect for redirect in redirects)
    )
