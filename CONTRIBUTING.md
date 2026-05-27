# Contributing

## Local Setup

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Checks

Run the same checks used by CI:

```sh
PYTHONPATH=src python -m pytest -n auto tests
PYTHONPATH=src python -m compileall src tests
python -m ruff check src tests
python -m ruff format --check src tests
```

Before opening a pull request, always run the two Ruff commands above. Passing
unit tests is not enough: CI has a separate formatting gate, and
`python -m ruff format --check src tests` will fail the PR if the formatter
would rewrite even one file.

The test suite is written with `unittest` and run through `pytest` in CI so
`pytest-xdist` can split tests across workers. It should not require network
access.

## Public Change Metadata

Do not add automated assistant, agent, model, bot, or tool attribution anywhere
in public repo metadata or delivery text. This includes pull request titles,
pull request descriptions, commit subjects, commit bodies, commit trailers,
generated-by footers, release notes, and issue comments.

Do not include agent handles or names, provider names, model names, tool names,
or non-human coauthor trailers such as `Co-authored-by` for generated work. Keep
public change metadata focused on the human-requested code change unless the
human explicitly asks for attribution in that exact artifact.

## Worktrees

Agents working in this repo must use a dedicated git worktree branched from the
latest `origin/main`, not the shared checkout. Multiple agents share the same
clone, and working directly on a checked-out branch lets one agent stash or
overwrite another's uncommitted changes. Worktrees give each task its own
working tree pinned to its own branch.

Before starting work:

```sh
git fetch origin
git worktree add ../<repo>-<task-slug> -b <branch-name> origin/main
cd ../<repo>-<task-slug>
```

Always branch from `origin/main`, not from whatever branch the shared checkout
happens to have. When the task is finished and the branch has been pushed and
merged (or abandoned), remove the worktree:

```sh
git worktree remove ../<repo>-<task-slug>
```

If you find your changes have been stashed by another agent, do not pop the
stash into a shared checkout. Recover by creating a fresh worktree off
`origin/main`, applying the stash there, and pushing from inside it.

## Development Notes

- Keep provider-specific behavior behind the provider adapters, runtime, session
  bridge, or channel/app-server modules.
- Prefer provider-neutral Slack UI and state where Codex and Claude can share
  behavior.
- Do not commit generated files such as `__pycache__`, `.egg-info`, local
  SQLite databases, or `.slackgentic` runtime logs.
- Treat Slack tokens and local transcript files as private developer data.
