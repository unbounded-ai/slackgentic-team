# Contributing

## Local Setup

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Checks

Run the same checks used by CI:

```sh
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m compileall src tests
python -m ruff check src tests
python -m ruff format --check src tests
```

Before opening a pull request, always run the two Ruff commands above. Passing
unit tests is not enough: CI has a separate formatting gate, and
`python -m ruff format --check src tests` will fail the PR if the formatter
would rewrite even one file.

The test suite uses `unittest` and should not require network access.

## Public Change Metadata

Do not add automated assistant, agent, model, bot, or tool attribution anywhere
in public repo metadata or delivery text. This includes pull request titles,
pull request descriptions, commit subjects, commit bodies, commit trailers,
generated-by footers, release notes, and issue comments.

Do not include agent handles or names, provider names, model names, tool names,
or non-human coauthor trailers such as `Co-authored-by` for generated work. Keep
public change metadata focused on the human-requested code change unless the
human explicitly asks for attribution in that exact artifact.

## Development Notes

- Keep provider-specific behavior behind the provider adapters, runtime, session
  bridge, or channel/app-server modules.
- Prefer provider-neutral Slack UI and state where Codex and Claude can share
  behavior.
- Do not commit generated files such as `__pycache__`, `.egg-info`, local
  SQLite databases, or `.slackgentic` runtime logs.
- Treat Slack tokens and local transcript files as private developer data.
