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

## Development Notes

- Keep provider-specific behavior behind the provider adapters, runtime, session
  bridge, or channel/app-server modules.
- Prefer provider-neutral Slack UI and state where Codex and Claude can share
  behavior.
- Do not commit generated files such as `__pycache__`, `.egg-info`, local
  SQLite databases, or `.slackgentic` runtime logs.
- Treat Slack tokens and local transcript files as private developer data.
