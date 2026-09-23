"""Install the bundled agent skills that teach Claude Code and Codex Slackgentic.

Both providers read the same ``SKILL.md`` layout: Claude Code from
``~/.claude/skills/<name>/`` and Codex from ``$CODEX_HOME/skills/<name>/``
(default ``~/.codex/skills``). Slackgentic owns the ``slackgentic*`` skill
directories it installs and replaces them wholesale on every install so
upgrades never leave stale reference files behind.
"""

from __future__ import annotations

import os
import shutil
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

SKILL_PROVIDERS = ("claude", "codex")
SKILLS_PACKAGE_DIR = "skills"


def bundled_skill_names() -> tuple[str, ...]:
    root = _bundled_skills_root()
    return tuple(
        sorted(
            item.name
            for item in root.iterdir()
            if item.is_dir() and item.joinpath("SKILL.md").is_file()
        )
    )


def skills_dir_for_provider(provider: str, home: Path | None = None) -> Path:
    base = home or Path.home()
    if provider == "claude":
        return base / ".claude" / "skills"
    if provider == "codex":
        codex_home = os.environ.get("CODEX_HOME") if home is None else None
        return (Path(codex_home) if codex_home else base / ".codex") / "skills"
    raise ValueError(f"unknown skill provider: {provider}")


def install_agent_skills(
    *,
    providers: tuple[str, ...] = SKILL_PROVIDERS,
    home: Path | None = None,
    only_existing: bool = False,
) -> list[Path]:
    """Copy every bundled skill into each provider's skills directory.

    With ``only_existing`` a provider is skipped unless its config directory
    (``~/.claude`` or the Codex home) already exists, so background refreshes
    never create config for a tool the user does not have.
    """

    installed: list[Path] = []
    root = _bundled_skills_root()
    for provider in providers:
        target_root = skills_dir_for_provider(provider, home)
        if only_existing and not target_root.parent.is_dir():
            continue
        for name in bundled_skill_names():
            target = target_root / name
            if target.exists():
                shutil.rmtree(target)
            _copy_tree(root.joinpath(name), target)
            installed.append(target)
    return installed


def _bundled_skills_root() -> Traversable:
    return resources.files("agent_harness").joinpath(SKILLS_PACKAGE_DIR)


def _copy_tree(source: Traversable, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        destination = target / item.name
        if item.is_dir():
            _copy_tree(item, destination)
        else:
            destination.write_bytes(item.read_bytes())
