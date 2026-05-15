# Repository Instructions

Follow `CONTRIBUTING.md`.

## Worktrees

Always work in a dedicated git worktree branched from the latest `origin/main`,
not the shared checkout. The shared checkout is reused across agents, and
working there directly causes one agent's changes to be stashed or overwritten
by another's. See the **Worktrees** section of `CONTRIBUTING.md` for the
commands.

## Public Change Metadata

Do not add automated assistant, agent, model, bot, or tool attribution anywhere
in public repo metadata or delivery text. This includes pull request titles,
pull request descriptions, commit subjects, commit bodies, commit trailers,
generated-by footers, release notes, and issue comments.

Do not include agent handles or names, provider names, model names, tool names,
or non-human coauthor trailers such as `Co-authored-by` for generated work. Keep
public change metadata focused on the human-requested code change unless the
human explicitly asks for attribution in that exact artifact.
