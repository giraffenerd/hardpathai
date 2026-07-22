# Repository Agent Instructions

## Working style

- Operate autonomously by default and complete normal in-scope work without asking for confirmation.
- Make reasonable implementation decisions, run relevant verification, and fix issues discovered during verification.
- Ask only when required information is unavailable, the requested outcome is materially ambiguous, or an action needs authority beyond the user's stated scope.
- Treat requests for cleanup, replacement, history rewriting, or moving files as authorization only for the explicitly named targets.
- Preserve unrelated user changes in the working tree.
- Platform security prompts and approval requirements still apply.

## Python projects

- Use `uv` for Python versions, dependencies, virtual environments, and lockfiles.
- Treat each unrelated directory under `projects/` as an independent Python project.
- Give every Python project its own `pyproject.toml`, `uv.lock`, `.python-version`, and ignored `.venv/`.
- Declare runtime dependencies in `pyproject.toml`; do not rely on undocumented global packages.
- Commit `uv.lock` and never commit `.venv/`, Python caches, downloaded runtimes, or package caches.
- Run a project with `uv run main.py` from that project's directory unless its manifest defines a more appropriate entry point.
- Use `uv add <package>` and `uv remove <package>` when changing dependencies so the manifest and lockfile stay synchronized.
- For a new standalone project, prefer `uv init --app projects/<project-name>`.
- Verify Python changes with `uv run` and any relevant tests before reporting completion.
- Allow `uv` to reuse its global download cache while keeping each project's environment and dependency declarations isolated.

## Communication

- Assume the user is familiar with frontend and npm workflows; explain Python conventions using concise npm analogies when useful.
- Lead with outcomes and commands the user can run.
