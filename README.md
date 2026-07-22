# Python Projects

Python projects in this repository use `uv` for Python versions, dependencies, virtual environments, and lockfiles.

## Install uv once

Windows:

```powershell
winget install --id=astral-sh.uv -e
```

macOS:

```bash
brew install uv
```

Linux and macOS alternative:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Run logistic regression

```powershell
cd projects\logistic-regression
uv run main.py
```

The first run downloads Python if needed, creates `.venv`, installs the exact dependencies from `uv.lock`, and runs the program. Later runs reuse the environment and package cache.

## Create project 2

```powershell
cd projects
uv init --app project2
cd project2
uv add requests
uv run main.py
```

Each project has its own `pyproject.toml`, `uv.lock`, and `.venv`, so dependencies can differ safely between projects.
