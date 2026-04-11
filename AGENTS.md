# Repository Guidelines

## Project Structure & Module Organization
Core library code lives in `src/hdr/`, with task implementations under `src/hdr/tasks/` (`std.py`, `coding.py`, `meta.py`, `mind.py`). Tests live in `tests/` and follow the same domain split, for example `tests/test_std.py`. User-facing docs are in `docs/tasks/`, and runnable examples are in `examples/` with paired `task.py` and `work.py` files. Documented solutions live in `docs/solutions/`, organized by category with YAML frontmatter such as `module`, `tags`, and `problem_type`; they are relevant when implementing or debugging in documented areas. Top-level references include `README.md` and `SKILL.md`.

## Build, Test, and Development Commands
Use Python 3.12 and `uv` for local setup.

- `uv venv .venv` creates the virtual environment.
- `uv pip install -e ".[dev]"` installs the package plus dev tools.
- `uv run pytest` runs the full test suite.
- `uv run pyright` runs static type checks for `src`, `tests`, and `examples`.
- `uv run ruff check .` runs linting.
- `uv run ruff format .` formats Python files.

If you are iterating on one area, prefer targeted runs such as `uv run pytest tests/test_meta.py`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, explicit type hints, and concise docstrings where the behavior is not obvious. Use `snake_case` for functions, variables, and test names; use `CamelCase` for task and helper classes such as `DirectoryCreated` or `TaskCreated`. Keep modules focused by task domain, and prefer adding new task types under `src/hdr/tasks/` instead of expanding unrelated files.

## Testing Guidelines
Tests use `pytest` with the `test_*.py` pattern configured in `pyproject.toml`. Add tests alongside the affected domain and name cases by behavior, for example `test_md_file_must_end_with_md`. Cover both success and failure paths, especially validation errors and type/verification boundaries. Run `uv run pytest` and `uv run pyright` before opening a PR.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit messages such as `Fix pyright errors` and `Add tests for TaskCreated meta-task`. Keep commits focused and descriptive. Pull requests should include a brief problem statement, the implementation summary, test results, and linked issues if applicable. Include command output or screenshots only when they clarify behavior changes in docs or examples.

## Configuration Tips
Runtime verification reads settings from `~/.hdr/config.yaml`, not from environment variables. On the first `Task.verify()` call, HDR creates that file if it does not exist; fill in `anthropic_auth_token` there before rerunning. `anthropic_model`, `anthropic_base_url`, and `verify_cache_dir` are optional config entries with repo-defined defaults. Avoid hardcoding secrets or absolute machine-specific paths in examples and tests.
