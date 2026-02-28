# CLAUDE.md

## Commands

```bash
uv run python server.py         # run MCP server
uv run python -c "..."          # run ad-hoc scripts
uv run ruff check --fix .       # lint (auto-fix)
uv run ruff format .            # format
```

Always use `uv run python` to execute Python — never use `.venv/bin/python` or bare `python`.

## Code Quality

Run ruff lint and format before every commit:

```bash
uv run ruff check --fix . && uv run ruff format .
```

## Code Style

- Python 3.13+
- Type hints required for public APIs

## Logging

- Use `logging.getLogger(__name__)` in each module. `logging.basicConfig` is configured once in `server.py`'s `main()`.
- Levels: `DEBUG` for troubleshooting details, `WARNING` for problem indicators.
- Keep logs minimal: API calls, cache gaps/holidays, tool invocations, and errors only.
- No logging in test files.
- `cache.py` is a data layer — no logs needed. Callers (`history.py`) cover it.
