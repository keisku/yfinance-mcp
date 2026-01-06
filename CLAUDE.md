# CLAUDE.md

Project-specific instructions for AI assistants working on this codebase.

## Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automatic changelog generation via git-cliff.

### Format

```
<type>[optional scope]: <description>

[optional body]
```

### Types

| Type | When to use |
|------|-------------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code restructuring without behavior change |
| `docs` | Documentation only |
| `style` | Formatting, linting (no logic change) |
| `test` | Adding or updating tests |
| `build` | Build system or external dependencies |
| `ci` | CI configuration files and scripts |
| `chore` | Maintenance tasks, dependencies |

### Breaking Changes

Add `!` after type for breaking changes:

```
feat!: change response format to delta-encoded
```

### Body Format

Body lines appear as sub-bullets in the changelog. Use `-` prefix:

```
feat: add adjusted close column

- history tool returns both c and ac
- technicals uses ac for calculations
- **BREAKING**: cache schema adds adj_close column
```

### Examples

```bash
feat: add fibonacci retracement indicator
fix(cache): handle missing trading days
perf: reduce memory allocation in resampling
docs: update README with new tool descriptions
chore: bump yfinance to 0.2.50
```

## Code Style

- Python 3.13+
- Formatting: `uv run ruff format`
- Linting: `uv run ruff check --fix`
- Type hints required for public APIs
- Tests: pytest with hypothesis for property-based testing

## Testing

```bash
uv run pytest              # all tests
uv run pytest -v --tb=short  # verbose with short traceback
```
