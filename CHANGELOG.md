# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.2.1] - 2026-01-06

### CI

- skip prerelease tags in release workflow ([5c69d65](https://github.com/keisku/yfinance-mcp/commit/5c69d654852008e07579f374f47d299a37804e31))

### Documentation

- clean CHANGELOG.md ([3dd199a](https://github.com/keisku/yfinance-mcp/commit/3dd199a66dd8d77223ae00dc7f06d62b16e28cf9))



## [0.2.0] - 2026-01-06

### Fixed

- preserve file changes when switching to main branch ([6fa14d1](https://github.com/keisku/yfinance-mcp/commit/6fa14d1864fa277fed8fb181548cb6d067246987))
  Tag checkout runs in detached HEAD. Stash changes before switching
  to main to prevent losing CHANGELOG.md and pyproject.toml updates.

### Added

- add adjusted close (ac) column for total return calculations ([e9ed62a](https://github.com/keisku/yfinance-mcp/commit/e9ed62a532080554fcaa1f957d71f7f93e282f08))
  - history tool returns both c (price-return) and ac (dividend/split-adjusted)
  - technicals tool uses ac for indicator calculations (falls back to c for indices)
  - cache schema adds adj_close column (breaking: clear cache before upgrade)
  - updated tests with ac/c relationship coverage

### CI

- add automated release workflow with git-cliff ([69917a4](https://github.com/keisku/yfinance-mcp/commit/69917a474a6e47d36f7de7246899930a8591ab0c))
  - Changelog generated from conventional commits on tag push
  - pyproject.toml version updated automatically during release
  - GitHub Release created with generated notes
  - git-cliff added as dev dependency for local preview

### Documentation

- add CLAUDE.md for AI assistant guidelines ([144a735](https://github.com/keisku/yfinance-mcp/commit/144a735f783e61116317ceea53232ee563456859))

### Styling

- ruff fix ([5b690dc](https://github.com/keisku/yfinance-mcp/commit/5b690dc519683cf93ebd67e2499e18d1d673374b))

## [0.1.0] - 2026-01-06

### Changed

- **BREAKING**: Delta-encoded split format for time series responses ([8fa2050](https://github.com/keisku/yfinance-mcp/commit/8fa2050))
  - Response structure changed from row-oriented to `cols/t0/dt/dt_unit/rows`, achieving ~56% token reduction vs JSON
  - TARGET_POINTS default increased from 120 to 200 (stays within 1-3K ideal token range)
  - Intraday data auto-detected: uses minutes for HH:MM timestamps, days otherwise
  - Schema hint added for LLM clarity: `_: "ts[i] = t0 + sum(dt[0..i])"`

### Fixed

- Reduce partial_data warning noise ([ff3e901](https://github.com/keisku/yfinance-mcp/commit/ff3e901))
  - Only warn when <50% of data is valid (warmup dominates), reducing false positives for long periods with short warmup indicators

## [0.0.2] - 2025-01-05

### Added

- MCP server metadata: version (from package), instructions, and website_url ([2366723](https://github.com/keisku/yfinance-mcp/commit/2366723))
