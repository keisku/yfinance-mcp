# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-01-08

### Added

- add expense_ratio for ETFs and mutual funds ([531fef3](https://github.com/keisku/yfinance-mcp/commit/531fef336364a944f2ffe6b759207971fe688937))
  - Add expense_ratio field to search_stock response for ETF/MUTUALFUND quote types
  - Implement Yahoo Finance Japan scraping fallback for JPX ETFs lacking netExpenseRatio
  - Add etf_expense table to DuckDB cache with 1-day TTL
  - Include retry logic with UA rotation for web scraping robustness
  - Add comprehensive tests for fetch_japan_etf_expense helper

### Documentation

- Clarify time range limit in history and technicals ([94a223a](https://github.com/keisku/yfinance-mcp/commit/94a223a0dffbf268277916caeeb090d8af52a1c0))
  State the threshold explicitly in the split instruction rather than
  repeating the max range twice.

### Miscellaneous

- Fix uv deprecation and urllib3 CVE ([8441da9](https://github.com/keisku/yfinance-mcp/commit/8441da93547a31cc6610415f5590c4fe0f07b29f))
- Set explicit PyPI index URL for uv ([54ddece](https://github.com/keisku/yfinance-mcp/commit/54ddece0c02090f27f4227df021c9bc45ac6e3b8))
  Ensures consistent package resolution by explicitly specifying the
  default PyPI index in uv configuration.

## [0.3.1] - 2026-01-07

### Fixed

- Use trading days consistently for interval selection ([07270e8](https://github.com/keisku/yfinance-mcp/commit/07270e81f48c74ba96d3265a482819abe8ca29d7))
  Convert calendar days to trading days (× 5/7) when calculating bar
  counts for interval selection and period validation. This fixes
  incorrect interval selection where weekly PPD was assumed to be 1/5
  but calculations used calendar days.
  Key changes:
  - MAX_PERIOD_DAYS now represents trading days (1000) not calendar days
  - select_interval() converts calendar to trading days for bar count
  - validate_date_range() uses trading days for limit checks
  - _merge_gaps() converts MAX_PERIOD_DAYS back to calendar days
  - get_valid_periods() compares against trading days
  Yahoo API limits (max_days in interval config) remain in calendar days
  as that's what the API enforces.

## [0.3.0] - 2026-01-06

### Added

- Add quote_type field for instrument identification ([ed373e8](https://github.com/keisku/yfinance-mcp/commit/ed373e824be00d72bae3f789af077713d15481a7))
  - Expose yfinance's quoteType (EQUITY, ETF, INDEX, etc.)
  - Helps distinguish investable instruments from indices

### CI

- remove unnecessary break lines in CHANGELOG ([f6aef17](https://github.com/keisku/yfinance-mcp/commit/f6aef1759e7a0e9e72ca83142ab5d953275773dc))
- suppress git-cliff warnings during calculating next version ([a94827c](https://github.com/keisku/yfinance-mcp/commit/a94827c581af41b9f24c1b9cc5019e5ae9321f11))
- pin python 3.13 ([9640d31](https://github.com/keisku/yfinance-mcp/commit/9640d317e8bd27df426b0eeb979855b6f36191a1))
- Block release if Test workflow not passing ([aa9c830](https://github.com/keisku/yfinance-mcp/commit/aa9c830a10a7fa06a3e49d2acf9ac5a9ec788086))
  - Check latest Test workflow status on main before releasing
  - Fail fast with error if tests have not passed
- Auto-calculate version with git-cliff ([cb090d0](https://github.com/keisku/yfinance-mcp/commit/cb090d05bd3a7f88832286e10241b0148a1f7797))
  - Use workflow_dispatch trigger instead of tag push
  - Calculate semver from commit types via --bumped-version
  - Add bump config: feat→minor, breaking→major
  - Restrict workflow to main branch only
- add uv sync to update lockfile during release ([a5b5fdb](https://github.com/keisku/yfinance-mcp/commit/a5b5fdb67447bec49059a6b4e983927622ff60e9))

### Miscellaneous

- update uv.lock ([7eabb02](https://github.com/keisku/yfinance-mcp/commit/7eabb02070354939c3992c070627afcbdc742215))

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
