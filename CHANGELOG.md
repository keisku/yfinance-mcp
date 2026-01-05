# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
