"""Pytest configuration for Yahoo Finance MCP tests."""

import os

import pytest

# Disable cache during tests so mocks work correctly
os.environ["YFINANCE_CACHE_DISABLED"] = "1"


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "crosscheck: cross-validation tests against pandas-ta (may be slow)",
    )


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Reset circuit breaker state before each test to prevent test pollution."""
    from yahoo_finance_mcp.server import reset_circuit_breaker_for_testing

    reset_circuit_breaker_for_testing()
    yield
    reset_circuit_breaker_for_testing()
