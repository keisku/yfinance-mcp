import pytest

from yfinance_mcp.errors import ValidationError
from yfinance_mcp.helpers import INTERVAL_ORDER, select_interval


@pytest.mark.parametrize("interval", ["5m", "15m", "30m", "1h", "1d", "1wk"])
def test_select_interval_explicit_returns_requested_when_within_range(interval: str) -> None:
    # Use a short window that stays within all intraday max_days limits.
    out = select_interval(period="1w", interval=interval, symbol="QQQ", exchange="NMS")
    assert out == interval


def test_select_interval_auto_returns_supported_value() -> None:
    out = select_interval(period="1w", interval="auto", symbol="QQQ", exchange="NMS")
    assert out in INTERVAL_ORDER


@pytest.mark.parametrize(
    "interval,period",
    [
        pytest.param("5m", "1y", id="5m_too_long"),
        pytest.param("15m", "1y", id="15m_too_long"),
        pytest.param("30m", "1y", id="30m_too_long"),
    ],
)
def test_select_interval_explicit_raises_when_period_exceeds_max_days(
    interval: str, period: str
) -> None:
    with pytest.raises(ValidationError, match=r"supports up to"):
        select_interval(period=period, interval=interval, symbol="QQQ", exchange="NMS")


@pytest.mark.parametrize(
    "interval,start,end",
    [
        pytest.param("30m", "2024-01-01", "2024-04-01", id="30m_date_range_too_long"),
        pytest.param("5m", "2024-01-01", "2024-04-01", id="5m_date_range_too_long"),
    ],
)
def test_select_interval_explicit_raises_when_date_range_exceeds_max_days(
    interval: str, start: str, end: str
) -> None:
    with pytest.raises(ValidationError, match=r"supports up to"):
        select_interval(start=start, end=end, interval=interval, symbol="QQQ", exchange="NMS")
