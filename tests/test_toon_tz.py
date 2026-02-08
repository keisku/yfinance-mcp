import pandas as pd

from yfinance_mcp.helpers import fmt_toon


def test_fmt_toon_converts_tz_aware_index_to_declared_tz() -> None:
    # 14:30 UTC is 09:30 in New York during standard time.
    idx = pd.DatetimeIndex([pd.Timestamp("2026-02-02T14:30:00Z")])
    df = pd.DataFrame({"c": [1.0]}, index=idx)

    out = fmt_toon(df, wrapper_key="bars", tz="America/New_York", interval="30m")

    assert "2026-02-02T09:30" in out
    assert "2026-02-02T14:30" not in out


def test_fmt_toon_converts_tz_naive_intraday_index_assuming_utc() -> None:
    # tz-naive but effectively UTC: should still render in local exchange time.
    idx = pd.DatetimeIndex([pd.Timestamp("2026-02-02T14:00:00")])
    df = pd.DataFrame({"c": [1.0]}, index=idx)

    out = fmt_toon(df, wrapper_key="bars", tz="America/New_York", interval="30m")

    # 14:00 UTC -> 09:00 NY (standard time)
    assert "2026-02-02T09:00" in out
    assert "2026-02-02T14:00" not in out
