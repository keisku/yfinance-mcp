import re

from yfinance_mcp.server import _handle_technicals


def _rows_count(toon_text: str) -> int:
    # toon-format prints rows like: "  rows[150]:" in the embedded text
    m = re.search(r"\brows\[(\d+)\]", toon_text)
    return int(m.group(1)) if m else -1


def test_technicals_downsample_toggle_never_reduces_when_disabled():
    # Use a longer window where interval selection may change.
    period = "2y"

    out_default = _handle_technicals(
        {"symbol": "QQQ", "period": period, "indicators": ["rsi", "macd", "bb"]}
    )
    n_default = _rows_count(out_default)
    assert 1 <= n_default <= 150  # cap when downsampling is enabled

    out_full = _handle_technicals(
        {
            "symbol": "QQQ",
            "period": period,
            "indicators": ["rsi", "macd", "bb"],
            "downsample": False,
        }
    )
    n_full = _rows_count(out_full)

    # Disabling downsample should never reduce the number of returned rows.
    assert n_full >= n_default


def test_technicals_downsample_noop_when_small():
    period = "6mo"  # usually <=150 daily bars

    out_default = _handle_technicals(
        {"symbol": "QQQ", "period": period, "indicators": ["rsi", "macd", "bb"]}
    )
    n_default = _rows_count(out_default)

    out_full = _handle_technicals(
        {
            "symbol": "QQQ",
            "period": period,
            "indicators": ["rsi", "macd", "bb"],
            "downsample": False,
        }
    )
    n_full = _rows_count(out_full)

    assert n_default == n_full
