from yfinance_mcp.helpers import configure_logging


def test_yfinance_log_file_disabled_disables_file_logging(monkeypatch):
    monkeypatch.setenv("YFINANCE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("YFINANCE_LOG_CONSOLE", "0")
    monkeypatch.setenv("YFINANCE_LOG_FILE", "disabled")

    logger = configure_logging()

    # root handlers should not include RotatingFileHandler when disabled
    handler_names = {type(h).__name__ for h in logger.root.handlers}
    assert "RotatingFileHandler" not in handler_names


def test_yfinance_log_file_empty_disables_file_logging(monkeypatch):
    monkeypatch.setenv("YFINANCE_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("YFINANCE_LOG_CONSOLE", "0")
    monkeypatch.setenv("YFINANCE_LOG_FILE", "")

    logger = configure_logging()

    handler_names = {type(h).__name__ for h in logger.root.handlers}
    assert "RotatingFileHandler" not in handler_names
