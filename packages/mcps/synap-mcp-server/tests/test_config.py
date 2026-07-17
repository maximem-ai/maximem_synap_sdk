"""Regression tests for config handling."""

import importlib
import logging


def test_lowercase_log_level_is_usable(monkeypatch):
    """Regression: docker-compose.staging sets LOG_LEVEL=debug (lowercase), but
    logging.basicConfig/setLevel reject lowercase level names. server.py must
    upper-case the value before handing it to logging."""
    monkeypatch.setenv("LOG_LEVEL", "debug")
    from synap_mcp_server import config

    importlib.reload(config)
    level = config.settings.log_level.upper()
    # _checkLevel raises ValueError on an unknown level; getLevelName returns the int.
    assert isinstance(logging.getLevelName(level), int)
    logging.getLogger("regression").setLevel(level)  # must not raise
