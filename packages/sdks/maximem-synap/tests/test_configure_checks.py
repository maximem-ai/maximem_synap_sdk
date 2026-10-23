"""configure() refuses bad values with InvalidInputError, and says when it
ignores a setting.

Found reading configure() against the docs (0.5.20): an out-of-range value
raised pydantic's ValidationError, an unknown log_level silently became
WARNING, any non-empty cache_backend meant SQLite, and an unknown setting (a
typo) vanished without a word.
"""
import logging

import pytest

from maximem_synap import InvalidInputError, MaximemSynapSDK


def _sdk():
    return MaximemSynapSDK(api_key="synap_test", _force_new=True)


@pytest.mark.parametrize("kwargs, message", [
    ({"session_timeout_minutes": 2}, "Invalid configuration"),
    ({"agent_context_window": "big"}, "Invalid configuration"),
    ({"log_level": "LOUD"}, "Invalid log_level"),
    ({"cache_backend": "redis"}, "Invalid cache_backend"),
])
def test_a_bad_value_is_an_invalid_input_error_and_changes_nothing(kwargs, message):
    sdk = _sdk()
    before = sdk._config.model_dump()
    with pytest.raises(InvalidInputError, match=message):
        sdk.configure(**kwargs)
    assert sdk._config.model_dump() == before


@pytest.mark.parametrize("kwargs", [
    {"log_level": "info"},
    {"log_level": "DEBUG"},
    {"cache_backend": None},
    {"cache_backend": "sqlite"},
    {"session_timeout_minutes": 60},
    {"agent_model": "gpt-4o", "agent_context_window": 128000},
    {"history_search_model": None},
])
def test_valid_values_are_applied(kwargs):
    sdk = _sdk()
    sdk.configure(**kwargs)
    for k, v in kwargs.items():
        assert getattr(sdk._config, k) == v


def test_an_unknown_setting_is_reported_not_raised(caplog):
    sdk = _sdk()
    with caplog.at_level(logging.WARNING):
        sdk.configure(log_levle="DEBUG")
    assert "log_levle" in caplog.text


def test_configure_after_initialize_is_refused():
    sdk = _sdk()
    sdk._initialized = True
    with pytest.raises(InvalidInputError, match="after initialization"):
        sdk.configure(log_level="INFO")
