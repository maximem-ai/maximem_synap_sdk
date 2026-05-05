"""Datetime parsing utilities."""

from datetime import datetime
from typing import Optional


def parse_iso_datetime(value) -> Optional[datetime]:
    """Parse an ISO 8601 string to datetime.

    Handles the trailing Z timezone designator, which fromisoformat()
    only supports natively from Python 3.11 onwards.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
