"""Local mode: talking to the open-source Synap server, which has no auth.

It is on when no API key is given (argument or SYNAP_API_KEY) and a base URL
is configured. A key always wins, so Cloud callers never see a difference.
"""

import os
from typing import Optional


def is_local_mode(api_key: Optional[str], base_url: Optional[str]) -> bool:
    """True when there is no API key and a base URL is set."""
    if api_key or os.environ.get("SYNAP_API_KEY"):
        return False
    return bool(base_url and base_url.strip())


def local_token() -> str:
    """The optional shared token the open-source server may require.

    Read from the environment only. Empty means send no Authorization header.
    """
    return os.environ.get("SYNAP_API_TOKEN", "").strip()
