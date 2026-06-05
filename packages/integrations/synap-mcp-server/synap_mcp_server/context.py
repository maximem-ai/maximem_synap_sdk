"""Per-request Bearer token, carried via a ContextVar.

The token differs on every request and must never be stored on the server. The HTTP
middleware (server.py) stashes it here at the start of each request; the REST client
(client.py) reads it when building outbound calls. ContextVar keeps it isolated across
concurrent async tasks, so two simultaneous requests with different keys never collide.
"""

from contextvars import ContextVar
from typing import Optional

_current_token: ContextVar[Optional[str]] = ContextVar("synap_bearer_token", default=None)


def set_token(token: Optional[str]) -> None:
    _current_token.set(token)


def get_token() -> Optional[str]:
    return _current_token.get()


class MissingTokenError(Exception):
    """Raised when a tool runs without a forwarded Bearer token."""
