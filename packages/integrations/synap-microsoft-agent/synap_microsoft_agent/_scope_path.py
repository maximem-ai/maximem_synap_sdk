"""Validation of a deeper-ladder ``scope_path`` at construction."""

from __future__ import annotations

from typing import Dict, Mapping, Optional


def checked_scope_path(
    scope_path: Optional[Mapping[str, str]], where: str
) -> Optional[Dict[str, str]]:
    """Return a copy of ``scope_path``, or ``None`` when it is unset or empty.

    Raises ``ValueError`` at once (not later, from a method that feeds the
    system prompt) when it is not a mapping, or when a level key or value is
    not a non-empty string, or a value contains ``/`` (it would make the
    ``/a/b/c`` owner label ambiguous).
    """
    if not scope_path:
        return None
    if not isinstance(scope_path, Mapping):
        raise ValueError(
            f"{where}: scope_path must map level keys to values, got {scope_path!r}")
    for key, value in scope_path.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                f"{where}: scope_path level keys must be non-empty strings, got {key!r}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{where}: scope_path value for level {key!r} must be a non-empty "
                f"string, got {value!r}")
        if "/" in value:
            raise ValueError(
                f"{where}: scope_path value for level {key!r} must not contain '/', "
                f"got {value!r}")
    return dict(scope_path)
