"""Scope path construction for Synap integrations.

Synap's scope chain narrows from WORLD -> CLIENT -> CUSTOMER -> USER. When
an integration caches a record locally (CrewAI) or annotates a framework
record with its origin scope (LlamaIndex, LangChain), it needs a consistent
path string.

Historically the integrations did this ad-hoc: llamaindex retriever built
one path, llamaindex memory built another; CrewAI defaulted to ``/user``
while LangChain left it blank. This module is the single source of truth.
"""

from __future__ import annotations

from typing import Optional


def default_scope(user_id: str, customer_id: Optional[str] = None) -> str:
    """Return the canonical default scope path for a record.

    - ``default_scope("alice")`` -> ``"/alice"``
    - ``default_scope("alice", "acme")`` -> ``"/acme/alice"``
    - ``default_scope("alice", "")`` -> ``"/alice"`` (empty string treated as absent)

    ``user_id`` is required; an empty ``user_id`` raises ``ValueError``
    because a record with no user scope is ambiguous in Synap's model.
    """
    if not user_id:
        raise ValueError("default_scope requires a non-empty user_id")
    if customer_id:
        return f"/{customer_id}/{user_id}"
    return f"/{user_id}"


__all__ = ["default_scope"]
