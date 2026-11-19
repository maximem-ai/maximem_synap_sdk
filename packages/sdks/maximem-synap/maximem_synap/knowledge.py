"""Knowledge namespace: describe and query the facts and entities extracted from
documents (D165, D167).

``describe`` orients: sources, document counts by type, and the instance
ontology's entity and fact types with their typed properties and relations.
``query`` returns complete, filtered sets, a count, or counts grouped by a
property, never a ranked top-k. Callers send a structured request, never query
text. Every response carries ``total``, ``complete`` and a ``coverage`` report
(documents in scope that are not yet processed or only lightly extracted), so an
agent can say "14 found; 3 contracts still processing".

Usage:
    schema = await sdk.knowledge.describe(customer_id="acme-corp")
    page = await sdk.knowledge.query(
        target="Obligation",
        where={"due_at": {"before": "2027-03-01"}, "status": "open"},
        related={"COUNTERPARTY": {"type": "Organization", "name": "Acme"}},
        customer_id="acme-corp",
    )
    n = await sdk.knowledge.query(target="Obligation", aggregate={"count": True})

Responses are returned as the server sends them (dicts). The routes are behind
the ``SYNAP_KNOWLEDGE_QUERY`` flag (404 while off) and need a Pro plan or above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .documents import _scope_params
from .models.errors import InvalidInputError
from .utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from .sdk import MaximemSynapSDK


_ROOT = "/api/v1/knowledge"
MAX_LIMIT = 500


class KnowledgeInterface:
    """``sdk.knowledge``: the knowledge routes (tech spec, endpoints.yaml additions)."""

    def __init__(self, sdk: "MaximemSynapSDK"):
        self._sdk = sdk

    async def _auth(self):
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        return await self._sdk._get_auth_context(correlation_id), correlation_id

    async def describe(
        self,
        *,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Sources, document counts by type, and the ontology's entity and fact
        types with their typed properties and relations, as the caller's scope
        sees them. Call this first."""
        self._sdk._check_customer_id(customer_id, where="knowledge.describe")
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.get(
            path=f"{_ROOT}/describe", auth_context=auth,
            params=_scope_params(user_id, customer_id, scope),
            correlation_id=correlation_id,
        )

    async def query(
        self,
        target: str,
        *,
        where: Optional[Dict[str, Any]] = None,
        related: Optional[Dict[str, Any]] = None,
        as_of: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        aggregate: Optional[Dict[str, Any]] = None,
        sort: Optional[List[Dict[str, str]]] = None,
        limit: Optional[int] = None,
        cursor: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """A structured query: filter, traverse (at most 2 hops) or aggregate.

        Args:
            target: an entity or fact type from ``describe``, e.g. ``"Obligation"``.
            where: property to a value or ``{eq|ne|in|before|after|between|gt|lt|
                contains|exists: ...}``.
            related: relation to a filter on the node at the other end; nesting
                is a further hop.
            as_of: an ISO date or datetime; current versions only when unset.
            document_ids: only facts from these documents, as ``context.fetch``.
            aggregate: ``{"count": True}`` or ``{"group_by": "<property>"}``.
            sort: ``[{"property": ..., "order": "asc" | "desc"}]``.
            limit: rows per page, at most 500 (the server's default is 100).
            cursor: the previous response's ``next_cursor``.

        Returns:
            ``items``, ``total``, ``complete`` (False when the 10,000-row cap cut
            the set), ``coverage``, ``next_cursor`` and, with ``group_by``,
            ``groups``.

        Raises:
            InvalidInputError: 422 ``unknown_type``, ``unknown_property``,
                ``bad_condition``, ``too_many_hops``, ``unknown_relation``,
                ``invalid_cursor`` or ``query_too_broad``.
        """
        self._sdk._check_customer_id(customer_id, where="knowledge.query")
        if not isinstance(target, str) or not target:
            raise InvalidInputError("target must be a non-empty string")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIMIT
        ):
            raise InvalidInputError(f"limit must be a whole number from 1 to {MAX_LIMIT}")
        body: Dict[str, Any] = {"target": target}
        for key, value in (
            ("where", where), ("related", related), ("as_of", as_of),
            ("document_ids", document_ids), ("aggregate", aggregate), ("sort", sort),
            ("limit", limit), ("cursor", cursor),
        ):
            if value is not None:
                body[key] = value
        if user_id is not None:
            body["user_id"] = user_id
        if customer_id is not None:
            body["customer_id"] = customer_id
        if scope is not None:
            if not isinstance(scope, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in scope.items()
            ):
                raise InvalidInputError("scope must be a flat dict of level key to id, both strings.")
            body["scope"] = scope
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.post(
            path=f"{_ROOT}/query", auth_context=auth, json=body, correlation_id=correlation_id,
        )
