"""Documents namespace: list, read, delete and restore documents (D91, D178, T12).

A document is addressed by its ``document_key``, the server's stable address
returned by ``memories.create`` and carried in every citation. A caller who
kept only its own ``document_id`` passes it with the same ``user_id``,
``customer_id`` or ``scope`` it was created with; the SDK then resolves it
through ``documents_lookup`` first, because a ``document_id`` is unique only
within a scope. When it names more than one document the lookup answers 409,
raised as :class:`AmbiguousDocumentError` listing each match's scope.

Usage:
    doc = await sdk.documents.get(result.document_key)
    versions = await sdk.documents.versions(document_id="msa-1", customer_id="acme")
    pdf = await sdk.documents.original(doc["document_key"], version=2)
    await sdk.documents.delete(document_id="msa-1", customer_id="acme")
    await sdk.documents.restore(doc["document_key"])
    tree = await sdk.documents.index(doc["document_key"], customer_id="acme")
    part = await sdk.documents.read(doc["document_key"], section_key="...", customer_id="acme")

Responses are returned as the server sends them (dicts), except
``original``, which returns the file's bytes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from urllib.parse import quote

from .memories.parts import compact_json
from .models.errors import InvalidInputError
from .utils.correlation import generate_correlation_id

if TYPE_CHECKING:
    from .sdk import MaximemSynapSDK


_ROOT = "/api/v1/documents"
_PAGES = re.compile(r"^(\d+)-(\d+)$")


def _scope_params(
    user_id: Optional[str], customer_id: Optional[str], scope: Optional[Dict[str, str]],
) -> Dict[str, str]:
    """The scope a document was created at, as query parameters.

    ``scope`` travels as a JSON string, as ``create_from_file`` sends it on its
    multipart route: a query string cannot carry a map.
    """
    params: Dict[str, str] = {}
    if user_id is not None:
        params["user_id"] = user_id
    if customer_id is not None:
        params["customer_id"] = customer_id
    if scope is not None:
        if not isinstance(scope, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in scope.items()
        ):
            raise InvalidInputError("scope must be a flat dict of level key to id, both strings.")
        params["scope"] = compact_json(scope)
    return params


class DocumentsInterface:
    """``sdk.documents``: the documents routes (tech spec, endpoints.yaml additions)."""

    def __init__(self, sdk: "MaximemSynapSDK"):
        self._sdk = sdk

    async def _auth(self):
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        return await self._sdk._get_auth_context(correlation_id), correlation_id

    async def _key(
        self,
        document_key: Optional[str],
        document_id: Optional[str],
        user_id: Optional[str],
        customer_id: Optional[str],
        scope: Optional[Dict[str, str]],
    ) -> str:
        """The ``document_key`` to address, looking a caller id up when needed."""
        if (document_key is None) == (document_id is None):
            raise InvalidInputError("Pass exactly one of document_key or document_id")
        if document_key is not None:
            if not isinstance(document_key, str) or not document_key:
                raise InvalidInputError("document_key must be a non-empty string")
            if user_id is not None or customer_id is not None or scope is not None:
                raise InvalidInputError(
                    "user_id, customer_id and scope resolve a document_id; "
                    "a document_key already names one document"
                )
            return document_key
        found = await self.lookup(
            document_id, user_id=user_id, customer_id=customer_id, scope=scope,
        )
        key = found.get("document_key") if isinstance(found, dict) else None
        if not key:
            raise InvalidInputError("documents_lookup returned no document_key")
        return key

    async def lookup(
        self,
        document_id: str,
        *,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Resolve a scope-qualified ``document_id`` to its document and key.

        Raises:
            ContextNotFoundError: no such document in that scope (404).
            AmbiguousDocumentError: several documents match (409); ``matches``
                lists their scopes.
        """
        self._sdk._check_customer_id(customer_id, where="documents.lookup")
        if not isinstance(document_id, str) or not document_id:
            raise InvalidInputError("document_id must be a non-empty string")
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.get(
            path=f"{_ROOT}/lookup",
            auth_context=auth,
            params={"document_id": document_id,
                    **_scope_params(user_id, customer_id, scope)},
            correlation_id=correlation_id,
        )

    async def list(
        self,
        *,
        document_type: Optional[str] = None,
        source_system: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """One page of documents. Each item carries its ``document_key`` and
        ``document_id``; pass the response's cursor for the next page. Only the
        filters given are sent."""
        self._sdk._check_customer_id(customer_id, where="documents.list")
        params: Dict[str, str] = {}
        if document_type is not None:
            params["document_type"] = document_type
        if source_system is not None:
            params["source_system"] = source_system
        params.update(_scope_params(user_id, customer_id, scope))
        if cursor is not None:
            params["cursor"] = cursor
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.get(
            path=_ROOT, auth_context=auth, params=params, correlation_id=correlation_id,
        )

    async def get(
        self,
        document_key: Optional[str] = None,
        *,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """The document's current version, status and ``processing_stage``."""
        self._sdk._check_customer_id(customer_id, where="documents.get")
        key = await self._key(document_key, document_id, user_id, customer_id, scope)
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.get(
            path=f"{_ROOT}/{quote(key, safe='')}", auth_context=auth,
            correlation_id=correlation_id,
        )

    async def versions(
        self,
        document_key: Optional[str] = None,
        *,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Every version of the document (D91)."""
        self._sdk._check_customer_id(customer_id, where="documents.versions")
        key = await self._key(document_key, document_id, user_id, customer_id, scope)
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.get(
            path=f"{_ROOT}/{quote(key, safe='')}/versions", auth_context=auth,
            correlation_id=correlation_id,
        )

    async def original(
        self,
        document_key: Optional[str] = None,
        *,
        version: int,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """The original file of one version, as bytes.

        Answered only when the document is visible to the caller; otherwise
        404, raised as ContextNotFoundError (D93).
        """
        self._sdk._check_customer_id(customer_id, where="documents.original")
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise InvalidInputError("version must be a positive integer")
        key = await self._key(document_key, document_id, user_id, customer_id, scope)
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.request(
            method="GET",
            path=f"{_ROOT}/{quote(key, safe='')}/versions/{version}/original",
            auth_context=auth,
            correlation_id=correlation_id,
            raw=True,
        )

    async def delete(
        self,
        document_key: Optional[str] = None,
        *,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
        permanent: bool = False,
    ) -> Dict[str, Any]:
        """Delete the document and everything made from it (D63).

        Soft by default: hidden from every read at once and restorable with
        ``restore`` until the instance's purge date (D177). ``permanent=True``
        removes it at once, with no undo, and needs a key with management
        access (D178).
        """
        self._sdk._check_customer_id(customer_id, where="documents.delete")
        if not isinstance(permanent, bool):
            raise InvalidInputError("permanent must be True or False")
        key = await self._key(document_key, document_id, user_id, customer_id, scope)
        auth, correlation_id = await self._auth()
        kwargs: Dict[str, Any] = {}
        if permanent:
            kwargs["params"] = {"permanent": "true"}
        return await self._sdk._http_transport.delete(
            path=f"{_ROOT}/{quote(key, safe='')}", auth_context=auth,
            correlation_id=correlation_id, **kwargs,
        )

    async def restore(
        self,
        document_key: Optional[str] = None,
        *,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Undo a soft delete before the purge date (D178).

        Raises DocumentIdInUseError (409) when a newer document now holds the
        same identity, and ContextNotFoundError (404) after the purge date.
        """
        self._sdk._check_customer_id(customer_id, where="documents.restore")
        key = await self._key(document_key, document_id, user_id, customer_id, scope)
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.post(
            path=f"{_ROOT}/{quote(key, safe='')}/restore", auth_context=auth,
            correlation_id=correlation_id,
        )

    async def _nav_key(
        self,
        document_key: Optional[str],
        document_id: Optional[str],
        user_id: Optional[str],
        customer_id: Optional[str],
        scope: Optional[Dict[str, str]],
    ) -> str:
        """The ``document_key`` for ``index`` and ``read``. Unlike the other
        methods, these send the scope with a key too: navigation reads under
        fetch's scope rule (D175), so a customer's document opens only for a call
        that names that customer."""
        if (document_key is None) == (document_id is None):
            raise InvalidInputError("Pass exactly one of document_key or document_id")
        if document_key is not None:
            if not isinstance(document_key, str) or not document_key:
                raise InvalidInputError("document_key must be a non-empty string")
            return document_key
        return await self._key(None, document_id, user_id, customer_id, scope)

    async def index(
        self,
        document_key: Optional[str] = None,
        *,
        version: Optional[int] = None,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """A document's title, summary and section tree (``section_key``,
        ``heading``, ``level``, ``page_start``, ``page_end``, ``summary``), of the
        current version or of ``version`` (D166).

        ``user_id``, ``customer_id`` and ``scope`` are the ones the document is read
        at, as in ``context.fetch``. A document the caller cannot see is a 404,
        raised as ContextNotFoundError (D93). Behind ``SYNAP_DOCS_NAVIGATION``
        (404 while off); Pro plan and above.
        """
        self._sdk._check_customer_id(customer_id, where="documents.index")
        if version is not None and (
            isinstance(version, bool) or not isinstance(version, int) or version < 1
        ):
            raise InvalidInputError("version must be a positive integer")
        params = _scope_params(user_id, customer_id, scope)
        key = await self._nav_key(document_key, document_id, user_id, customer_id, scope)
        if version is not None:
            params = {"version": str(version), **params}
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.get(
            path=f"{_ROOT}/{quote(key, safe='')}/index", auth_context=auth,
            params=params, correlation_id=correlation_id,
        )

    async def read(
        self,
        document_key: Optional[str] = None,
        *,
        section_key: Optional[str] = None,
        pages: Optional[Union[str, Tuple[int, int]]] = None,
        document_id: Optional[str] = None,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        scope: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """The text of one section or one page range of the current version, with
        its ``citation`` (D166). Pass exactly one of ``section_key`` (from
        ``index``) or ``pages`` (``"3-5"`` or ``(3, 5)``).

        At most 20 pages or 8,000 tokens per call: more raises InvalidInputError
        (422 ``read_too_large``) with ``code`` set. A document still processing
        raises ContextNotFoundError with ``code == "document_not_ready"``. Behind
        ``SYNAP_DOCS_NAVIGATION`` (404 while off); Pro plan and above.
        """
        self._sdk._check_customer_id(customer_id, where="documents.read")
        if (section_key is None) == (pages is None):
            raise InvalidInputError("Pass exactly one of section_key or pages")
        params = _scope_params(user_id, customer_id, scope)
        if section_key is not None:
            if not isinstance(section_key, str) or not section_key:
                raise InvalidInputError("section_key must be a non-empty string")
            params = {"section_key": section_key, **params}
        else:
            if isinstance(pages, (tuple, list)):
                if len(pages) != 2 or not all(
                    isinstance(n, int) and not isinstance(n, bool) for n in pages
                ):
                    raise InvalidInputError("pages is (from, to), two whole page numbers")
                pages = f"{pages[0]}-{pages[1]}"
            if not isinstance(pages, str):
                raise InvalidInputError("pages is 'from-to' or a (from, to) pair")
            match = _PAGES.match(pages.strip())
            if match is None or int(match.group(1)) < 1 or int(match.group(2)) < int(match.group(1)):
                raise InvalidInputError("pages is from-to, whole page numbers from 1, e.g. 3-5")
            params = {"pages": pages.strip(), **params}
        key = await self._nav_key(document_key, document_id, user_id, customer_id, scope)
        auth, correlation_id = await self._auth()
        return await self._sdk._http_transport.get(
            path=f"{_ROOT}/{quote(key, safe='')}/read", auth_context=auth,
            params=params, correlation_id=correlation_id,
        )
