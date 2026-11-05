"""Interface for deleting data by scope."""

import time
from urllib.parse import quote
from typing import Any, Dict, List, Optional, Union

from ..models.errors import InvalidInputError
from ..utils.correlation import generate_correlation_id
from .models import DATA_SCOPES, MAX_RESTORE_IDS, DeleteDataResult, RestoreResult

# The most calls `wait=True` makes. The server deletes at most a fixed number of
# rows per call and asks to be called again; 50 calls is far beyond any real size.
MAX_WAIT_CALLS = 50

_ID_FIELD = {"user": "user_id", "customer": "customer_id", "conversation": "conversation_id"}


def build_delete_body(
    *,
    scope: Any,
    instance_id: str,
    user_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    dry_run: bool = False,
    confirm: Optional[str] = None,
    instances: Optional[Union[str, List[str]]] = None,
    permanent: bool = False,
) -> Dict[str, Any]:
    """Validate and build the request body. Raises before any request is sent.

    Without ``instances`` or ``permanent`` the body is byte-identical to earlier
    versions (``permanent`` is sent only when True; the server refuses unknown fields).
    """
    if not isinstance(permanent, bool):
        raise InvalidInputError("permanent must be a boolean")
    if isinstance(scope, dict):
        raise InvalidInputError(
            "scope must be one of: user, conversation, customer. A dict is not "
            "accepted here (the rung-path dict used by memories.create and fetch "
            "is a different parameter)."
        )
    if not isinstance(scope, str) or scope not in DATA_SCOPES:
        raise InvalidInputError(
            f"Invalid scope '{scope}'. Expected one of: {', '.join(DATA_SCOPES)}"
        )
    if not isinstance(dry_run, bool):
        raise InvalidInputError("dry_run must be a boolean")

    ids = {"user_id": user_id, "customer_id": customer_id, "conversation_id": conversation_id}
    for name, value in ids.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise InvalidInputError(f"{name} must not be empty")

    field = _ID_FIELD[scope]
    target = ids[field]
    if target is None:
        raise InvalidInputError(f"{field} is required for scope='{scope}'")

    if not dry_run:
        if confirm is None or confirm == "":
            raise InvalidInputError(
                f"confirm is required unless dry_run=True: pass the {field} being "
                "deleted as confirm"
            )
        if confirm != target:
            raise InvalidInputError(f"confirm must equal the {field} being deleted")

    if instances is not None:
        instances = _check_instances(instances)

    body = {
        "scope": scope,
        "instance_id": instance_id if instances is None else None,
        "user_id": user_id,
        "customer_id": customer_id,
        "conversation_id": conversation_id,
        "dry_run": dry_run,
        "confirm": confirm,
    }
    if instances is not None:
        body["instance_ids"] = instances
    if permanent:
        body["permanent"] = True
    return body


def _check_instances(instances: Any) -> Union[str, List[str]]:
    if instances == "all":
        return "all"
    if isinstance(instances, str) or not isinstance(instances, (list, tuple)):
        raise InvalidInputError(
            'instances must be "all" or a list of instance ids')
    out: List[str] = []
    for i in instances:
        if not isinstance(i, str) or not i.strip():
            raise InvalidInputError("instances must not contain an empty instance id")
        if i.strip() not in out:
            out.append(i.strip())
    if not out:
        raise InvalidInputError('instances is empty: name at least one instance, or "all"')
    return out


def build_restore_body(
    memory_ids: Any, deletion_id: Any, instance_id: str
) -> Dict[str, Any]:
    """Validate and build the restore body. Raises before any request is sent."""
    if (memory_ids is None) == (deletion_id is None):
        raise InvalidInputError("send exactly one of memory_ids or deletion_id")
    if deletion_id is not None:
        if not isinstance(deletion_id, str) or not deletion_id.strip():
            raise InvalidInputError("deletion_id must not be empty")
        return {"deletion_id": deletion_id.strip(), "instance_id": instance_id}
    if isinstance(memory_ids, str) or not isinstance(memory_ids, (list, tuple)):
        raise InvalidInputError("memory_ids must be a list of memory ids")
    ids = [str(i) for i in memory_ids]
    if not ids:
        raise InvalidInputError("memory_ids must not be empty")
    if len(ids) > MAX_RESTORE_IDS:
        raise InvalidInputError(f"at most {MAX_RESTORE_IDS} memory_ids per call")
    if any(not i.strip() for i in ids):
        raise InvalidInputError("memory_ids must not contain empty ids")
    return {"memory_ids": ids, "instance_id": instance_id}


class DataInterface:
    """
    Delete personal data by scope.

    Usage:
        # See what would go, change nothing
        preview = await sdk.data.delete(scope="user", user_id="user-123", dry_run=True)

        # Delete everything held about one user (B2B: also customer_id)
        result = await sdk.data.delete(
            scope="user", user_id="user-123", confirm="user-123",
        )
        if not result.fully_removed:
            print("still present in:", result.still_present_in)
    """

    def __init__(self, sdk: "MaximemSynapSDK"):
        self._sdk = sdk

    async def delete(
        self,
        scope: str,
        user_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        dry_run: bool = False,
        confirm: Optional[str] = None,
        wait: bool = True,
        instances: Optional[Union[str, List[str]]] = None,
        permanent: bool = False,
    ) -> DeleteDataResult:
        """
        Delete data for one user, one conversation or one customer.

        scope="conversation" removes that conversation's turns, summaries,
        telemetry and turn buffer ONLY. Memories learned from it are kept: use
        scope="user" or memories.delete to remove them. Data in another
        instance of the same client is not touched unless you ask for it with
        `instances`; stores that cannot be scoped by instance are skipped and
        named in `still_present_in`.

        scope="customer" needs a management key (management:* or *); a runtime
        key is refused with 403.

        Memories are soft-deleted by default: hidden from every read at once,
        restorable with ``restore(deletion_id=result.deletion_id)`` until
        ``result.purge_after`` (the instance's deletion period, 0 to 30 days,
        30 by default), then purged for good; ``status`` is
        ``"pending_purge"``. Conversations, profile and source text are removed
        at once and do not come back with a restore. ``permanent=True``
        removes the memories now too and needs a management-level key. Any
        valid Synap API key of the instance with permission to delete may call
        it; it can only delete inside its own instance and client.

        Args:
            scope: "user", "conversation" or "customer". Client-wide delete is
                not supported.
            user_id: Required for scope="user" and scope="conversation".
            customer_id: Required on B2B for all three scopes; rejected on B2C.
            conversation_id: Required for scope="conversation".
            dry_run: Count what would be removed and change nothing. No
                `confirm` needed.
            confirm: Required unless `dry_run`. Must equal the id of the scope
                being deleted (user_id, customer_id or conversation_id).
            wait: Repeat the call while the server says `continue_` (its size
                cap stopped the call), up to 50 calls. False returns after one.
            instances: Delete in several instances of the client at once:
                "all", or a list of instance ids. Needs a management key that
                is not bound to one instance (a key bound to an instance may
                only name its own). The result then carries one entry per
                instance in `instances`, and the top-level `status`,
                `fully_removed` and `still_present_in` cover all of them.
                Omit it to delete in this SDK's instance only, as before.
            permanent: Remove the memories now instead of after the deletion
                period. Needs a management-level key (``delete:permanent``,
                ``delete:*``, ``management:*`` or ``*``).

        Returns:
            DeleteDataResult. A store that could not be cleared is in
            `still_present_in` and `fully_removed` is False; it is not raised.

        Raises:
            InvalidInputError: bad arguments (caught before any request), or
                400 from the server (for example scope="customer" on B2C).
            AuthenticationError: 401, or 403 (key not allowed to delete, or
                not this instance's).
            ConflictError: 409 (id belongs to another customer, or a delete of
                the same target is already running).
            RateLimitError: 429.
        """
        self._sdk._check_customer_id(customer_id, where="data.delete")
        self._sdk._ensure_initialized()
        body = build_delete_body(
            scope=scope,
            instance_id=self._sdk.instance_id,
            user_id=user_id,
            customer_id=customer_id,
            conversation_id=conversation_id,
            dry_run=dry_run,
            confirm=confirm,
            instances=instances,
            permanent=permanent,
        )

        result: Optional[DeleteDataResult] = None
        for _ in range(MAX_WAIT_CALLS if wait else 1):
            correlation_id = generate_correlation_id(self._sdk.instance_id)
            auth_context = await self._sdk._get_auth_context(correlation_id)
            raw = await self._sdk._http_transport.post(
                path="/v1/data/delete",
                auth_context=auth_context,
                json=body,
                correlation_id=correlation_id,
            )
            result = DeleteDataResult(**raw)
            if dry_run or not result.continue_:
                break

        assert result is not None
        if not dry_run and result.status in (
            "complete", "pending_purge", "partial", "in_progress"
        ):
            result.local_cleared = self._sdk._evict_deleted(
                scope, user_id, customer_id, conversation_id
            )
        return result

    async def restore(
        self,
        memory_ids: Optional[List[str]] = None,
        deletion_id: Optional[str] = None,
    ) -> RestoreResult:
        """
        Bring deleted memories back, until their purge date.

        A deleted memory stays hidden and restorable for the instance's
        deletion period (0 to 30 days, 30 by default) and is then purged for
        good. Restoring needs the same key permissions as delete, and only
        reaches memories of this instance.

        Args:
            memory_ids: Up to 200 memory ids.
            deletion_id: Instead of ids, every memory one ``data.delete`` call
                hid (its result's ``deletion_id``). Send exactly one of the two.

        Returns:
            RestoreResult. Ids that could not be restored (not found, not
            deleted, or past the purge date) are in `not_restorable_ids`; that
            is a result, not an exception.

        Raises:
            InvalidInputError: bad arguments (caught before any request).
            AuthenticationError: 401, or 403 (key not allowed).
        """
        body = build_restore_body(memory_ids, deletion_id, self._sdk.instance_id)
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        auth_context = await self._sdk._get_auth_context(correlation_id)
        raw = await self._sdk._http_transport.post(
            path="/v1/data/restore",
            auth_context=auth_context,
            json=body,
            correlation_id=correlation_id,
        )
        return RestoreResult(**raw)

    async def status(self, deletion_id: str) -> DeleteDataResult:
        """Read the audit record of a delete (status, counts, still_present_in)."""
        if not isinstance(deletion_id, str) or not deletion_id.strip():
            raise InvalidInputError("deletion_id is required")
        self._sdk._ensure_initialized()
        correlation_id = generate_correlation_id(self._sdk.instance_id)
        auth_context = await self._sdk._get_auth_context(correlation_id)
        raw = await self._sdk._http_transport.get(
            path=f"/v1/data/delete/{quote(deletion_id.strip(), safe='')}",
            auth_context=auth_context,
            params={"instance_id": self._sdk.instance_id},
            correlation_id=correlation_id,
        )
        return DeleteDataResult(**raw)
