"""Models for `sdk.data.delete`."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

DATA_SCOPES = ("user", "conversation", "customer")


class DeleteDataResult(BaseModel):
    """Result of `sdk.data.delete` / `sdk.data.status`.

    Check ``fully_removed`` before telling an end user their data is gone.
    ``status == "partial"`` means Postgres is clear but a store named in
    ``still_present_in`` could not be cleared. That is a result, not an
    exception: the rest of the data really was deleted.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    deletion_id: Optional[str] = None
    # complete | pending_purge | partial | in_progress | dry_run
    # pending_purge: hidden from every read now, restorable with
    # `sdk.data.restore` until `purge_after`, then removed for good.
    status: str = "unknown"
    scope: Optional[str] = None
    dry_run: bool = False
    found: bool = False
    # Rows / documents / nodes removed per store (would remove, for a dry run).
    counts: Dict[str, Any] = Field(default_factory=dict)
    stores: Dict[str, Any] = Field(default_factory=dict)
    still_present_in: List[str] = Field(default_factory=list)
    fully_removed: bool = False  # pessimistic: true only when the server said so
    # True when the server's size cap stopped the call. `wait=True` repeats the
    # call for you; with `wait=False` call again with the same arguments.
    continue_: bool = Field(default=False, alias="continue")
    kept: List[str] = Field(default_factory=list)
    # Stores this delete does not reach at all, in plain words. ``complete``
    # means every store the delete targets was cleared, not that nothing is
    # left in these. Empty from servers that predate the field.
    not_covered: List[str] = Field(default_factory=list)
    note: Optional[str] = None
    # Conversation scope keeps the memories learned from the conversation.
    memories_kept: bool = False
    audit: Optional[str] = None
    # What the SDK cleared on this machine after the server finished.
    local_cleared: List[str] = Field(default_factory=list)
    # Only for a call with `instances`: one result per instance (each the shape
    # of a one-instance result, plus `instance_id`; a refused instance has
    # `status="refused"` and `error`), the instances the time budget did not
    # reach (the next call does them), and every audit id.
    instances: List[Dict[str, Any]] = Field(default_factory=list)
    not_started: List[str] = Field(default_factory=list)
    deletion_ids: List[str] = Field(default_factory=list)
    # With status "pending_purge": when the deleted memories are purged for
    # good (ISO 8601, UTC). Until then they can be restored.
    purge_after: Optional[str] = None


MAX_RESTORE_IDS = 200


class RestoreResult(BaseModel):
    """Result of `sdk.data.restore`.

    ``restored_ids`` are live again and served by the next fetch.
    ``not_restorable_ids`` were not found in this instance, were not deleted,
    or are past their purge date. ``still_missing_in`` names a store that could
    not be brought back in line; the memory is restored either way.
    """

    model_config = ConfigDict(extra="allow")

    restored_ids: List[str] = Field(default_factory=list)
    not_restorable_ids: List[str] = Field(default_factory=list)
    still_missing_in: List[str] = Field(default_factory=list)
    reindexed_ids: List[str] = Field(default_factory=list)
    fully_restored: bool = False
