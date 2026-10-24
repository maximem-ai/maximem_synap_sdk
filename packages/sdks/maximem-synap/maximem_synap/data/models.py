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
    # complete | partial | in_progress | dry_run
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
    note: Optional[str] = None
    # Conversation scope keeps the memories learned from the conversation.
    memories_kept: bool = False
    audit: Optional[str] = None
    # What the SDK cleared on this machine after the server finished.
    local_cleared: List[str] = Field(default_factory=list)
